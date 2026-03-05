#!/usr/bin/env python3
"""
Degradation experiment for DiffusionRenderer.

Two experiments:
  iterative - Round-trip inverse+forward rendering N times, measuring quality decay
  control   - Same G-buffers re-rendered with cycling env maps, showing stable quality

Usage:
  # Experiment 1: iterative degradation
  python run_degradation_experiment.py iterative \
      --data_dir source_data_polyhaven \
      --workspace degradation_workspace \
      --scenes all_purpose_cleaner_env_0 \
      --n_iterations 50

  # Experiment 2: control (same buffer, cycling envs)
  python run_degradation_experiment.py control \
      --data_dir source_data_polyhaven \
      --workspace degradation_workspace \
      --object all_purpose_cleaner \
      --source_variant env_0 \
      --n_iterations 50
"""

import os
import sys
import json
import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import imageio
import torch

# ---------------------------------------------------------------------------
# Reusable helpers from run_polyhaven_experiment.py
# ---------------------------------------------------------------------------


def recover_hdr_from_pngs(ldr_png, hdr_png):
    """Approximate linear HDR from preprocessed _ldr.png / _hdr.png pair."""
    hdr_clipped = np.power(np.clip(ldr_png, 1e-6, 1.0), 2.2)
    mask = (hdr_clipped > 0.02) & (hdr_clipped < 0.9) & (hdr_png > 0.02)
    if mask.any():
        log_vals = np.log1p(10.0 * hdr_clipped[mask])
        M = float(np.median(log_vals / hdr_png[mask]))
    else:
        M = float(np.log1p(10.0))
    M = max(M, 1.0)
    hdr_recovered = np.expm1(hdr_png * M) / 10.0
    return np.clip(hdr_recovered, 0.0, 65504.0)


def apply_model_hdr_mapping(hdr_np, device):
    """Model-expected tone mapping (Reinhard + sRGB, log1p/10000 + sRGB)."""
    import src.data.rendering_utils as util
    hdr_t = torch.from_numpy(hdr_np).float().to(device)
    env_ldr = util.rgb2srgb(util.reinhard(hdr_t, max_point=16).clamp(0, 1))
    env_log = util.rgb2srgb((torch.log1p(hdr_t) / np.log1p(10000)).clamp(0, 1))
    return env_ldr.cpu(), env_log.cpu()


def composite_on_white(rgb, alpha, bg_value=255):
    """Composite RGB uint8 onto white using alpha [0-255]."""
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    a = np.clip(alpha.astype(np.float32) / 255.0, 0, 1)[:, :, np.newaxis]
    rgb_f = rgb.astype(np.float32) / 255.0
    bg = bg_value / 255.0
    return ((rgb_f * a + bg * (1 - a)).clip(0, 1) * 255).astype(np.uint8)


def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(img1, img2):
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, channel_axis=2, data_range=255)


def compute_lpips_val(pred, gt, lpips_fn, device):
    pred_t = torch.from_numpy(pred).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    gt_t = torch.from_numpy(gt).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    pred_t = pred_t.to(device) * 2 - 1
    gt_t = gt_t.to(device) * 2 - 1
    with torch.no_grad():
        return lpips_fn(pred_t, gt_t).item()


def touch(path):
    import io
    with io.open(path, "ab"):
        os.utime(path, None)


# ---------------------------------------------------------------------------
# Env-map loading
# ---------------------------------------------------------------------------


def load_envmaps_for_frames(envmap_dir, frame_ids, target_h, target_w, device):
    """Load per-frame env maps and apply model tone mapping.

    Returns env_ldr, env_log tensors of shape (1, F, C, H, W) in [0,1].
    """
    from src.data.rendering_utils import envmap_vec

    env_ldr_list, env_log_list = [], []
    for fid in frame_ids:
        ldr_path = os.path.join(envmap_dir, f"{fid}_ldr.png")
        hdr_path = os.path.join(envmap_dir, f"{fid}_hdr.png")
        if not os.path.exists(ldr_path) or not os.path.exists(hdr_path):
            raise FileNotFoundError(f"Missing env map for frame {fid}")

        ldr_img = Image.open(ldr_path).convert("RGB").resize((target_w, target_h), Image.BILINEAR)
        hdr_img = Image.open(hdr_path).convert("RGB").resize((target_w, target_h), Image.BILINEAR)
        ldr_np = np.asarray(ldr_img).astype(np.float32) / 255.0
        hdr_np = np.asarray(hdr_img).astype(np.float32) / 255.0

        hdr_recovered = recover_hdr_from_pngs(ldr_np, hdr_np)
        e_ldr, e_log = apply_model_hdr_mapping(hdr_recovered, device)
        env_ldr_list.append(e_ldr)
        env_log_list.append(e_log)

    env_ldr = torch.stack(env_ldr_list).unsqueeze(0).permute(0, 1, 4, 2, 3).to(device)
    env_log = torch.stack(env_log_list).unsqueeze(0).permute(0, 1, 4, 2, 3).to(device)

    env_nrm = envmap_vec((target_h, target_w), device=device) * 0.5 + 0.5
    env_nrm = env_nrm.unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 2, 3).expand_as(env_ldr)

    return env_ldr, env_log, env_nrm


# ---------------------------------------------------------------------------
# Pipeline loaders
# ---------------------------------------------------------------------------


def load_inverse_pipeline(config_path, device, weight_dtype=torch.float16):
    """Load the RGB->X (inverse rendering / delighting) pipeline."""
    from omegaconf import OmegaConf
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from src.pipelines.pipeline_rgbx import RGBXVideoDiffusionPipeline

    cfg = OmegaConf.load(config_path)
    cond_mode = cfg.get("cond_mode", "skip")
    use_det = cfg.get("use_deterministic_mode", False)

    missing_kwargs = {"cond_mode": cond_mode, "use_deterministic_mode": use_det}
    model_subs = os.listdir(cfg.inference_model_weights) if os.path.exists(cfg.inference_model_weights) else []
    if "image_encoder" not in model_subs:
        missing_kwargs["image_encoder"] = CLIPVisionModelWithProjection.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="image_encoder")
    if "feature_extractor" not in model_subs:
        missing_kwargs["feature_extractor"] = CLIPImageProcessor.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="feature_extractor")

    pipeline = RGBXVideoDiffusionPipeline.from_pretrained(cfg.inference_model_weights, **missing_kwargs)
    pipeline = pipeline.to(device).to(weight_dtype)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline, cfg


def load_forward_pipeline(config_path, device, weight_dtype=torch.float16):
    """Load the X->RGB (forward rendering / relighting) pipeline."""
    from omegaconf import OmegaConf
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
    from src.models.custom_unet_st import UNetCustomSpatioTemporalConditionModel
    from src.models.env_encoder import EnvEncoder
    from src.pipelines.pipeline_rgbx import RGBXVideoDiffusionPipeline

    cfg = OmegaConf.load(config_path)
    cond_mode = cfg.get("cond_mode", "env")
    use_det = cfg.get("use_deterministic_mode", False)

    env_encoder = None
    image_encoder, feature_extractor = None, None
    if cond_mode == "env":
        env_encoder = EnvEncoder.from_pretrained(cfg.inference_model_weights, subfolder="env_encoder")
    elif cond_mode == "image":
        feature_extractor = CLIPImageProcessor.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="feature_extractor")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="image_encoder")

    vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.inference_model_weights, subfolder="vae")
    unet = UNetCustomSpatioTemporalConditionModel.from_pretrained(cfg.inference_model_weights, subfolder="unet")
    scheduler = EulerDiscreteScheduler.from_pretrained(cfg.inference_model_weights, subfolder="scheduler")

    for m in [image_encoder, vae, unet, env_encoder]:
        if m is not None:
            m.to(device, dtype=weight_dtype)

    pipeline = RGBXVideoDiffusionPipeline(
        vae=vae, image_encoder=image_encoder, feature_extractor=feature_extractor,
        unet=unet, scheduler=scheduler, env_encoder=env_encoder,
        scale_cond_latents=cfg.model_pipeline.get("scale_cond_latents", False),
        cond_mode=cond_mode,
    )
    pipeline.scheduler.register_to_config(timestep_spacing="trailing")
    try:
        pipeline.load_lora_weights(cfg.inference_model_weights, subfolder="lora", adapter_name="real-lora")
    except Exception:
        print("No LoRA weights found, using base weights")

    pipeline = pipeline.to(device).to(weight_dtype)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline, cfg


# ---------------------------------------------------------------------------
# Core render helpers
# ---------------------------------------------------------------------------

GBUF_PASSES = ["basecolor", "normal", "depth", "roughness", "metallic"]


def run_inverse(pipeline, cfg, images_pil, device, n_frames=24):
    """Run inverse rendering on a list of PIL images.

    Args:
        images_pil: list of PIL.Image (actual frames, will be padded to n_frames)

    Returns:
        dict  {gbuf_label: list[PIL.Image]} with len == len(images_pil)
    """
    from utils.utils_rgbx import convert_rgba_to_rgb_pil
    from utils.utils_rgbx_inference import resize_upscale_without_padding

    try:
        inference_height, inference_width = cfg.inference_res
    except Exception:
        inference_height, inference_width = 512, 512

    processed = []
    width, height = None, None
    for ind, pil_img in enumerate(images_pil):
        pil_img = convert_rgba_to_rgb_pil(pil_img, background_color=(0, 0, 0))
        if ind == 0:
            width, height = pil_img.size
            if width != inference_width or height != inference_height:
                pil_img = resize_upscale_without_padding(pil_img, inference_height, inference_width)
                width, height = pil_img.size
        else:
            if pil_img.size != (width, height):
                pil_img = pil_img.resize((width, height), Image.BILINEAR)
        processed.append(np.asarray(pil_img))

    n_actual = len(processed)
    while len(processed) < n_frames:
        processed.append(processed[-1])

    input_arr = np.stack(processed, axis=0)[None, ...].astype(np.float32) / 255.0
    cond_images = {"rgb": input_arr}
    cond_labels = {"rgb": "vae"}

    seed = cfg.get("seed", 0)
    autocast_enabled = cfg.get("autocast", True)

    result = {}
    for pass_name in GBUF_PASSES:
        cond_images["input_context"] = pass_name
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

        if torch.backends.mps.is_available():
            ctx = nullcontext()
        else:
            ctx = torch.autocast(device.type, enabled=autocast_enabled)

        with ctx:
            frames = pipeline(
                cond_images, cond_labels,
                height=height, width=width,
                num_frames=n_frames,
                num_inference_steps=cfg.get("inference_n_steps", 20),
                min_guidance_scale=cfg.get("inference_min_guidance_scale", 1.0),
                max_guidance_scale=cfg.get("inference_max_guidance_scale", 1.0),
                fps=cfg.get("fps", 7),
                motion_bucket_id=cfg.get("motion_bucket_id", 127),
                noise_aug_strength=cfg.get("cond_aug", 0),
                generator=generator,
                decode_chunk_size=cfg.get("decode_chunk_size", None),
            ).frames[0]

        result[pass_name] = frames[:n_actual]

    return result, (width, height)


def run_forward(pipeline, cfg, gbuffers, env_ldr, env_log, env_nrm, device, n_frames=24):
    """Run forward rendering given G-buffers and env maps.

    Args:
        gbuffers: dict {label: list[PIL.Image]}
        env_ldr, env_log, env_nrm: tensors (1, F_padded, C, H, W)

    Returns:
        list[PIL.Image] with len == n_actual_frames
    """
    from utils.utils_rgbx_inference import resize_upscale_without_padding

    try:
        inference_height, inference_width = cfg.inference_res
    except Exception:
        inference_height, inference_width = 512, 512

    cond_labels = cfg.model_pipeline.cond_images
    cond_gbuf_labels = [x for x in list(cond_labels.keys()) if "env_" not in x]

    first_label = cond_gbuf_labels[0]
    n_actual = len(gbuffers[first_label])

    cond_images = {}
    width, height = None, None
    for gbuf_label in cond_gbuf_labels:
        frames_pil = gbuffers[gbuf_label]
        arr_list = []
        for ind, pil_img in enumerate(frames_pil):
            img = pil_img.convert("RGB")
            if ind == 0:
                width, height = img.size
                if width != inference_width or height != inference_height:
                    img = resize_upscale_without_padding(img, inference_height, inference_width)
                    width, height = img.size
            else:
                if img.size != (width, height):
                    img = img.resize((width, height), Image.BILINEAR)
            arr_list.append(np.asarray(img))

        while len(arr_list) < n_frames:
            arr_list.append(arr_list[-1])

        gbuf_arr = np.stack(arr_list, axis=0)[None, ...].astype(np.float32) / 255.0
        cond_images[gbuf_label] = torch.from_numpy(gbuf_arr).permute(0, 1, 4, 2, 3).to(device)

    cond_images["env_ldr"] = env_ldr
    cond_images["env_log"] = env_log
    cond_images["env_nrm"] = env_nrm

    seed = cfg.get("seed", 0)
    autocast_enabled = cfg.get("autocast", True)
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    if torch.backends.mps.is_available():
        ctx = nullcontext()
    else:
        ctx = torch.autocast(device.type, enabled=autocast_enabled)

    with ctx:
        frames = pipeline(
            cond_images, cond_labels,
            height=height, width=width,
            num_frames=n_frames,
            num_inference_steps=cfg.get("inference_n_steps", 20),
            min_guidance_scale=cfg.get("inference_min_guidance_scale", 1.2),
            max_guidance_scale=cfg.get("inference_max_guidance_scale", 1.2),
            fps=cfg.get("fps", 7),
            motion_bucket_id=cfg.get("motion_bucket_id", 127),
            noise_aug_strength=cfg.get("cond_aug", 0),
            generator=generator,
            cross_attention_kwargs={"scale": cfg.get("lora_scale", 0.0)},
            dynamic_guidance=False,
            decode_chunk_size=cfg.get("decode_chunk_size", None),
        ).frames[0]

    return frames[:n_actual]


# ---------------------------------------------------------------------------
# Metrics on a set of frames
# ---------------------------------------------------------------------------


def compute_frame_metrics(pred_frames, gt_dir, frame_ids, lpips_fn, device):
    """Compute per-frame and mean PSNR/SSIM/LPIPS.

    Both predicted and GT are composited on white using GT alpha before comparison.
    """
    psnr_list, ssim_list, lpips_list = [], [], []

    for idx, fid in enumerate(frame_ids):
        gt_path = os.path.join(gt_dir, f"{fid}.png")
        if not os.path.exists(gt_path):
            continue

        pred_arr = np.asarray(pred_frames[idx].convert("RGB"))
        gt_pil = Image.open(gt_path)

        if gt_pil.mode == "RGBA":
            gt_np = np.asarray(gt_pil)
            alpha = gt_np[:, :, 3]
            gt_rgb = composite_on_white(gt_np[:, :, :3], alpha)
            pred_arr = composite_on_white(pred_arr, alpha)
        else:
            gt_rgb = np.asarray(gt_pil.convert("RGB"))

        if gt_rgb.shape[:2] != pred_arr.shape[:2]:
            pred_arr = np.asarray(
                Image.fromarray(pred_arr).resize(
                    (gt_rgb.shape[1], gt_rgb.shape[0]), Image.BILINEAR))

        psnr_list.append(compute_psnr(pred_arr, gt_rgb))
        ssim_list.append(compute_ssim(pred_arr, gt_rgb))
        lpips_list.append(compute_lpips_val(pred_arr, gt_rgb, lpips_fn, device))

    if not psnr_list:
        return None

    return {
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
        "lpips": float(np.mean(lpips_list)),
        "psnr_per_frame": [float(v) for v in psnr_list],
        "ssim_per_frame": [float(v) for v in ssim_list],
        "lpips_per_frame": [float(v) for v in lpips_list],
    }


# ---------------------------------------------------------------------------
# Model swapping helpers
# ---------------------------------------------------------------------------


def move_pipeline(pipeline, target_device):
    """Move a pipeline to target_device (for memory-constrained setups)."""
    pipeline = pipeline.to(target_device)
    if target_device == torch.device("cpu"):
        torch.cuda.empty_cache()
    return pipeline


# ---------------------------------------------------------------------------
# Experiment 1: iterative degradation
# ---------------------------------------------------------------------------


def run_iterative(args):
    """Iterative round-trip degradation experiment."""
    import lpips as lpips_module

    data_dir = args.data_dir
    workspace = args.workspace
    n_iters = args.n_iterations
    save_buffers_every = args.save_buffers_every
    swap_models = args.swap_models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips_module.LPIPS(net="alex").to(device)

    print("Loading inverse pipeline...")
    inv_pipeline, inv_cfg = load_inverse_pipeline(args.rgbx_config, device)
    n_frames = inv_cfg.get("inference_n_frames", 24)

    print("Loading forward pipeline...")
    if swap_models:
        inv_pipeline = move_pipeline(inv_pipeline, torch.device("cpu"))
    fwd_pipeline, fwd_cfg = load_forward_pipeline(args.xrgb_config, device)
    if swap_models:
        fwd_pipeline = move_pipeline(fwd_pipeline, torch.device("cpu"))
        inv_pipeline = move_pipeline(inv_pipeline, device)

    env_resolution = tuple(fwd_cfg.model_pipeline.get("env_resolution", [512, 512]))

    scenes = args.scenes
    if not scenes:
        scenes = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

    out_base = os.path.join(workspace, "iterative")

    for scene_name in scenes:
        print(f"\n{'='*60}")
        print(f"Scene: {scene_name}")
        print(f"{'='*60}")

        scene_out = os.path.join(out_base, scene_name)
        summary_path = os.path.join(scene_out, "summary.json")
        if os.path.exists(summary_path):
            print(f"  Already completed, skipping. Delete {summary_path} to re-run.")
            continue

        input_dir = os.path.join(data_dir, scene_name, "input_images")
        gt_dir = os.path.join(data_dir, scene_name, "target_images")
        envmap_dir = os.path.join(data_dir, scene_name, "envmaps")

        if not all(os.path.isdir(d) for d in [input_dir, gt_dir, envmap_dir]):
            print(f"  Missing directories, skipping")
            continue

        frame_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith(".png")])
        if not frame_ids:
            print(f"  No frames found, skipping")
            continue

        current_images = [Image.open(os.path.join(input_dir, f"{fid}.png")) for fid in frame_ids]

        padded_frame_ids = list(frame_ids)
        while len(padded_frame_ids) < n_frames:
            padded_frame_ids.append(padded_frame_ids[-1])

        env_ldr, env_log, env_nrm = load_envmaps_for_frames(
            envmap_dir, padded_frame_ids, env_resolution[0], env_resolution[1], device)

        metrics_history = []
        last_completed = -1

        for it_dir in sorted(Path(scene_out).glob("iter_*")) if os.path.isdir(scene_out) else []:
            done_file = it_dir / "done.txt"
            if done_file.exists():
                it_num = int(it_dir.name.split("_")[1])
                last_completed = max(last_completed, it_num)

        if last_completed >= 0:
            metrics_file = os.path.join(scene_out, "metrics_partial.json")
            if os.path.exists(metrics_file):
                with open(metrics_file) as f:
                    metrics_history = json.load(f)

            last_dir = os.path.join(scene_out, f"iter_{last_completed:04d}", "rendered")
            if os.path.isdir(last_dir):
                current_images = [
                    Image.open(os.path.join(last_dir, f"{fid}.png"))
                    for fid in frame_ids
                    if os.path.exists(os.path.join(last_dir, f"{fid}.png"))
                ]
            print(f"  Resuming from iteration {last_completed + 1}")

        for iteration in tqdm(range(last_completed + 1, n_iters), desc=f"  {scene_name}"):
            iter_dir = os.path.join(scene_out, f"iter_{iteration:04d}")
            rendered_dir = os.path.join(iter_dir, "rendered")
            os.makedirs(rendered_dir, exist_ok=True)

            done_path = os.path.join(iter_dir, "done.txt")
            if os.path.exists(done_path):
                continue

            # --- Inverse render ---
            if swap_models:
                fwd_pipeline = move_pipeline(fwd_pipeline, torch.device("cpu"))
                inv_pipeline = move_pipeline(inv_pipeline, device)

            gbuffers, (w, h) = run_inverse(inv_pipeline, inv_cfg, current_images, device, n_frames)

            if iteration % save_buffers_every == 0:
                buf_dir = os.path.join(iter_dir, "buffers")
                os.makedirs(buf_dir, exist_ok=True)
                for label, pil_list in gbuffers.items():
                    for fi, pil_img in enumerate(pil_list):
                        pil_img.save(os.path.join(buf_dir, f"{frame_ids[fi]}.{label}.png"))

            # --- Forward render ---
            if swap_models:
                inv_pipeline = move_pipeline(inv_pipeline, torch.device("cpu"))
                fwd_pipeline = move_pipeline(fwd_pipeline, device)

            rendered_frames = run_forward(
                fwd_pipeline, fwd_cfg, gbuffers, env_ldr, env_log, env_nrm, device, n_frames)

            for fi, pil_img in enumerate(rendered_frames):
                pil_img.save(os.path.join(rendered_dir, f"{frame_ids[fi]}.png"))

            # --- Metrics ---
            m = compute_frame_metrics(rendered_frames, gt_dir, frame_ids, lpips_fn, device)
            if m:
                m["iteration"] = iteration
                metrics_history.append(m)
                print(f"    iter {iteration}: PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}  LPIPS={m['lpips']:.4f}")

            with open(os.path.join(scene_out, "metrics_partial.json"), "w") as f:
                json.dump(metrics_history, f, indent=2)

            touch(done_path)

            current_images = rendered_frames

        with open(summary_path, "w") as f:
            json.dump({
                "experiment": "iterative",
                "scene": scene_name,
                "n_iterations": n_iters,
                "metrics": metrics_history,
            }, f, indent=2)

        partial = os.path.join(scene_out, "metrics_partial.json")
        if os.path.exists(partial):
            os.remove(partial)

        print(f"  Done. Summary saved to {summary_path}")

    generate_plots(workspace, "iterative", scenes)


# ---------------------------------------------------------------------------
# Experiment 2: control (same buffer, cycling env maps)
# ---------------------------------------------------------------------------


def collect_env_variants(data_dir, object_name):
    """Find all scene dirs that belong to the same object.

    Returns list of (variant_name, scene_dir) sorted.
    """
    variants = []
    for d in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, d)
        if not os.path.isdir(full):
            continue
        if d.startswith(object_name + "_"):
            suffix = d[len(object_name) + 1:]
            variants.append((suffix, d))
    return variants


def run_control(args):
    """Control experiment: same G-buffers, cycling env maps from all variants."""
    import lpips as lpips_module

    data_dir = args.data_dir
    workspace = args.workspace
    n_iters = args.n_iterations
    swap_models = args.swap_models
    object_name = args.object
    source_variant = args.source_variant

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips_module.LPIPS(net="alex").to(device)

    print("Loading inverse pipeline...")
    inv_pipeline, inv_cfg = load_inverse_pipeline(args.rgbx_config, device)
    n_frames = inv_cfg.get("inference_n_frames", 24)

    print("Loading forward pipeline...")
    if swap_models:
        inv_pipeline = move_pipeline(inv_pipeline, torch.device("cpu"))
    fwd_pipeline, fwd_cfg = load_forward_pipeline(args.xrgb_config, device)

    env_resolution = tuple(fwd_cfg.model_pipeline.get("env_resolution", [512, 512]))

    source_scene = f"{object_name}_{source_variant}"
    input_dir = os.path.join(data_dir, source_scene, "input_images")
    frame_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith(".png")])

    padded_frame_ids = list(frame_ids)
    while len(padded_frame_ids) < n_frames:
        padded_frame_ids.append(padded_frame_ids[-1])

    # --- Inverse render once ---
    print(f"Inverse rendering {source_scene} to get B_0...")
    if swap_models:
        fwd_pipeline = move_pipeline(fwd_pipeline, torch.device("cpu"))
        inv_pipeline = move_pipeline(inv_pipeline, device)

    source_images = [Image.open(os.path.join(input_dir, f"{fid}.png")) for fid in frame_ids]
    gbuffers, (w, h) = run_inverse(inv_pipeline, inv_cfg, source_images, device, n_frames)

    if swap_models:
        inv_pipeline = move_pipeline(inv_pipeline, torch.device("cpu"))
        fwd_pipeline = move_pipeline(fwd_pipeline, device)

    out_base = os.path.join(workspace, "control", object_name)
    os.makedirs(out_base, exist_ok=True)

    buf_dir = os.path.join(out_base, "buffers_B0")
    os.makedirs(buf_dir, exist_ok=True)
    for label, pil_list in gbuffers.items():
        for fi, pil_img in enumerate(pil_list):
            pil_img.save(os.path.join(buf_dir, f"{frame_ids[fi]}.{label}.png"))

    # --- Collect env map sets from all variants ---
    variants = collect_env_variants(data_dir, object_name)
    if not variants:
        print(f"No variants found for object {object_name}")
        return

    env_pool = []
    for suffix, scene_dir in variants:
        envmap_dir = os.path.join(data_dir, scene_dir, "envmaps")
        gt_dir = os.path.join(data_dir, scene_dir, "target_images")
        if os.path.isdir(envmap_dir) and os.path.isdir(gt_dir):
            env_pool.append({
                "variant": suffix,
                "scene_dir": scene_dir,
                "envmap_dir": envmap_dir,
                "gt_dir": gt_dir,
            })

    print(f"Collected {len(env_pool)} env map sets from variants: "
          f"{[e['variant'] for e in env_pool]}")

    summary_path = os.path.join(out_base, "summary.json")
    if os.path.exists(summary_path):
        print(f"Already completed, skipping. Delete {summary_path} to re-run.")
        generate_plots(workspace, "control", [object_name])
        return

    metrics_history = []

    for iteration in tqdm(range(n_iters), desc=f"  control ({object_name})"):
        env_info = env_pool[iteration % len(env_pool)]
        iter_dir = os.path.join(out_base, f"iter_{iteration:04d}")
        rendered_dir = os.path.join(iter_dir, "rendered")
        os.makedirs(rendered_dir, exist_ok=True)

        done_path = os.path.join(iter_dir, "done.txt")
        if os.path.exists(done_path):
            partial_file = os.path.join(out_base, "metrics_partial.json")
            if os.path.exists(partial_file):
                with open(partial_file) as f:
                    saved = json.load(f)
                    if len(saved) > iteration:
                        metrics_history = saved
            continue

        env_ldr, env_log, env_nrm = load_envmaps_for_frames(
            env_info["envmap_dir"], padded_frame_ids,
            env_resolution[0], env_resolution[1], device)

        rendered_frames = run_forward(
            fwd_pipeline, fwd_cfg, gbuffers, env_ldr, env_log, env_nrm, device, n_frames)

        for fi, pil_img in enumerate(rendered_frames):
            pil_img.save(os.path.join(rendered_dir, f"{frame_ids[fi]}.png"))

        m = compute_frame_metrics(rendered_frames, env_info["gt_dir"], frame_ids, lpips_fn, device)
        if m:
            m["iteration"] = iteration
            m["variant"] = env_info["variant"]
            metrics_history.append(m)
            print(f"    iter {iteration} ({env_info['variant']}): "
                  f"PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}  LPIPS={m['lpips']:.4f}")

        with open(os.path.join(out_base, "metrics_partial.json"), "w") as f:
            json.dump(metrics_history, f, indent=2)

        touch(done_path)

    with open(summary_path, "w") as f:
        json.dump({
            "experiment": "control",
            "object": object_name,
            "source_variant": source_variant,
            "n_iterations": n_iters,
            "env_pool_variants": [e["variant"] for e in env_pool],
            "metrics": metrics_history,
        }, f, indent=2)

    partial = os.path.join(out_base, "metrics_partial.json")
    if os.path.exists(partial):
        os.remove(partial)

    print(f"  Done. Summary saved to {summary_path}")
    generate_plots(workspace, "control", [object_name])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def generate_plots(workspace, experiment_type, names):
    """Generate degradation curve plots from summary JSONs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    plot_dir = os.path.join(workspace, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for name in names:
        if experiment_type == "iterative":
            summary_path = os.path.join(workspace, "iterative", name, "summary.json")
        else:
            summary_path = os.path.join(workspace, "control", name, "summary.json")

        if not os.path.exists(summary_path):
            continue

        with open(summary_path) as f:
            data = json.load(f)

        metrics = data["metrics"]
        if not metrics:
            continue

        iters = [m["iteration"] for m in metrics]
        psnr = [m["psnr"] for m in metrics]
        ssim = [m["ssim"] for m in metrics]
        lpips_vals = [m["lpips"] for m in metrics]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"{experiment_type.title()} - {name}", fontsize=14)

        axes[0].plot(iters, psnr, "b-o", markersize=3)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("PSNR (dB)")
        axes[0].set_title("PSNR")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(iters, ssim, "g-o", markersize=3)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("SSIM")
        axes[1].set_title("SSIM")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(iters, lpips_vals, "r-o", markersize=3)
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("LPIPS")
        axes[2].set_title("LPIPS")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f"{experiment_type}_{name}.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {plot_path}")


def generate_comparison_plot(workspace, scene_name, object_name):
    """Overlay iterative vs control on the same plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping comparison plot")
        return

    iter_path = os.path.join(workspace, "iterative", scene_name, "summary.json")
    ctrl_path = os.path.join(workspace, "control", object_name, "summary.json")

    if not os.path.exists(iter_path) or not os.path.exists(ctrl_path):
        return

    with open(iter_path) as f:
        iter_data = json.load(f)
    with open(ctrl_path) as f:
        ctrl_data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Degradation Comparison - {scene_name}", fontsize=14)

    for dataset, label, style in [
        (iter_data["metrics"], "Iterative", "b-o"),
        (ctrl_data["metrics"], "Control", "r-s"),
    ]:
        if not dataset:
            continue
        iters = [m["iteration"] for m in dataset]
        psnr = [m["psnr"] for m in dataset]
        ssim = [m["ssim"] for m in dataset]
        lp = [m["lpips"] for m in dataset]

        axes[0].plot(iters, psnr, style, markersize=3, label=label)
        axes[1].plot(iters, ssim, style, markersize=3, label=label)
        axes[2].plot(iters, lp, style, markersize=3, label=label)

    for ax, title, ylabel in zip(axes, ["PSNR", "SSIM", "LPIPS"],
                                  ["PSNR (dB)", "SSIM", "LPIPS"]):
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_dir = os.path.join(workspace, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, f"comparison_{scene_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Comparison plot saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Degradation experiment for DiffusionRenderer")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # -- shared args helper --
    def add_common(p):
        p.add_argument("--data_dir", required=True)
        p.add_argument("--workspace", default="degradation_workspace")
        p.add_argument("--n_iterations", type=int, default=50)
        p.add_argument("--rgbx_config", default="configs/rgbx_inference.yaml")
        p.add_argument("--xrgb_config", default="configs/xrgb_inference.yaml")
        p.add_argument("--swap_models", action="store_true",
                        help="Swap models to CPU when not in use (saves GPU memory)")

    p_iter = subparsers.add_parser("iterative", help="Round-trip degradation experiment")
    add_common(p_iter)
    p_iter.add_argument("--scenes", nargs="*", default=None,
                        help="Scene names to process (default: all)")
    p_iter.add_argument("--save_buffers_every", type=int, default=5,
                        help="Save G-buffers every N iterations")

    p_ctrl = subparsers.add_parser("control", help="Same-buffer control experiment")
    add_common(p_ctrl)
    p_ctrl.add_argument("--object", required=True,
                        help="Object name (e.g. all_purpose_cleaner)")
    p_ctrl.add_argument("--source_variant", default="env_0",
                        help="Variant to use for initial inverse rendering")

    p_plot = subparsers.add_parser("plot", help="Generate plots from existing results")
    p_plot.add_argument("--workspace", default="degradation_workspace")
    p_plot.add_argument("--scene", help="Scene name for iterative results")
    p_plot.add_argument("--object", help="Object name for control results")

    args = parser.parse_args()

    if args.cmd == "iterative":
        run_iterative(args)
    elif args.cmd == "control":
        run_control(args)
    elif args.cmd == "plot":
        if args.scene:
            generate_plots(args.workspace, "iterative", [args.scene])
        if args.object:
            generate_plots(args.workspace, "control", [args.object])
        if args.scene and args.object:
            generate_comparison_plot(args.workspace, args.scene, args.object)


if __name__ == "__main__":
    main()
