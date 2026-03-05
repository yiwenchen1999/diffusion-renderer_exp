#!/usr/bin/env python3
"""
Run DiffusionRenderer experiment on source_data_polyhaven.

Supports three steps:
  prepare  - Create symlink structure for inverse rendering
  forward  - Run forward rendering with per-frame environment maps
  evaluate - Compute PSNR, SSIM, LPIPS against ground truth

Full pipeline:
  # Step 0: Prepare input for inverse rendering
  python run_polyhaven_experiment.py prepare \
    --data_dir source_data_polyhaven --workspace polyhaven_workspace

  # Step 1: Run inverse rendering (existing script)
  python inference_svd_rgbx.py --config configs/rgbx_inference.yaml \
    inference_input_dir=polyhaven_workspace/inverse_input \
    inference_save_dir=polyhaven_workspace/delighting \
    chunk_mode=all model_passes="['basecolor','normal','depth','roughness','metallic']"

  # Step 2: Run forward rendering with per-frame env maps
  python run_polyhaven_experiment.py forward \
    --data_dir source_data_polyhaven \
    --workspace polyhaven_workspace \
    --config configs/xrgb_inference.yaml

  # Step 3: Evaluate
  python run_polyhaven_experiment.py evaluate \
    --data_dir source_data_polyhaven \
    --workspace polyhaven_workspace
"""

import os
import sys
import argparse
import json
import glob
import csv
from contextlib import nullcontext
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import imageio
import torch

# ─────────────────────────── Step 0: prepare ──────────────────────────


def prepare(args):
    """Create symlink directory structure for inverse rendering."""
    data_dir = args.data_dir
    workspace = args.workspace
    inverse_input_dir = os.path.join(workspace, "inverse_input")
    os.makedirs(inverse_input_dir, exist_ok=True)

    scene_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    # Deduplicate: scenes with same object but different env share input_images.
    # We still create per-scene symlink folders so delighting output is per-scene.
    created = 0
    for scene_name in scene_dirs:
        input_images_dir = os.path.join(data_dir, scene_name, "input_images")
        if not os.path.isdir(input_images_dir):
            print(f"[WARN] {scene_name}: no input_images/ folder, skipping")
            continue

        target_dir = os.path.join(inverse_input_dir, scene_name)
        if os.path.exists(target_dir):
            continue

        os.makedirs(target_dir, exist_ok=True)
        for fname in sorted(os.listdir(input_images_dir)):
            src = os.path.abspath(os.path.join(input_images_dir, fname))
            dst = os.path.join(target_dir, fname)
            if not os.path.exists(dst):
                os.symlink(src, dst)
        created += 1

    print(f"Prepared {created} scene folders in {inverse_input_dir}")
    print(f"\nNext: run inverse rendering:")
    print(f"  python inference_svd_rgbx.py --config configs/rgbx_inference.yaml \\")
    print(f"    inference_input_dir={inverse_input_dir} \\")
    print(f"    inference_save_dir={os.path.join(workspace, 'delighting')} \\")
    print(f"    chunk_mode=all \\")
    print(f"    model_passes=\"['basecolor','normal','depth','roughness','metallic']\"")


# ─────────────────────── Step 2: forward rendering ─────────────────────


def recover_hdr_from_pngs(ldr_png, hdr_png):
    """
    Recover original HDR from preprocessing outputs.

    preprocess_objaverse.py saved:
      _ldr.png = clip(hdr, 0, 1) ^ (1/2.2)
      _hdr.png = log1p(10 * hdr) / max(log1p(10 * hdr))

    We reverse-engineer the unknown per-image max M using LDR pixels
    where hdr <= 1, then recover the full HDR including highlights.

    Args:
        ldr_png: float32 array [H, W, 3] in [0, 1] from _ldr.png
        hdr_png: float32 array [H, W, 3] in [0, 1] from _hdr.png

    Returns:
        hdr_recovered: float32 array [H, W, 3], linear HDR values
    """
    hdr_clipped = np.power(np.clip(ldr_png, 1e-6, 1.0), 2.2)

    # Estimate M = max(log1p(10 * hdr)) using mid-range pixels
    # where both images have reliable values (avoid near-zero and saturated)
    mask = (hdr_clipped > 0.02) & (hdr_clipped < 0.9) & (hdr_png > 0.02)

    if mask.any():
        log_vals = np.log1p(10.0 * hdr_clipped[mask])
        M_estimates = log_vals / hdr_png[mask]
        M = float(np.median(M_estimates))
    else:
        M = float(np.log1p(10.0))

    M = max(M, 1.0)

    hdr_recovered = np.expm1(hdr_png * M) / 10.0
    hdr_recovered = np.clip(hdr_recovered, 0.0, 65504.0)
    return hdr_recovered


def apply_model_hdr_mapping(hdr_np, device):
    """
    Apply the model's expected tone mapping (from utils/utils_env_proj.py hdr_mapping).

    Returns env_ldr and env_log as float32 tensors [H, W, 3] in [0, 1].
    """
    import src.data.rendering_utils as util

    hdr_t = torch.from_numpy(hdr_np).float().to(device)
    env_ldr = util.rgb2srgb(util.reinhard(hdr_t, max_point=16).clamp(0, 1))
    env_log = util.rgb2srgb(
        (torch.log1p(hdr_t) / np.log1p(10000)).clamp(0, 1)
    )
    return env_ldr.cpu(), env_log.cpu()


def load_per_frame_envmaps(envmap_dir, frame_ids, target_h, target_w, device):
    """
    Load per-frame _ldr.png and _hdr.png, recover original HDR,
    then apply the model's tone mapping (reinhard+sRGB for LDR,
    log1p/log1p(10000)+sRGB for log).

    Returns env_ldr and env_log tensors of shape (1, F, C, H, W) in [0, 1].
    """
    env_ldr_list = []
    env_log_list = []
    for fid in frame_ids:
        ldr_path = os.path.join(envmap_dir, f"{fid}_ldr.png")
        hdr_path = os.path.join(envmap_dir, f"{fid}_hdr.png")

        if not os.path.exists(ldr_path) or not os.path.exists(hdr_path):
            raise FileNotFoundError(
                f"Missing env map for frame {fid}: {ldr_path} or {hdr_path}"
            )

        ldr_img = Image.open(ldr_path).convert("RGB")
        hdr_img = Image.open(hdr_path).convert("RGB")

        if ldr_img.size != (target_w, target_h):
            ldr_img = ldr_img.resize((target_w, target_h), Image.BILINEAR)
        if hdr_img.size != (target_w, target_h):
            hdr_img = hdr_img.resize((target_w, target_h), Image.BILINEAR)

        ldr_np = np.asarray(ldr_img).astype(np.float32) / 255.0
        hdr_np = np.asarray(hdr_img).astype(np.float32) / 255.0

        hdr_recovered = recover_hdr_from_pngs(ldr_np, hdr_np)
        env_ldr, env_log = apply_model_hdr_mapping(hdr_recovered, device)

        env_ldr_list.append(env_ldr)
        env_log_list.append(env_log)

    env_ldr = torch.stack(env_ldr_list, dim=0)  # (F, H, W, 3)
    env_log = torch.stack(env_log_list, dim=0)

    env_ldr = env_ldr.unsqueeze(0).permute(0, 1, 4, 2, 3).to(device)  # (1, F, 3, H, W)
    env_log = env_log.unsqueeze(0).permute(0, 1, 4, 2, 3).to(device)
    return env_ldr, env_log


def resolve_model_dtype(cfg, device):
    """Resolve model dtype from config, fallback safe on CPU."""
    key = str(cfg.get("model_dtype", cfg.get("weight_dtype", "fp16"))).lower()
    if device.type == "cpu":
        return torch.float32
    if key in ("fp32", "float32"):
        return torch.float32
    return torch.float16


def build_inverse_pipeline(cfg, device, weight_dtype):
    """Build inverse RGB->X pipeline similar to inference_svd_rgbx.py."""
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from src.pipelines.pipeline_rgbx import RGBXVideoDiffusionPipeline

    cond_mode = cfg.get("cond_mode", "skip")
    use_deterministic_mode = cfg.get("use_deterministic_mode", False)
    missing_kwargs = {
        "cond_mode": cond_mode,
        "use_deterministic_mode": use_deterministic_mode,
    }
    if os.path.exists(cfg.inference_model_weights):
        model_weights_subfolders = os.listdir(cfg.inference_model_weights)
    else:
        model_weights_subfolders = []
    if "image_encoder" not in model_weights_subfolders:
        missing_kwargs["image_encoder"] = CLIPVisionModelWithProjection.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="image_encoder"
        )
    if "feature_extractor" not in model_weights_subfolders:
        missing_kwargs["feature_extractor"] = CLIPImageProcessor.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="feature_extractor"
        )
    pipeline = RGBXVideoDiffusionPipeline.from_pretrained(
        cfg.inference_model_weights, **missing_kwargs
    )
    pipeline = pipeline.to(device).to(weight_dtype)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def build_forward_pipeline(cfg, device, weight_dtype):
    """Build forward X->RGB pipeline similar to forward_rendering()."""
    from accelerate import PartialState
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
    from src.models.custom_unet_st import UNetCustomSpatioTemporalConditionModel
    from src.models.env_encoder import EnvEncoder
    from src.pipelines.pipeline_rgbx import RGBXVideoDiffusionPipeline

    cond_mode = cfg.get("cond_mode", "env")
    use_deterministic_mode = cfg.get("use_deterministic_mode", False)
    _ = PartialState()  # kept for parity with existing initialization path

    text_encoder, image_encoder, vae, env_encoder = None, None, None, None
    feature_extractor = None
    if cond_mode == "env":
        env_encoder = EnvEncoder.from_pretrained(
            cfg.inference_model_weights, subfolder="env_encoder"
        )
    elif cond_mode == "image":
        feature_extractor = CLIPImageProcessor.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="feature_extractor"
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="image_encoder"
        )

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        cfg.inference_model_weights, subfolder="vae"
    )
    unet = UNetCustomSpatioTemporalConditionModel.from_pretrained(
        cfg.inference_model_weights, subfolder="unet"
    )
    scheduler = EulerDiscreteScheduler.from_pretrained(
        cfg.inference_model_weights, subfolder="scheduler"
    )
    for module in [text_encoder, image_encoder, vae, unet, env_encoder]:
        if module is not None:
            module.to(device, dtype=weight_dtype)

    pipeline = RGBXVideoDiffusionPipeline(
        vae=vae,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        unet=unet,
        scheduler=scheduler,
        env_encoder=env_encoder,
        scale_cond_latents=cfg.model_pipeline.get("scale_cond_latents", False),
        cond_mode=cond_mode,
    )
    pipeline.scheduler.register_to_config(timestep_spacing="trailing")
    try:
        pipeline.load_lora_weights(
            cfg.inference_model_weights, subfolder="lora", adapter_name="real-lora"
        )
    except Exception:
        print("No LoRA weights found, using base weights")
    pipeline = pipeline.to(device).to(weight_dtype)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def list_env_frame_ids(scene_env_dir):
    """List frame IDs that have both _ldr and _hdr env maps."""
    ldr_ids = set()
    hdr_ids = set()
    for fname in sorted(os.listdir(scene_env_dir)):
        if fname.endswith("_ldr.png"):
            ldr_ids.add(fname[:-8])  # strip "_ldr.png"
        elif fname.endswith("_hdr.png"):
            hdr_ids.add(fname[:-8])  # strip "_hdr.png"
    return sorted(list(ldr_ids & hdr_ids))


def cyclic_schedule(frame_ids, iters):
    """Repeat frame IDs cyclically to target iteration count."""
    if not frame_ids:
        return []
    return [frame_ids[i % len(frame_ids)] for i in range(iters)]


def pil_to_cond_tensor_rgb(pil_img, device):
    """Convert RGB PIL image to (1,1,3,H,W) float tensor in [0,1]."""
    arr = np.asarray(pil_img.convert("RGB")).astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
    return ten.to(device)


def run_inverse_single_image(
    inverse_pipeline,
    rgb_pil,
    inverse_cfg,
    model_passes,
    device,
):
    """Inverse render one image into G-buffer tensors for forward rendering."""
    from utils.utils_rgbx import convert_rgba_to_rgb_pil
    from utils.utils_rgbx_inference import resize_upscale_without_padding

    rgb_pil = convert_rgba_to_rgb_pil(rgb_pil, background_color=(0, 0, 0))
    try:
        target_h, target_w = inverse_cfg.inference_res
    except Exception:
        target_h, target_w = 512, 512
    if rgb_pil.size != (target_w, target_h):
        rgb_pil = resize_upscale_without_padding(rgb_pil, target_h, target_w)
    arr = np.asarray(rgb_pil).astype(np.float32) / 255.0

    n_frames = int(inverse_cfg.get("inference_n_frames", 24))
    input_images = np.stack([arr] * n_frames, axis=0)[None, ...]  # (1,F,H,W,3)
    cond_images = {"rgb": input_images}
    cond_labels = {"rgb": "vae"}
    if inverse_cfg.get("cond_mode", "skip") == "image":
        cond_images["clip_img"] = input_images[:, 0:1, ...]
        cond_labels["clip_img"] = "clip"

    if inverse_cfg.get("seed", None) is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(int(inverse_cfg.seed))

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(device.type, enabled=inverse_cfg.get("autocast", True))

    outputs = {}
    for inference_pass in model_passes:
        cond_images["input_context"] = inference_pass
        with autocast_ctx:
            pred_frames = inverse_pipeline(
                cond_images,
                cond_labels,
                height=arr.shape[0],
                width=arr.shape[1],
                num_frames=n_frames,
                num_inference_steps=inverse_cfg.get("inference_n_steps", 20),
                min_guidance_scale=inverse_cfg.get("inference_min_guidance_scale", 1.0),
                max_guidance_scale=inverse_cfg.get("inference_max_guidance_scale", 1.0),
                fps=inverse_cfg.get("fps", 7),
                motion_bucket_id=inverse_cfg.get("motion_bucket_id", 127),
                noise_aug_strength=inverse_cfg.get("cond_aug", 0),
                generator=generator,
                decode_chunk_size=inverse_cfg.get("decode_chunk_size", None),
            ).frames[0]
        outputs[inference_pass] = pil_to_cond_tensor_rgb(pred_frames[0], device)
    return outputs


def run_forward_single_image(
    forward_pipeline,
    gbuffer_tensors,
    envmap_dir,
    frame_id,
    forward_cfg,
    device,
):
    """Forward render one image from G-buffer tensors and one env frame ID."""
    from src.data.rendering_utils import envmap_vec

    cond_labels = forward_cfg.model_pipeline.cond_images
    cond_gbuf_labels = [x for x in list(cond_labels.keys()) if "env_" not in x]
    env_resolution = tuple(forward_cfg.model_pipeline.get("env_resolution", [512, 512]))
    n_frames = int(forward_cfg.get("inference_n_frames", 24))

    cond_images = {}
    height = width = None
    for label in cond_gbuf_labels:
        if label not in gbuffer_tensors:
            raise KeyError(f"Missing gbuffer tensor: {label}")
        ten = gbuffer_tensors[label]
        if ten.shape[1] != 1:
            ten = ten[:, :1, ...]
        if height is None:
            height = int(ten.shape[-2])
            width = int(ten.shape[-1])
        cond_images[label] = ten.expand(1, n_frames, ten.shape[2], ten.shape[3], ten.shape[4]).contiguous()

    env_ids = [frame_id] * n_frames
    env_ldr, env_log = load_per_frame_envmaps(
        envmap_dir,
        env_ids,
        env_resolution[0],
        env_resolution[1],
        device,
    )
    cond_images["env_ldr"] = env_ldr
    cond_images["env_log"] = env_log
    env_nrm = envmap_vec(env_resolution, device=device) * 0.5 + 0.5
    cond_images["env_nrm"] = (
        env_nrm.unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 2, 3).expand_as(env_ldr)
    )

    if forward_cfg.get("seed", None) is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(int(forward_cfg.seed))

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(device.type, enabled=forward_cfg.get("autocast", True))

    with autocast_ctx:
        out = forward_pipeline(
            cond_images,
            cond_labels,
            height=height,
            width=width,
            num_frames=n_frames,
            num_inference_steps=forward_cfg.get("inference_n_steps", 20),
            min_guidance_scale=forward_cfg.get("inference_min_guidance_scale", 1.2),
            max_guidance_scale=forward_cfg.get("inference_max_guidance_scale", 1.2),
            fps=forward_cfg.get("fps", 7),
            motion_bucket_id=forward_cfg.get("motion_bucket_id", 127),
            noise_aug_strength=forward_cfg.get("cond_aug", 0),
            generator=generator,
            cross_attention_kwargs={"scale": forward_cfg.get("lora_scale", 0.0)},
            dynamic_guidance=False,
            decode_chunk_size=forward_cfg.get("decode_chunk_size", None),
        ).frames[0]
    return out[0]


def evaluate_pred_against_gt(pred_rgb_u8, gt_path):
    """Evaluate one predicted RGB image against GT with white compositing."""
    gt_pil = Image.open(gt_path)
    if gt_pil.mode == "RGBA":
        gt_rgba = np.asarray(gt_pil)
        alpha = gt_rgba[:, :, 3]
        gt_img = composite_on_white(gt_rgba[:, :, :3], alpha)
        pred_img = composite_on_white(pred_rgb_u8, alpha)
    else:
        gt_img = np.asarray(gt_pil.convert("RGB"))
        pred_img = pred_rgb_u8
    if gt_img.shape != pred_img.shape:
        pred_img = np.asarray(
            Image.fromarray(pred_img).resize((gt_img.shape[1], gt_img.shape[0]), Image.BILINEAR)
        )
    return pred_img, gt_img


def run_degradation_experiment(args):
    """Run two degradation experiments on one PolyHaven scene."""
    from omegaconf import OmegaConf
    import lpips as lpips_module

    data_dir = args.data_dir
    workspace = args.workspace
    scene_name = args.scene
    iters = int(args.iters)
    mode = args.mode

    scene_dir = os.path.join(data_dir, scene_name)
    input_dir = os.path.join(scene_dir, "input_images")
    target_dir = os.path.join(scene_dir, "target_images")
    env_dir = os.path.join(scene_dir, "envmaps")
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(f"Scene not found: {scene_dir}")
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Missing input_images: {input_dir}")
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"Missing target_images: {target_dir}")
    if not os.path.isdir(env_dir):
        raise FileNotFoundError(f"Missing envmaps: {env_dir}")

    env_frame_ids = list_env_frame_ids(env_dir)
    if len(env_frame_ids) == 0:
        raise RuntimeError(f"No valid envmaps found in {env_dir}")
    schedule = cyclic_schedule(env_frame_ids, iters)

    inverse_cfg = OmegaConf.load(args.config_rgbx)
    forward_cfg = OmegaConf.load(args.config_xrgb)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inverse_dtype = resolve_model_dtype(inverse_cfg, device)
    forward_dtype = resolve_model_dtype(forward_cfg, device)

    print("[degrade] building inverse pipeline...")
    inverse_pipeline = build_inverse_pipeline(inverse_cfg, device, inverse_dtype)
    print("[degrade] building forward pipeline...")
    forward_pipeline = build_forward_pipeline(forward_cfg, device, forward_dtype)
    model_passes = list(inverse_cfg.get("model_passes", ["basecolor", "normal", "depth", "roughness", "metallic"]))

    lpips_fn = lpips_module.LPIPS(net="alex").to(device)

    start_fid = schedule[0]
    start_input_path = os.path.join(input_dir, f"{start_fid}.png")
    if not os.path.exists(start_input_path):
        input_candidates = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
        if not input_candidates:
            raise RuntimeError(f"No input pngs found in {input_dir}")
        start_input_path = os.path.join(input_dir, input_candidates[0])
    initial_pil = Image.open(start_input_path).convert("RGB")

    out_root = os.path.join(workspace, "degradation", scene_name)
    os.makedirs(out_root, exist_ok=True)

    def run_mode_chain():
        mode_dir = os.path.join(out_root, "chain")
        os.makedirs(mode_dir, exist_ok=True)
        metrics = []
        viz_frames = []

        current_buffers = run_inverse_single_image(
            inverse_pipeline, initial_pil, inverse_cfg, model_passes, device
        )
        current_img_pil = initial_pil
        for idx, fid in enumerate(tqdm(schedule, desc=f"Chain degradation [{scene_name}]")):
            pred_pil = run_forward_single_image(
                forward_pipeline, current_buffers, env_dir, fid, forward_cfg, device
            )
            pred_u8 = np.asarray(pred_pil.convert("RGB"))
            gt_path = os.path.join(target_dir, f"{fid}.png")
            if not os.path.exists(gt_path):
                print(f"[WARN] missing GT for frame {fid}, skip metric")
                psnr = ssim = lp = float("nan")
                pred_eval = pred_u8
            else:
                pred_eval, gt_eval = evaluate_pred_against_gt(pred_u8, gt_path)
                psnr = compute_psnr(pred_eval, gt_eval)
                ssim = compute_ssim(pred_eval, gt_eval)
                lp = compute_lpips_batch([pred_eval], [gt_eval], lpips_fn, device)[0]

            iter_path = os.path.join(mode_dir, f"iter_{idx:03d}.png")
            Image.fromarray(pred_eval).save(iter_path)
            viz_frames.append(pred_eval)
            metrics.append({
                "iter": idx,
                "frame_id": fid,
                "image_path": iter_path,
                "psnr": float(psnr),
                "ssim": float(ssim),
                "lpips": float(lp),
            })

            current_img_pil = Image.fromarray(pred_eval)
            if idx < len(schedule) - 1:
                current_buffers = run_inverse_single_image(
                    inverse_pipeline, current_img_pil, inverse_cfg, model_passes, device
                )

        if args.save_video and viz_frames:
            imageio.mimsave(
                os.path.join(out_root, "viz_chain.mp4"),
                viz_frames,
                fps=int(args.video_fps),
                codec="h264",
            )
        return metrics

    def run_mode_fixed_buffer():
        mode_dir = os.path.join(out_root, "fixed_buffer")
        os.makedirs(mode_dir, exist_ok=True)
        metrics = []
        viz_frames = []

        fixed_buffers = run_inverse_single_image(
            inverse_pipeline, initial_pil, inverse_cfg, model_passes, device
        )
        for idx, fid in enumerate(tqdm(schedule, desc=f"Fixed-buffer relighting [{scene_name}]")):
            pred_pil = run_forward_single_image(
                forward_pipeline, fixed_buffers, env_dir, fid, forward_cfg, device
            )
            pred_u8 = np.asarray(pred_pil.convert("RGB"))
            gt_path = os.path.join(target_dir, f"{fid}.png")
            if not os.path.exists(gt_path):
                print(f"[WARN] missing GT for frame {fid}, skip metric")
                psnr = ssim = lp = float("nan")
                pred_eval = pred_u8
            else:
                pred_eval, gt_eval = evaluate_pred_against_gt(pred_u8, gt_path)
                psnr = compute_psnr(pred_eval, gt_eval)
                ssim = compute_ssim(pred_eval, gt_eval)
                lp = compute_lpips_batch([pred_eval], [gt_eval], lpips_fn, device)[0]

            iter_path = os.path.join(mode_dir, f"iter_{idx:03d}.png")
            Image.fromarray(pred_eval).save(iter_path)
            viz_frames.append(pred_eval)
            metrics.append({
                "iter": idx,
                "frame_id": fid,
                "image_path": iter_path,
                "psnr": float(psnr),
                "ssim": float(ssim),
                "lpips": float(lp),
            })

        if args.save_video and viz_frames:
            imageio.mimsave(
                os.path.join(out_root, "viz_fixed_buffer.mp4"),
                viz_frames,
                fps=int(args.video_fps),
                codec="h264",
            )
        return metrics

    def summarize(metrics):
        vals_psnr = [m["psnr"] for m in metrics if np.isfinite(m["psnr"])]
        vals_ssim = [m["ssim"] for m in metrics if np.isfinite(m["ssim"])]
        vals_lp = [m["lpips"] for m in metrics if np.isfinite(m["lpips"])]
        return {
            "n_iters": len(metrics),
            "n_valid": len(vals_psnr),
            "psnr_mean": float(np.mean(vals_psnr)) if vals_psnr else None,
            "psnr_std": float(np.std(vals_psnr)) if vals_psnr else None,
            "ssim_mean": float(np.mean(vals_ssim)) if vals_ssim else None,
            "ssim_std": float(np.std(vals_ssim)) if vals_ssim else None,
            "lpips_mean": float(np.mean(vals_lp)) if vals_lp else None,
            "lpips_std": float(np.std(vals_lp)) if vals_lp else None,
        }

    summary = {
        "scene": scene_name,
        "iters": iters,
        "schedule": schedule,
        "modes": {},
    }

    if mode in ("chain", "both"):
        chain_metrics = run_mode_chain()
        with open(os.path.join(out_root, "metrics_chain.json"), "w") as f:
            json.dump({"summary": summarize(chain_metrics), "per_iter": chain_metrics}, f, indent=2)
        with open(os.path.join(out_root, "metrics_chain.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["iter", "frame_id", "image_path", "psnr", "ssim", "lpips"])
            writer.writeheader()
            writer.writerows(chain_metrics)
        summary["modes"]["chain"] = summarize(chain_metrics)

    if mode in ("fixed_buffer", "both"):
        fixed_metrics = run_mode_fixed_buffer()
        with open(os.path.join(out_root, "metrics_fixed_buffer.json"), "w") as f:
            json.dump({"summary": summarize(fixed_metrics), "per_iter": fixed_metrics}, f, indent=2)
        with open(os.path.join(out_root, "metrics_fixed_buffer.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["iter", "frame_id", "image_path", "psnr", "ssim", "lpips"])
            writer.writeheader()
            writer.writerows(fixed_metrics)
        summary["modes"]["fixed_buffer"] = summarize(fixed_metrics)

    with open(os.path.join(out_root, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[degrade] done. outputs: {out_root}")


def forward_rendering(args):
    """Run forward rendering with per-frame environment maps."""
    import omegaconf
    from omegaconf import OmegaConf
    from accelerate import Accelerator, PartialState
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from peft import LoraConfig
    from peft.utils import set_peft_model_state_dict
    from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
    from src.models.custom_unet_st import UNetCustomSpatioTemporalConditionModel
    from src.models.env_encoder import EnvEncoder
    from src.pipelines.pipeline_rgbx import RGBXVideoDiffusionPipeline
    from utils.utils_rgbx import convert_rgba_to_rgb_pil
    from utils.utils_rgbx_inference import (
        touch, find_images_recursive, base_plus_ext,
        group_images_into_videos, split_list_with_overlap,
        resize_upscale_without_padding,
    )
    from src.data.rendering_utils import envmap_vec

    cfg = OmegaConf.load(args.config)
    cli_overrides = []
    if args.extra:
        cli_overrides = args.extra
    cli = OmegaConf.from_dotlist(cli_overrides)
    cfg = OmegaConf.merge(cfg, cli)

    data_dir = args.data_dir
    workspace = args.workspace
    delighting_dir = os.path.join(workspace, "delighting")
    output_dir = os.path.join(workspace, "relighting")

    try:
        inference_height, inference_width = cfg.inference_res
    except Exception:
        inference_height, inference_width = 512, 512

    accelerator = Accelerator()
    cond_mode = cfg.get("cond_mode", "env")
    use_deterministic_mode = cfg.get("use_deterministic_mode", False)

    weight_dtype = torch.float16 if cfg.get("autocast", True) else torch.float32

    # ── Build model pipeline ──
    missing_kwargs = {"cond_mode": cond_mode, "use_deterministic_mode": use_deterministic_mode}
    model_weights_subfolders = os.listdir(cfg.inference_model_weights)
    distributed_state = PartialState()

    text_encoder, image_encoder, vae, env_encoder = None, None, None, None
    tokenizer, feature_extractor = None, None

    if cond_mode == "env":
        env_encoder = EnvEncoder.from_pretrained(
            cfg.inference_model_weights, subfolder="env_encoder"
        )
    elif cond_mode == "image":
        feature_extractor = CLIPImageProcessor.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="feature_extractor"
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="image_encoder"
        )

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        cfg.inference_model_weights, subfolder="vae"
    )
    unet = UNetCustomSpatioTemporalConditionModel.from_pretrained(
        cfg.inference_model_weights, subfolder="unet"
    )
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        cfg.inference_model_weights, subfolder="scheduler"
    )

    for module in [text_encoder, image_encoder, vae, unet, env_encoder]:
        if module is not None:
            module.to(distributed_state.device, dtype=weight_dtype)

    pipeline = RGBXVideoDiffusionPipeline(
        vae=vae,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        unet=unet,
        scheduler=noise_scheduler,
        env_encoder=env_encoder,
        scale_cond_latents=cfg.model_pipeline.get("scale_cond_latents", False),
        cond_mode=cond_mode,
    )
    pipeline.scheduler.register_to_config(timestep_spacing="trailing")
    try:
        pipeline.load_lora_weights(
            cfg.inference_model_weights, subfolder="lora", adapter_name="real-lora"
        )
    except Exception:
        print("No LoRA weights found, using base weights")

    pipeline = pipeline.to(distributed_state.device).to(weight_dtype)
    pipeline.set_progress_bar_config(disable=True)

    inference_n_frames = cfg.get("inference_n_frames", 24)
    env_resolution = tuple(cfg.model_pipeline.get("env_resolution", [512, 512]))
    cond_labels = cfg.model_pipeline.cond_images
    cond_gbuf_labels = [x for x in list(cond_labels.keys()) if "env_" not in x]

    # ── Discover scenes ──
    scene_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    os.makedirs(output_dir, exist_ok=True)

    # ── Process each scene ──
    all_scenes = []
    for scene_name in scene_dirs:
        envmap_dir = os.path.join(data_dir, scene_name, "envmaps")
        if not os.path.isdir(envmap_dir):
            print(f"[WARN] {scene_name}: no envmaps/ folder, skipping")
            continue

        scene_delighting = os.path.join(delighting_dir, scene_name)
        if not os.path.isdir(scene_delighting):
            print(f"[WARN] {scene_name}: no delighting output, skipping")
            continue

        all_scenes.append(scene_name)

    with distributed_state.split_between_processes(all_scenes) as local_scenes:
        for scene_name in tqdm(local_scenes, desc="Forward rendering"):
            scene_output_dir = os.path.join(output_dir, scene_name)
            if os.path.exists(os.path.join(scene_output_dir, "done.txt")):
                print(f"Skipping {scene_name}: already processed")
                continue

            envmap_dir = os.path.join(data_dir, scene_name, "envmaps")
            scene_delighting = os.path.join(delighting_dir, scene_name)

            # Find all G-buffer files and determine available frames
            all_files = sorted(os.listdir(scene_delighting))
            frame_set = set()
            for f in all_files:
                parts = f.split(".")
                if len(parts) >= 4:
                    frame_set.add((parts[0], parts[1]))  # (chunk, frame_idx)

            frame_keys = sorted(frame_set)
            if not frame_keys:
                print(f"[WARN] {scene_name}: no G-buffer frames found, skipping")
                continue

            n_actual_frames = len(frame_keys)
            n_frames = inference_n_frames

            # Load G-buffers, padding to inference_n_frames by repeating last frame
            cond_images = {}
            width, height = None, None
            for gbuf_label in cond_gbuf_labels:
                gbuf_list = []
                for i in range(min(n_actual_frames, n_frames)):
                    chunk_id, frame_id = frame_keys[i]
                    gbuf_path = os.path.join(
                        scene_delighting, f"{chunk_id}.{frame_id}.{gbuf_label}.png"
                    )
                    if not os.path.exists(gbuf_path):
                        print(f"[WARN] {scene_name}: missing {gbuf_label} for frame {frame_id}")
                        break
                    img = Image.open(gbuf_path).convert("RGB")
                    if width is None:
                        width, height = img.size
                        if width != inference_width or height != inference_height:
                            img = resize_upscale_without_padding(
                                img, inference_height, inference_width
                            )
                            width, height = img.size
                    else:
                        if img.size != (width, height):
                            img = img.resize((width, height), Image.BILINEAR)
                    gbuf_list.append(np.asarray(img))

                if not gbuf_list:
                    break

                # Pad by repeating last frame
                while len(gbuf_list) < n_frames:
                    gbuf_list.append(gbuf_list[-1])

                gbuf_array = (
                    np.stack(gbuf_list, axis=0)[None, ...].astype(np.float32) / 255.0
                )
                if cfg.get("use_fixed_frame_ind", False):
                    fi = cfg.get("fixed_frame_ind", 0)
                    gbuf_array = np.concatenate(
                        [gbuf_array[:, fi : fi + 1, ...]] * n_frames, axis=1
                    )
                cond_images[gbuf_label] = (
                    torch.from_numpy(gbuf_array)
                    .permute(0, 1, 4, 2, 3)
                    .to(distributed_state.device)
                )

            if len(cond_images) < len(cond_gbuf_labels):
                print(f"[WARN] {scene_name}: incomplete G-buffers, skipping")
                continue

            # Determine frame IDs for env map lookup
            # The original input images were named like 00006.png, 00053.png, etc.
            # The delighting output is named as 0000.0000.rgb.png, 0000.0001.rgb.png, etc.
            # We need to map delighting frame indices back to original frame IDs.
            input_images_dir = os.path.join(data_dir, scene_name, "input_images")
            original_frame_ids = sorted([
                os.path.splitext(f)[0]
                for f in os.listdir(input_images_dir)
                if f.endswith(".png")
            ])

            # Use actual frame IDs; pad to inference_n_frames by repeating last
            env_frame_ids = original_frame_ids[:n_frames]
            while len(env_frame_ids) < n_frames:
                env_frame_ids.append(env_frame_ids[-1])

            # Load per-frame env maps (already padded)
            try:
                env_ldr, env_log = load_per_frame_envmaps(
                    os.path.join(data_dir, scene_name, "envmaps"),
                    env_frame_ids,
                    env_resolution[0],
                    env_resolution[1],
                    distributed_state.device,
                )
            except FileNotFoundError as e:
                print(f"[WARN] {scene_name}: {e}, skipping")
                continue

            cond_images["env_ldr"] = env_ldr
            cond_images["env_log"] = env_log

            env_nrm = envmap_vec(env_resolution, device=distributed_state.device) * 0.5 + 0.5
            cond_images["env_nrm"] = (
                env_nrm.unsqueeze(0)
                .unsqueeze(0)
                .permute(0, 1, 4, 2, 3)
                .expand_as(env_ldr)
            )

            # Run inference
            if cfg.seed is not None:
                generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)
            else:
                generator = None

            if torch.backends.mps.is_available():
                autocast_ctx = nullcontext()
            else:
                autocast_ctx = torch.autocast(
                    accelerator.device.type, enabled=cfg.get("autocast", True)
                )

            with autocast_ctx:
                inference_image_list = pipeline(
                    cond_images,
                    cond_labels,
                    height=height,
                    width=width,
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

            # Save only the actual (non-padded) frames that have ground truth
            # Composite onto white background using GT alpha for fair comparison
            target_dir = os.path.join(data_dir, scene_name, "target_images")
            n_save = min(len(original_frame_ids), len(inference_image_list))
            os.makedirs(scene_output_dir, exist_ok=True)
            viz_frames = []
            for idx in range(n_save):
                fid = original_frame_ids[idx]
                pil_img = inference_image_list[idx]
                pred_arr = np.asarray(pil_img.convert("RGB"))
                gt_path = os.path.join(target_dir, f"{fid}.png")
                if os.path.exists(gt_path):
                    gt_pil = Image.open(gt_path)
                    if gt_pil.mode == "RGBA":
                        alpha = np.asarray(gt_pil)[:, :, 3]
                        pred_arr = composite_on_white(pred_arr, alpha)
                save_path = os.path.join(scene_output_dir, f"{fid}.png")
                Image.fromarray(pred_arr).save(save_path)
                viz_frames.append(pred_arr)

            if cfg.get("save_video", True):
                video_path = os.path.join(scene_output_dir, "viz.mp4")
                imageio.mimsave(
                    video_path,
                    viz_frames,
                    fps=cfg.get("save_video_fps", 10),
                    codec="h264",
                )

            touch(os.path.join(scene_output_dir, "done.txt"))
            print(f"[OK] {scene_name}: {n_save} frames rendered")


# ─────────────────────── Step 3: evaluate ──────────────────────────


def composite_on_white(rgb, alpha, bg_value=255):
    """
    Composite RGB image onto white background using alpha mask.

    Args:
        rgb: np array [H, W, 3] uint8
        alpha: np array [H, W] or [H, W, 1], values in [0, 255]
    Returns:
        Compositied image [H, W, 3] uint8
    """
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    alpha = np.clip(alpha.astype(np.float32) / 255.0, 0, 1)
    rgb_f = rgb.astype(np.float32) / 255.0
    bg = np.full((*rgb.shape[:2], rgb.shape[2]), bg_value / 255.0, dtype=np.float32)
    a = alpha[:, :, np.newaxis]
    out = (rgb_f * a + bg * (1 - a)).clip(0, 1) * 255
    return out.astype(np.uint8)


def compute_psnr(img1, img2):
    """Compute PSNR between two uint8 images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(img1, img2):
    """Compute SSIM using skimage."""
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, channel_axis=2, data_range=255)


def compute_lpips_batch(pred_images, gt_images, lpips_fn, device):
    """Compute LPIPS for a batch of image pairs."""
    scores = []
    for pred, gt in zip(pred_images, gt_images):
        pred_t = torch.from_numpy(pred).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        gt_t = torch.from_numpy(gt).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        pred_t = pred_t.to(device) * 2 - 1  # LPIPS expects [-1, 1]
        gt_t = gt_t.to(device) * 2 - 1
        with torch.no_grad():
            score = lpips_fn(pred_t, gt_t).item()
        scores.append(score)
    return scores


def evaluate(args):
    """Compute PSNR, SSIM, LPIPS between predictions and ground truth."""
    import lpips as lpips_module

    data_dir = args.data_dir
    workspace = args.workspace
    output_dir = os.path.join(workspace, "relighting")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips_module.LPIPS(net="alex").to(device)

    scene_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    all_psnr, all_ssim, all_lpips = [], [], []
    per_scene_results = {}

    for scene_name in tqdm(scene_dirs, desc="Evaluating"):
        target_dir = os.path.join(data_dir, scene_name, "target_images")
        pred_dir = os.path.join(output_dir, scene_name)

        if not os.path.isdir(target_dir) or not os.path.isdir(pred_dir):
            continue

        target_files = sorted([
            f for f in os.listdir(target_dir) if f.endswith(".png")
        ])

        scene_psnr, scene_ssim, scene_lpips = [], [], []
        pred_imgs, gt_imgs = [], []

        for tf in target_files:
            gt_path = os.path.join(target_dir, tf)
            pred_path = os.path.join(pred_dir, tf)
            if not os.path.exists(pred_path):
                continue

            gt_pil = Image.open(gt_path)
            pred_img = np.asarray(Image.open(pred_path).convert("RGB"))

            if gt_pil.mode == "RGBA":
                gt_arr = np.asarray(gt_pil)
                alpha = gt_arr[:, :, 3]
                gt_img = composite_on_white(gt_arr[:, :, :3], alpha)
                pred_img = composite_on_white(pred_img, alpha)
            else:
                gt_img = np.asarray(gt_pil.convert("RGB"))

            if gt_img.shape != pred_img.shape:
                pred_img = np.asarray(
                    Image.fromarray(pred_img).resize(
                        (gt_img.shape[1], gt_img.shape[0]), Image.BILINEAR
                    )
                )

            scene_psnr.append(compute_psnr(pred_img, gt_img))
            scene_ssim.append(compute_ssim(pred_img, gt_img))
            pred_imgs.append(pred_img)
            gt_imgs.append(gt_img)

        if pred_imgs:
            scene_lpips_scores = compute_lpips_batch(
                pred_imgs, gt_imgs, lpips_fn, device
            )
            scene_lpips.extend(scene_lpips_scores)

        if scene_psnr:
            mean_psnr = np.mean(scene_psnr)
            mean_ssim = np.mean(scene_ssim)
            mean_lpips = np.mean(scene_lpips) if scene_lpips else float("nan")

            per_scene_results[scene_name] = {
                "psnr": float(mean_psnr),
                "ssim": float(mean_ssim),
                "lpips": float(mean_lpips),
                "n_frames": len(scene_psnr),
            }
            all_psnr.extend(scene_psnr)
            all_ssim.extend(scene_ssim)
            all_lpips.extend(scene_lpips)

            print(
                f"  {scene_name}: PSNR={mean_psnr:.2f} SSIM={mean_ssim:.4f} "
                f"LPIPS={mean_lpips:.4f} ({len(scene_psnr)} frames)"
            )

    # Aggregate
    if all_psnr:
        print("\n" + "=" * 60)
        print(f"Overall ({len(per_scene_results)} scenes, {len(all_psnr)} frames):")
        print(f"  PSNR:  {np.mean(all_psnr):.2f} +/- {np.std(all_psnr):.2f}")
        print(f"  SSIM:  {np.mean(all_ssim):.4f} +/- {np.std(all_ssim):.4f}")
        if all_lpips:
            print(
                f"  LPIPS: {np.mean(all_lpips):.4f} +/- {np.std(all_lpips):.4f}"
            )
        print("=" * 60)

        results = {
            "overall": {
                "psnr_mean": float(np.mean(all_psnr)),
                "psnr_std": float(np.std(all_psnr)),
                "ssim_mean": float(np.mean(all_ssim)),
                "ssim_std": float(np.std(all_ssim)),
                "lpips_mean": float(np.mean(all_lpips)) if all_lpips else None,
                "lpips_std": float(np.std(all_lpips)) if all_lpips else None,
                "n_scenes": len(per_scene_results),
                "n_frames": len(all_psnr),
            },
            "per_scene": per_scene_results,
        }
        results_path = os.path.join(workspace, "eval_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
    else:
        print("No valid scenes found for evaluation.")


# ─────────────────────── CLI ──────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run DiffusionRenderer experiment on polyhaven data"
    )
    subparsers = parser.add_subparsers(dest="step", required=True)

    # prepare
    p_prep = subparsers.add_parser("prepare", help="Prepare inverse rendering input")
    p_prep.add_argument("--data_dir", required=True, help="Path to source_data_polyhaven")
    p_prep.add_argument(
        "--workspace", default="polyhaven_workspace", help="Workspace directory"
    )

    # forward
    p_fwd = subparsers.add_parser(
        "forward", help="Forward rendering with per-frame env maps"
    )
    p_fwd.add_argument("--data_dir", required=True, help="Path to source_data_polyhaven")
    p_fwd.add_argument(
        "--workspace", default="polyhaven_workspace", help="Workspace directory"
    )
    p_fwd.add_argument(
        "--config",
        default="configs/xrgb_inference.yaml",
        help="Forward rendering config",
    )
    p_fwd.add_argument("extra", nargs="*", help="Extra omegaconf overrides")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Compute evaluation metrics")
    p_eval.add_argument("--data_dir", required=True, help="Path to source_data_polyhaven")
    p_eval.add_argument(
        "--workspace", default="polyhaven_workspace", help="Workspace directory"
    )

    # degradation
    p_deg = subparsers.add_parser(
        "degrade", help="Run iterative degradation experiments on one scene"
    )
    p_deg.add_argument("--data_dir", required=True, help="Path to source_data_polyhaven")
    p_deg.add_argument("--workspace", default="polyhaven_workspace", help="Workspace directory")
    p_deg.add_argument("--scene", required=True, help="Scene folder name in data_dir")
    p_deg.add_argument("--iters", type=int, default=50, help="Number of relighting iterations")
    p_deg.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["chain", "fixed_buffer", "both"],
        help="Degradation mode",
    )
    p_deg.add_argument(
        "--config_rgbx",
        default="configs/rgbx_inference.yaml",
        help="Inverse rendering config",
    )
    p_deg.add_argument(
        "--config_xrgb",
        default="configs/xrgb_inference.yaml",
        help="Forward rendering config",
    )
    p_deg.add_argument(
        "--save_video",
        action="store_true",
        help="Save iteration videos",
    )
    p_deg.add_argument("--video_fps", type=int, default=10, help="Video FPS")

    args = parser.parse_args()

    if args.step == "prepare":
        prepare(args)
    elif args.step == "forward":
        forward_rendering(args)
    elif args.step == "evaluate":
        evaluate(args)
    elif args.step == "degrade":
        run_degradation_experiment(args)


if __name__ == "__main__":
    main()
