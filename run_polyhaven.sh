#!/bin/bash
# Full pipeline to run DiffusionRenderer on source_data_polyhaven
set -e

DATA_DIR="source_data_polyhaven"
WORKSPACE="polyhaven_workspace"

echo "=== Step 0: Prepare input structure ==="
python run_polyhaven_experiment.py prepare \
    --data_dir $DATA_DIR \
    --workspace $WORKSPACE

echo ""
echo "=== Step 1: Inverse rendering (estimate G-buffers) ==="
python inference_svd_rgbx.py --config configs/rgbx_inference.yaml \
    inference_input_dir=${WORKSPACE}/inverse_input \
    inference_save_dir=${WORKSPACE}/delighting \
    chunk_mode=all \
    model_passes="['basecolor','normal','depth','roughness','metallic']"

echo ""
echo "=== Step 2: Forward rendering (per-frame env maps) ==="
python run_polyhaven_experiment.py forward \
    --data_dir $DATA_DIR \
    --workspace $WORKSPACE \
    --config configs/xrgb_inference.yaml \
    lora_scale=0.0

echo ""
echo "=== Step 3: Evaluate (PSNR / SSIM / LPIPS) ==="
pip install lpips scikit-image -q
python run_polyhaven_experiment.py evaluate \
    --data_dir $DATA_DIR \
    --workspace $WORKSPACE

echo ""
echo "=== Done! Results in ${WORKSPACE}/eval_results.json ==="
