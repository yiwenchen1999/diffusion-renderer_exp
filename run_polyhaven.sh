#!/bin/bash
# Full pipeline to run DiffusionRenderer on source_data_polyhaven
# Optional: BENCHMARK=1 or --benchmark to run benchmark (100 frames, timing + peak GPU/CPU)
set -e

DATA_DIR="relight_metadata_smallsplit"
WORKSPACE="polyhaven_workspace_figure"
# Benchmark: 1 scene (~4 frames) for quick timing/resource check
N_FRAMES=4

# Check for benchmark mode
BENCHMARK=0
for arg in "$@"; do
    if [ "$arg" = "--benchmark" ]; then
        BENCHMARK=1
        break
    fi
done
if [ "$BENCHMARK" = "1" ] || [ -n "$BENCHMARK_MODE" ]; then
    BENCHMARK=1
fi

if [ "$BENCHMARK" = "1" ]; then
    echo "=== BENCHMARK MODE: ${N_FRAMES} frames ==="
    python run_polyhaven_experiment.py benchmark \
        --data_dir "$DATA_DIR" \
        --workspace "$WORKSPACE" \
        --n_frames $N_FRAMES \
        lora_scale=0.0
    echo ""
    echo "=== Done! Report in ${WORKSPACE}/benchmark_report.json ==="
    exit 0
fi

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
