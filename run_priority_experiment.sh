#!/bin/bash
# Run DiffusionRenderer on priority sub-dataset for quick qualitative results

set -e

SOURCE_DATA="../LVSMExp/source_data_polyhaven"
PRIORITY_LIST="relight_metadata_smallsplit"
SUBSET_DATA="source_data_polyhaven_smallsplit"
WORKSPACE="polyhaven_workspace_smallsplit"

echo "=== Step 0: Create priority sub-dataset ==="
python create_priority_subset.py \
    --source "$SOURCE_DATA" \
    --priority_list "$PRIORITY_LIST" \
    --output "$SUBSET_DATA"

echo ""
echo "=== Step 1: Prepare inverse input ==="
python run_polyhaven_experiment.py prepare \
    --data_dir "$SUBSET_DATA" \
    --workspace "$WORKSPACE"

echo ""
echo "=== Step 2: Inverse rendering ==="
rm -rf "${WORKSPACE}/delighting"
python inference_svd_rgbx.py --config configs/rgbx_inference.yaml \
    inference_input_dir=${WORKSPACE}/inverse_input \
    inference_save_dir=${WORKSPACE}/delighting \
    chunk_mode=all overlap_n_frames=0 \
    model_passes="['basecolor','normal','depth','roughness','metallic']"

echo ""
echo "=== Step 3: Forward rendering ==="
python run_polyhaven_experiment.py forward \
    --data_dir "$SUBSET_DATA" \
    --workspace "$WORKSPACE" \
    --config configs/xrgb_inference.yaml \
    lora_scale=0.0

echo ""
echo "=== Step 4: Evaluate ==="
python run_polyhaven_experiment.py evaluate \
    --data_dir "$SUBSET_DATA" \
    --workspace "$WORKSPACE"

echo ""
echo "=== Done! Results in ${WORKSPACE}/eval_results.json ==="
