#!/bin/bash
set -e

# Degradation experiment for DiffusionRenderer
# Runs both iterative and control experiments on the polyhaven dataset.
#
# Adjust SCENES, OBJECT, N_ITERS as needed.

DATA_DIR="source_data_polyhaven"
WORKSPACE="degradation_workspace"
RGBX_CFG="configs/rgbx_inference.yaml"
XRGB_CFG="configs/xrgb_inference.yaml"

# Which scene(s) to test (iterative experiment)
SCENES="all_purpose_cleaner_env_0"

# Object name and source variant (control experiment)
OBJECT="all_purpose_cleaner"
SOURCE_VARIANT="env_0"

# Number of round-trip iterations
N_ITERS=50

# Save G-buffer snapshots every N iterations
SAVE_BUF_EVERY=5

# Add --swap_models if GPU memory is tight
EXTRA_FLAGS=""

echo "=============================================="
echo " Experiment 1: Iterative Degradation"
echo "=============================================="
python run_degradation_experiment.py iterative \
    --data_dir "$DATA_DIR" \
    --workspace "$WORKSPACE" \
    --scenes $SCENES \
    --n_iterations $N_ITERS \
    --save_buffers_every $SAVE_BUF_EVERY \
    --rgbx_config "$RGBX_CFG" \
    --xrgb_config "$XRGB_CFG" \
    $EXTRA_FLAGS

echo ""
echo "=============================================="
echo " Experiment 2: Control (Same Buffer)"
echo "=============================================="
python run_degradation_experiment.py control \
    --data_dir "$DATA_DIR" \
    --workspace "$WORKSPACE" \
    --object "$OBJECT" \
    --source_variant "$SOURCE_VARIANT" \
    --n_iterations $N_ITERS \
    --rgbx_config "$RGBX_CFG" \
    --xrgb_config "$XRGB_CFG" \
    $EXTRA_FLAGS

echo ""
echo "=============================================="
echo " Generating comparison plots"
echo "=============================================="
python run_degradation_experiment.py plot \
    --workspace "$WORKSPACE" \
    --scene "$SCENES" \
    --object "$OBJECT"

echo ""
echo "Done! Results in: $WORKSPACE/"
echo "  Iterative: $WORKSPACE/iterative/$SCENES/summary.json"
echo "  Control:   $WORKSPACE/control/$OBJECT/summary.json"
echo "  Plots:     $WORKSPACE/plots/"
