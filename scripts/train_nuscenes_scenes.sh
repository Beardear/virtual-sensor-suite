#!/bin/bash
# Train SplatAD on multiple diverse nuScenes scenes (sequential)
# Usage: bash train_nuscenes_scenes.sh [scene_id ...]
# Default: trains on the curated diverse set

set -e
source /workspace/SplatAD/venv_SplatAD/bin/activate
cd /workspace/SplatAD/neurad-studio

OUTPUT_BASE="/workspace/SplatAD/neurad-studio/outputs"

# Curated diverse scene set (see devlog 2026-04-11)
declare -A SCENES
SCENES[0796]="construction-singapore"    # Construction, bus, cars, scooter (singapore-queenstown)
SCENES[0712]="busy-intersection-boston"   # Day intersection, bridge, cones (boston-seaport)
SCENES[0637]="rain-reflections-boston"    # Rain, water reflections, jaywalker (boston-seaport)
SCENES[1050]="night-hollandvillage"      # Night, bus, truck, scooter (singapore-hollandvillage)
SCENES[1053]="night-rain-hollandvillage" # Night+rain, congestion (singapore-hollandvillage)

# Use provided scenes or default set
if [ $# -gt 0 ]; then
    SCENE_IDS=("$@")
else
    SCENE_IDS=(0796 0712 0637 1050 1053)
fi

for scene_id in "${SCENE_IDS[@]}"; do
    label="${SCENES[$scene_id]:-scene}"
    exp_name="nuscenes-${scene_id}-${label}"

    # Check if already trained (has a checkpoint)
    existing=$(find "$OUTPUT_BASE/$exp_name" -name "step-*.ckpt" 2>/dev/null | head -1)
    if [ -n "$existing" ]; then
        echo "[$exp_name] Already has checkpoint, skipping: $existing"
        continue
    fi

    echo "=========================================="
    echo "Training: $exp_name (scene-$scene_id)"
    echo "=========================================="

    log_file="$OUTPUT_BASE/train_${scene_id}.log"

    PYTHONUNBUFFERED=1 ns-train splatad \
        --experiment-name "$exp_name" \
        --vis tensorboard \
        --output-dir "$OUTPUT_BASE" \
        nuscenes-data \
        --data /workspace/SplatAD/data/nuscenes \
        --sequence "$scene_id" \
        --version v1.0-trainval \
        2>&1 | tee "$log_file"

    echo ""
    echo "[$exp_name] Training complete."
    echo ""
done

echo "All scenes trained. Run eval_nuscenes.sh to compute metrics."
