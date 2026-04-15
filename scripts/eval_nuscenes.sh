#!/bin/bash
# Batch evaluation of SplatAD checkpoints on nuScenes scenes
# Usage: bash eval_nuscenes.sh [checkpoint_dir]
# If no checkpoint_dir is given, evaluates all nuscenes-* experiments in the default output dir.

set -e
source /workspace/SplatAD/venv_SplatAD/bin/activate

OUTPUT_BASE="/workspace/SplatAD/neurad-studio/outputs"
EVAL_DIR="/workspace/virtual-sensor-suite/eval_results"
mkdir -p "$EVAL_DIR"

# Find all nuScenes SplatAD configs
if [ -n "$1" ]; then
    CONFIGS=("$1/config.yml")
else
    CONFIGS=($(find "$OUTPUT_BASE" -path "*/nuscenes-*/splatad/*/config.yml" -type f 2>/dev/null | sort))
fi

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "No nuScenes SplatAD configs found."
    exit 1
fi

echo "Found ${#CONFIGS[@]} checkpoint(s) to evaluate:"
for cfg in "${CONFIGS[@]}"; do
    echo "  $cfg"
done
echo ""

for cfg in "${CONFIGS[@]}"; do
    # Extract experiment name from path
    exp_name=$(echo "$cfg" | grep -oP 'nuscenes-[^/]+')
    timestamp=$(echo "$cfg" | grep -oP '\d{4}-\d{2}-\d{2}_\d{6}')
    run_id="${exp_name}_${timestamp}"

    echo "=========================================="
    echo "Evaluating: $run_id"
    echo "=========================================="

    metrics_file="$EVAL_DIR/${run_id}_metrics.json"
    render_dir="$EVAL_DIR/${run_id}_renders"

    if [ -f "$metrics_file" ]; then
        echo "  Metrics already exist, skipping. Delete $metrics_file to re-evaluate."
        continue
    fi

    # Run evaluation
    ns-eval \
        --load-config "$cfg" \
        --output-path "$metrics_file" \
        --render-output-path "$render_dir"

    echo "  Saved metrics to: $metrics_file"
    echo "  Saved renders to: $render_dir"
    echo ""
done

# Aggregate results
echo ""
echo "=========================================="
echo "RESULTS SUMMARY"
echo "=========================================="

python3 << 'PYEOF'
import json, glob, os

eval_dir = os.environ.get("EVAL_DIR", "/workspace/virtual-sensor-suite/eval_results")
results = []

for f in sorted(glob.glob(os.path.join(eval_dir, "*_metrics.json"))):
    with open(f) as fp:
        data = json.load(fp)
    r = data.get("results", {})
    results.append({
        "experiment": data.get("experiment_name", "?"),
        "psnr": r.get("psnr", "N/A"),
        "ssim": r.get("ssim", "N/A"),
        "lpips": r.get("lpips", "N/A"),
        "chamfer": r.get("chamfer_distance", "N/A"),
        "intensity_rmse": r.get("intensity_rmse", "N/A"),
        "ray_drop_acc": r.get("ray_drop_accuracy", "N/A"),
    })

if not results:
    print("No results found yet.")
else:
    # Print table
    header = f"{'Experiment':<40} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'Chamfer':>10} {'Int RMSE':>10} {'RayDrop':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        print(f"{r['experiment']:<40} {fmt(r['psnr']):>8} {fmt(r['ssim']):>8} {fmt(r['lpips']):>8} {fmt(r['chamfer']):>10} {fmt(r['intensity_rmse']):>10} {fmt(r['ray_drop_acc']):>10}")
PYEOF

# Export PLY for SuperSplat visualization
echo ""
echo "=========================================="
echo "PLY EXPORT (for SuperSplat)"
echo "=========================================="

for cfg in "${CONFIGS[@]}"; do
    exp_name=$(echo "$cfg" | grep -oP 'nuscenes-[^/]+')
    timestamp=$(echo "$cfg" | grep -oP '\d{4}-\d{2}-\d{2}_\d{6}')
    run_id="${exp_name}_${timestamp}"
    ply_dir="$EVAL_DIR/${run_id}_ply"

    if [ -f "$ply_dir/splat.ply" ]; then
        echo "  PLY already exists for $run_id, skipping."
        continue
    fi

    echo "Exporting PLY for $run_id..."
    ns-export gaussian-splat \
        --load-config "$cfg" \
        --output-dir "$ply_dir"
    echo "  Saved to: $ply_dir/splat.ply"
done
