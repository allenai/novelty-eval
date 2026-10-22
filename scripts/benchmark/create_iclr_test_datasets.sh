#!/usr/bin/env bash
# Create multiple versions of the ICLR test dataset by running
# create_benchmark_instances.py with different configs in sequence,
# then copies all outputs into a single bundle folder for easy sharing.
#
# Usage:
#   ./scripts/create_iclr_test_datasets.sh
#
# Bundle location: output/benchmark_instances/bundle_<timestamp>/
#   Each config gets a subfolder named after the config file (without extension).
#
# Add or remove configs from the CONFIGS array below to control which
# dataset versions are generated.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/src"
export SECRETS="${PROJECT_ROOT}/secrets.toml"
export OUTPUT_DIR="${PROJECT_ROOT}/output"

ICLR_SCRIPT="${PROJECT_ROOT}/src/novelty_eval/benchmark_data/create_benchmark_instances.py"

# ── Configs to run ────────────────────────────────────────────────────────────
CONFIGS=(
    "${PROJECT_ROOT}/src/novelty_eval/benchmark_data/config/create_pairwise_benchmark_instances.yaml"
    "${PROJECT_ROOT}/src/novelty_eval/benchmark_data/config/create_pointwise_benchmark_instances.yaml"
)

# ── Bundle folder (created once, shared across all runs) ─────────────────────
BUNDLE_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_DIR="${PROJECT_ROOT}/output/benchmark_instances/bundle_${BUNDLE_TIMESTAMP}"
mkdir -p "$BUNDLE_DIR"
echo "Bundle folder: $BUNDLE_DIR"

# ── Run ───────────────────────────────────────────────────────────────────────
TOTAL=${#CONFIGS[@]}
FAILED=0

for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    IDX=$((i + 1))
    CONFIG_NAME="$(basename "$CONFIG" .yaml)"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$IDX/$TOTAL] $CONFIG_NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Read output_dir from the config (falls back to PROJECT_ROOT/output if missing)
    CONFIG_OUTPUT_DIR="$(python3 -c "
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('output_dir', '$PROJECT_ROOT/output'))
")"

    # Snapshot of existing timestamped subdirs before the run
    BEFORE="$(ls -d "${CONFIG_OUTPUT_DIR}"/[0-9]* 2>/dev/null || true)"

    if python3 "$ICLR_SCRIPT" --config "$CONFIG"; then
        echo "  ✓ Done: $CONFIG_NAME"

        # Find the new subdir created by this run (newest that wasn't there before)
        NEW_DIR="$(comm -13 \
            <(echo "$BEFORE" | sort) \
            <(ls -d "${CONFIG_OUTPUT_DIR}"/[0-9]* 2>/dev/null | sort) \
            | tail -1)"

        if [[ -n "$NEW_DIR" ]]; then
            DEST="${BUNDLE_DIR}/${CONFIG_NAME}"
            cp -r "$NEW_DIR" "$DEST"
            echo "  → Copied to bundle: ${CONFIG_NAME}/"
        else
            echo "  ⚠ Could not find new output dir to copy." >&2
        fi
    else
        echo "  ✗ FAILED: $CONFIG_NAME" >&2
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Completed $((TOTAL - FAILED))/$TOTAL configs successfully."
echo "  Bundle: $BUNDLE_DIR"
if [[ $FAILED -gt 0 ]]; then
    echo "  $FAILED config(s) failed — see output above."
    exit 1
fi
