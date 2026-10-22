#!/bin/bash

# Serve the blind human-annotation UI for judge novelty mistakes.
# See src/novelty_eval/judge_error_annotation/README.md
#
# Usage:
#   ./scripts/run_judge_error_annotation.sh <RUN_DIR> [OUT_CSV] [PORT]
#
# Example:
#   ./scripts/run_judge_error_annotation.sh \
#     output/ablation_sweeps/20260518_110534/pairwise-vanilla-ai_current/pairwise-vanilla-ai_current-claude-opus-4-6

set -euo pipefail

# Get the project root directory (assuming script is in scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Activate the conda env if conda is available; otherwise fall back to the
# system python3 (this tool only needs pyyaml + the stdlib).
for CONDA_SH in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "$HOME/opt/anaconda3/etc/profile.d/conda.sh" \
    "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"; do
    if [ -f "$CONDA_SH" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_SH"
        conda activate "${PROJECT_ROOT}/myenv" 2>/dev/null || conda activate myenv 2>/dev/null || true
        break
    fi
done

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

RUN_DIR="${1:-output/ablation_sweeps/20260518_110534/pairwise-vanilla-ai_current/pairwise-vanilla-ai_current-claude-opus-4-6}"
OUT_CSV="${2:-${PROJECT_ROOT}/output/judge_error_annotations/annotations.csv}"
PORT="${3:-8000}"

echo "Project Root: ${PROJECT_ROOT}"
echo "Run dir:      ${RUN_DIR}"
echo "Out CSV:      ${OUT_CSV}"

python3 -m novelty_eval.judge_error_annotation.annotate_judge_errors \
    --run-dir "${RUN_DIR}" \
    --out-csv "${OUT_CSV}" \
    --port "${PORT}"
