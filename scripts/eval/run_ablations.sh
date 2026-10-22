#!/usr/bin/env bash
# Run the ablation study for comparative idea quality evaluation.
#
# Usage:
#   ./scripts/run_ablations.sh [args...]
#
# All arguments are passed through to run_ablations.py.
#
# Examples:
#   ./scripts/run_ablations.sh
#   ./scripts/run_ablations.sh --ablations current,c1
#   ./scripts/run_ablations.sh --skip-create --ablations current
#   ./scripts/run_ablations.sh --model gpt-5.2 --n-runs 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/src"
export SECRETS="${PROJECT_ROOT}/secrets.toml"
export OUTPUT_DIR="${PROJECT_ROOT}/output"

python3 "${PROJECT_ROOT}/src/novelty_eval/ablation/run_ablations.py" "$@"
