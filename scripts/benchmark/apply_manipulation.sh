#!/usr/bin/env bash
# Apply an abstract manipulation prompt to every idea in an existing dataset YAML.
#
# Reads a pairwise or pointwise benchmark_instances.yaml, calls the LLM for
# each unique abstract using the given Jinja2 template, and writes a new YAML
# with manipulated abstracts alongside a manipulation_debug.md report.
#
# Usage:
#   ./scripts/apply_manipulation.sh \
#       --dataset output/benchmark_instances/pairwise_data/<ts>/benchmark_instances.yaml \
#       --template src/novelty_eval/benchmark_data/templates/extract_research_plan.jinja2 \
#       [--model gpt-4o-mini] \
#       [--output path/to/output.yaml] \
#       [--max-workers 8]
#
# Default output: <dataset_dir>/manipulated/benchmark_instances.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/src"
export SECRETS="${PROJECT_ROOT}/secrets.toml"
export OUTPUT_DIR="${PROJECT_ROOT}/output"

APPLY_SCRIPT="${PROJECT_ROOT}/src/novelty_eval/benchmark_data/utility/apply_manipulation.py"

python3 "$APPLY_SCRIPT" "$@"
