#!/bin/bash
# GPT + web_search (arXiv) novelty judge that emits its cited papers, grounded via
# Semantic Scholar.
#
# Usage:
#   ./run_web_search_novelty_judge.sh                          # live mode
#   ./run_web_search_novelty_judge.sh --mode submit            # submit batch
#   ./run_web_search_novelty_judge.sh --mode collect           # collect batch
#   ./run_web_search_novelty_judge.sh --config <path> --mode submit
#   ./run_web_search_novelty_judge.sh --mode collect --batch-id <id>

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

CONFIG="${PROJECT_ROOT}/src/novelty_eval/retrieval/web_search_novelty_judge.yaml"

# Pull out an optional --config; pass everything else (--mode, --batch-id) through.
PASS_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    *) PASS_ARGS+=("$1"); shift ;;
  esac
done

python3 "${PROJECT_ROOT}/src/novelty_eval/retrieval/web_search_novelty_judge.py" \
  --config "$CONFIG" "${PASS_ARGS[@]}"
