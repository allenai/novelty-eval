#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

CONFIG="${PROJECT_ROOT}/src/novelty_eval/retrieval/retrieve_candidates.yaml"

if [[ "$1" == "--config" ]]; then
  CONFIG="$2"
fi

python3 "${PROJECT_ROOT}/src/novelty_eval/retrieval/retrieve_candidates.py" \
  --config "$CONFIG"
