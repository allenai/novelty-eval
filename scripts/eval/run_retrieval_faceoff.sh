#!/usr/bin/env bash
# Thin wrapper: run the retrieval face-off using the packaged driver + config.
# Usage: ./scripts/run_retrieval_faceoff.sh sample          # draw a matched subset only
#        ./scripts/run_retrieval_faceoff.sh submit [ROOT]   # start retrieval (defaults to latest sample)
#        ./scripts/run_retrieval_faceoff.sh poll [ROOT]     # (batch mode) collect + score
#        ./scripts/run_retrieval_faceoff.sh recompute [ROOT]  # re-score without re-running the LLM
#        ./scripts/run_retrieval_faceoff.sh resubmit [ROOT]   # (batch mode) retry dropped ideas, then poll
# In live mode (use_batch_api: false) `submit` runs the rest of the pipeline; `poll` is a no-op.
# Override the config with CONFIG=/path/to/config.yaml.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD/src}"
CONFIG="${CONFIG:-src/novelty_eval/retrieval_faceoff/config.yaml}"
exec python3 -m novelty_eval.retrieval_faceoff.run --config "$CONFIG" "$@"
