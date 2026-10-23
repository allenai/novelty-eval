#!/usr/bin/env bash
# Build the dataset folder and upload it. Needs `hf auth login` with a write token.
#
# Usage:
#   ./scripts/hf/upload.sh <repo-id>     e.g. allenai/novelty-judge-bench
#
# The repo id is required: `hf upload` creates a missing repo as public.

set -euo pipefail

REPO_ID="${1:?usage: $0 <repo-id>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/output/hf_dataset"

python3 "${SCRIPT_DIR}/build_dataset.py" --data-dir "${PROJECT_ROOT}/data" --out "${BUILD_DIR}"

hf upload "${REPO_ID}" "${BUILD_DIR}" . --repo-type dataset
