#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH -c1
#SBATCH --mem-per-cpu=4g

# Resolve project root relative to this script's location, not $PWD
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$PROJECT_ROOT/src"
export SECRETS="$PROJECT_ROOT/secrets.toml"
export OUTPUT_DIR="$PROJECT_ROOT/output"

# Run the accuracy sweep from the project root so all relative paths work
cd "$PROJECT_ROOT" || exit 1

python3.12 src/novelty_eval/run_benchmark_sweep.py \
  --config src/novelty_eval/config/accuracy_sweep_config.yaml \
  --parallel \
  --skip-on-error

