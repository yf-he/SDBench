#!/bin/bash
set -euo pipefail

# Usage:
#   bash scripts/run_five_models.sh /path/to/dataset.sdbench.jsonl [limit]
# Example:
#   bash scripts/run_five_models.sh \
#     \
#     "/Users/yufei/Desktop/SDBench/converted/test-00000-of-00001.sdbench.jsonl" 30

DATASET_PATH=${1:-""}
LIMIT=${2:-30}

if [ -z "$DATASET_PATH" ]; then
  echo "ERROR: Please provide dataset path (sdbench JSONL)." >&2
  exit 1
fi

# Ensure we run from repo root for imports
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "Dataset: $DATASET_PATH"
echo "Case limit: $LIMIT"

# Five target models (agent-side only). Gatekeeper/Judge fixed to 4o-mini in config.
python benchmark_agents.py \
  "$DATASET_PATH" \
  --models \
  openai/gpt-4o-mini \
  openai/gpt-5-mini \
  anthropic/claude-sonnet-4.5 \
  google/gemini-2.5-flash-preview-09-2025 \
  deepseek/deepseek-v3.2-exp \
  --limit "$LIMIT"

echo "All done."


