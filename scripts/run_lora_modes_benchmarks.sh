#!/usr/bin/env bash
set -euo pipefail

# Run LoRA experiments for MRPC and CoLA with ranks 4, 8, and 16
# in both standard LoRA mode and LoRA + truncated SVD initialization.
#
# Optional overrides:
#   PYTHON_BIN=python3
#   CONFIG=configs/train.yaml
#   OUTPUT_LOG=outputs/lora_modes_mrpc_cola.jsonl

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
if [ -x "${DEFAULT_PYTHON_BIN}" ]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
CONFIG_PATH="${CONFIG:-${ROOT_DIR}/configs/train.yaml}"
OUTPUT_LOG="${OUTPUT_LOG:-${ROOT_DIR}/outputs/lora_modes_mrpc_cola.jsonl}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_finetuning.py" \
  --config "${CONFIG_PATH}" \
  --tasks mrpc cola \
  --rank-sweep 4 8 16 \
  --pretraining-modes standard truncated_svd \
  --output-log "${OUTPUT_LOG}" \
  "$@"
