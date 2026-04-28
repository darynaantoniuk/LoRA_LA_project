#!/usr/bin/env bash
set -euo pipefail

# Evaluate every fine-tuned checkpoint under outputs/**/checkpoint-*
# Results are written to outputs/all_checkpoint_evals/*.json
#
# Usage:
#   bash scripts/evaluate_all_checkpoints.sh
#   CONFIG=configs/train.yaml PYTHON_BIN=python3 bash scripts/evaluate_all_checkpoints.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUTS_DIR="${ROOT_DIR}/outputs"
RESULTS_DIR="${OUTPUTS_DIR}/all_checkpoint_evals"
CONFIG_PATH="${CONFIG:-${ROOT_DIR}/configs/train.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${RESULTS_DIR}"

shopt -s nullglob
checkpoint_dirs=( "${OUTPUTS_DIR}"/*/checkpoint-* )

if [ "${#checkpoint_dirs[@]}" -eq 0 ]; then
  echo "No checkpoints found under ${OUTPUTS_DIR}"
  exit 0
fi

echo "Found ${#checkpoint_dirs[@]} checkpoint directories."

evaluated=0
skipped=0
failed=0

for checkpoint_path in "${checkpoint_dirs[@]}"; do
  if [ ! -d "${checkpoint_path}" ]; then
    continue
  fi

  run_dir_name="$(basename "$(dirname "${checkpoint_path}")")"
  checkpoint_name="$(basename "${checkpoint_path}")"

  # Extract task from pattern: <task>_(full|lora)_r<rank>
  task_name="$(echo "${run_dir_name}" | sed -E 's/_(full|lora)_r[0-9]+$//')"
  if [ -z "${task_name}" ] || [ "${task_name}" = "${run_dir_name}" ]; then
    echo "Skipping ${checkpoint_path} (cannot infer task from ${run_dir_name})"
    skipped=$((skipped + 1))
    continue
  fi

  output_file="${RESULTS_DIR}/${run_dir_name}_${checkpoint_name}.json"
  echo "Evaluating: task=${task_name} checkpoint=${checkpoint_path}"

  if "${PYTHON_BIN}" "${ROOT_DIR}/src/evaluate.py" \
      --config "${CONFIG_PATH}" \
      --task "${task_name}" \
      --checkpoint_path "${checkpoint_path}" \
      --output_file "${output_file}"; then
    evaluated=$((evaluated + 1))
  else
    echo "Failed: ${checkpoint_path}"
    failed=$((failed + 1))
  fi
done

echo
echo "Done."
echo "Evaluated: ${evaluated}"
echo "Skipped:   ${skipped}"
echo "Failed:    ${failed}"
echo "Results:   ${RESULTS_DIR}"

if [ "${failed}" -gt 0 ]; then
  exit 1
fi
