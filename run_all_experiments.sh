#!/usr/bin/env bash
set -euo pipefail

# Root of the repository
cd "$(dirname "$0")"

# Activate virtual environment if present
if [[ -f "venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "venv/bin/activate"
fi

CONFIG_DIR="./configs"
RESULTS_DIR="./results"
PROPOSED_METHOD="fedasync"
FAILED_CONFIGS=()

mkdir -p "${RESULTS_DIR}/metrics"
mkdir -p "${RESULTS_DIR}/plots"

echo "Running all config files under ${CONFIG_DIR}..."
for cfg in "${CONFIG_DIR}"/*.yaml; do
  if [[ ! -f "${cfg}" ]]; then
    continue
  fi

  echo
  echo "================================================================"
  echo "Running config: ${cfg}"
  echo "================================================================"

  set +e
  python main.py --config "${cfg}"
  exit_code=$?
  set -e

  if [[ ${exit_code} -ne 0 ]]; then
    echo "[ERROR] Experiment failed for config: ${cfg} (exit code: ${exit_code})"
    FAILED_CONFIGS+=("${cfg}")
  else
    echo "[OK] Completed config: ${cfg}"
  fi

done

echo
if [[ ${#FAILED_CONFIGS[@]} -gt 0 ]]; then
  echo "WARNING: ${#FAILED_CONFIGS[@]} config(s) failed during execution:"
  for failed in "${FAILED_CONFIGS[@]}"; do
    echo "  - ${failed}"
  done
  echo "Proceeding to generate figures from successful experiments..."
fi

echo "Generating final aggregated figures and results table..."
python generate_figures.py --results-dir "${RESULTS_DIR}" --proposed "${PROPOSED_METHOD}"

echo
if [[ ${#FAILED_CONFIGS[@]} -gt 0 ]]; then
  echo "Completed with some failures. See above for failed configs."
  echo "Results directory: ${RESULTS_DIR}"
  echo "Metrics CSV: ${RESULTS_DIR}/metrics/results_table.csv"
  echo "Plots directory: ${RESULTS_DIR}/plots"
  exit 1
fi

echo "All done."
