#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-250000}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results}"
RENDER_MODE="${RENDER_MODE:-}"

VARIANTS=(easy medium hard)
RUN_SEEDS=(0 1 2)
WARNING_RADII=(0.10 0.20 0.28 0.40 0.60 0.80)

slugify_number() {
  printf '%s' "$1" | tr '.' 'p' | tr '-' 'm'
}

TOTAL_RUNS=$(( ${#VARIANTS[@]} * ${#RUN_SEEDS[@]} * ${#WARNING_RADII[@]} ))
RUN_INDEX=0
COMPLETED_RUNS=0
SKIPPED_RUNS=0
START_TIME="$(date +%s)"

format_duration() {
  local seconds="$1"
  printf '%02d:%02d:%02d' \
    "$((seconds / 3600))" \
    "$(((seconds % 3600) / 60))" \
    "$((seconds % 60))"
}

print_progress() {
  local status="$1"
  local elapsed
  local remaining
  elapsed="$(( $(date +%s) - START_TIME ))"
  remaining="$(( TOTAL_RUNS - RUN_INDEX ))"
  echo "[sweep] progress status=${status} processed=${RUN_INDEX}/${TOTAL_RUNS} completed=${COMPLETED_RUNS} skipped=${SKIPPED_RUNS} remaining=${remaining} elapsed=$(format_duration "${elapsed}")"
}

echo "[sweep] shield total_runs=${TOTAL_RUNS} total_timesteps=${TOTAL_TIMESTEPS} save_freq=${SAVE_FREQ}"

for variant in "${VARIANTS[@]}"; do
  for seed in "${RUN_SEEDS[@]}"; do
    for radius in "${WARNING_RADII[@]}"; do
      RUN_INDEX=$((RUN_INDEX + 1))
      radius_slug="$(slugify_number "${radius}")"
      output_dir="${OUTPUT_ROOT}/shield/${variant}/seed${seed}_radius${radius_slug}"

      if [[ -f "${output_dir}/final_model.zip" && -f "${output_dir}/evaluation_manifest.json" ]]; then
        echo "[sweep] (${RUN_INDEX}/${TOTAL_RUNS}) skipping existing run ${output_dir}"
        SKIPPED_RUNS=$((SKIPPED_RUNS + 1))
        print_progress "skipped"
        continue
      fi

      cmd=(
        python scripts/train_baseline.py
        --baseline shield
        --variant "${variant}"
        --seed "${seed}"
        --total-timesteps "${TOTAL_TIMESTEPS}"
        --save-freq "${SAVE_FREQ}"
        --shield-warning-radius "${radius}"
        --output-dir "${output_dir}"
      )

      if [[ -n "${RENDER_MODE}" ]]; then
        cmd+=(--render "${RENDER_MODE}")
      fi

      echo "[sweep] (${RUN_INDEX}/${TOTAL_RUNS}) running shield variant=${variant} seed=${seed} radius=${radius}"
      printf '[sweep] command:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
      "${cmd[@]}"
      COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
      print_progress "completed"
    done
  done
done

echo "[sweep] shield sweep complete completed=${COMPLETED_RUNS} skipped=${SKIPPED_RUNS} elapsed=$(format_duration "$(( $(date +%s) - START_TIME ))")"
