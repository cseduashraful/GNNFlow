#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  ./run_profile_sweep.sh [optional-config-file]

Description:
  Run a profiling sweep across multiple model / dataset / batch-size
  combinations. Edit the USER CONFIG block in this script directly, or pass a
  shell config file that overrides the variables.

Example:
  ./run_profile_sweep.sh
  ./run_profile_sweep.sh ./scripts/profile_sweep_config.sh
EOF
}

if [[ $# -gt 1 ]]; then
    usage
    exit 1
fi

#
# USER CONFIG
#
# Edit these values directly, or override them from an external config file.
#

DATASETS=(REDDIT)
MODELS=(TGN TGAT)
BATCH_SIZES=(4000 8000)

CACHE="LRUCache"
EDGE_CACHE_RATIO="0.2"
NODE_CACHE_RATIO="0.2"
TIME_WINDOW="0"
NPROC_PER_NODE="4"

EPOCH="50"
NUM_WORKERS="8"
NUM_CHUNKS="8"
PRINT_FREQ="100"
SEED="42"
INGESTION_BATCH_SIZE="10000000"

PROFILE_ONLY="1"
PROFILE_WAIT="1"
PROFILE_WARMUP="1"
PROFILE_ACTIVE="6"
PROFILE_REPEAT="1"
PROFILE_ROW_LIMIT="50"
PROFILE_GPU_SAMPLE_INTERVAL="0.2"
PROFILE_RECORD_SHAPES="1"
PROFILE_WITH_STACK="0"
PROFILE_WITH_FLOPS="0"
PROFILE_EXPORT_MEMORY_TIMELINE="0"

PYTHON_BIN="python"
LOG_DIR="${REPO_ROOT}/profiles/logs"
PROFILE_DIR_ROOT="${REPO_ROOT}/profiles"
DRY_RUN="0"

# Extra arguments appended after the explicit profiler arguments.
EXTRA_ARGS=()

if [[ $# -eq 1 ]]; then
    # shellcheck source=/dev/null
    source "$1"
fi

mkdir -p "${LOG_DIR}"
mkdir -p "${PROFILE_DIR_ROOT}"

printf 'Profiling sweep configuration\n'
printf '  datasets: %s\n' "${DATASETS[*]}"
printf '  models: %s\n' "${MODELS[*]}"
printf '  batch sizes: %s\n' "${BATCH_SIZES[*]}"
printf '  cache: %s\n' "${CACHE}"
printf '  world size: %s\n' "${NPROC_PER_NODE}"
printf '  profile-only: %s\n' "${PROFILE_ONLY}"
printf '  profile dir: %s\n' "${PROFILE_DIR_ROOT}"
printf '  log dir: %s\n' "${LOG_DIR}"

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for batch_size in "${BATCH_SIZES[@]}"; do
            run_name="profile_${model}_${dataset}_${CACHE}_bs${batch_size}_e${EDGE_CACHE_RATIO}_n${NODE_CACHE_RATIO}_tw${TIME_WINDOW}_ws${NPROC_PER_NODE}"
            log_path="${LOG_DIR}/${run_name}.log"

            common_args=(
                offline_edge_prediction.py
                --model "${model}"
                --data "${dataset}"
                --cache "${CACHE}"
                --edge-cache-ratio "${EDGE_CACHE_RATIO}"
                --node-cache-ratio "${NODE_CACHE_RATIO}"
                --snapshot-time-window "${TIME_WINDOW}"
                --ingestion-batch-size "${INGESTION_BATCH_SIZE}"
                --epoch "${EPOCH}"
                --num-workers "${NUM_WORKERS}"
                --num-chunks "${NUM_CHUNKS}"
                --print-freq "${PRINT_FREQ}"
                --seed "${SEED}"
                --batch-size "${batch_size}"
                --profile
                --profile-dir "${PROFILE_DIR_ROOT}"
                --profile-wait "${PROFILE_WAIT}"
                --profile-warmup "${PROFILE_WARMUP}"
                --profile-active "${PROFILE_ACTIVE}"
                --profile-repeat "${PROFILE_REPEAT}"
                --profile-row-limit "${PROFILE_ROW_LIMIT}"
                --profile-gpu-sample-interval "${PROFILE_GPU_SAMPLE_INTERVAL}"
            )

            if [[ "${PROFILE_ONLY}" == "1" ]]; then
                common_args+=(--profile-only)
            fi

            if [[ "${PROFILE_RECORD_SHAPES}" == "0" ]]; then
                common_args+=(--no-profile-record-shapes)
            fi

            if [[ "${PROFILE_WITH_STACK}" == "1" ]]; then
                common_args+=(--profile-with-stack)
            fi

            if [[ "${PROFILE_WITH_FLOPS}" == "1" ]]; then
                common_args+=(--profile-with-flops)
            fi

            if [[ "${PROFILE_EXPORT_MEMORY_TIMELINE}" == "1" ]]; then
                common_args+=(--profile-export-memory-timeline)
            fi

            common_args+=("${EXTRA_ARGS[@]}")

            if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
                cmd=(
                    torchrun
                    --nnodes=1
                    --nproc_per_node="${NPROC_PER_NODE}"
                    --standalone
                    "${common_args[@]}"
                )
            else
                cmd=(
                    "${PYTHON_BIN}"
                    "${common_args[@]}"
                )
            fi

            printf '\n=== %s ===\n' "${run_name}"
            printf 'log: %s\n' "${log_path}"
            printf '%q ' "${cmd[@]}"
            printf '\n'

            if [[ "${DRY_RUN}" == "1" ]]; then
                continue
            fi

            (
                cd "${SCRIPT_DIR}"
                OMP_NUM_THREADS=8 "${cmd[@]}"
            ) >"${log_path}" 2>&1
        done
    done
done
