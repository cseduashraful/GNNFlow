#!/bin/bash

# Example config file for scripts/run_profile_sweep.sh
#
# Usage:
#   ./scripts/run_profile_sweep.sh ./scripts/profile_sweep_config.sh

DATASETS=(REDDIT)

# Add or remove models here.
MODELS=(
  TGN
  TGAT
  GRAPHSAGE
)

# Add or remove batch sizes here.
BATCH_SIZES=(
  2000
  4000
  8000
  16000
  32000
)

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
DRY_RUN="0"

# You can add extra arguments if needed, for example:
# EXTRA_ARGS=(--profile-with-stack)
EXTRA_ARGS=()
