#!/bin/bash

MODEL=$1
DATA=$2
CACHE="${3:-LFUCache}"
EDGE_CACHE_RATIO="${4:-0.2}" # default 20% of cache
NODE_CACHE_RATIO="0.2" # default 20% of cache
TIME_WINDOW="0" # default 0
NPROC_PER_NODE=1

is_positive_int() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

# Supported forms:
# 1) MODEL DATA [CACHE] [EDGE_CACHE_RATIO]
# 2) MODEL DATA CACHE EDGE_CACHE_RATIO NPROC_PER_NODE
# 3) MODEL DATA CACHE EDGE_CACHE_RATIO NODE_CACHE_RATIO NPROC_PER_NODE
# 4) MODEL DATA CACHE EDGE_CACHE_RATIO NODE_CACHE_RATIO TIME_WINDOW NPROC_PER_NODE
if [[ $# -ge 5 ]]; then
    if [[ $# -eq 5 ]] && is_positive_int "$5"; then
        NPROC_PER_NODE=$5
    else
        NODE_CACHE_RATIO="$5"
    fi
fi

if [[ $# -ge 6 ]]; then
    if [[ $# -eq 6 ]] && is_positive_int "$6"; then
        NPROC_PER_NODE=$6
    else
        TIME_WINDOW="$6"
    fi
fi

if [[ $# -ge 7 ]]; then
    NPROC_PER_NODE=$7
fi

if [[ $NPROC_PER_NODE -gt 1 ]]; then
    cmd="torchrun \
        --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \
        --standalone \
        offline_edge_prediction.py --model $MODEL --data $DATA \
        --cache $CACHE --edge-cache-ratio $EDGE_CACHE_RATIO \
        --node-cache-ratio $NODE_CACHE_RATIO --snapshot-time-window $TIME_WINDOW \
        --ingestion-batch-size 10000000"
else
    cmd="python offline_edge_prediction.py --model $MODEL --data $DATA \
        --cache $CACHE --edge-cache-ratio $EDGE_CACHE_RATIO \
        --node-cache-ratio $NODE_CACHE_RATIO --snapshot-time-window $TIME_WINDOW \
        --ingestion-batch-size 10000000"
fi

echo $cmd
OMP_NUM_THREADS=8 exec $cmd > ${MODEL}_${DATA}_${CACHE}_${EDGE_CACHE_RATIO}_${NODE_CACHE_RATIO}_${TIME_WINDOW}_presampling.log 2>&1
