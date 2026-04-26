# GNNFlow

A comprehensive framework for training graph neural networks on dynamic graphs.

NB: this is an ongoing work.

## Clone

Clone the `profile` branch and initialize the submodules:

```sh
git clone --branch profile --recursive <repo-url>
cd GNNFlow
git submodule update --init --recursive
```

If you already cloned the repository without submodules, run:

```sh
git submodule update --init --recursive
```

## Build

The `profile` branch has been tested on a Linux cluster environment with:
- Python 3.10
- PyTorch 2.3.1
- DGL with a matching CUDA build
- gcc 9.4.0
- CUDA 11.8

Dependencies:
- torch
- dgl (CUDA version)

On the cluster, load the compiler and CUDA modules first, then build:

```sh
module purge
module load gcc/9.4.0
module load cuda/11.8
python setup.py install
```

This installs both the Python package and the `libgnnflow` CUDA extension into the current environment.

## Prepare data

```sh
cd scripts
./download_data.sh
cd ..
```

Notes:
- `scripts/download_data.sh` currently downloads the REDDIT dataset by default.
- The script works with `aria2c`, `curl`, or `wget`.
- If you want additional datasets, uncomment the corresponding lines in `scripts/download_data.sh`.

## Run

Run the launcher from inside `scripts/`. The script uses relative paths, so invoking it from the repository root is not supported.

Verified single-node, 4-GPU REDDIT training command:

```sh
cd scripts
./run_offline.sh TGN REDDIT LRUCache 0.2 0.2 0 4
```

Argument order for `scripts/run_offline.sh`:
- `MODEL`
- `DATA`
- `CACHE`
- `EDGE_CACHE_RATIO`
- `NODE_CACHE_RATIO`
- `TIME_WINDOW`
- `NPROC_PER_NODE`

Any additional arguments after `NPROC_PER_NODE` are passed directly to
`offline_edge_prediction.py`. This is useful for profiler flags and batch-size
overrides.

The launcher redirects output to a log file in `scripts/`. For the command above, check:

```sh
tail -f TGN_REDDIT_LRUCache_0.2_0.2_0_presampling.log
```

## Profile training

The `profile` branch includes a Torch-profiler-backed training mode in
`scripts/offline_edge_prediction.py`. It is designed for short, repeatable
profiling runs on the training loop so you can compare:

- end-to-end step time
- sampling time
- feature fetch time
- memory fetch / update / write-back time
- model forward time
- loss + backward + optimizer time
- GPU peak allocated / reserved memory
- process RSS
- GPU utilization and GPU memory utilization (sampled during profiling)

Recommended REDDIT profiling command from `scripts/`:

```sh
./run_offline.sh TGN REDDIT LRUCache 0.2 0.2 0 4 \
  --batch-size 4000 \
  --profile \
  --profile-only \
  --profile-wait 1 \
  --profile-warmup 1 \
  --profile-active 6 \
  --profile-repeat 1
```

For a different algorithm or batch size, change `MODEL` or `--batch-size`. For
example:

```sh
./run_offline.sh TGAT REDDIT LRUCache 0.2 0.2 0 4 \
  --batch-size 1200 \
  --profile \
  --profile-only
```

Profiler artifacts are written under `profiles/` at the repository root. Each
run gets its own directory, and each rank writes:

- `summary.json`: structured per-rank runtime, memory, and GPU-utilization summary
- `traces/`: Torch profiler traces for TensorBoard / Chrome trace viewers
- `key_averages_*.txt`: top profiler tables sorted by CPU time, CUDA time, and memory

For distributed runs, rank 0 also writes:

- `summary_all_ranks.json`: aggregate summary across all ranks

Optional profiling flags:

- `--profile-only`: stop after the profiling window instead of running full training
- `--profile-dir <path>`: change where profiler outputs are written
- `--profile-with-stack`: collect stack traces for deeper operator analysis
- `--profile-with-flops`: include FLOPs estimates where supported
- `--profile-export-memory-timeline`: export a memory timeline if supported by the installed torch version
- `--no-profile-record-shapes`: disable operator shape capture if you want lower profiler overhead

## Plot profiler results

Use `scripts/plot_profiler_results.py` to generate comparison plots from the
saved profiler summaries. The script reads runs from `profiles/`, filters them
by config, keeps the latest run per config by default, and generates filenames
based on the selected models, datasets, and batch sizes.

Example:

```sh
python scripts/plot_profiler_results.py \
  --models TGN TGAT \
  --datasets REDDIT \
  --batch-sizes 600 1200 4000
```

This writes plots under `profiles/plots/`, including:

- overview dashboard
- training-stage breakdown
- training-stage share
- setup-stage breakdown
- batch-scaling comparison plot when multiple batch sizes are present

Useful flags:

- `--profiles-dir <path>`: read profiler outputs from a different directory
- `--output-dir <path>`: write plots somewhere else
- `--caches <names...>`: filter by cache policy
- `--world-sizes <ints...>`: filter by number of ranks / GPUs
- `--edge-cache-ratios <floats...>`: filter by edge cache ratio
- `--node-cache-ratios <floats...>`: filter by node cache ratio
- `--snapshot-time-windows <floats...>`: filter by time-window setting
- `--all-matching-runs`: include every matching run instead of only the latest per config
