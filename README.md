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

The launcher redirects output to a log file in `scripts/`. For the command above, check:

```sh
tail -f TGN_REDDIT_LRUCache_0.2_0.2_0_presampling.log
```
