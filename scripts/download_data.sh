#!/bin/bash
# from https://github.com/amazon-research/tgl/blob/main/down.sh

set -euo pipefail

download_file() {
  local output_dir="$1"
  local url="$2"
  local filename

  mkdir -p "$output_dir"
  filename="$(basename "$url")"

  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 16 -d "$output_dir" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail --output "$output_dir/$filename" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$output_dir/$filename" "$url"
  else
    echo "Error: no supported downloader found. Install aria2, curl, or wget." >&2
    exit 1
  fi
}

download_file ../data/MOOC https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MOOC/edges.csv

download_file ../data/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/edge_features.pt
download_file ../data/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/edges.csv
download_file ../data/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/labels.csv

download_file ../data/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/edge_features.pt
download_file ../data/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/edges.csv
download_file ../data/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/labels.csv

download_file ../data/LASTFM https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/LASTFM/edges.csv

download_file ../data/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/node_features.pt
download_file ../data/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/labels.csv
download_file ../data/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/edges.csv
download_file ../data/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/edge_features.pt

download_file ../data/MAG https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MAG/labels.csv
download_file ../data/MAG https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MAG/edges.csv
download_file ../data/MAG https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MAG/node_features.pt
