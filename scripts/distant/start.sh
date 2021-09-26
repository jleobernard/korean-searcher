#!/bin/bash
set -e
DIR=$(realpath "$(dirname "$0")")
cd $DIR/../..
root_dir=$(pwd)
source $root_dir/.venv/bin/activate
python3 com/leo/koreanparser/main.py --conf $DIR/.env