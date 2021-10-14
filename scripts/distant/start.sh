#!/bin/bash
set -e
DIR=$(realpath "$(dirname "$0")")
cd $DIR/../..
root_dir=$(pwd)
source $root_dir/.venv/bin/activate
export PYTHONPATH=$root_dir
export GOOGLE_APPLICATION_CREDENTIALS=/Users/leo/.secrets/laflemme-3b669604d7e6.json
$root_dir/.venv/bin/python3 com/leo/koreanparser/main.py --conf $DIR/.env