#!/bin/bash
set -e
DIR=$(realpath "$(dirname "$0")")
source $DIR/../../.venv/bin/activate
python3 com/leo/koreanparser/main.py --conf $DIR/.env