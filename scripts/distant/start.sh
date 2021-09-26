#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
source $DIR/../../.venv/bin/activate
python3 com/leo/koreanparser/main.py --conf $DIR/.env