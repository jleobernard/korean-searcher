#!/bin/bash
set -e
DIR=$(realpath "$(dirname "$0")")
cd $DIR/../..
root_dir=$(pwd)
rsync -vrza $root_dir/server/* mandela:/opt/containers/kosubs/docker/app
ssh mandela "cd /opt/containers/kosubs && mv docker/app/requirements.txt docker && docker-compose up -d --build"