#!/bin/bash
set -e
DIR=$(realpath "$(dirname "$0")")
cd $DIR/../..
root_dir=$(pwd)
rsync -vrza $root_dir/server/* ks-leo-noport:/opt/containers/kosubs/docker/app
ssh ks-leo-noport "cd /opt/containers/kosubs && docker-compose up -d --build"