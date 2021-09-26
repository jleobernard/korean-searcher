#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
if test -f "$DIR/.conf"; then
  source "$DIR/.conf"
elif test -f "$DIR/../.conf"; then
  source "$DIR/../.conf"
else
  echo "Aucun fichier de configuration $DIR/.conf ou $DIR/../.conf trouvé"
  exit -1
fi
source $DIR/../../.venv/bin/activate
python3 com/leo/koreanparser/main.py $DIR/.env