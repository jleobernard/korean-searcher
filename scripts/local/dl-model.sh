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

echo "Réalisation de la copie de sauvegarde..."
MYDATE="`date +%Y%m%dH%H%M`"
distant_history_dir="$distant_model_dir/history"
distant_file="$distant_model_dir/model.pt"
read -p "Commentaire (une seule ligne) ? " comment
comment=${comment:-"na"}
ssh $distant_server "mkdir -p $distant_history_dir && touch $distant_history_dir/comments.txt && if test -f $distant_file; then cp $distant_file $distant_history_dir/model.pt-$MYDATE.old; echo 'model.pt-$MYDATE.old=$comment' >> $distant_history_dir/comments.txt; else echo 'Aucun fichier à sauvegarder'; fi"
scp "$DIR/../../com/leo/koreanparser/dl/data/models/best.pt" $distant_server:$distant_file