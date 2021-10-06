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
function process_file() {
    file=$1
    read -p "Copier le fichier $file ? [Y/n] " answer
    answer=${answer:-"y"}
    if [[ "$answer" = "y" ]]; then
      echo "Copie du fichier sur le serveur"
      cp $file $distant_income_dir
      date >  $distant_income_dir/$file.ready
      if [[ $? -eq 0 ]]; then
        echo "Copie réalisée avec succès @ $distant_income_dir"
        read -p "Supprimer le fichier local $file ? [Y/n] " answer
        answer=${answer:-"y"}
        if [[ "$answer" = "y" ]]; then
          rm -f $file
          if [[ $? -eq 0 ]]; then
            echo "Fichier local supprimé avec succès ($file)"
          fi
        fi
      else
          echo "Une erreur est survenue lors de la recopie du fichier sur $distant_server:$distant_income_dir"
      fi
    else
      echo "Fichier ignoré ($file)"
    fi
}
cd $local_income_dir
files=$(ls . | grep '.*.webm')
for file in $files
do
  process_file $file
done