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

source $project_path/.venv/bin/activate
export PYTHONPATH=$project_path
export GOOGLE_APPLICATION_CREDENTIALS="$gac"


function process_file() {
    file=$1
    read -p "Analyser le fichier $file ? [Y/n] " answer
    answer=${answer:-"y"}
    if [[ "$answer" = "y" ]]; then
      python3 $project_path/com/leo/koreanparser/main.py --conf $project_path/scripts/distant/.env --file "$file"
      if [[ $? -eq 0 ]]; then
        echo "Anaylse du fichier $file terminée"
        read -p "Recharger les sous-titres du serveur ? [Y/n] " answer
        answer=${answer:-"y"}
        if [[ "$answer" = "y" ]]; then
          curl "$endpoint/api/kosubs:reload"
        fi
        read -p "Supprimer les fichiers temporaires de $work_directory ? [y/N] " answer
        answer=${answer:-"n"}
        if [[ "$answer" = "y" ]]; then
          find $work_directory -type f -exec rm -f {} \;
          if [[ $? -eq 0 ]]; then
            echo "Fichier temporaires supprimés"
          fi
        fi
      else
          echo "Une erreur est survenue lors de l'analyse du fichier $file"
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