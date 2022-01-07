#!/bin/bash
set -e


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

my_env=""
if test -f "$DIR/.conf"; then
  my_env="$DIR/.conf"
elif test -f "$DIR/../.conf"; then
  my_env="$DIR/../.conf"
else
  echo "Aucun fichier de configuration $DIR/.conf ou $DIR/../.conf trouvé"
  exit -1
fi
source "$my_env"

source $project_path/.venv/bin/activate
export PYTHONPATH=$project_path
export GOOGLE_APPLICATION_CREDENTIALS="$gac"


function process_file() {
    file=$1
    echo "Analyser du fichier $file"
    python3 $project_path/com/leo/koreanparser/main.py --conf $my_env --file "$file"
    if [[ $? -eq 0 ]]; then
      echo "Anaylse du fichier $file terminée"
      echo "Recharge des sous-titres du serveur"
      curl "$endpoint/api/kosubs:reload"
      echo "Suppression des fichiers temporaires de $work_directory"
      find $work_directory -type f -exec rm -f {} \;
      echo "Suppression du fichier $file"
      rm -f $file
      echo "Fichiers temporaires supprimés"
    else
        echo "Une erreur est survenue lors de l'analyse du fichier $file"
    fi
}
cd $income_dir
files=$(ls . | grep '.*.webm')
for file in $files
do
  process_file $file
done