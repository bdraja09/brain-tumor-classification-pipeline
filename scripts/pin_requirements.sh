#!/usr/bin/env bash
# Fige les versions des 3 requirements.txt d'après l'environnement Python ACTIF.
# À lancer depuis le venv qui charge correctement vos modèles :
#
#   source <votre_venv>/bin/activate
#   scripts/pin_requirements.sh
#
# Indispensable pour TensorFlow/Keras (.h5) et scikit-learn (.joblib) : une autre
# version peut empêcher le chargement des modèles.
set -euo pipefail
cd "$(dirname "$0")/.."
FREEZE="$(pip freeze)"

pin() {   # $1 = fichier de sortie, $2 = regex des paquets à garder
  { echo "$FREEZE" | grep -iE "^($2)==" || true; } | sort -u > "$1"
}
add_missing() {   # $1 = fichier, suivants = paquets à ajouter s'ils manquent
  local f="$1"; shift
  for p in "$@"; do
    grep -qiE "^${p}(==|$)" "$f" || echo "$p" >> "$f"
  done
}
headless() {   # opencv-python -> opencv-python-headless (même version)
  sed -i -E 's/^opencv-python(-headless)?==/opencv-python-headless==/I' "$1"
  sort -u -o "$1" "$1"
}

# Détection : TensorFlow sans GPU
pin fog/detection/requirements.txt "tensorflow|tensorflow-cpu|keras|h5py|numpy|fastapi|uvicorn|python-multipart|opencv-python|opencv-python-headless"
sed -i -E 's/^tensorflow==/tensorflow-cpu==/I' fog/detection/requirements.txt
headless fog/detection/requirements.txt
add_missing fog/detection/requirements.txt fastapi uvicorn python-multipart numpy tensorflow-cpu opencv-python-headless

# Segmentation
pin fog/segmentation/requirements.txt "numpy|fastapi|uvicorn|python-multipart|opencv-python|opencv-python-headless"
headless fog/segmentation/requirements.txt
add_missing fog/segmentation/requirements.txt fastapi uvicorn python-multipart numpy opencv-python-headless

# Classification
pin fog/classification/requirements.txt "numpy|scikit-learn|scikit-image|joblib|fastapi|uvicorn|python-multipart|opencv-python|opencv-python-headless"
headless fog/classification/requirements.txt
add_missing fog/classification/requirements.txt fastapi uvicorn python-multipart numpy scikit-learn scikit-image joblib opencv-python-headless

for f in detection segmentation classification; do
  echo "=== fog/$f/requirements.txt"; cat "fog/$f/requirements.txt"
done