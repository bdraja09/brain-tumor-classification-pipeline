#!/usr/bin/env bash
# Lance UN nœud directement avec Python (sans Docker), pour déboguer le code.
#
#   source venv/bin/activate
#   scripts/run_local.sh detection      8001 detection-1
#   scripts/run_local.sh segmentation   8002 segmentation-1
#   scripts/run_local.sh classification 8003 classification-1
#
# Variables optionnelles : MODEL_PATH, DELAY_MS, MAX_CONCURRENCY, DET_THRESHOLD, SEG_THRESHOLD
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

KIND="${1:?usage : run_local.sh <detection|segmentation|classification> <port> [nom]}"
PORT="${2:?port manquant}"
export NODE_NAME="${3:-$KIND-local}"

case "$KIND" in
  detection)      export MODEL_PATH="${MODEL_PATH:-$ROOT/models/tumor_detection_model.h5}" ;;
  classification) export MODEL_PATH="${MODEL_PATH:-$ROOT/models/brain_tumor_rf/random_forest.joblib}" ;;
  segmentation)   ;;
  *) echo "Type inconnu : $KIND" >&2; exit 1 ;;
esac

cd "fog/$KIND"
echo ">> $NODE_NAME ($KIND) sur http://127.0.0.1:$PORT"
exec python -m uvicorn main:app --host 127.0.0.1 --port "$PORT"