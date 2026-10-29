"""
Nœud fog : détection de tumeur (CNN binaire, entrée 128x128 en niveaux de gris).

Endpoints :
  GET  /health   -> état du nœud (503 si le modèle n'est pas chargé)
  GET  /stats    -> compteurs (requêtes, erreurs, en cours, latence moyenne)
  POST /detect   -> reçoit une image (multipart), renvoie la probabilité de tumeur

Variables d'environnement (voir .env.example) :
  NODE_NAME      nom affiché dans les réponses (ex. detection-1)
  MODEL_PATH     chemin du modèle .h5 dans le conteneur
  DET_THRESHOLD  seuil de décision (0.4, comme dans l'application)
  TF_THREADS     nombre de threads TensorFlow (aligné sur les CPU du conteneur)
  DELAY_MS       délai artificiel par requête, pour rendre la répartition visible en démo
  MAX_CONCURRENCY nombre d'inférences simultanées par réplica (1 par défaut)
"""
import os
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile

NODE_NAME = os.getenv("NODE_NAME", "detection-1")
MODEL_PATH = os.getenv("MODEL_PATH", "/models/tumor_detection_model.h5")
THRESHOLD = float(os.getenv("DET_THRESHOLD", "0.4"))
DEFAULT_SIZE = int(os.getenv("IMG_SIZE", "128"))
DELAY_MS = float(os.getenv("DELAY_MS", "0"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "1"))

# Limiter les threads TensorFlow : le conteneur n'a qu'une part du processeur
try:
    n_threads = int(os.getenv("TF_THREADS", "1"))
    tf.config.threading.set_intra_op_parallelism_threads(n_threads)
    tf.config.threading.set_inter_op_parallelism_threads(n_threads)
except RuntimeError:
    pass

state = {
    "model": None,
    "img_size": DEFAULT_SIZE,
    "requests": 0,
    "errors": 0,
    "in_flight": 0,
    "total_ms": 0.0,
    "started_at": time.time(),
}
stats_lock = threading.Lock()
infer_sem = threading.Semaphore(MAX_CONCURRENCY)  # capacité du réplica


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement unique au démarrage du conteneur
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        shape = model.input_shape  # attendu : (None, 128, 128, 1)
        if shape and len(shape) == 4 and shape[1]:
            state["img_size"] = int(shape[1])
        size = state["img_size"]
        # Inférence à blanc : la première prédiction est toujours plus lente
        model(np.zeros((1, size, size, 1), dtype=np.float32), training=False)
        state["model"] = model
        print(f"[{NODE_NAME}] Modèle chargé : {MODEL_PATH} "
              f"(entrée {shape}, seuil {THRESHOLD})", flush=True)
    except Exception as e:
        print(f"[{NODE_NAME}] ERREUR chargement du modèle ({MODEL_PATH}) : {e}", flush=True)
    yield


app = FastAPI(title=f"Nœud {NODE_NAME}", lifespan=lifespan)


def preprocess(image_bytes: bytes, size: int) -> np.ndarray:
    """Même préparation que l'application : couleur -> redimension -> gris -> [0,1]."""
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image illisible (formats acceptés : jpg, png, bmp)")
    img = cv2.resize(img, (size, size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = gray.astype(np.float32) / 255.0
    return x.reshape(1, size, size, 1)


@app.get("/health")
def health():
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"status": "ok", "node": NODE_NAME}


@app.get("/stats")
def stats():
    with stats_lock:
        n = state["requests"]
        return {
            "node": NODE_NAME,
            "requests": n,
            "errors": state["errors"],
            "in_flight": state["in_flight"],
            "avg_latency_ms": round(state["total_ms"] / n, 2) if n else 0.0,
            "uptime_s": round(time.time() - state["started_at"], 1),
        }


# « def » et non « async def » : FastAPI exécute la fonction dans un thread,
# la boucle d'événements reste libre pendant le calcul.
@app.post("/detect")
def detect(file: UploadFile = File(...)):
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    with stats_lock:
        state["in_flight"] += 1
    start = time.perf_counter()
    ok = False
    try:
        x = preprocess(file.file.read(), state["img_size"])
        with infer_sem:
            if DELAY_MS > 0:
                time.sleep(DELAY_MS / 1000.0)
            pred = model(x, training=False).numpy()
        prob = float(np.ravel(pred)[0])
        tumor = prob > THRESHOLD
        ok = True
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "node": NODE_NAME,
            "tumor": tumor,
            "probability": round(prob, 4),
            "confidence": round((prob if tumor else 1.0 - prob) * 100, 1),
            "threshold": THRESHOLD,
            "latency_ms": round(elapsed_ms, 2),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        with stats_lock:
            state["in_flight"] -= 1
            state["requests"] += 1
            state["total_ms"] += elapsed_ms
            if not ok:
                state["errors"] += 1