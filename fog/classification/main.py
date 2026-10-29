"""
Nœud fog : classification (HOG + Random Forest).

Endpoints :
  GET  /health    -> état du nœud (503 si le modèle n'est pas chargé)
  GET  /stats     -> compteurs simples (requêtes, erreurs, en cours, latence moyenne)
  POST /classify  -> reçoit une image (multipart), renvoie les probabilités des 4 classes

Variables d'environnement : NODE_NAME, MODEL_PATH, RF_N_JOBS, DELAY_MS, MAX_CONCURRENCY
"""
import os
import threading
import time
from contextlib import asynccontextmanager

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from skimage.feature import hog

NODE_NAME = os.getenv("NODE_NAME", "classification-1")
MODEL_PATH = os.getenv("MODEL_PATH", "/models/brain_tumor_rf/random_forest.joblib")
DELAY_MS = float(os.getenv("DELAY_MS", "0"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "1"))

# Doit rester identique au notebook d'entraînement
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
HOG_SIZE = (96, 96)

state = {
    "model": None,
    "requests": 0,
    "errors": 0,
    "in_flight": 0,
    "total_ms": 0.0,
    "started_at": time.time(),
}
lock = threading.Lock()
infer_sem = threading.Semaphore(MAX_CONCURRENCY)  # capacité du réplica


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Le modèle est chargé une seule fois au démarrage du conteneur
    try:
        model = joblib.load(MODEL_PATH)
        # Un seul thread par requête : le CPU du conteneur est limité
        model.n_jobs = int(os.getenv("RF_N_JOBS", "1"))
        state["model"] = model
        print(f"[{NODE_NAME}] Modèle chargé : {MODEL_PATH} "
              f"({model.n_features_in_} features)", flush=True)
    except Exception as e:
        print(f"[{NODE_NAME}] ERREUR chargement du modèle ({MODEL_PATH}) : {e}", flush=True)
    yield


app = FastAPI(title=f"Nœud {NODE_NAME}", lifespan=lifespan)


def extract_features(image_bytes: bytes) -> np.ndarray:
    """HOG identique au notebook : image grise 96x96 -> (1, 4356)."""
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image illisible (formats acceptés : jpg, png, bmp)")
    img = cv2.resize(img, HOG_SIZE)
    feats = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return feats.astype(np.float32).reshape(1, -1)


@app.get("/health")
def health():
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"status": "ok", "node": NODE_NAME}


@app.get("/stats")
def stats():
    with lock:
        n = state["requests"]
        return {
            "node": NODE_NAME,
            "requests": n,
            "errors": state["errors"],
            "in_flight": state["in_flight"],
            "avg_latency_ms": round(state["total_ms"] / n, 2) if n else 0.0,
            "uptime_s": round(time.time() - state["started_at"], 1),
        }


# Fonction « def » (et non « async def ») : FastAPI l'exécute dans un thread,
# ce qui évite de bloquer la boucle d'événements pendant le calcul.
@app.post("/classify")
def classify(file: UploadFile = File(...)):
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    with lock:
        state["in_flight"] += 1
    start = time.perf_counter()
    ok = False
    try:
        X = extract_features(file.file.read())
        if X.shape[1] != model.n_features_in_:
            raise ValueError(
                f"{X.shape[1]} features calculées, {model.n_features_in_} attendues"
            )
        with infer_sem:
            if DELAY_MS > 0:
                time.sleep(DELAY_MS / 1000.0)
            proba = model.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        ok = True
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "node": NODE_NAME,
            "label": CLASSES[idx],
            "confidence": round(float(proba[idx]) * 100, 1),
            "probabilities": {c: round(float(p) * 100, 1) for c, p in zip(CLASSES, proba)},
            "latency_ms": round(elapsed_ms, 2),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        with lock:
            state["in_flight"] -= 1
            state["requests"] += 1
            state["total_ms"] += elapsed_ms
            if not ok:
                state["errors"] += 1