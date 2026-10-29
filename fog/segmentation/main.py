"""
Nœud fog : segmentation de la tumeur (OpenCV, sans modèle à charger).

Reprend la méthode de l'application : redimensionnement 128x128, niveaux de gris,
flou gaussien, seuillage, fermeture morphologique, plus grande composante connexe.

Endpoints :
  GET  /health                          -> état du nœud
  GET  /stats                           -> compteurs
  POST /segment[?include_mask=true]     -> surface, pourcentage, boîte englobante

Variables d'environnement :
  NODE_NAME, SEG_THRESHOLD (180), IMG_SIZE (128), DELAY_MS, MAX_CONCURRENCY
"""
import base64
import os
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

NODE_NAME = os.getenv("NODE_NAME", "segmentation-1")
SEG_THRESHOLD = int(os.getenv("SEG_THRESHOLD", "180"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "128"))
DELAY_MS = float(os.getenv("DELAY_MS", "0"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "1"))

state = {"requests": 0, "errors": 0, "in_flight": 0, "total_ms": 0.0,
         "started_at": time.time()}
stats_lock = threading.Lock()
infer_sem = threading.Semaphore(MAX_CONCURRENCY)  # capacité du réplica

app = FastAPI(title=f"Nœud {NODE_NAME}")


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return np.zeros_like(mask)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask)
    out[labels == largest] = 255
    return out


def run_segmentation(image_bytes: bytes) -> np.ndarray:
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image illisible (formats acceptés : jpg, png, bmp)")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, SEG_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return keep_largest_component(mask)


@app.get("/health")
def health():
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


# « def » et non « async def » : exécuté dans un thread, la boucle reste libre.
@app.post("/segment")
def segment(file: UploadFile = File(...), include_mask: bool = False):
    with stats_lock:
        state["in_flight"] += 1
    start = time.perf_counter()
    ok = False
    try:
        data = file.file.read()
        with infer_sem:
            if DELAY_MS > 0:
                time.sleep(DELAY_MS / 1000.0)
            mask = run_segmentation(data)

        surface = int(np.count_nonzero(mask))
        result = {
            "node": NODE_NAME,
            "found": surface > 0,
            "surface_px": surface,
            "percentage": round(surface / (IMG_SIZE * IMG_SIZE) * 100, 2),
            "bbox": list(cv2.boundingRect(mask)) if surface > 0 else None,  # x, y, w, h
            "image_size": IMG_SIZE,
            "threshold": SEG_THRESHOLD,
        }
        if include_mask:
            _, png = cv2.imencode(".png", mask)
            result["mask_png_base64"] = base64.b64encode(png.tobytes()).decode("ascii")
        ok = True
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return result
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