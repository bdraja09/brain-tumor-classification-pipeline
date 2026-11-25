"""
Agent de coordination fog (un par machine).

Cas 1 : passage de relais avec accusé de réception (ACK).
    L'agent d'origine exécute ses étapes locales, puis envoie l'image à l'agent
    de l'autre machine (POST /handoff). Celui-ci vérifie le hash SHA-256,
    enregistre l'image sur disque et répond par un ACK avant de la traiter.

Cas 2 : panne, rapport et redemande de l'image.
    Sans ACK après plusieurs essais, l'origine garde l'image et le job passe
    en EN_ATTENTE_PAIR. Quand l'autre agent revient, il envoie un rapport
    (POST /report), reçoit la liste des jobs qu'il n'a pas reçus, redemande
    chaque image (GET /jobs/{id}/image), vérifie le hash et envoie son ACK.

Tout est journalisé (console + table « events ») et conservé dans SQLite,
donc un redémarrage de l'agent ne perd ni les jobs ni les images.
"""
import asyncio
import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

# ------------------------------------------------------------------ configuration
AGENT = os.getenv("AGENT_NAME", "agent")
PEER_URL = os.getenv("PEER_URL", "").rstrip("/")          # ex. http://172.20.10.3:8090

PIPELINE = ["detection", "segmentation", "classification"]
NODES = {                                                   # étape -> (URL du nœud local, route)
    "detection": (os.getenv("DET_URL", "").rstrip("/"), "/detect"),
    "segmentation": (os.getenv("SEG_URL", "").rstrip("/"), "/segment"),
    "classification": (os.getenv("CLS_URL", "").rstrip("/"), "/classify"),
}
# Étapes faites ICI pour un job soumis à cet agent ; le reste part chez l'autre agent.
LOCAL_STEPS = [s for s in os.getenv("LOCAL_STEPS", "detection,segmentation").split(",") if s]
STOP_IF_NO_TUMOR = os.getenv("STOP_IF_NO_TUMOR", "1") == "1"

ACK_TIMEOUT_S = float(os.getenv("ACK_TIMEOUT_S", "5"))
RETRY_DELAYS_S = [float(x) for x in os.getenv("RETRY_DELAYS_S", "1,2,4").split(",") if x]
HEARTBEAT_S = float(os.getenv("HEARTBEAT_S", "3"))
PENDING_S = float(os.getenv("PENDING_S", "5"))
POLL_S = float(os.getenv("POLL_S", "2"))

DATA = Path(os.getenv("DATA_DIR", "/data"))
JOBS_DIR = DATA / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ base SQLite
_db = sqlite3.connect(str(DATA / "agent.db"), check_same_thread=False)
_db.row_factory = sqlite3.Row
_db.execute("PRAGMA journal_mode=WAL")
_db.executescript("""
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY, role TEXT, state TEXT, sha256 TEXT,
    steps TEXT, partial TEXT, result TEXT, corrupt_first INTEGER DEFAULT 0,
    attempts INTEGER DEFAULT 0, note TEXT, created REAL, updated REAL);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, ts REAL, event TEXT, detail TEXT);
""")
_db.commit()
_lock = threading.Lock()


def sql(query, args=()):
    with _lock:
        cur = _db.execute(query, args)
        rows = cur.fetchall()
        _db.commit()
    return rows


def get_job(job_id):
    rows = sql("SELECT * FROM jobs WHERE job_id=?", (job_id,))
    return dict(rows[0]) if rows else None


def update(job_id, **fields):
    for k in ("steps", "partial", "result"):
        if k in fields and not isinstance(fields[k], str) and fields[k] is not None:
            fields[k] = json.dumps(fields[k])
    fields["updated"] = time.time()
    cols = ", ".join(f"{k}=?" for k in fields)
    sql(f"UPDATE jobs SET {cols} WHERE job_id=?", (*fields.values(), job_id))


def create_job(job_id, role, sha256, state, steps=None, partial=None, corrupt_first=0):
    now = time.time()
    sql("INSERT INTO jobs (job_id, role, state, sha256, steps, partial, corrupt_first, created, updated)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (job_id, role, state, sha256, json.dumps(steps or []), json.dumps(partial or {}),
         corrupt_first, now, now))


def log(job_id, event, detail=""):
    ts = time.time()
    sql("INSERT INTO events (job_id, ts, event, detail) VALUES (?,?,?,?)", (job_id, ts, event, detail))
    clock = time.strftime("%H:%M:%S", time.localtime(ts)) + f".{int(ts * 1000) % 1000:03d}"
    print(f"[{AGENT}] {clock} job={job_id:<10} {event:<22} {detail}", flush=True)


def image_path(job_id):
    return JOBS_DIR / f"{job_id}.img"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------ état en mémoire
state = {"peer_up": None}
inflight: set = set()        # jobs dont un envoi est en cours (évite les envois en double)
tasks: set = set()           # références aux tâches de fond


def spawn(coro):
    t = asyncio.create_task(coro)
    tasks.add(t)
    t.add_done_callback(tasks.discard)


# ------------------------------------------------------------------ exécution des étapes
async def run_steps(job_id, data: bytes, steps, partial):
    """Appelle les nœuds locaux. Retourne (résultats, raison d'arrêt ou None)."""
    async with httpx.AsyncClient(timeout=30) as client:
        for step in steps:
            url, route = NODES[step]
            if not url:
                raise RuntimeError(f"nœud local « {step} » non configuré")
            log(job_id, "ETAPE_DEBUT", step)
            t0 = time.perf_counter()
            r = await client.post(f"{url}{route}", files={"file": ("image.jpg", data)})
            r.raise_for_status()
            res = r.json()
            partial[step] = res
            log(job_id, "ETAPE_FIN",
                f"{step} sur {res.get('node')} en {(time.perf_counter() - t0) * 1000:.0f} ms")
            if step == "detection" and STOP_IF_NO_TUMOR and not res.get("tumor"):
                return partial, "aucune tumeur détectée : pipeline arrêté"
    return partial, None


def finish(job_id, partial, stopped=None):
    result = {"steps": partial, "stopped": stopped, "origin": AGENT}
    update(job_id, state="TERMINE", result=result, partial=partial)
    log(job_id, "TERMINE", stopped or "toutes les étapes sont faites")


# ------------------------------------------------------------------ côté ORIGINE
async def process_origin(job_id):
    job = get_job(job_id)
    data = image_path(job_id).read_bytes()
    local = [s for s in PIPELINE if s in LOCAL_STEPS]
    remote = [s for s in PIPELINE if s not in LOCAL_STEPS]
    try:
        update(job_id, state="LOCAL_EN_COURS")
        partial, stopped = await run_steps(job_id, data, local, json.loads(job["partial"] or "{}"))
    except Exception as e:  # noqa: BLE001
        update(job_id, state="ECHEC", note=f"étape locale : {type(e).__name__} {e}")
        log(job_id, "ECHEC", f"étape locale : {type(e).__name__} {e}")
        return
    if stopped or not remote:
        finish(job_id, partial, stopped)
        return
    update(job_id, state="HANDOFF", steps=remote, partial=partial)
    await try_handoff(job_id)


async def try_handoff(job_id) -> bool:
    """Envoie l'image au pair avec réessais. True si ACK reçu."""
    if job_id in inflight:
        return False
    inflight.add(job_id)
    try:
        job = get_job(job_id)
        data = image_path(job_id).read_bytes()
        delays = [0.0] + RETRY_DELAYS_S
        for i, delay in enumerate(delays):
            if delay:
                await asyncio.sleep(delay)
            payload = data
            if job["corrupt_first"] and i == 0:          # démonstration : image altérée en route
                payload = bytes([data[0] ^ 0xFF]) + data[1:]
                update(job_id, corrupt_first=0)
                log(job_id, "SIMULATION", "image volontairement altérée pour tester le NACK")
            update(job_id, attempts=job["attempts"] + i + 1)
            log(job_id, "HANDOFF_ENVOI", f"tentative {i + 1}/{len(delays)} vers {PEER_URL or '(PEER_URL vide)'}")
            try:
                async with httpx.AsyncClient(timeout=ACK_TIMEOUT_S) as client:
                    r = await client.post(
                        f"{PEER_URL}/handoff",
                        data={"job_id": job_id, "sha256": job["sha256"], "steps": job["steps"],
                              "partial": job["partial"], "origin": AGENT},
                        files={"file": ("image.jpg", payload)})
                body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                if r.status_code == 200 and body.get("ack"):
                    update(job_id, state="ACK_RECU")
                    log(job_id, "ACK_RECU", f"de {body.get('agent')} : {body.get('bytes')} octets, hash OK")
                    return True
                log(job_id, "NACK", f"HTTP {r.status_code} {body.get('reason', r.text[:80])}")
            except Exception as e:  # noqa: BLE001
                log(job_id, "PAS_D_ACK", f"{type(e).__name__} : le pair ne répond pas")
        update(job_id, state="EN_ATTENTE_PAIR")
        log(job_id, "EN_ATTENTE_PAIR", "image et résultats conservés ; reprise au retour du pair")
        return False
    finally:
        inflight.discard(job_id)


async def poll_result(job_id):
    try:
        async with httpx.AsyncClient(timeout=ACK_TIMEOUT_S) as client:
            r = await client.get(f"{PEER_URL}/handoff/{job_id}")
    except Exception:  # noqa: BLE001
        return                                           # pair injoignable : on réessaiera
    if r.status_code != 200:
        return
    body = r.json()
    if body.get("state") == "TERMINE":
        job = get_job(job_id)
        partial = json.loads(job["partial"] or "{}")
        partial.update(body["result"]["steps"])
        log(job_id, "RESULTAT_RECU", f"étapes distantes faites par {body['result'].get('agent')}")
        finish(job_id, partial, body["result"].get("stopped"))
    elif body.get("state") == "ECHEC":
        update(job_id, state="ECHEC", note=body.get("note", ""))
        log(job_id, "ECHEC", f"échec chez le pair : {body.get('note', '')}")


# ------------------------------------------------------------------ côté DISTANT
async def process_remote(job_id):
    job = get_job(job_id)
    data = image_path(job_id).read_bytes()
    try:
        update(job_id, state="EN_COURS")
        partial, stopped = await run_steps(job_id, data, json.loads(job["steps"]), {})
        update(job_id, state="TERMINE",
               result={"steps": partial, "stopped": stopped, "agent": AGENT})
        log(job_id, "TERMINE", "résultat prêt, l'origine peut le récupérer")
    except Exception as e:  # noqa: BLE001
        update(job_id, state="ECHEC", note=f"{type(e).__name__} {e}")
        log(job_id, "ECHEC", f"{type(e).__name__} {e}")


async def pull_missing(ids):
    """Redemande à l'origine les images que ce nœud n'avait pas reçues (cas 2)."""
    for jid in ids:
        if get_job(jid):
            continue
        log(jid, "REDEMANDE_IMAGE", f"image jamais reçue : je la redemande à {PEER_URL}")
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                meta = (await client.get(f"{PEER_URL}/jobs/{jid}/meta")).json()
                img = await client.get(f"{PEER_URL}/jobs/{jid}/image")
                data = img.content
                if sha(data) != meta["sha256"]:
                    log(jid, "HASH_INVALIDE", "image redemandée corrompue, abandon (nouvel essai plus tard)")
                    continue
                image_path(jid).write_bytes(data)
                create_job(jid, "remote", meta["sha256"], "RECU", steps=meta["steps"])
                log(jid, "IMAGE_RECUE", f"{len(data)} octets, hash OK")
                await client.post(f"{PEER_URL}/jobs/{jid}/ack", json={"sha256": meta["sha256"], "node": AGENT})
                log(jid, "ACK_ENVOYE", "accusé de réception envoyé à l'origine")
            spawn(process_remote(jid))
        except Exception as e:  # noqa: BLE001
            log(jid, "REDEMANDE_ECHEC", f"{type(e).__name__} : nouvel essai au prochain rapport")


# ------------------------------------------------------------------ boucles de fond
async def heartbeat_loop():
    while True:
        up = False
        if PEER_URL:
            try:
                async with httpx.AsyncClient(timeout=2) as client:
                    up = (await client.get(f"{PEER_URL}/health")).status_code == 200
            except Exception:  # noqa: BLE001
                up = False
        if up != state["peer_up"]:
            log("-", "PAIR_DISPONIBLE" if up else "PAIR_INDISPONIBLE", PEER_URL)
            state["peer_up"] = up
        await asyncio.sleep(HEARTBEAT_S)


async def report_loop():
    """Au démarrage : informe le pair de notre retour, puis redemande ce qu'on a raté."""
    if not PEER_URL:
        return
    last = sql("SELECT job_id FROM jobs WHERE role='remote' ORDER BY created DESC LIMIT 1")
    body = {"agent": AGENT, "reason": "demarrage_ou_retour_apres_panne",
            "last_job_seen": last[0]["job_id"] if last else None}
    while True:
        try:
            async with httpx.AsyncClient(timeout=4) as client:
                r = await client.post(f"{PEER_URL}/report", json=body)
            pending = r.json().get("pending", [])
            log("-", "RAPPORT_ENVOYE", f"au pair ; jobs à récupérer : {pending or 'aucun'}")
            await pull_missing(pending)
            return
        except Exception:  # noqa: BLE001
            await asyncio.sleep(HEARTBEAT_S)


async def pending_loop():
    while True:
        await asyncio.sleep(PENDING_S)
        if not state["peer_up"]:
            continue
        for row in sql("SELECT job_id FROM jobs WHERE role='origin' AND state='EN_ATTENTE_PAIR'"):
            spawn(try_handoff(row["job_id"]))


async def result_loop():
    while True:
        await asyncio.sleep(POLL_S)
        for row in sql("SELECT job_id FROM jobs WHERE role='origin' AND state='ACK_RECU'"):
            await poll_result(row["job_id"])


@asynccontextmanager
async def lifespan(app):
    # Reprise après redémarrage de CET agent
    for row in sql("SELECT job_id, role, state FROM jobs WHERE state IN ('RECU','LOCAL_EN_COURS','EN_COURS','HANDOFF')"):
        jid, role = row["job_id"], row["role"]
        log(jid, "REPRISE", f"job interrompu ({row['state']}) : relancé au démarrage")
        if role == "origin":
            if row["state"] == "HANDOFF":
                update(jid, state="EN_ATTENTE_PAIR")
            else:
                spawn(process_origin(jid))
        else:
            spawn(process_remote(jid))
    log("-", "DEMARRAGE", f"local={LOCAL_STEPS} pair={PEER_URL or '(non défini)'}")
    for loop in (heartbeat_loop, report_loop, pending_loop, result_loop):
        spawn(loop())
    yield


app = FastAPI(title=f"Agent {AGENT}", lifespan=lifespan)


# ------------------------------------------------------------------ API
@app.get("/health")
def health():
    return {"status": "ok", "agent": AGENT, "peer_up": state["peer_up"], "local_steps": LOCAL_STEPS}


@app.post("/jobs", status_code=202)
async def submit(file: UploadFile = File(...), simulate_corruption: bool = False):
    data = await file.read()
    if not data:
        raise HTTPException(400, "fichier vide")
    job_id = uuid.uuid4().hex[:8]
    image_path(job_id).write_bytes(data)
    create_job(job_id, "origin", sha(data), "RECU", corrupt_first=int(simulate_corruption))
    log(job_id, "JOB_RECU", f"{len(data)} octets, sha256={sha(data)[:12]}…")
    spawn(process_origin(job_id))
    return {"job_id": job_id, "agent": AGENT, "suivi": f"/jobs/{job_id}"}


def _view(job, with_events=True):
    out = {k: job[k] for k in ("job_id", "role", "state", "attempts", "note")}
    out["steps_distantes"] = json.loads(job["steps"] or "[]")
    out["result"] = json.loads(job["result"]) if job["result"] else None
    if with_events:
        out["events"] = [
            {"t": time.strftime("%H:%M:%S", time.localtime(e["ts"])) + f".{int(e['ts'] * 1000) % 1000:03d}",
             "event": e["event"], "detail": e["detail"]}
            for e in sql("SELECT * FROM events WHERE job_id=? ORDER BY id", (job["job_id"],))]
    return out


@app.get("/jobs")
def list_jobs():
    return [_view(dict(r), with_events=False) for r in sql("SELECT * FROM jobs ORDER BY created DESC LIMIT 50")]


@app.get("/jobs/{job_id}")
def job_detail(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "job inconnu")
    return _view(job)


@app.get("/jobs/{job_id}/meta")
def job_meta(job_id: str):
    job = get_job(job_id)
    if not job or job["role"] != "origin":
        raise HTTPException(404, "job inconnu")
    return {"job_id": job_id, "sha256": job["sha256"], "steps": json.loads(job["steps"]),
            "partial": json.loads(job["partial"] or "{}")}


@app.get("/jobs/{job_id}/image")
def job_image(job_id: str):
    job = get_job(job_id)
    if not job or job["role"] != "origin" or not image_path(job_id).exists():
        raise HTTPException(404, "image inconnue")
    return Response(image_path(job_id).read_bytes(), media_type="application/octet-stream",
                    headers={"X-SHA256": job["sha256"]})


@app.post("/jobs/{job_id}/ack")
def job_ack(job_id: str, body: dict):
    """ACK envoyé par le pair après une redemande d'image (cas 2)."""
    job = get_job(job_id)
    if not job or job["role"] != "origin":
        raise HTTPException(404, "job inconnu")
    if body.get("sha256") != job["sha256"]:
        raise HTTPException(422, "hash différent")
    if job["state"] in ("EN_ATTENTE_PAIR", "HANDOFF"):
        update(job_id, state="ACK_RECU")
        log(job_id, "ACK_RECU", f"de {body.get('node')} après redemande de l'image")
    return {"ok": True}


@app.post("/handoff")
async def handoff(file: UploadFile = File(...), job_id: str = Form(...), sha256: str = Form(...),
                  steps: str = Form(...), partial: str = Form("{}"), origin: str = Form("")):
    data = await file.read()
    if get_job(job_id):
        log(job_id, "HANDOFF_DOUBLON", "déjà reçu : ACK renvoyé, pas de retraitement")
        return {"ack": True, "job_id": job_id, "bytes": len(data), "agent": AGENT}
    got = sha(data)
    if got != sha256:
        log(job_id, "HASH_INVALIDE", f"reçu {got[:12]}… attendu {sha256[:12]}… : NACK")
        return JSONResponse({"ack": False, "reason": "hash invalide", "agent": AGENT}, status_code=422)
    image_path(job_id).write_bytes(data)
    create_job(job_id, "remote", sha256, "RECU", steps=json.loads(steps), partial=json.loads(partial))
    log(job_id, "HANDOFF_RECU", f"de {origin} : {len(data)} octets, hash OK, image enregistrée -> ACK")
    spawn(process_remote(job_id))
    return {"ack": True, "job_id": job_id, "sha256": got, "bytes": len(data), "agent": AGENT}


@app.get("/handoff/{job_id}")
def handoff_status(job_id: str):
    job = get_job(job_id)
    if not job or job["role"] != "remote":
        raise HTTPException(404, "job inconnu")
    return {"state": job["state"], "note": job["note"],
            "result": json.loads(job["result"]) if job["result"] else None}


@app.post("/report")
def report(body: dict):
    """Rapport d'un pair qui revient : on lui indique les jobs qu'il n'a pas reçus."""
    state["peer_up"] = True
    log("-", "RAPPORT_RECU", f"{body.get('agent')} annonce son retour ({body.get('reason')}, "
                            f"dernier job vu : {body.get('last_job_seen')})")
    pending = [r["job_id"] for r in sql(
        "SELECT job_id FROM jobs WHERE role='origin' AND state IN ('EN_ATTENTE_PAIR','HANDOFF') ORDER BY created")]
    return {"agent": AGENT, "pending": pending}
