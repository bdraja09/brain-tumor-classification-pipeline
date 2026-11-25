#!/usr/bin/env python3
"""
Démonstration des cas 1 et 2 : soumet une image à un agent et affiche la chronologie.

  python scripts/demo_agent.py --agent http://localhost:8090 --image samples/irm.jpg
  python scripts/demo_agent.py --agent http://localhost:8090 --image samples/irm.jpg --corrupt   # test du NACK
  python scripts/demo_agent.py --agent http://localhost:8090 --job a1b2c3d4                      # suivre un job existant
  python scripts/demo_agent.py --agent http://localhost:8090 --list                              # lister les jobs

Bibliothèque standard uniquement.
"""
import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
import uuid

FINAL = ("TERMINE", "ECHEC")


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def submit(base, path, corrupt):
    boundary = uuid.uuid4().hex
    with open(path, "rb") as f:
        data = f.read()
    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"image\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    url = f"{base}/jobs" + ("?simulate_corruption=true" if corrupt else "")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def show(job):
    print(f"\njob {job['job_id']}  état : {job['state']}")
    for e in job["events"]:
        print(f"  {e['t']}  {e['event']:<18} {e['detail']}")
    res = job.get("result")
    if res:
        print("\nrésultat :")
        steps = res["steps"]
        d, s, c = steps.get("detection"), steps.get("segmentation"), steps.get("classification")
        if d:
            print(f"  détection      {d.get('node'):<18} tumeur={d.get('tumor')} proba={d.get('probability')}")
        if s:
            print(f"  segmentation   {s.get('node'):<18} surface={s.get('surface_px')}px bbox={s.get('bbox')}")
        if c:
            print(f"  classification {c.get('node'):<18} {c.get('label')} {c.get('confidence')}%")
        if res.get("stopped"):
            print(f"  arrêt : {res['stopped']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="ex. http://localhost:8090")
    ap.add_argument("--image")
    ap.add_argument("--job")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--corrupt", action="store_true", help="altère l'image à l'envoi : test du NACK")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()
    base = a.agent.rstrip("/")

    try:
        if a.list:
            for j in get(f"{base}/jobs"):
                print(f"{j['job_id']}  {j['role']:<7} {j['state']:<16} tentatives={j['attempts']}")
            return 0
        job_id = a.job
        if not job_id:
            if not a.image:
                ap.error("--image requis")
            job_id = submit(base, a.image, a.corrupt)["job_id"]
            print(f"job soumis : {job_id}")
        if a.no_wait:
            return 0
        t0, last = time.time(), None
        while time.time() - t0 < a.timeout:
            job = get(f"{base}/jobs/{job_id}")
            if job["state"] != last:
                print(f"  ... {job['state']}")
                last = job["state"]
            if job["state"] in FINAL:
                show(job)
                return 0 if job["state"] == "TERMINE" else 1
            time.sleep(1)
        show(get(f"{base}/jobs/{job_id}"))
        print(f"\nPas fini après {a.timeout} s (normal si le pair est en panne : relancez avec --job {job_id}).")
        return 2
    except (urllib.error.URLError, OSError) as e:
        print(f"Agent injoignable ({e}). Vérifiez l'adresse et `docker compose ps`.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
