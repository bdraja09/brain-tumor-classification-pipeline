#!/usr/bin/env python3
"""
Soumet PLUSIEURS images à un agent (une toutes les --pause secondes) et suit leur avancement.
Avec un DELAY_MS élevé sur les nœuds, plusieurs jobs sont en cours en même temps : on peut alors
arrêter un nœud ou un agent en plein travail et voir ce qui arrive à chaque job.

  python scripts/demo_batch.py --agent http://localhost:8090 --folder database --count 6
  python scripts/demo_batch.py --agent http://localhost:8090 --folder database --count 10 --pause 1

Bibliothèque standard uniquement.
"""
import argparse
import json
import mimetypes
import random
import sys
import time
import urllib.request
import uuid
from pathlib import Path

EXT = {".jpg", ".jpeg", ".png", ".bmp"}
FINAL = ("TERMINE", "ECHEC")


def get(url, timeout=8):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def submit(base, path):
    boundary = uuid.uuid4().hex
    data = path.read_bytes()
    ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{base}/jobs", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)["job_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--folder", required=True, help="dossier d'images (recherche récursive)")
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--pause", type=float, default=0.5, help="secondes entre deux soumissions")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--seed", type=int, default=1, help="même graine = mêmes images")
    a = ap.parse_args()
    base = a.agent.rstrip("/")

    files = sorted(p for p in Path(a.folder).rglob("*") if p.suffix.lower() in EXT)
    if not files:
        print(f"Aucune image trouvée dans « {a.folder} ». Vérifiez le chemin (dir {a.folder}).")
        return 1
    random.Random(a.seed).shuffle(files)
    files = files[:a.count]
    print(f"{len(files)} image(s) envoyées à {base}, une toutes les {a.pause} s\n")

    jobs, t_start = {}, {}
    for p in files:
        try:
            jid = submit(base, p)
        except OSError as e:
            print(f"Agent injoignable ({e})")
            return 1
        jobs[jid] = p.name
        t_start[jid] = time.time()
        print(f"  {time.strftime('%H:%M:%S')}  soumis {jid}  {p.name}")
        time.sleep(a.pause)

    print("\nSuivi (Ctrl+C pour arrêter le suivi, les jobs continuent) :")
    done, last_line, warned = {}, "", False
    t0 = time.time()
    while len(done) < len(jobs) and time.time() - t0 < a.timeout:
        counts = {}
        for jid in jobs:
            if jid in done:
                counts["TERMINE" if done[jid]["state"] == "TERMINE" else "ECHEC"] = \
                    counts.get("TERMINE" if done[jid]["state"] == "TERMINE" else "ECHEC", 0) + 1
                continue
            try:
                j = get(f"{base}/jobs/{jid}")
                warned = False
            except OSError:
                if not warned:
                    print(f"  {time.strftime('%H:%M:%S')}  agent injoignable (arrêté ?) : j'attends son retour")
                    warned = True
                continue
            counts[j["state"]] = counts.get(j["state"], 0) + 1
            if j["state"] in FINAL:
                j["duree"] = time.time() - t_start[jid]
                done[jid] = j
        line = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        if line and line != last_line:
            print(f"  {time.strftime('%H:%M:%S')}  {line}")
            last_line = line
        time.sleep(1)

    print(f"\n{'job':<10} {'image':<22} {'état':<16} {'durée':>6}  nœuds / résultat")
    print("-" * 100)
    for jid, name in jobs.items():
        j = done.get(jid)
        if not j:
            try:
                j = get(f"{base}/jobs/{jid}")
                j["duree"] = time.time() - t_start[jid]
            except OSError:
                print(f"{jid:<10} {name[:22]:<22} {'(agent injoignable)':<16}")
                continue
        steps = (j.get("result") or {}).get("steps", {})
        nodes = " > ".join(v.get("node", "?") for v in steps.values()) or (j.get("note") or "")
        c = steps.get("classification")
        extra = f"  -> {c.get('label')} {c.get('confidence')}%" if c else ""
        stop = (j.get("result") or {}).get("stopped")
        extra += f"  ({stop})" if stop else ""
        print(f"{jid:<10} {name[:22]:<22} {j['state']:<16} {j.get('duree', 0):>5.1f}s  {nodes}{extra}")
    ok = sum(1 for j in done.values() if j["state"] == "TERMINE")
    print(f"\n{ok}/{len(jobs)} terminés, {len(done) - ok} en échec, {len(jobs) - len(done)} encore en cours.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nSuivi arrêté.")
