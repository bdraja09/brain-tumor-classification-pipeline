#!/usr/bin/env python3
"""
Test de bout en bout des nœuds fog (bibliothèque standard uniquement).

Pour chaque nœud testé : GET /health puis POST de l'image sur son endpoint.
Ensuite, compare les deux réplicas d'une même tâche (s'ils ont été testés tous les deux).

Placement des nœuds :
  machine A : detection-1, segmentation-2, classification-2
  machine B : detection-2, segmentation-1, classification-1

Modes d'adressage des nœuds LOCAUX (ceux de cette machine) :
  host  http://localhost:<port publié>        (avec docker-compose.dev.yml, ou nœuds lancés en local)
  dns   http://<nom du nœud>:<port interne>   (depuis un conteneur du sous-réseau)
  ip    http://<IP fixe>:<port interne>       (depuis un conteneur du sous-réseau)

Les nœuds de l'AUTRE machine ne sont testés que si --remote-host est donné
(http://<IP de l'autre machine>:<port publié>).

Exemples :
  # machine A, nœuds locaux seulement
  python scripts/smoke_test.py --machine a --image samples/irm.jpg
  # machine A, avec l'autre machine (IP du réseau local) : teste aussi la cohérence des réplicas
  python scripts/smoke_test.py --machine a --remote-host 192.168.1.20 --image samples/irm.jpg
  # depuis le sous-réseau Docker de la machine A
  docker run --rm --network fog-net -v "$PWD/scripts:/s:ro" -v "$PWD/samples:/samples:ro" \
      python:3.11-slim python /s/smoke_test.py --machine a --mode dns --image /samples/irm.jpg

Code de sortie : 0 si tout ce qui a été testé est conforme, 1 sinon.
"""
import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

# nom, tâche, endpoint, port publié, port interne, IP fixe
NODES = [
    ("detection-1",      "detection",      "/detect",   8001, 8001, "172.28.0.11"),
    ("detection-2",      "detection",      "/detect",   8011, 8001, "172.28.0.12"),
    ("segmentation-1",   "segmentation",   "/segment",  8002, 8002, "172.28.0.21"),
    ("segmentation-2",   "segmentation",   "/segment",  8012, 8002, "172.28.0.22"),
    ("classification-1", "classification", "/classify", 8003, 8003, "172.28.0.31"),
    ("classification-2", "classification", "/classify", 8013, 8003, "172.28.0.32"),
]

MACHINES = {
    "a": {"detection-1", "segmentation-2", "classification-2"},
    "b": {"detection-2", "segmentation-1", "classification-1"},
}


def node_url(node, args, local):
    """URL du nœud, ou None s'il n'est pas à tester."""
    name, _, _, host_port, port, ip = node
    if name in local:
        if args.mode == "host":
            return f"http://localhost:{host_port}"
        if args.mode == "dns":
            return f"http://{name}:{port}"
        return f"http://{ip}:{port}"
    if args.remote_host:
        return f"http://{args.remote_host}:{host_port}"
    return None


def get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def post_image(url, path, timeout):
    boundary = uuid.uuid4().hex
    with open(path, "rb") as f:
        data = f.read()
    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    head = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(path)}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n").encode()
    body = head + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def summary(task, res):
    if task == "detection":
        return f"tumeur={res['tumor']} proba={res['probability']} conf={res['confidence']}%"
    if task == "segmentation":
        return f"surface={res['surface_px']}px ({res['percentage']}%) bbox={res['bbox']}"
    return f"{res['label']} {res['confidence']}%"


def same(task, a, b):
    if task == "detection":
        return a["tumor"] == b["tumor"] and abs(a["probability"] - b["probability"]) < 1e-3
    if task == "segmentation":
        return a["surface_px"] == b["surface_px"]
    return a["label"] == b["label"] and abs(a["confidence"] - b["confidence"]) < 0.2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="IRM de test (jpg, png, bmp)")
    ap.add_argument("--machine", choices=["a", "b"], help="jeu de nœuds local (a ou b)")
    ap.add_argument("--local", help="liste de nœuds locaux séparés par des virgules (remplace --machine)")
    ap.add_argument("--mode", choices=["host", "dns", "ip"], default="host",
                    help="adressage des nœuds locaux (défaut : host)")
    ap.add_argument("--remote-host", help="IP de l'autre machine, pour tester ses nœuds")
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"Image introuvable : {args.image}")

    if args.local:
        local = {n.strip() for n in args.local.split(",") if n.strip()}
    elif args.machine:
        local = MACHINES[args.machine]
    else:
        local = {n[0] for n in NODES}      # tout est supposé local
    unknown = local - {n[0] for n in NODES}
    if unknown:
        sys.exit(f"Nœud(s) inconnu(s) : {', '.join(sorted(unknown))}")

    print(f"Nœuds locaux : {', '.join(sorted(local))}  (mode {args.mode})")
    print(f"Autre machine : {args.remote_host or 'non testée'}   Image : {args.image}\n")
    print(f"{'nœud':<18}{'où':<9}{'santé':<8}{'temps':>9}  résultat")
    print("-" * 82)

    results, failures = {}, 0
    for node in NODES:
        name, task, endpoint = node[0], node[1], node[2]
        url = node_url(node, args, local)
        where = "local" if name in local else "distant"
        if url is None:
            print(f"{name:<18}{where:<9}{'-':<8}{'':>9}  non testé")
            continue
        try:
            get_json(url + "/health", args.timeout)
        except Exception as e:
            print(f"{name:<18}{where:<9}{'KO':<8}{'':>9}  health : {type(e).__name__}")
            failures += 1
            continue
        try:
            t0 = time.perf_counter()
            res = post_image(url + endpoint, args.image, args.timeout)
            dt = (time.perf_counter() - t0) * 1000
            results[name] = res
            print(f"{name:<18}{where:<9}{'ok':<8}{dt:>7.0f}ms  {summary(task, res)}")
        except urllib.error.HTTPError as e:
            print(f"{name:<18}{where:<9}{'ok':<8}{'':>9}  HTTP {e.code} : {e.read().decode()[:80]}")
            failures += 1
        except Exception as e:
            print(f"{name:<18}{where:<9}{'ok':<8}{'':>9}  {type(e).__name__} : {e}")
            failures += 1

    print("\nCohérence des réplicas")
    print("-" * 82)
    for task in ("detection", "segmentation", "classification"):
        names = [n[0] for n in NODES if n[1] == task]
        if all(n in results for n in names):
            ok = same(task, results[names[0]], results[names[1]])
            print(f"{task:<16}{'identiques' if ok else 'DIFFÉRENTS'}")
            failures += 0 if ok else 1
        else:
            print(f"{task:<16}non comparé (réplica non testé ou sans réponse)")

    print("\n" + ("TOUT CE QUI A ÉTÉ TESTÉ EST CONFORME" if failures == 0 else f"{failures} problème(s) détecté(s)"))
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()