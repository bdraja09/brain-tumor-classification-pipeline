# Agents de coordination : cas 1 (ACK) et cas 2 (reprise après panne)

Un **agent** (`fog/agent/`) tourne sur chaque machine, à côté de ses trois nœuds. Il ne remplace
aucun nœud : il enchaîne les appels et gère l'envoi d'une machine à l'autre.

```
 Machine B                                   Machine A
 ┌──────────────────────────┐   /handoff    ┌──────────────────────────┐
 │ agent-b                  │ ────────────► │ agent-a                  │
 │  ├─ detection-2          │ ◄──── ACK ─── │  └─ classification-2     │
 │  └─ segmentation-1       │ ◄─ résultat ─ │                          │
 └──────────────────────────┘   /report     └──────────────────────────┘
                              ◄────────────►  (retour après panne)
```

Par défaut, l'agent qui reçoit un job fait **détection + segmentation** sur ses nœuds locaux,
puis envoie l'image à l'autre machine pour la **classification** (variable `LOCAL_STEPS`).
Si la détection ne trouve pas de tumeur, le pipeline s'arrête sans rien envoyer.

## Cas 1 : accusé de réception

1. L'agent d'origine envoie `POST /handoff` (image + `job_id` + SHA-256 + étapes restantes).
2. L'agent récepteur **vérifie le hash**, **enregistre l'image sur disque**, puis répond **ACK**
   (ou **NACK** HTTP 422 si le hash ne correspond pas).
3. Après l'ACK, le récepteur traite l'image ; l'origine récupère le résultat (`GET /handoff/{id}`).
4. Réessais en cas d'absence d'ACK (délais `RETRY_DELAYS_S`), et idempotence par `job_id`
   (un doublon renvoie l'ACK sans retraiter).

## Cas 2 : panne, rapport et redemande de l'image

1. Sans ACK après tous les essais : l'image et les résultats partiels sont **conservés**, le job passe
   en `EN_ATTENTE_PAIR`. L'agent surveille le pair (heartbeat `/health`).
2. Quand l'agent en panne **redémarre**, il envoie un **rapport** `POST /report` à l'autre.
3. L'autre répond avec la liste des jobs que le revenant n'a pas reçus.
4. Le revenant **redemande l'image** (`GET /jobs/{id}/image`), vérifie le hash, envoie son ACK
   (`POST /jobs/{id}/ack`) et traite le job.
5. Jobs et images sont dans SQLite + un volume Docker : un redémarrage ne perd rien.

## Installation (sur chaque machine, depuis la racine du dépôt)

1. Récupérer les fichiers (branche dédiée pour éviter les conflits) :
   ```powershell
   git checkout -b feature/agent
   ```
   puis copier `fog/agent/`, `fog/docker-compose.agent.yml`, `scripts/demo_agent.py`, `docs/AGENT.md`.
2. Dans le `.env`, ajouter l'adresse de **l'autre** machine (port 8090) :
   ```
   PEER_URL=http://<IP de l'autre machine>:8090
   ```
3. Ouvrir le port 8090 (PowerShell administrateur) :
   ```powershell
   New-NetFirewallRule -DisplayName "Fog agent" -Direction Inbound -Protocol TCP -LocalPort 8090 -Action Allow -Profile Any
   ```
4. Lancer :
   ```powershell
   docker compose -f docker-compose.yml -f fog/docker-compose.dev.yml -f fog/docker-compose.agent.yml up -d --build
   docker compose ps
   ```
   Un conteneur `agent-a` (profil machine-a) ou `agent-b` (profil machine-b) s'ajoute aux trois nœuds.

## Démonstration

Journal en direct : `docker compose logs -f agent-b` (ou `agent-a`).

**Cas 1, normal** (sur la machine B) :
```powershell
python scripts\demo_agent.py --agent http://localhost:8090 --image samples\irm.jpg
```
On voit `HANDOFF_ENVOI`, `ACK_RECU`, `RESULTAT_RECU`, et la classification faite par un nœud de l'autre machine.

**Cas 1, image corrompue** (test du NACK) :
```powershell
python scripts\demo_agent.py --agent http://localhost:8090 --image samples\irm.jpg --corrupt
```
Première tentative `NACK`, deuxième `ACK_RECU`.

**Cas 2, panne** :
1. Sur la machine A : `docker stop agent-a`
2. Sur la machine B : `python scripts\demo_agent.py --agent http://localhost:8090 --image samples\irm.jpg --no-wait`
   (noter le `job_id`). Après les essais : `PAS_D_ACK` puis `EN_ATTENTE_PAIR`.
3. Sur la machine A : `docker start agent-a`
4. Sur B : `python scripts\demo_agent.py --agent http://localhost:8090 --job <job_id>`
   → `ACK_RECU après redemande de l'image`, puis `TERMINE`.
   Sur A (`docker compose logs agent-a`) : `RAPPORT_ENVOYE`, `REDEMANDE_IMAGE`, `IMAGE_RECUE`, `ACK_ENVOYE`.

Lister les jobs : `python scripts\demo_agent.py --agent http://localhost:8090 --list`.

## Tester avec plusieurs images et des nœuds lents (pour pouvoir les interrompre)

Avec une seule image, un job dure moins d'une seconde : impossible d'arrêter un nœud en plein travail.
Deux réglages rendent la panne observable :

1. **Ralentir les nœuds** : ajouter `DELAY_MS=2000` dans le `.env` des deux machines, puis relancer
   la même commande `docker compose ... up -d` (les conteneurs sont recréés avec le nouveau délai).
2. **Envoyer plusieurs images** (une toutes les secondes, elles se chevauchent) :
   ```powershell
   python scripts\demo_batch.py --agent http://localhost:8090 --folder database --count 6 --pause 1
   ```
   `--folder` : un dossier d'images (recherche récursive, voir `dir database`). Le script affiche le nombre de
   jobs par état, puis un tableau final : nœuds utilisés, classe, durée.

Chaque nœud traite une requête à la fois (`MAX_CONCURRENCY=1`) : les autres attendent leur tour, et un
appel local est abandonné après 30 s. Gardez donc `DELAY_MS` x nombre d'images sous environ 25 s.

Pendant que le lot tourne, arrêter un agent (`docker stop agent-a`) puis le relancer (`docker start agent-a`) :
les jobs en cours sont repris et aucun n'est perdu.

Arrêter un **nœud** (par exemple `docker stop classification-2`) pendant un job fait passer ce job en `ECHEC` :
l'agent ne bascule pas encore vers le réplica de l'autre machine (amélioration possible).

## Variables (.env)

| Variable | Rôle | Défaut |
|---|---|---|
| `PEER_URL` | Adresse de l'agent de l'autre machine | (obligatoire) |
| `LOCAL_STEPS` | Étapes faites en local, le reste part chez le pair | `detection,segmentation` |
| `STOP_IF_NO_TUMOR` | Arrêt du pipeline si pas de tumeur | `1` |
| `ACK_TIMEOUT_S` | Attente maximale d'un ACK | `5` |
| `RETRY_DELAYS_S` | Pauses entre les réessais | `1,2,4` |

## Limites connues

- Le pair est supposé joignable en HTTP ; la sécurité (TLS, authentification) n'est pas traitée.
- Si un nœud **local** tombe (par exemple `segmentation-1`), le job passe en `ECHEC` : le basculement
  vers le réplica de l'autre machine est une amélioration possible.
- L'image reste sur le disque de l'origine tant que le job n'est pas supprimé (pas de nettoyage automatique).
