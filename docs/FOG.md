# Brain tumor fog node: détection

Nœud fog de détection de tumeur cérébrale (CNN binaire), conteneurisé avec Docker.
Le même code tourne sur plusieurs machines : seul le fichier `.env` change.

> Projet académique. Outil d'aide à la décision, ne remplace pas un avis médical.

## Structure

```
brain-tumer-fog/
├── fog/
│   ├── Dockerfile          # recette de l'image
│   ├── main.py             # service FastAPI (/health, /stats, /detect)
│   └── requirements.txt    # dépendances Python
├── docker-compose.yml      # lancement du nœud
├── .env.example            # configuration à copier en .env
└── README.md
```

## Modèle (non versionné)

Placer le CNN dans `models/` :

```
models/tumor_detection_model.h5
```

Pour une autre machine, le copier par `scp` ou depuis le Google Drive du projet.

## Figer les versions (une fois)

Depuis l'environnement Python qui charge correctement le modèle :

```bash
pip freeze | grep -iE "^(tensorflow|keras|h5py|numpy|fastapi|uvicorn|python-multipart)==" \
  | sed 's/^tensorflow==/tensorflow-cpu==/' > fog/requirements.txt
echo "opencv-python-headless" >> fog/requirements.txt
for p in fastapi uvicorn python-multipart; do
  grep -qi "^$p" fog/requirements.txt || echo "$p" >> fog/requirements.txt
done
```

## Lancer un nœud

```bash
cp .env.example .env          # machine 2 : mettre NODE_NAME=detection-2
docker compose up -d --build
docker compose ps             # attendre l'état "healthy" (environ 1 minute)
docker compose logs -f
```

## Tester

```bash
curl http://localhost:8001/health
curl -s -X POST -F "file=@irm.jpg" http://localhost:8001/detect
curl http://localhost:8001/stats
```

Depuis l'autre machine, remplacer `localhost` par l'adresse IP de la machine du nœud
et ouvrir le port 8001 dans son pare-feu.

## Réponse de `/detect`

```json
{"node":"detection-1","tumor":true,"probability":0.931,"confidence":93.1,
 "threshold":0.4,"latency_ms":41.2}
```

## Variables

| Variable | Rôle | Défaut |
|---|---|---|
| `NODE_NAME` | Nom du nœud | `detection-1` |
| `HOST_PORT` | Port publié sur la machine | `8001` |
| `DET_THRESHOLD` | Seuil de décision | `0.4` |
| `TF_THREADS` | Threads TensorFlow | `2` |
| `DELAY_MS` | Délai artificiel (démo) | `0` |
| `CPUS`, `MEM_LIMIT` | Limites du conteneur | `2.0`, `2g` |