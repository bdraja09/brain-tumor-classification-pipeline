# 🧠 Brain Tumor Classification Pipeline

Pipeline de diagnostic assisté de tumeurs cérébrales par imagerie IRM, combinant Deep Learning (CNN, U-Net) et Machine Learning classique (Random Forest sur descripteurs HOG).

> ⚠️ **Projet académique / démonstration.** Cet outil est une aide à la décision et **ne remplace pas un avis médical**.

## 🔬 Fonctionnement

Le pipeline s'exécute en 3 phases, pilotées depuis l'interface **NeuroVision IA** :

| Phase | Méthode | Résultat |
|---|---|---|
| 1. Détection | CNN binaire (entrée 128×128, niveaux de gris) | Tumeur / sain + confiance |
| 2. Segmentation | Seuillage + morphologie (U-Net dans `src/training`) | Masque, surface, pourcentage |
| 3. Classification | Random Forest sur HOG (96×96, 4356 features) | Gliome, méningiome, hypophysaire, pas de tumeur + probabilité |

Le Random Forest atteint environ **82 %** d'accuracy sur le jeu de test du dataset d'entraînement ([Brain Tumor MRI Dataset, M. Nickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)). Les performances peuvent baisser sur des images issues d'autres sources (contraste, scanner, recadrage).

## 📁 Architecture du projet

```
brain-tumor-classification-pipeline/
├── config.py                       # Configuration centralisée (chemins, seuils)
├── requirements.txt                # Dépendances Python
├── .gitignore                      # Fichiers ignorés par Git
│
├── src/                            # Code source
│   ├── training/                   # Scripts d'entraînement
│   │   ├── train_detection.py      #   CNN détection binaire
│   │   ├── train_segmentation.py   #   U-Net segmentation
│   │   └── train_classification.py #   Classification des types de tumeur
│   ├── inference/                  # Scripts de test / inférence
│   │   └── test_segmentation.py    #   Test de la segmentation
│   └── app/                        # Application GUI
│       └── pipeline.py             #   Interface NeuroVision IA
│
├── models/                         # Modèles entraînés (non versionnés, voir ci-dessous)
│   ├── tumor_detection_model.h5    #   CNN de détection
│   └── brain_tumor_rf/
│       ├── random_forest.joblib    #   Random Forest (HOG)
│       └── metadata.json           #   Classes et paramètres HOG (versionné)
│
├── data/                           # Datasets (non versionnés)
└── outputs/                        # Résultats d'entraînement (non versionnés)
```

## 🚀 Installation

```bash
git clone https://github.com/bdraja09/brain-tumor-classification-pipeline.git
cd brain-tumor-classification-pipeline

python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

> Le Random Forest a été entraîné avec scikit-learn sur Google Colab. Utilisez la même version de scikit-learn que celle indiquée dans `requirements.txt`, sinon le chargement du modèle peut échouer.

## 📦 Modèles et données (non versionnés)

En raison de la **limite de taille de GitHub (100 Mo)** et pour éviter de publier des données médicales, les éléments suivants **ne sont pas dans le dépôt** :

- Modèle CNN de détection (`.h5`)
- Modèle U-Net (`.keras`)
- Modèle Random Forest (`.joblib`)
- Datasets d'entraînement et de test

**Google Drive** (modèles et datasets) :
👉 https://drive.google.com/drive/folders/1aDh1BquiP_WZHqCHrg8pNNNtzw9zwAa7?usp=sharing

**Random Forest** : également hébergé sur Hugging Face (dépôt privé `raja-dev/brain_tumer_classifier`, accès sur demande) :

```bash
huggingface-cli login
huggingface-cli download raja-dev/brain_tumer_classifier --local-dir models/brain_tumor_rf
```

Après téléchargement, placez les fichiers aux emplacements indiqués dans l'arborescence ci-dessus (les chemins sont définis dans `config.py`).

## ▶️ Utilisation

### Lancer l'application GUI
```bash
python src/app/pipeline.py
```

Étapes dans l'interface : **1.** Charger IRM → **2.** Détection CNN → **3.** Segmentation → **4.** Classification. Le bouton *Réinitialiser* vide toutes les zones d'affichage.

### Entraîner les modèles
```bash
python src/training/train_detection.py        # CNN de détection
python src/training/train_segmentation.py     # U-Net
python src/training/train_classification.py   # Classification des types de tumeur
```

### Tester la segmentation
```bash
python src/inference/test_segmentation.py
```

## ⚠️ Limites connues

- La segmentation par seuil fixe fonctionne sur des IRM très contrastées ; elle est moins fiable sur d'autres jeux d'images.
- Le Random Forest sur HOG est sensible au changement de domaine ; un résultat avec une probabilité inférieure à 60 % est signalé comme incertain dans l'interface.
- Le **grade** tumoral ne peut pas être estimé par ce pipeline (il nécessite une analyse histologique) ; l'interface affiche uniquement une indication de taille.

## 📄 Licence et avertissement

Les datasets utilisés sont soumis à leurs licences respectives (le dataset Kaggle est sous CC BY 4.0). Ce logiciel est fourni à des fins éducatives et ne doit pas être utilisé pour un diagnostic clinique.