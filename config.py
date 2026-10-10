# ============================================================
# Configuration centralisée du projet
# Brain Tumor Classification Pipeline
# ============================================================

import os
from pathlib import Path

# ============================================================
# Racine du projet (détectée automatiquement)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================
# Chemins des données
# ============================================================
DATA_DIR = PROJECT_ROOT / "data"

# Détection binaire (tumeur / pas de tumeur)
DETECTION_DATA_DIR = DATA_DIR / "detection"

# Segmentation U-Net (images + masques)
SEGMENTATION_DATA_DIR = DATA_DIR / "segmentation" / "brain_tumor_data"

# ============================================================
# Chemins des modèles sauvegardés
# ============================================================
MODELS_DIR = PROJECT_ROOT / "models"
DETECTION_MODEL_PATH = MODELS_DIR / "tumor_detection_model.h5"
SEGMENTATION_MODEL_PATH = MODELS_DIR / "best_unetmodel.keras"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "best_cnn_model.keras"
CLASSIFIER_RF_PATH = MODELS_DIR / "brain_tumor_rf/random_forest.joblib"

# ============================================================
# Chemins des sorties (graphiques, métriques)
# ============================================================
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"
TRAINING_CURVES_PATH = OUTPUTS_DIR / "training_curves.png"

# ============================================================
# Hyperparamètres communs
# ============================================================
IMG_SIZE = 128
TARGET_SIZE = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE = 32

# Détection
DETECTION_EPOCHS = 20
DETECTION_THRESHOLD = 0.4

# Segmentation
SEGMENTATION_EPOCHS = 50
SEGMENTATION_LR = 1e-4

# Classification (4 classes)
CLASSIFICATION_EPOCHS = 50
CLASSIFICATION_CLASSES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
SEGMENTATION_CLASSES = ['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']
DETECTION_CATEGORIES = ['no', 'yes']

# ============================================================
# Création automatique des dossiers de sortie
# ============================================================
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
