# 🧠 Brain Tumor Classification Pipeline

Pipeline complet de diagnostic de tumeurs cérébrales par imagerie IRM utilisant le Deep Learning.

## 📁 Architecture du projet

```
brain-tumor-classification-pipeline/
├── config.py                      # Configuration centralisée (chemins, hyperparamètres)
├── requirements.txt               # Dépendances Python
├── .gitignore                     # Fichiers ignorés par Git
│
├── data/                          # Datasets
│   ├── detection/                 # Détection binaire (tumeur / sain)
│   │   ├── no/                    #   Images sans tumeur
│   │   └── yes/                   #   Images avec tumeur
│   └── segmentation/              # Segmentation U-Net
│       └── brain_tumor_data/
│           ├── train/
│           ├── val/
│           └── test/
│
├── src/                           # Code source
│   ├── training/                  # Scripts d'entraînement
│   │   ├── train_detection.py     #   CNN détection binaire
│   │   ├── train_segmentation.py  #   U-Net segmentation
│   │   └── train_classification.py #  CNN classification 4 classes
│   ├── inference/                 # Scripts de test / inférence
│   │   └── test_segmentation.py   #   Test U-Net + Random Forest
│   └── app/                       # Application GUI
│       └── pipeline.py            #   Interface NeuroVision IA
│
├── models/                        # Modèles entraînés sauvegardés
│   └── tumor_detection_model.h5
│
└── outputs/                       # Résultats d'entraînement
    ├── confusion_matrix.png
    └── training_curves.png
```

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## ▶️ Utilisation

### Lancer l'application GUI
```bash
python src/app/pipeline.py
```

### Entraîner le modèle de détection
```bash
python src/training/train_detection.py
```

### Entraîner le modèle U-Net (segmentation)
```bash
python src/training/train_segmentation.py
```

### Entraîner le modèle de classification (4 classes)
```bash
python src/training/train_classification.py
```

### Tester la segmentation + classification
```bash
python src/inference/test_segmentation.py
```

---

## ⚠️ Fichiers volumineux (Google Drive)

En raison de la **limite de taille imposée par GitHub (100 MB)**,  
les éléments suivants **ne sont pas stockés directement dans le repository** :

- ✅ Modèle entraîné **U-NET (.keras)**
- ✅ Dataset d'entraînement
- ✅ Dataset de test

📂 **Tous ces fichiers sont disponibles via Google Drive :**  
🔗 **Lien Google Drive :**  
👉 *https://drive.google.com/drive/folders/1aDh1BquiP_WZHqCHrg8pNNNtzw9zwAa7?usp=sharing*

---
