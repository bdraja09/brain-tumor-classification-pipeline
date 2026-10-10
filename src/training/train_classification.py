# ============================================================
# CNN Direct pour Classification de Tumeurs Cérébrales
# Architecture : ResNet-like avec Data Augmentation
# ============================================================

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Ajouter la racine du projet au path pour importer config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (CLASSIFICATION_MODEL_PATH, CLASSIFICATION_CLASSES,
                    CLASSIFICATION_EPOCHS, MODELS_DIR)

# Configuration
sns.set_style("whitegrid")
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = CLASSIFICATION_EPOCHS
classes = CLASSIFICATION_CLASSES

# ============================================================
# 1. Chargement et Préparation des Données
# ============================================================
def load_images_and_labels(data_dir, classes, img_size=128):
    """Charge toutes les images et leurs labels"""
    images = []
    labels = []
    image_paths = []
    
    for idx, class_name in enumerate(classes):
        class_folder = os.path.join(data_dir, class_name)
        if not os.path.exists(class_folder):
            print(f"⚠️  Dossier introuvable : {class_folder}")
            continue
        
        count = 0
        for image_name in os.listdir(class_folder):
            if image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(class_folder, image_name)
                
                # Lire et redimensionner l'image
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                image_resized = cv2.resize(image, (img_size, img_size))
                image_norm = image_resized / 255.0  # Normalisation
                
                images.append(image_norm)
                labels.append(idx)  # Index de la classe
                image_paths.append(img_path)
                count += 1
        
        print(f"✓ {count} images chargées pour '{class_name}'")
    
    # Convertir en numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Ajouter dimension channel (grayscale)
    images = np.expand_dims(images, axis=-1)
    
    return images, labels, image_paths

# ============================================================
# 2. Data Augmentation (CRITIQUE pour améliorer glioma)
# ============================================================
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomFlip("horizontal"),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.2),
])

# ============================================================
# 3. Architecture CNN Optimisée
# ============================================================
def create_improved_cnn(input_shape=(128, 128, 1), num_classes=4):
    """
    CNN avec architecture ResNet-like
    - Blocs résiduels pour éviter vanishing gradient
    - Batch Normalization pour stabilité
    - Dropout pour régularisation
    """
    
    inputs = keras.Input(shape=input_shape)
    
    # Data augmentation (appliquée pendant l'entraînement)
    x = data_augmentation(inputs)
    
    # Bloc 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Bloc 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Bloc 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Bloc 4
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
    # Couches denses
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Sortie
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='brain_tumor_cnn')
    return model

# ============================================================
# 4. Chargement des Données
# ============================================================
# NOTE : Ces chemins pointent vers un dataset externe (4 classes).
# Placez vos données dans data/classification/Training/ et data/classification/Testing/
# ou modifiez ces chemins selon votre environnement.
train_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'classification', 'Training')
test_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'classification', 'Testing')

print("📥 Chargement des données d'entraînement...")
X_train_full, y_train_full, train_paths = load_images_and_labels(train_dir, classes, IMG_SIZE)

print("\n📥 Chargement des données de test...")
X_test, y_test, test_paths = load_images_and_labels(test_dir, classes, IMG_SIZE)

# Split train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

print(f"\n✓ Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

# Calculer les poids des classes pour gérer le déséquilibre
from sklearn.utils.class_weight import compute_class_weight
class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights_array))
print(f"\n⚖️  Poids des classes: {class_weights}")

# ============================================================
# 5. Création et Compilation du Modèle
# ============================================================
print("\n🏗️  Construction du modèle CNN...")
model = create_improved_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=len(classes))
model.summary()

# Compilation avec learning rate adaptatif
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# 6. Callbacks pour l'entraînement
# ============================================================
callbacks = [
    # Arrêt anticipé si pas d'amélioration
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Réduction du learning rate
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    
    # Sauvegarder le meilleur modèle
    ModelCheckpoint(
        str(CLASSIFICATION_MODEL_PATH),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ============================================================
# 7. Entraînement
# ============================================================
print("\n🚀 Début de l'entraînement...")
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# 8. Évaluation sur le Test Set
# ============================================================
print("\n📊 Évaluation sur le test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Prédictions
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# ============================================================
# 9. Visualisations
# ============================================================

# Courbes d'apprentissage
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0].set_title('Accuracy pendant l\'entraînement', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'], label='Train', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[1].set_title('Loss pendant l\'entraînement', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes,
            cbar_kws={'label': 'Nombre de prédictions'},
            linewidths=2, linecolor='white', square=True,
            annot_kws={"size": 14, "weight": "bold"})
plt.title('Matrice de Confusion - CNN Direct', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('Vraie Classe', fontsize=14, fontweight='bold')
plt.xlabel('Classe Prédite', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Rapport détaillé
print("\n📋 Rapport de Classification:")
print(classification_report(y_test, y_pred, target_names=classes))

# Accuracy par classe
print("\n📈 Statistiques par classe:")
for i, cls in enumerate(classes):
    indices_cls = y_test == i
    if np.sum(indices_cls) > 0:
        acc_cls = np.mean(y_pred[indices_cls] == y_test[indices_cls])
        print(f"  • {cls}: {acc_cls*100:.2f}% ({np.sum(indices_cls)} images)")

# ============================================================
# 10. Visualisation des Prédictions
# ============================================================
print("\n🖼️  Génération de la grille de visualisation...")

num_examples = min(15, len(X_test))
rows = 5
cols = 3

fig = plt.figure(figsize=(18, rows*4))

for idx in range(num_examples):
    image = X_test[idx]
    true_label = y_test[idx]
    pred_label = y_pred[idx]
    confidence = y_pred_probs[idx][pred_label] * 100
    
    true_class = classes[true_label]
    pred_class = classes[pred_label]
    
    is_correct = true_label == pred_label
    border_color = 'green' if is_correct else 'red'
    status_icon = '✓' if is_correct else '✗'
    
    # Position dans la grille
    position = idx + 1
    ax = plt.subplot(rows, cols, position)
    
    # Afficher l'image
    ax.imshow(image.squeeze(), cmap='gray')
    ax.axis('off')
    
    # Titre avec informations
    title_text = f"{status_icon} Prédit: {pred_class} ({confidence:.1f}%)\nRéel: {true_class}"
    ax.set_title(title_text, fontsize=10, fontweight='bold', 
                color=border_color, pad=10)
    
    # Bordure colorée
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(3)
        spine.set_visible(True)

plt.suptitle('Résultats CNN - Images Test', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

# ============================================================
# 11. Analyse des Erreurs
# ============================================================
print("\n❌ Analyse des Erreurs:")
print("="*70)

errors = []
for i in range(len(X_test)):
    if y_pred[i] != y_test[i]:
        errors.append({
            'Image': os.path.basename(test_paths[i]),
            'Prédit': classes[y_pred[i]],
            'Réel': classes[y_test[i]],
            'Confiance': f"{y_pred_probs[i][y_pred[i]]*100:.1f}%"
        })

if len(errors) > 0:
    errors_df = pd.DataFrame(errors)
    print(f"\nNombre total d'erreurs: {len(errors)} / {len(y_test)} ({len(errors)/len(y_test)*100:.1f}%)")
    print("\nTop 10 erreurs:")
    print(errors_df.head(10).to_string(index=False))
    
    print("\n🔀 Confusions les plus fréquentes:")
    confusion_pairs = {}
    for err in errors:
        pair = f"{err['Réel']} → {err['Prédit']}"
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    
    sorted_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    for pair, count in sorted_confusions[:5]:
        print(f"  • {pair}: {count} fois")
else:
    print("🎉 Aucune erreur ! Classification parfaite !")

print("\n" + "="*70)
print("\n✅ Entraînement terminé ! Modèle sauvegardé dans 'best_cnn_model.keras'")