import os
import sys
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Ajouter la racine du projet au path pour importer config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (DETECTION_DATA_DIR, DETECTION_MODEL_PATH,
                    CONFUSION_MATRIX_PATH, TRAINING_CURVES_PATH,
                    IMG_SIZE as CONFIG_IMG_SIZE)

tf.keras.backend.clear_session()
models = tf.keras.models
layers = tf.keras.layers
Input = tf.keras.layers.Input
BatchNormalization = tf.keras.layers.BatchNormalization
Dropout = tf.keras.layers.Dropout
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

dataset_path = str(DETECTION_DATA_DIR)

data = []
labels = [] 

IMG_SIZE = 128

categories = ['no','yes']

print("📂 Chargement des données...")
for label, category in enumerate(categories):
    dataset = os.path.join(dataset_path, category)    
    for i in os.listdir(dataset):
        img = cv2.imread(os.path.join(dataset, i), cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue
        
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        data.append(img)
        labels.append(label)

data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.int32) 

print(f"✅ Données chargées : {data.shape}")
print(f"   - Classe 0 (no) : {np.sum(labels == 0)} images")
print(f"   - Classe 1 (yes) : {np.sum(labels == 1)} images")

data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# Construction du modèle
model = models.Sequential([
    Input(shape=(128, 128, 1)),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n🏗️ Architecture du modèle :")
model.summary()

# ✅ ENTRAÎNEMENT D'ABORD
print('\n🚀 Entraînement en cours...')
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=20,
    validation_data=(x_test, y_test),
    verbose=1
)

print('\n✅ Entraînement terminé !')

# ✅ SAUVEGARDE APRÈS ENTRAÎNEMENT
model.save(
    str(DETECTION_MODEL_PATH),
    save_format='h5',
    include_optimizer=True
)
print(f"\n💾 Modèle sauvegardé dans '{DETECTION_MODEL_PATH}'")
print(f"   Input shape : {model.input_shape}")
print(f"   Output shape : {model.output_shape}")

# Évaluation
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n📊 Précision sur le test : {test_acc:.2%}")

# Prédictions
y_pred_prob = model.predict(x_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

# Analyse des prédictions
print("\n🔍 Distribution des prédictions :")
print(f"   Min : {y_pred_prob.min():.4f}")
print(f"   Max : {y_pred_prob.max():.4f}")
print(f"   Moyenne classe 0 (no) : {y_pred_prob[y_test == 0].mean():.4f}")
print(f"   Moyenne classe 1 (yes) : {y_pred_prob[y_test == 1].mean():.4f}")

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("\n📈 Matrice de confusion :")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["no", "yes"],
    yticklabels=["no", "yes"]
)
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title("Matrice de confusion - Détection de tumeur")
plt.tight_layout()
plt.savefig(str(CONFUSION_MATRIX_PATH), dpi=150)
print(f"💾 Matrice sauvegardée : {CONFUSION_MATRIX_PATH}")
plt.show()

# Rapport de classification
print("\n📋 Rapport détaillé :")
print(classification_report(
    y_test,
    y_pred,
    target_names=["no (sain)", "yes (tumeur)"],
    digits=4
))

# Courbes d'apprentissage
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Précision par époque')
plt.xlabel('Époque')
plt.ylabel('Précision')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Perte par époque')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(str(TRAINING_CURVES_PATH), dpi=150)
print(f"💾 Courbes sauvegardées : {TRAINING_CURVES_PATH}")
plt.show()

print("\n✅ Entraînement complet terminé !")