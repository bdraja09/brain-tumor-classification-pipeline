# ============================================================
# Brain Tumor Segmentation + Classification Pipeline
# U-Net (segmentation) + Random Forest (classification)
# ============================================================

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Ajouter la racine du projet au path pour importer config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import SEGMENTATION_MODEL_PATH, SEGMENTATION_CLASSES

# ============================================================
# U-Net metrics
# ============================================================
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def iou(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# ============================================================
# Configuration
# ============================================================
target_size = (128, 128)
classes = SEGMENTATION_CLASSES

# NOTE : Ces chemins pointent vers un dataset externe (4 classes).
# Placez vos données dans data/classification/Training/ et data/classification/Testing/
# ou modifiez ces chemins selon votre environnement.
train_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'classification', 'Training')
test_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'classification', 'Testing')

model_path = str(SEGMENTATION_MODEL_PATH)

# ============================================================
# Load U-Net model
# ============================================================
model = load_model(
    model_path,
    custom_objects={
        'dice_coefficient': dice_coefficient,
        'dice_loss': dice_loss,
        'iou': iou
    }
)
print("✓ U-Net chargé")

# ============================================================
# Segmentation function
# ============================================================
def predict_single_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size)
    image_norm = image / 255.0
    image_input = np.expand_dims(image_norm, axis=(0, -1))

    pred = model.predict(image_input, verbose=0)
    mask = (pred[0] > 0.5).astype(np.uint8)

    return image, mask

# ============================================================
# Feature extraction
# ============================================================
def extract_features(data_dir):
    features, labels, paths = [], [], []

    for cls in classes:
        folder = os.path.join(data_dir, cls)
        if not os.path.exists(folder):
            continue

        for img_name in os.listdir(folder):
            if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(folder, img_name)
                _, mask = predict_single_image(path)

                tumor_ratio = np.mean(mask)
                tumor_area = np.sum(mask)
                tumor_height = np.sum(np.any(mask, axis=1))
                tumor_width = np.sum(np.any(mask, axis=0))

                features.append([
                    tumor_ratio,
                    tumor_area,
                    tumor_height,
                    tumor_width
                ])

                labels.append(cls)
                paths.append(path)

    return np.array(features), np.array(labels), paths

# ============================================================
# Load data
# ============================================================
X_train, y_train, train_paths = extract_features(train_dir)
X_test, y_test, test_paths   = extract_features(test_dir)

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Testing samples : {len(X_test)}")

# ============================================================
# Random Forest Classifier (improved)
# ============================================================
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)

# ============================================================
# Prediction
# ============================================================
preds = clf.predict(X_test)

# ============================================================
# Evaluation
# ============================================================
print("\n📊 Classification Report:")
print(classification_report(y_test, preds))

# ============================================================
# Confusion Matrix
# ============================================================
cm = confusion_matrix(y_test, preds, labels=classes)

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes,
            yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Tumor Classification")
plt.show()

# ============================================================
# Visualization with labels on images
# ============================================================
for path, pred, true_label in zip(test_paths[:10], preds[:10], y_test[:10]):
    image, mask = predict_single_image(path)

    plt.figure(figsize=(10,4))

    # Original image
    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title("Image originale")

    plt.text(
        5, 15,
        f"Vrai : {true_label}\nPrédit : {pred}",
        color='yellow',
        fontsize=10,
        bbox=dict(facecolor='black', alpha=0.6)
    )

    # Segmentation mask
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.title("Masque de la tumeur")

    plt.show()
