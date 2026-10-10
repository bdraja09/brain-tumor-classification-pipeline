import os
import sys
import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import tensorflow as tf
import joblib
from skimage.feature import hog

# Ajouter la racine du projet au path pour importer config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (DETECTION_MODEL_PATH, CLASSIFIER_RF_PATH,
                    DETECTION_THRESHOLD, IMG_SIZE as CONFIG_IMG_SIZE)

# ================= CONFIG UI =================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class BrainTumorPipeline(ctk.CTk):

    # ================= INIT =================
    def __init__(self):
        super().__init__()

        self.title("NeuroVision IA - Pipeline Diagnostic Complet")
        self.geometry("1400x800")

        # --- CNN détection / segmentation ---
        self.IMG_SIZE = 128
        self.DET_THRESHOLD = 0.4  # Seuil de décision (tumeur si > 0.4)

        # --- Random Forest (doit être identique au notebook d'entraînement) ---
        self.CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
        self.CLASS_FR = {
            "glioma": "Gliome",
            "meningioma": "Méningiome",
            "notumor": "Pas de tumeur",
            "pituitary": "Tumeur hypophysaire",
        }
        self.HOG_SIZE = (96, 96)
        self.RF_MIN_CONFIDENCE = 60  # % en dessous duquel le résultat est jugé incertain

        self.current_image = None      # BGR 128x128 (affichage + CNN)
        self.original_gray = None      # Gris pleine résolution uint8 (HOG)
        self.current_mask = None

        self.detection_result = None
        self.detection_confidence = 0.0
        self.segmentation_done = False
        self.classification_done = False

        self.load_models()
        self.build_ui()

    # ================= MODELS =================
    def load_models(self):
        """Charge les modèles avec gestion d'erreurs"""
        try:
            # compile=False pour éviter les warnings
            self.detection_model = tf.keras.models.load_model(
                str(DETECTION_MODEL_PATH),
                compile=False
            )

            self.detection_model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            print("✅ CNN Détection chargé et recompilé")
            print(f"   Input shape attendu : {self.detection_model.input_shape}")

        except Exception as e:
            self.detection_model = None
            messagebox.showerror("Erreur", f"Modèle détection introuvable :\n{e}")

        try:
            self.classifier_model = joblib.load(str(CLASSIFIER_RF_PATH))
            print("✅ Random Forest chargé")
            print(f"   Features attendues : {self.classifier_model.n_features_in_}")
        except Exception as e:
            self.classifier_model = None
            print(f"⚠️  Random Forest non chargé : {e}")
            print("   → Classification par taille activée")

    # ================= UI =================
    def build_ui(self):
        """Construction de l'interface graphique"""
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # === SIDEBAR ===
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(10, weight=1)

        # Logo
        logo = ctk.CTkLabel(
            self.sidebar,
            text="🧠 NEUROVISION IA",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        logo.grid(row=0, column=0, pady=(20, 5), padx=20)

        subtitle = ctk.CTkLabel(
            self.sidebar,
            text="Pipeline Diagnostic en 3 Phases",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        subtitle.grid(row=1, column=0, pady=(0, 20))

        # Boutons
        self.btn_load = ctk.CTkButton(
            self.sidebar,
            text="📂 1. Charger IRM",
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.load_image
        )
        self.btn_load.grid(row=2, column=0, pady=8, padx=20, sticky="ew")

        self.btn_detect = ctk.CTkButton(
            self.sidebar,
            text="🔍 2. Détection CNN",
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#3498db",
            hover_color="#2980b9",
            state="disabled",
            command=self.run_detection
        )
        self.btn_detect.grid(row=3, column=0, pady=8, padx=20, sticky="ew")

        self.btn_segment = ctk.CTkButton(
            self.sidebar,
            text="🎯 3. Segmentation",
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#27ae60",
            hover_color="#229954",
            state="disabled",
            command=self.run_segmentation
        )
        self.btn_segment.grid(row=4, column=0, pady=8, padx=20, sticky="ew")

        self.btn_classify = ctk.CTkButton(
            self.sidebar,
            text="📊 4. Classification",
            height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#e67e22",
            hover_color="#d35400",
            state="disabled",
            command=self.run_classification
        )
        self.btn_classify.grid(row=5, column=0, pady=8, padx=20, sticky="ew")

        # Rapport
        report_frame = ctk.CTkFrame(self.sidebar, fg_color="#1a1a1a", corner_radius=8)
        report_frame.grid(row=6, column=0, pady=15, padx=15, sticky="nsew")

        ctk.CTkLabel(
            report_frame,
            text="📋 RAPPORT DIAGNOSTIC",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=10)

        ctk.CTkFrame(report_frame, height=1, fg_color="gray").pack(fill="x", padx=15)

        # Phase 1
        ctk.CTkLabel(
            report_frame,
            text="PHASE 1 : DÉTECTION",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#3498db"
        ).pack(pady=(10, 5))

        self.status_label = ctk.CTkLabel(
            report_frame,
            text="Statut : En attente",
            text_color="gray"
        )
        self.status_label.pack(pady=2)

        self.confidence_label = ctk.CTkLabel(
            report_frame,
            text="Confiance : --",
            text_color="gray"
        )
        self.confidence_label.pack(pady=2)

        ctk.CTkFrame(report_frame, height=1, fg_color="gray").pack(fill="x", padx=15, pady=8)

        # Phase 2
        ctk.CTkLabel(
            report_frame,
            text="PHASE 2 : SEGMENTATION",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#27ae60"
        ).pack(pady=5)

        self.surface_label = ctk.CTkLabel(
            report_frame,
            text="Surface : --",
            text_color="gray"
        )
        self.surface_label.pack(pady=2)

        self.percent_label = ctk.CTkLabel(
            report_frame,
            text="Pourcentage : --",
            text_color="gray"
        )
        self.percent_label.pack(pady=2)

        ctk.CTkFrame(report_frame, height=1, fg_color="gray").pack(fill="x", padx=15, pady=8)

        # Phase 3
        ctk.CTkLabel(
            report_frame,
            text="PHASE 3 : CLASSIFICATION",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#e67e22"
        ).pack(pady=5)

        self.class_label = ctk.CTkLabel(
            report_frame,
            text="Type : --",
            text_color="gray",
            wraplength=260
        )
        self.class_label.pack(pady=2)

        self.size_label = ctk.CTkLabel(
            report_frame,
            text="Taille : --",
            text_color="gray"
        )
        self.size_label.pack(pady=(2, 15))

        # Reset
        self.btn_reset = ctk.CTkButton(
            self.sidebar,
            text="🔄 Réinitialiser",
            fg_color="#95a5a6",
            hover_color="#7f8c8d",
            command=self.reset_pipeline
        )
        self.btn_reset.grid(row=11, column=0, pady=10, padx=20, sticky="ew")

        # Mention légale
        ctk.CTkLabel(
            self.sidebar,
            text="Outil d'aide à la décision.\nNe remplace pas un avis médical.",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).grid(row=12, column=0, pady=(0, 10), padx=20)

        # === MAIN AREA ===
        self.main = ctk.CTkFrame(self, fg_color="transparent")
        self.main.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        self.main.grid_columnconfigure((0, 1, 2), weight=1)
        self.main.grid_rowconfigure(0, weight=1)

        # Canvas 1 : Original
        frame1 = ctk.CTkFrame(self.main, fg_color="#1a1a1a", corner_radius=8)
        frame1.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        ctk.CTkLabel(frame1, text="📸 IRM ORIGINALE", font=ctk.CTkFont(size=13, weight="bold")).pack(pady=8)
        self.canvas_orig = ctk.CTkLabel(frame1, text="Aucune image", fg_color="#2c2c2c", width=350, height=350, corner_radius=8)
        self.canvas_orig.pack(padx=8, pady=8)

        # Canvas 2 : Segmentation
        frame2 = ctk.CTkFrame(self.main, fg_color="#1a1a1a", corner_radius=8)
        frame2.grid(row=0, column=1, padx=8, pady=8, sticky="nsew")
        ctk.CTkLabel(frame2, text="🎯 SEGMENTATION", font=ctk.CTkFont(size=13, weight="bold")).pack(pady=8)
        self.canvas_seg = ctk.CTkLabel(frame2, text="En attente", fg_color="#2c2c2c", width=350, height=350, corner_radius=8)
        self.canvas_seg.pack(padx=8, pady=8)

        # Canvas 3 : Résultat
        frame3 = ctk.CTkFrame(self.main, fg_color="#1a1a1a", corner_radius=8)
        frame3.grid(row=0, column=2, padx=8, pady=8, sticky="nsew")
        ctk.CTkLabel(frame3, text="✅ RÉSULTAT FINAL", font=ctk.CTkFont(size=13, weight="bold")).pack(pady=8)
        self.canvas_result = ctk.CTkLabel(frame3, text="En attente", fg_color="#2c2c2c", width=350, height=350, corner_radius=8)
        self.canvas_result.pack(padx=8, pady=8)

    # ================= PIPELINE =================

    def load_image(self):
        """Étape 1 : Chargement image"""
        path = filedialog.askopenfilename(
            title="Sélectionner une IRM",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("Tous", "*.*")]
        )

        if not path:
            return

        try:
            img = cv2.imread(path)
            if img is None:
                messagebox.showerror("Erreur", "Impossible de charger l'image")
                return

            # Image originale en gris, pleine résolution : sert au HOG du Random Forest
            self.original_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            self.current_image = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            self.update_canvas(self.canvas_orig, self.current_image)

            self.reset_results()
            self.btn_detect.configure(state="normal")
            self.status_label.configure(text="Statut : Image chargée ✓", text_color="white")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur chargement : {e}")

    def run_detection(self):
        """Phase 1 : Détection CNN"""
        if self.current_image is None or self.detection_model is None:
            messagebox.showwarning("Attention", "Image ou modèle manquant")
            return

        try:
            # 1. Niveaux de gris
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

            # 2. Resize
            gray = cv2.resize(gray, (self.IMG_SIZE, self.IMG_SIZE))

            # 3. Normalisation [0, 1] en float32
            gray_normalized = gray.astype(np.float32) / 255.0

            # 4. (128, 128) -> (1, 128, 128, 1)
            img_input = np.expand_dims(gray_normalized, axis=0)
            img_input = np.expand_dims(img_input, axis=-1)

            print(f"✅ Shape final : {img_input.shape}")
            print(f"✅ Dtype : {img_input.dtype}")
            print(f"✅ Min/Max : {img_input.min():.3f} / {img_input.max():.3f}")
            print(f"✅ Input shape attendu par le modèle : {self.detection_model.input_shape}")

            # 5. Prédiction
            prediction = self.detection_model.predict(img_input, verbose=0)

            if isinstance(prediction, np.ndarray):
                if prediction.shape == (1, 1):
                    pred_value = float(prediction[0][0])
                elif prediction.shape == (1,):
                    pred_value = float(prediction[0])
                else:
                    pred_value = float(prediction.flatten()[0])
            else:
                pred_value = float(prediction)

            self.detection_confidence = pred_value
            self.detection_result = pred_value > self.DET_THRESHOLD

            print(f"✅ Prédiction brute : {pred_value:.4f}")
            print(f"✅ Décision : {'TUMEUR' if self.detection_result else 'SAIN'}")

            if self.detection_result:
                self.status_label.configure(
                    text="Statut : 🔴 TUMEUR DÉTECTÉE",
                    text_color="#e74c3c"
                )
                self.confidence_label.configure(
                    text=f"Confiance : {self.detection_confidence*100:.1f}%",
                    text_color="#e74c3c"
                )

                self.btn_segment.configure(state="normal")
                self.canvas_seg.configure(text="Cliquez sur 'Segmentation'")

                messagebox.showinfo(
                    "Tumeur Détectée",
                    f"Tumeur détectée avec {self.detection_confidence*100:.1f}% de confiance.\n\n"
                    "Passez à la segmentation."
                )
            else:
                self.status_label.configure(
                    text="Statut : ✅ CERVEAU SAIN",
                    text_color="#27ae60"
                )
                self.confidence_label.configure(
                    text=f"Confiance : {(1-self.detection_confidence)*100:.1f}%",
                    text_color="#27ae60"
                )

                self.btn_segment.configure(state="disabled")
                self.btn_classify.configure(state="disabled")

                self.canvas_seg.configure(text="Aucune segmentation\n(Cerveau sain)")
                self.canvas_result.configure(text="Diagnostic : SAIN\nPipeline arrêté")

                messagebox.showinfo(
                    "Cerveau Sain",
                    f"Aucune tumeur détectée.\n"
                    f"Confiance : {(1-self.detection_confidence)*100:.1f}%\n\n"
                    "Le pipeline s'arrête ici."
                )

        except Exception as e:
            messagebox.showerror("Erreur Détection", f"Erreur : {e}\n\n{str(e.__class__.__name__)}")
            print(f"❌ Erreur complète : {e}")
            import traceback
            traceback.print_exc()

    def run_segmentation(self):
        """Phase 2 : Segmentation par seuillage"""
        if not self.detection_result:
            messagebox.showwarning("Attention", "Aucune tumeur détectée")
            return

        try:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Seuil fixe : adapté aux IRM très contrastées uniquement.
            # Pour d'autres bases, essayer Otsu :
            # _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, mask = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = self.keep_largest_component(mask)

            self.current_mask = mask
            surface_px = int(np.sum(mask > 0))

            if surface_px == 0:
                messagebox.showwarning("Attention", "Aucune région segmentée")
                return

            overlay = self.current_image.copy()
            overlay[mask > 0] = [0, 0, 255]
            self.update_canvas(self.canvas_seg, overlay)

            percentage = (surface_px / (self.IMG_SIZE * self.IMG_SIZE)) * 100

            self.surface_label.configure(
                text=f"Surface : {surface_px} px",
                text_color="white"
            )
            self.percent_label.configure(
                text=f"Pourcentage : {percentage:.2f}%",
                text_color="white"
            )

            self.segmentation_done = True
            self.btn_classify.configure(state="normal")
            self.canvas_result.configure(text="Cliquez sur 'Classification'")

            messagebox.showinfo(
                "Segmentation OK",
                f"Surface : {surface_px} px ({percentage:.2f}%)"
            )

        except Exception as e:
            messagebox.showerror("Erreur Segmentation", str(e))

    def run_classification(self):
        """Phase 3 : Classification (Random Forest sur HOG)"""
        if not self.segmentation_done:
            messagebox.showwarning("Attention", "Segmentation d'abord")
            return

        try:
            surface_px = int(np.sum(self.current_mask > 0))
            warning = ""

            if self.classifier_model is not None:
                X = self.extract_features()

                if X.shape[1] != self.classifier_model.n_features_in_:
                    raise ValueError(
                        f"Nombre de features : {X.shape[1]}, "
                        f"attendu : {self.classifier_model.n_features_in_}"
                    )

                proba = self.classifier_model.predict_proba(X)[0]
                idx = int(np.argmax(proba))
                key = self.CLASSES[idx]
                conf = float(proba[idx]) * 100
                tumor_type = f"{self.CLASS_FR[key]} ({conf:.0f}%)"

                if conf < self.RF_MIN_CONFIDENCE:
                    warning += f"\n⚠️ Résultat incertain (confiance < {self.RF_MIN_CONFIDENCE}%)"
                if key == "notumor":
                    warning += "\n⚠️ Contradiction : le CNN a détecté une tumeur"

                print({c: f"{p*100:.1f}%" for c, p in zip(self.CLASSES, proba)})
            else:
                # Mode de secours si le RF n'est pas chargé
                if surface_px < 500:
                    tumor_type = "Petite lésion"
                elif surface_px < 1500:
                    tumor_type = "Tumeur moyenne"
                else:
                    tumor_type = "Masse volumineuse"

            # Taille (le grade tumoral nécessite une histologie, il n'est pas estimable ici)
            if surface_px < 500:
                size_txt, urgency_color = "Petite", "#27ae60"
            elif surface_px < 1500:
                size_txt, urgency_color = "Moyenne", "#f39c12"
            else:
                size_txt, urgency_color = "Volumineuse", "#e74c3c"

            # Résultat final
            final = self.current_image.copy()
            final[self.current_mask > 0] = [255, 0, 0]

            contours, _ = cv2.findContours(self.current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(final, contours, -1, (0, 255, 255), 2)

            cv2.putText(final, tumor_type, (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(final, f"Taille : {size_txt}", (5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            self.update_canvas(self.canvas_result, final)

            # Rapport
            self.class_label.configure(text=f"Type : {tumor_type}", text_color="white")
            self.size_label.configure(text=f"Taille : {size_txt}", text_color=urgency_color)

            self.classification_done = True

            messagebox.showinfo(
                "Classification Terminée",
                f"Type : {tumor_type}\nTaille : {size_txt}{warning}\n\n"
                "Outil d'aide à la décision, ne remplace pas un avis médical."
            )

        except Exception as e:
            messagebox.showerror("Erreur Classification", str(e))

    # ================= UTILS =================

    def keep_largest_component(self, mask):
        """Garde la plus grande composante connexe"""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n <= 1:
            return np.zeros_like(mask)

        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        result = np.zeros_like(mask)
        result[labels == largest] = 255
        return result

    def extract_features(self):
        """HOG identique au notebook d'entraînement -> tableau (1, 4356)"""
        if self.original_gray is None:
            raise ValueError("Image originale absente, rechargez l'IRM")

        img = cv2.resize(self.original_gray, self.HOG_SIZE)
        feats = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )
        return feats.astype(np.float32).reshape(1, -1)

    def update_canvas(self, widget, img):
        """Affiche une image dans un widget"""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        ctk_img = ctk.CTkImage(pil_img, pil_img, size=(350, 350))
        widget.configure(image=ctk_img, text="")
        widget.image = ctk_img

    def clear_canvas(self, widget, text):
        """Vide réellement une case image (configure(image=None) ne marche pas avec CTkLabel)"""
        blank = ctk.CTkImage(
            Image.new("RGB", (350, 350), "#2c2c2c"),
            Image.new("RGB", (350, 350), "#2c2c2c"),
            size=(350, 350)
        )
        widget.configure(image=blank, text=text)
        widget.image = blank  # garder une référence

    def reset_results(self):
        """Reset des résultats intermédiaires"""
        self.detection_result = None
        self.detection_confidence = 0.0
        self.segmentation_done = False
        self.classification_done = False
        self.current_mask = None

        self.clear_canvas(self.canvas_seg, "En attente")
        self.clear_canvas(self.canvas_result, "En attente")

        self.btn_segment.configure(state="disabled")
        self.btn_classify.configure(state="disabled")

        self.confidence_label.configure(text="Confiance : --", text_color="gray")
        self.surface_label.configure(text="Surface : --", text_color="gray")
        self.percent_label.configure(text="Pourcentage : --", text_color="gray")
        self.class_label.configure(text="Type : --", text_color="gray")
        self.size_label.configure(text="Taille : --", text_color="gray")

    def reset_pipeline(self):
        """Reset complet"""
        self.current_image = None
        self.original_gray = None
        self.reset_results()

        self.clear_canvas(self.canvas_orig, "Aucune image")
        self.btn_detect.configure(state="disabled")
        self.status_label.configure(text="Statut : En attente", text_color="gray")


# ================= MAIN =================
if __name__ == "__main__":
    app = BrainTumorPipeline()
    app.mainloop()