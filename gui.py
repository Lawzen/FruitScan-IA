import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import sv_ttk
import customtkinter as ctk
from datetime import datetime
import json
import os

class ModernFruitClassifierGUI:
    def __init__(self, model_path):
        self.window = tk.Tk()
        self.window.title("🍎 FruitScan IA")
        self.window.geometry("700x700")
        self.window.configure(bg='#f0f0f0')
        sv_ttk.set_theme("light")

        try:
            self.model = tf.keras.models.load_model(model_path)
            self.load_classes(model_path)
            self.initialize_history()
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement du modèle : {str(e)}")
            self.window.destroy()
            return

        self.current_image = None
        self.photo = None

        self.main_container = ttk.Frame(self.window, padding=5)
        self.main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)

        self.create_widgets()
        self.update_history_display()

    def initialize_history(self):
        """
        Initialise l'historique depuis le json s'il existe déjà.
        """
        self.history = []
        self.history_file = "classification_history.json"
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
            except Exception as e:
                print(f"Impossible de lire le fichier d'historique : {e}")
                self.history = []
        else:
            self.history = []

    def save_to_history(self, result_dict):
        self.history.append(result_dict)
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history[-50:], f, indent=2)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'historique : {e}")

    def update_history_display(self):
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        for item in self.history[-10:]:
            self.history_tree.insert('', 'end', values=(item['date'], item['prediction'], item['confidence']))

    def load_classes(self, model_path):
        self.class_names = []
        with open(f"{model_path}_classes.txt", 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

    def create_widgets(self):
        self.main_container.columnconfigure(0, weight=1)
        self.main_container.columnconfigure(1, weight=1)

        self.create_header()
        self.create_image_section()
        self.create_controls()
        self.create_results_section()
        self.create_history_section()
        self.create_footer()

    def create_header(self):
        header_frame = ttk.Frame(self.main_container)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Label(
            header_frame,
            text="FruitScan IA",
            font=('Helvetica', 20, 'bold'),
            foreground='#2E7D32'
        ).pack(pady=5)

        ttk.Label(
            header_frame,
            text="Utilise l'IA pour identifier vos fruits !",
            font=('Helvetica', 12),
            foreground='#666666'
        ).pack()

    def create_image_section(self):
        self.image_frame = ttk.LabelFrame(self.main_container, text="Image", padding=5)
        self.image_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(expand=True, fill='both')
        self.show_placeholder()

    def show_placeholder(self):
        """
        Affiche par défaut un placeholder en attendant l'image.
        """
        placeholder = Image.new('RGB', (300, 300), '#f0f0f0')
        self.photo = ImageTk.PhotoImage(placeholder)
        self.image_label.configure(image=self.photo)

    def create_controls(self):
        control_frame = ttk.Frame(self.main_container)
        control_frame.grid(row=2, column=0, sticky="ew", pady=10)

        ttk.Style().configure("Action.TButton", padding=6)

        self.load_button = ttk.Button(
            control_frame,
            text="📁 Charger une image",
            command=self.load_image,
            style="Action.TButton"
        )
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.classify_button = ttk.Button(
            control_frame,
            text="🔍 Classifier",
            command=self.classify_image,
            state='disabled',
            style="Action.TButton"
        )
        self.classify_button.pack(side=tk.LEFT, padx=5)

    def create_results_section(self):
        self.result_frame = ttk.LabelFrame(self.main_container, text="Résultats", padding=5)
        self.result_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        self.result_label = ttk.Label(
            self.result_frame,
            text="En attente d'une image...",
            font=('Helvetica', 12),
            wraplength=280
        )
        self.result_label.pack(pady=10)

        self.progress = ttk.Progressbar(
            self.result_frame,
            orient="horizontal",
            length=200,
            mode="determinate"
        )
        self.progress.pack(pady=5)

    def create_history_section(self):
        history_frame = ttk.LabelFrame(self.main_container, text="Historique", padding=5)
        history_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)

        columns = ('date', 'prediction', 'confidence')
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show='headings', height=5)

        for col in columns:
            self.history_tree.heading(col, text=col.capitalize())
            self.history_tree.column(col, width=130)

        self.history_tree.pack(fill='x', pady=5)

    def create_footer(self):
        footer_frame = ttk.Frame(self.main_container)
        footer_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)

        ttk.Label(
            footer_frame,
            text="© 2024 Lawzen - Tous droits réservés",
            font=('Helvetica', 8),
            foreground='#666666'
        ).pack()

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Sélectionne une image de fruit",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        if file_path:
            try:
                image = Image.open(file_path)
                image.thumbnail((300, 300), Image.Resampling.LANCZOS)
                background = Image.new('RGB', (300, 300), 'white')
                offset = ((300 - image.size[0]) // 2, (300 - image.size[1]) // 2)
                background.paste(image, offset)

                self.current_image = image
                self.photo = ImageTk.PhotoImage(background)
                self.image_label.configure(image=self.photo)
                self.classify_button.configure(state='normal')
                self.result_label.configure(
                    text="Image chargée. Cliquez sur 'Classifier' pour l'analyser."
                )
                self.progress['value'] = 0
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors du chargement de l'image : {str(e)}")

    def classify_image(self):
        if self.current_image is None:
            return
        try:
            self.progress['value'] = 0
            self.window.update_idletasks()

            img_array = tf.keras.preprocessing.image.img_to_array(
                self.current_image.resize((100, 100))
            )
            img_array = tf.expand_dims(img_array, 0)

            self.progress['value'] = 30
            self.window.update_idletasks()

            predictions = self.model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            predicted_class = self.class_names[np.argmax(score)]
            confidence = 100 * np.max(score)

            self.progress['value'] = 60
            self.window.update_idletasks()

            result_text = (
                f"🎯 Résultat de la classification :\n\n"
                f"🍎 Fruit détecté : {predicted_class}\n"
                f"📊 Niveau de confiance : {confidence:.2f}%"
            )
            self.result_label.configure(text=result_text)

            result_dict = {
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'prediction': predicted_class,
                'confidence': f"{confidence:.2f}%"
            }
            self.save_to_history(result_dict)
            self.update_history_display()

            self.progress['value'] = 100
            self.window.update_idletasks()
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la classification : {str(e)}")
            self.progress['value'] = 0

    def run(self):
        """
        Lance l'app.
        """
        self.window.mainloop()

def main():
    MODEL_PATH = "fruit_classifier_model.keras"
    app = ModernFruitClassifierGUI(MODEL_PATH)
    app.run()

if __name__ == "__main__":
    main()
