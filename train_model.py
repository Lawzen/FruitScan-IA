import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os

class FruitClassifier:
    def __init__(self, data_dir, img_height=100, img_width=100):
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.class_names = None

    def prepare_dataset(self, batch_size=32, validation_split=0.2):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=batch_size
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=batch_size
        )

        self.class_names = train_ds.class_names
        print("Classes trouvées :", self.class_names)

        # Optimisation des performances
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        return train_ds, val_ds

    def create_model(self, num_classes):
        self.model = models.Sequential([
            layers.Rescaling(1./255, input_shape=(self.img_height, self.img_width, 3)),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

        # Compilation du modèle
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        self.model.summary()

    def train_model(self, train_ds, val_ds, epochs=10):
        print("\nDébut de l'entraînement...")
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Précision (entraînement)')
        plt.plot(history.history['val_accuracy'], label='Précision (validation)')
        plt.xlabel('Epoch')
        plt.ylabel('Précision')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Perte (entraînement)')
        plt.plot(history.history['val_loss'], label='Perte (validation)')
        plt.xlabel('Epoch')
        plt.ylabel('Perte')
        plt.legend()

        plt.show()
        return history

    def save_model(self, model_path):
        self.model.save(model_path)
        print(f"\nModèle sauvegardé dans : {model_path}")

        with open(f"{model_path}_classes.txt", 'w') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        print(f"Noms des classes sauvegardés dans : {model_path}_classes.txt")

def main():
    # Configuration
    DATA_DIR = "fruits_dataset"
    MODEL_PATH = "fruit_classifier_model.keras"

    print("Début de la préparation du modèle...")

    # Création et entraînement du modèle
    classifier = FruitClassifier(DATA_DIR)
    train_ds, val_ds = classifier.prepare_dataset()

    classifier.create_model(len(classifier.class_names))
    history = classifier.train_model(train_ds, val_ds, epochs=15)

    # Sauvegarde du modèle
    classifier.save_model(MODEL_PATH)

if __name__ == "__main__":
    main()
