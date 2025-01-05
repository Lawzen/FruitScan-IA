# FruitScan IA 🍎🍊🍓

## Description 📄
Ce projet a été réalisé au cours de ma dernière année à la HELHa pour le cours d'IA. Cette intelligence artificielle est un classificateur d'images de fruits utilisant TensorFlow et Keras. Il comprend un modèle de classification d'images capable de reconnaître différentes catégories de fruits à partir d'images. Une interface graphique moderne a également été développée pour rendre l'application conviviale et intuitive.

---

## Fonctionnalités principales ✨
- **Classification d'images de fruits**
- **Entraînement et validation de modèles personnalisés**
- **Affichage des résultats avec des graphiques de précision/perte**
- **Interface graphique Python avec Tkinter et CustomTkinter**
- **Historique des fruits format JSON**

---

## Structure du projet 📂
```
classement_fruit/
│
├── fruits_dataset/                          # Dossier contenant les images de fruits pour l'entraînement
│
├── venv/                                    # Environnement virtuel Python
│
├── classification_history.json              # Historique des classifications
│
├── fruit_classifier_model.keras             # Modèle de classification sauvegardé
│
├── fruit_classifier_model.keras_classes.txt # Noms des classes de fruits
│
├── gui.py                                   # Interface graphique principale
│
├── train_model.py                           # Script d'entraînement du modèle
│
├── requirements.txt                         # Dépendances Python
└── README.md                                # Ce fichier <3
```

---

## Installation et exécution 🚀

### 1. Cloner le dépôt :
```
git clone https://github.com/Lawzen/FruitScan-IA.git
cd FruitScan-IA
```

### 2. Utilise un interpréteur python :
```
python 3.9 (utilisé pour ma part)
```

### 3. Installer les dépendances :
```
pip install -r requirements.txt
```

---

## Entraîner le modèle 🏋️‍♂️
```
python train_model.py
```
Ce script entraîne le modèle en utilisant les images dans le dossier `fruits_dataset`. Les résultats sont affichés sous forme de graphiques.

---

## Lancer l'interface graphique 🖥️
```
python gui.py
```
Une interface graphique pour charger une image. l'IA se chargera de classer le fruit correspondant.

---

## Ajouter des fruits 🥭🍇
1. Ajoute de nouvelles images de fruits dans le dossier `fruits_dataset` en créant un dossier correspondant au nom du fruit.
2. Relance l'entraînement pour que le modèle prenne en compte les nouvelles classes.

---

## Dépendances 🧰
- **TensorFlow**
- **Keras**
- **Tkinter**
- **CustomTkinter**
- **sv_ttk**
- **Matplotlib**
- **Pillow**

Installe toutes tes dépendances avec :
```
pip install -r requirements.txt
```

---

## Auteur 👨‍💻
Développé par **Lawzen** - 2024

---

## Licence 📜
Ce projet est sous licence MIT. Vous pouvez librement l'utiliser et le modifier selon vos besoins.

