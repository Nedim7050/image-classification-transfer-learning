# 🖼️ Image Classification avec Transfer Learning

Projet complet de classification d'images utilisant le transfer learning avec PyTorch et OpenCV. Ce projet inclut des outils pour l'exploration de données, l'entraînement de modèles, l'inférence et une interface web interactive avec Streamlit.

## 📋 Table des Matières

- [Fonctionnalités](#fonctionnalités)
- [Structure du Projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Datasets](#datasets)
- [Entraînement](#entraînement)
- [Inférence](#inférence)
- [Application Streamlit](#application-streamlit)
- [Conversion ONNX](#conversion-onnx)
- [Fine-tuning Avancé](#fine-tuning-avancé)
- [Mixed Precision Training](#mixed-precision-training)
- [Dépannage](#dépannage)

## ✨ Fonctionnalités

- 🔬 **Exploration de données** : Notebooks Jupyter pour analyser et visualiser les datasets
- 🚀 **Transfer Learning** : Support pour ResNet50 et EfficientNet avec poids pré-entraînés
- 📊 **Entraînement complet** : Script CLI avec checkpointing, early stopping, et logging TensorBoard
- 🎯 **Inférence** : Prédictions top-k avec probabilités
- 🌐 **Interface Web** : Application Streamlit interactive pour tester les modèles
- 📦 **Export ONNX** : Conversion pour déploiement léger
- 🔧 **OpenCV Support** : Alternative de preprocessing avec OpenCV
- ⚡ **Mixed Precision** : Support AMP pour accélérer l'entraînement

## 📁 Structure du Projet

```
image-classification-transfer-learning/
├── data/                      # Dossier pour les datasets (optionnel)
│   ├── train/                 # Images d'entraînement par classe
│   │   ├── class1/
│   │   └── class2/
│   └── val/                   # Images de validation par classe
│       ├── class1/
│       └── class2/
├── notebooks/
│   ├── 01_EDA_images.ipynb    # Exploration et analyse des données
│   └── 02_training_transfer_learning.ipynb  # Entraînement et visualisation
├── src/
│   ├── __init__.py
│   ├── dataset.py             # Chargement de datasets (CIFAR-10, ImageFolder)
│   ├── train.py               # Script d'entraînement CLI
│   ├── infer.py               # Fonctions d'inférence
│   └── utils.py               # Utilitaires (métriques, visualisation)
├── app/
│   └── app.py                 # Application Streamlit
├── scripts/
│   └── convert_to_onnx.py     # Conversion PyTorch → ONNX
├── models/                     # Modèles sauvegardés (créé automatiquement)
├── logs/                       # Logs TensorBoard (créé automatiquement)
├── requirements.txt
├── README.md
└── LICENSE
```

## ⚠️ Compatibilité Python

**Important**: Ce projet fonctionne mieux avec Python 3.11 ou 3.12. Si vous utilisez Python 3.14 (alpha), vous pouvez rencontrer des problèmes de compatibilité. Voir [PYTHON314_FIX.md](PYTHON314_FIX.md) pour les solutions.

## 🧪 Tests Rapides

Avant de commencer, testez que tout fonctionne:

```bash
# Test complet automatique
python test_project.py

# Test d'entraînement rapide (1 epoch)
python src/train.py --use_cifar10 --epochs 1 --batch_size 16

# Test de l'application Streamlit
streamlit run app/app.py
```

📖 **Guide complet de test**: Voir [TESTING_GUIDE.md](TESTING_GUIDE.md)

## 🚀 Installation

### Prérequis

- Python 3.8+
- pip ou conda
- (Optionnel) CUDA pour l'entraînement GPU

### Installation des Dépendances

1. **Créer un environnement virtuel** (recommandé):

```bash
# Avec venv
python -m venv venv

# Activer l'environnement
# Sur Windows:
venv\Scripts\activate
# Sur Linux/Mac:
source venv/bin/activate
```

2. **Installer les dépendances**:

```bash
pip install -r requirements.txt
```

### Installation de PyTorch avec CUDA (optionnel)

Pour utiliser le GPU, installez PyTorch avec support CUDA depuis [pytorch.org](https://pytorch.org/get-started/locally/):

```bash
# Exemple pour CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📖 Utilisation

### 1. Exploration des Données

Lancez le notebook d'exploration pour analyser votre dataset:

```bash
jupyter notebook notebooks/01_EDA_images.ipynb
```

Ce notebook permet de:
- Visualiser la distribution des classes
- Afficher des exemples d'images par classe
- Analyser les statistiques des images

### 2. Préparation des Données

#### Option A: Utiliser CIFAR-10 (Démo rapide)

Le dataset CIFAR-10 sera téléchargé automatiquement lors de l'exécution.

#### Option B: Utiliser un Dataset Local

Organisez vos images dans la structure suivante:

```
data/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       ├── img1.jpg
│       └── img2.jpg
└── val/
    ├── class1/
    └── class2/
```

Si vous n'avez qu'un dossier `train/`, le script divisera automatiquement les données en train/val.

## 🎯 Entraînement

### Méthode 1: Script CLI

```bash
# Entraînement basique avec CIFAR-10
python src/train.py --use_cifar10 --epochs 10 --batch_size 32 --lr 0.001

# Entraînement avec dataset local
python src/train.py --data_dir ./data --epochs 20 --batch_size 32 --lr 0.001

# Entraînement avec GPU et Mixed Precision
python src/train.py --use_cifar10 --epochs 10 --device cuda --use_amp

# Entraînement avec EfficientNet
python src/train.py --use_cifar10 --model efficientnet --epochs 10

# Entraînement avec early stopping
python src/train.py --use_cifar10 --epochs 50 --early_stopping_patience 5
```

#### Arguments Principaux

- `--data_dir`: Chemin vers le dossier de données (défaut: `./data`)
- `--use_cifar10`: Utiliser CIFAR-10 au lieu d'ImageFolder
- `--epochs`: Nombre d'époques (défaut: 10)
- `--batch_size`: Taille du batch (défaut: 32)
- `--lr`: Learning rate (défaut: 0.001)
- `--model`: Architecture (`resnet50` ou `efficientnet`, défaut: `resnet50`)
- `--pretrained`: Utiliser des poids pré-entraînés (défaut: True)
- `--freeze_backbone`: Geler le backbone initialement
- `--unfreeze_last_n`: Débloquer les N dernières couches
- `--device`: Device (`cuda` ou `cpu`, auto-détecté par défaut)
- `--use_amp`: Utiliser Mixed Precision Training
- `--img_size`: Taille d'image (défaut: 224)
- `--save_dir`: Dossier pour sauvegarder les checkpoints (défaut: `./models`)
- `--log_dir`: Dossier pour les logs TensorBoard (défaut: `./logs`)
- `--early_stopping_patience`: Patience pour early stopping (None pour désactiver)
- `--resume`: Reprendre depuis un checkpoint

### Méthode 2: Notebook Jupyter

```bash
jupyter notebook notebooks/02_training_transfer_learning.ipynb
```

Le notebook offre une interface interactive pour:
- Configurer les hyperparamètres
- Visualiser les courbes d'entraînement en temps réel
- Afficher la matrice de confusion
- Tester des prédictions sur des images

### Visualisation avec TensorBoard

```bash
tensorboard --logdir logs
```

Ouvrez votre navigateur sur `http://localhost:6006` pour visualiser:
- Loss et accuracy (train/val)
- Learning rate
- Graphiques en temps réel

## 🔮 Inférence

### Utilisation en Python

```python
from src.infer import predict

# Prédiction sur une image
results = predict(
    image_path='path/to/image.jpg',
    model_path='models/best_model.pth',
    topk=3,
    device='cpu'
)

# Afficher les résultats
for class_name, prob in results:
    print(f"{class_name}: {prob*100:.2f}%")
```

### Utilisation en Ligne de Commande

```bash
python -c "from src.infer import predict; print(predict('image.jpg', 'models/best_model.pth'))"
```

## 🌐 Application Streamlit

Lancez l'application web interactive:

```bash
streamlit run app/app.py
```

L'application permet de:
- 📤 Uploader des images
- 🎯 Faire des prédictions avec visualisation top-k
- 📊 Afficher les probabilités avec graphiques
- 📷 Utiliser des images d'exemple (CIFAR-10)

### Fonctionnalités de l'App

- **Upload d'images**: Support PNG, JPG, JPEG, BMP, GIF
- **Images d'exemple**: Sélection d'exemples CIFAR-10
- **Visualisation**: Graphiques en barres pour les probabilités
- **Configuration**: Choix du modèle, device, top-k, taille d'image

## 📦 Conversion ONNX

Convertir le modèle PyTorch en format ONNX pour un déploiement léger:

```bash
python scripts/convert_to_onnx.py --model_path models/best_model.pth --output_path models/model.onnx
```

### Utilisation du Modèle ONNX

```python
import onnxruntime as ort
import numpy as np

# Charger le modèle ONNX
session = ort.InferenceSession('models/model.onnx')

# Préparer l'input (image préprocessée)
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Prédiction
outputs = session.run(['output'], {'input': input_data})
predictions = outputs[0]
```

### Avantages ONNX

- ✅ Format standardisé pour le déploiement
- ✅ Plus léger que PyTorch
- ✅ Compatible avec ONNX Runtime (CPU/GPU)
- ✅ Support multi-plateforme

## 🔧 Fine-tuning Avancé

### Débloquer Progressivement les Couches

Pour un fine-tuning plus fin, vous pouvez débloquer progressivement les couches:

```bash
# Étape 1: Entraîner avec backbone gelé
python src/train.py --use_cifar10 --freeze_backbone --epochs 5

# Étape 2: Débloquer les 2 dernières couches
python src/train.py --use_cifar10 --freeze_backbone --unfreeze_last_n 2 --resume models/best_model.pth --epochs 5

# Étape 3: Débloquer tout et fine-tune avec learning rate plus faible
python src/train.py --use_cifar10 --lr 0.0001 --resume models/best_model.pth --epochs 10
```

### Stratégie de Fine-tuning Recommandée

1. **Phase 1**: Geler le backbone, entraîner seulement le classifieur (5-10 epochs)
2. **Phase 2**: Débloquer les dernières couches (2-3 layers), learning rate réduit (5-10 epochs)
3. **Phase 3**: Fine-tuning complet avec learning rate très faible (10-20 epochs)

## ⚡ Mixed Precision Training

Le Mixed Precision Training (AMP) peut accélérer l'entraînement de 1.5x à 2x sur GPU:

```bash
python src/train.py --use_cifar10 --use_amp --device cuda
```

**Avantages:**
- 🚀 Entraînement plus rapide
- 💾 Moins de mémoire GPU utilisée
- ✅ Précision similaire à FP32

**Note**: Nécessite un GPU compatible (compute capability >= 7.0 pour Tensor Cores).

## 📊 Datasets Recommandés

### Datasets Populaires pour la Classification d'Images

1. **CIFAR-10/CIFAR-100**: 10/100 classes, images 32x32
   - Téléchargement automatique via le code

2. **PlantVillage**: Classification de maladies de plantes
   - [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

3. **GTSRB (German Traffic Sign Recognition)**: Panneaux de signalisation
   - [Site officiel](https://benchmark.ini.rub.de/gtsrb_dataset.html)

4. **Food-101**: 101 catégories de nourriture
   - [Kaggle](https://www.kaggle.com/datasets/kmader/food41)

5. **Stanford Dogs**: 120 races de chiens
   - [Site officiel](http://vision.stanford.edu/aditya86/ImageNetDogs/)

### Préparation d'un Dataset Personnalisé

1. Organisez vos images dans `data/train/<class>/` et `data/val/<class>/`
2. Assurez-vous que les images sont dans un format supporté (JPG, PNG, etc.)
3. Utilisez `--data_dir ./data` lors de l'entraînement

## 💾 Tailles de Checkpoints

Les checkpoints PyTorch peuvent être volumineux:

- **ResNet50**: ~100 MB
- **EfficientNet-B0**: ~20 MB
- **Modèle ONNX**: Généralement 30-50% plus petit

**Conseil**: Utilisez `git-lfs` pour versionner les modèles, ou excluez-les avec `.gitignore`.

## 🐛 Dépannage

### Problèmes Courants

#### 1. Erreur "CUDA out of memory"

**Solution**: Réduisez la taille du batch:
```bash
python src/train.py --batch_size 16  # ou 8
```

#### 2. Modèle non trouvé dans Streamlit

**Solution**: Vérifiez que le chemin est correct:
```bash
# Le chemin doit être relatif au dossier racine
streamlit run app/app.py
# Puis utilisez: models/best_model.pth
```

#### 3. Erreur d'import dans les notebooks

**Solution**: Assurez-vous d'exécuter les notebooks depuis le dossier `notebooks/`:
```python
# La première cellule ajoute le parent au path
sys.path.append('..')
```

#### 4. OpenCV ne charge pas les images

**Solution**: Vérifiez que les chemins sont corrects et que les images existent. Utilisez `use_opencv=False` pour utiliser PIL à la place.

### Support GPU

Vérifiez que CUDA est disponible:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## 📝 Exemples d'Utilisation Complète

### Exemple 1: Entraînement Rapide sur CIFAR-10

```bash
# 1. Entraîner
python src/train.py --use_cifar10 --epochs 5 --batch_size 64

# 2. Tester dans Streamlit
streamlit run app/app.py

# 3. Convertir en ONNX
python scripts/convert_to_onnx.py --model_path models/best_model.pth
```

### Exemple 2: Dataset Personnalisé

```bash
# 1. Organiser les données dans data/train/ et data/val/
# 2. Entraîner
python src/train.py --data_dir ./data --epochs 20 --batch_size 32 --lr 0.001

# 3. Utiliser le modèle
python -c "from src.infer import predict; print(predict('test.jpg', 'models/best_model.pth'))"
```

## 🤝 Contribution

Les contributions sont les bienvenues! N'hésitez pas à:
- Ouvrir des issues pour signaler des bugs
- Proposer de nouvelles fonctionnalités
- Soumettre des pull requests

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🙏 Remerciements

- PyTorch et torchvision pour les modèles pré-entraînés
- Streamlit pour l'interface web
- La communauté open source pour les outils et datasets

## 🧪 Tests et Validation

Avant de déployer, exécutez les tests:

```bash
# Test automatique complet
python test_project.py

# Tests manuels
python src/train.py --use_cifar10 --epochs 1  # Test entraînement
streamlit run app/app.py  # Test application
```

📖 **Guide détaillé**: [TESTING_GUIDE.md](TESTING_GUIDE.md)

## 🚀 Déploiement sur Streamlit Cloud

1. **Pousser sur GitHub**:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Déployer sur Streamlit Cloud**:
   - Allez sur [share.streamlit.io](https://share.streamlit.io)
   - Connectez votre compte GitHub
   - Sélectionnez votre repo
   - Point d'entrée: `app/app.py`
   - Cliquez sur "Deploy"

3. **Configuration**:
   - Le fichier `.streamlit/config.toml` est déjà configuré
   - Les dépendances sont dans `requirements.txt`
   - Les modèles doivent être dans le repo ou hébergés ailleurs

## 📚 Ressources Additionnelles

- [Documentation PyTorch](https://pytorch.org/docs/)
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [ONNX Documentation](https://onnx.ai/)
- [Streamlit Cloud](https://docs.streamlit.io/streamlit-community-cloud)

---

**Bon entraînement! 🚀**

