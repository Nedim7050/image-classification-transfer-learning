# 📦 Guide d'Installation

## Installation Standard

```bash
pip install -r requirements.txt
```

## Problèmes avec Python 3.14?

Si vous utilisez Python 3.14 (très récent), certains packages peuvent ne pas être disponibles. Utilisez cette approche:

### Option 1: Installation par étapes

```bash
# 1. Installer les dépendances essentielles
pip install -r requirements-core.txt

# 2. Installer PyTorch (peut nécessiter une version spécifique)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Ou avec CUDA:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Installer les autres packages
pip install opencv-python pillow matplotlib seaborn streamlit numpy scikit-learn tqdm tensorboard jupyter ipywidgets

# 4. ONNX (optionnel - peut ne pas être disponible pour Python 3.14)
pip install onnx
# onnxruntime peut ne pas être disponible - le script ONNX est optionnel
```

### Option 2: Utiliser Python 3.11 ou 3.12 (Recommandé)

Pour une meilleure compatibilité, utilisez Python 3.11 ou 3.12:

```bash
# Créer un environnement avec Python 3.12
python3.12 -m venv venv
# ou
py -3.12 -m venv venv

# Activer
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Installer
pip install -r requirements.txt
```

### Option 3: Installation minimale (sans ONNX)

Si vous n'avez pas besoin de la conversion ONNX:

```bash
pip install -r requirements-core.txt
```

## Vérification de l'Installation

```bash
python -c "import torch; import torchvision; import cv2; import streamlit; print('✅ Toutes les dépendances sont installées')"
```

## Problèmes Courants

### Erreur: "No module named 'torch'"

```bash
# Réinstaller PyTorch
pip uninstall torch torchvision
pip install torch torchvision
```

### Erreur: "onnxruntime not found"

C'est normal si vous utilisez Python 3.14. Le script ONNX est optionnel. Vous pouvez:
- Utiliser Python 3.11 ou 3.12
- Ignorer la conversion ONNX (pas nécessaire pour l'entraînement ou Streamlit)

### Erreur: "opencv-python not found"

```bash
pip install opencv-python
```

## Installation avec CUDA (GPU)

Pour utiliser le GPU:

```bash
# Vérifier la version CUDA
nvidia-smi

# Installer PyTorch avec CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Vérification Finale

Après installation, testez:

```bash
python test_project.py
```

