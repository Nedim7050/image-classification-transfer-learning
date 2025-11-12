# 🧪 Guide de Test du Projet

Ce guide vous explique comment tester votre projet avant de le pousser sur GitHub et de le déployer sur Streamlit Cloud.

## 📋 Checklist Pré-Déploiement

### 1. Tests Automatiques

Exécutez le script de test complet:

```bash
python test_project.py
```

Ce script vérifie:
- ✅ Installation de toutes les dépendances
- ✅ Disponibilité CUDA (optionnel)
- ✅ Structure des fichiers
- ✅ Chargement du dataset CIFAR-10
- ✅ Création du modèle
- ✅ Étape d'entraînement
- ✅ Inférence
- ✅ Application Streamlit
- ✅ Script ONNX

### 2. Tests Manuels

#### A. Test du Chargement des Données

```bash
python -c "from src.dataset import load_cifar10_dataset; train, val, classes = load_cifar10_dataset('./data', 224, 16, False, 0); print(f'✅ Dataset chargé: {len(classes)} classes')"
```

#### B. Test d'Entraînement Rapide (1 epoch)

```bash
python src/train.py --use_cifar10 --epochs 1 --batch_size 16 --save_dir ./models
```

Vérifiez que:
- Le modèle s'entraîne sans erreur
- Un checkpoint est créé dans `./models/`
- Le fichier `best_model.pth` est créé

#### C. Test d'Inférence

Après l'entraînement, testez l'inférence:

```python
from src.infer import predict
import numpy as np
from PIL import Image

# Créer une image de test
test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
Image.fromarray(test_image).save('test_image.jpg')

# Prédire
results = predict('test_image.jpg', 'models/best_model.pth', topk=3)
print(results)
```

#### D. Test de l'Application Streamlit

```bash
streamlit run app/app.py
```

Vérifiez que:
- L'application se lance sans erreur
- Vous pouvez charger un modèle
- Vous pouvez uploader une image
- Les prédictions s'affichent correctement

**Note**: Si vous n'avez pas encore de modèle entraîné, l'app affichera un message d'erreur - c'est normal.

#### E. Test des Notebooks

```bash
jupyter notebook notebooks/
```

Testez chaque notebook:
1. `01_EDA_images.ipynb` - Doit charger et visualiser les données
2. `02_training_transfer_learning.ipynb` - Doit entraîner un modèle

### 3. Vérifications de Code

#### A. Vérifier .gitignore

Assurez-vous que `.gitignore` exclut:
- `*.pth`, `*.pt` (modèles)
- `data/` (datasets)
- `logs/` (logs TensorBoard)
- `__pycache__/`
- `venv/`, `env/`

```bash
# Vérifier ce qui sera commité
git status
```

#### B. Vérifier les Chemins Relatifs

Tous les chemins dans le code doivent être relatifs:
- ✅ `'./data'` ou `'../data'`
- ❌ `'C:/Users/...'` ou chemins absolus

#### C. Vérifier les Imports

Tous les imports doivent fonctionner:

```bash
python -c "from src.dataset import *; from src.train import *; from src.infer import *; from src.utils import *; print('✅ Tous les imports fonctionnent')"
```

### 4. Test de Conversion ONNX (Optionnel)

```bash
# D'abord, entraînez un modèle
python src/train.py --use_cifar10 --epochs 1

# Ensuite, convertissez-le
python scripts/convert_to_onnx.py --model_path models/best_model.pth --output_path models/model.onnx --test
```

### 5. Préparation pour GitHub

#### A. Créer un Fichier .streamlit/config.toml (Optionnel)

Pour Streamlit Cloud, créez:

```bash
mkdir -p .streamlit
```

Créez `.streamlit/config.toml`:

```toml
[server]
headless = true
port = 8501

[browser]
gatherUsageStats = false
```

#### B. Vérifier requirements.txt

Assurez-vous que toutes les dépendances sont listées:

```bash
pip freeze > requirements_check.txt
# Comparez avec requirements.txt
```

#### C. Créer un Fichier .streamlit/secrets.toml (Si nécessaire)

Si votre app nécessite des secrets (API keys, etc.), créez `.streamlit/secrets.toml.example`:

```toml
# Exemple de fichier secrets
# Copiez ce fichier en secrets.toml et remplissez les valeurs
API_KEY = "your_api_key_here"
```

### 6. Test Local Complet

Simulez un environnement propre:

```bash
# Créer un nouvel environnement virtuel
python -m venv test_env
test_env\Scripts\activate  # Windows
# source test_env/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt

# Exécuter les tests
python test_project.py

# Tester l'entraînement
python src/train.py --use_cifar10 --epochs 1

# Tester Streamlit
streamlit run app/app.py
```

## 🚀 Préparation pour Streamlit Cloud

### 1. Structure Requise

Votre projet doit avoir:
```
image-classification-transfer-learning/
├── app/
│   └── app.py          # Point d'entrée Streamlit
├── src/                # Code source
├── requirements.txt    # Dépendances
├── README.md          # Documentation
└── .streamlit/        # Config (optionnel)
    └── config.toml
```

### 2. Vérifications Spécifiques Streamlit

#### A. Chemins dans app.py

Vérifiez que tous les chemins sont relatifs:

```python
# ✅ Bon
model_path = 'models/best_model.pth'
data_dir = '../data'

# ❌ Mauvais
model_path = 'C:/Users/.../models/best_model.pth'
```

#### B. Gestion des Fichiers Manquants

L'app doit gérer gracieusement l'absence de modèle:

```python
if not os.path.exists(model_path):
    st.warning("Modèle non trouvé. Entraînez d'abord un modèle.")
```

#### C. Taille des Fichiers

Streamlit Cloud a des limites:
- Fichiers > 1GB peuvent poser problème
- Utilisez Git LFS pour les gros modèles ou hébergez-les ailleurs

### 3. Test de Déploiement Local

Testez comme si c'était Streamlit Cloud:

```bash
# Dans un dossier propre
git clone <votre-repo> test_deploy
cd test_deploy
pip install -r requirements.txt
streamlit run app/app.py
```

## 📝 Checklist Finale

Avant de pousser sur GitHub:

- [ ] `python test_project.py` passe tous les tests
- [ ] Entraînement rapide (1 epoch) fonctionne
- [ ] Inférence fonctionne
- [ ] Streamlit app se lance localement
- [ ] `.gitignore` exclut les fichiers volumineux
- [ ] `requirements.txt` est à jour
- [ ] `README.md` est complet
- [ ] Pas de chemins absolus dans le code
- [ ] Pas de secrets hardcodés
- [ ] Tous les imports fonctionnent

Avant de déployer sur Streamlit:

- [ ] Repo GitHub est à jour
- [ ] `app/app.py` est le point d'entrée
- [ ] `requirements.txt` est correct
- [ ] Pas de fichiers > 1GB à commit
- [ ] Documentation README est claire

## 🐛 Résolution de Problèmes Courants

### Erreur: "Module not found"

```bash
# Vérifiez que vous êtes dans le bon environnement
pip install -r requirements.txt
```

### Erreur: "CUDA out of memory"

Réduisez la taille du batch:
```bash
python src/train.py --batch_size 8  # Au lieu de 32
```

### Erreur Streamlit: "File not found"

Vérifiez les chemins relatifs dans `app/app.py`. Utilisez:
```python
import os
BASE_DIR = Path(__file__).parent.parent
model_path = BASE_DIR / 'models' / 'best_model.pth'
```

### Erreur: "Dataset not found"

Pour CIFAR-10, il sera téléchargé automatiquement. Pour un dataset local, vérifiez la structure:
```
data/
├── train/
│   └── class1/
└── val/
    └── class1/
```

## ✅ Commandes Rapides de Test

```bash
# Test complet
python test_project.py

# Test entraînement rapide
python src/train.py --use_cifar10 --epochs 1 --batch_size 16

# Test Streamlit
streamlit run app/app.py

# Test inférence
python -c "from src.infer import predict; print(predict('test.jpg', 'models/best_model.pth'))"

# Vérifier structure
python -c "from pathlib import Path; files = ['src/train.py', 'app/app.py', 'requirements.txt']; print('✅' if all(Path(f).exists() for f in files) else '❌')"
```

## 🎯 Prochaines Étapes

Une fois tous les tests passés:

1. **Commit sur GitHub**:
   ```bash
   git add .
   git commit -m "Initial commit: Image classification project"
   git push origin main
   ```

2. **Déployer sur Streamlit Cloud**:
   - Allez sur [streamlit.io](https://streamlit.io)
   - Connectez votre repo GitHub
   - Sélectionnez `app/app.py` comme point d'entrée
   - Déployez!

3. **Partager votre modèle**:
   - Option 1: Utilisez Git LFS pour les modèles
   - Option 2: Hébergez les modèles sur Google Drive/Dropbox
   - Option 3: Utilisez Hugging Face Hub

---

**Bon test! 🚀**

