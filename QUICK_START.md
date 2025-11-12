# 🚀 Démarrage Rapide

## Installation Rapide (Python 3.11 ou 3.12)

```bash
# 1. Créer un environnement virtuel
python -m venv venv
# ou avec Python 3.12 spécifiquement:
py -3.12 -m venv venv

# 2. Activer l'environnement
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Tester l'installation
python test_project.py
```

## Si vous avez des problèmes avec Python 3.14

Voir [PYTHON314_FIX.md](PYTHON314_FIX.md) pour les solutions.

## Installation Minimale (sans ONNX)

Si vous avez des problèmes avec certains packages:

```bash
pip install -r requirements-core.txt
```

## Test Rapide

```bash
# Test d'entraînement (1 epoch, ~2-3 minutes)
python src/train.py --use_cifar10 --epochs 1 --batch_size 16

# Test Streamlit
streamlit run app/app.py
```

## Prochaines Étapes

1. ✅ Installation réussie
2. 📖 Lire le [README.md](README.md) pour les instructions complètes
3. 🧪 Exécuter les notebooks dans `notebooks/`
4. 🚀 Entraîner votre modèle
5. 🌐 Déployer sur Streamlit Cloud

