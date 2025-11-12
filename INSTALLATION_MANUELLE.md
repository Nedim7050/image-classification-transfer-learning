# 📝 Installation Manuelle - Guide Étape par Étape

## ✅ Vous avez Python 3.11 installé - Parfait!

Suivez ces étapes dans l'ordre:

## Étape 1: Créer l'environnement virtuel avec Python 3.11

```powershell
# Dans PowerShell, dans le dossier du projet:
py -3.11 -m venv venv311
```

## Étape 2: Activer l'environnement

```powershell
.\venv311\Scripts\Activate.ps1
```

**Si vous avez une erreur d'exécution de script**, exécutez d'abord:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Puis réessayez l'activation.

## Étape 3: Vérifier que vous utilisez Python 3.11

```powershell
python --version
```

Vous devriez voir: `Python 3.11.x`

## Étape 4: Mettre à jour pip

```powershell
python -m pip install --upgrade pip
```

## Étape 5: Installer les dépendances

```powershell
pip install -r requirements.txt
```

**Si vous avez des problèmes avec numpy**, essayez:
```powershell
pip install numpy --only-binary :all:
pip install -r requirements.txt --no-deps
pip install torch torchvision opencv-python streamlit scikit-learn tqdm tensorboard jupyter ipywidgets matplotlib seaborn pillow
```

## Étape 6: Vérifier l'installation

```powershell
python test_project.py
```

## Étape 7: Test rapide d'entraînement

```powershell
python src/train.py --use_cifar10 --epochs 1 --batch_size 16
```

## 🎯 Commandes Complètes (Copier-Coller)

```powershell
# 1. Créer l'environnement
py -3.11 -m venv venv311

# 2. Activer (si erreur, voir ci-dessus)
.\venv311\Scripts\Activate.ps1

# 3. Vérifier
python --version

# 4. Mettre à jour pip
python -m pip install --upgrade pip

# 5. Installer (méthode 1 - complète)
pip install -r requirements.txt

# OU méthode 2 - si problèmes avec numpy
pip install numpy --only-binary :all:
pip install torch torchvision opencv-python streamlit scikit-learn tqdm tensorboard jupyter ipywidgets matplotlib seaborn pillow

# 6. Tester
python test_project.py
```

## ⚠️ Important

**À chaque fois que vous ouvrez un nouveau terminal**, vous devez réactiver l'environnement:

```powershell
.\venv311\Scripts\Activate.ps1
```

Vous saurez que l'environnement est activé car vous verrez `(venv311)` au début de votre ligne de commande.

## ✅ Checklist

- [ ] Environnement créé avec `py -3.11 -m venv venv311`
- [ ] Environnement activé (vous voyez `(venv311)`)
- [ ] `python --version` affiche Python 3.11.x
- [ ] Dépendances installées
- [ ] `python test_project.py` passe les tests

