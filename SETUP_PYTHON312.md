# 🐍 Installation et Configuration Python 3.12

## Étape 1: Vérifier les Versions Python Disponibles

```bash
py -0
```

Cette commande liste toutes les versions Python installées sur votre système.

## Étape 2: Si Python 3.12 n'est pas installé

### Option A: Télécharger depuis python.org (Recommandé)

1. Allez sur [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Téléchargez Python 3.12.x (dernière version stable)
3. **Important**: Cochez "Add Python to PATH" lors de l'installation
4. Installez

### Option B: Utiliser le Windows Store

1. Ouvrez le Microsoft Store
2. Recherchez "Python 3.12"
3. Installez

## Étape 3: Vérifier l'Installation

```bash
py -3.12 --version
```

Vous devriez voir: `Python 3.12.x`

## Étape 4: Créer un Environnement Virtuel avec Python 3.12

```bash
# Dans le dossier du projet
cd C:\Users\najdm\image-classification-transfer-learning

# Créer l'environnement avec Python 3.12
py -3.12 -m venv venv312

# Activer l'environnement
venv312\Scripts\activate
```

## Étape 5: Mettre à jour pip

```bash
python -m pip install --upgrade pip
```

## Étape 6: Installer les Dépendances

```bash
pip install -r requirements.txt
```

## Étape 7: Vérifier l'Installation

```bash
python test_project.py
```

## Étape 8: Test Rapide

```bash
# Test d'entraînement (1 epoch)
python src/train.py --use_cifar10 --epochs 1 --batch_size 16

# Test Streamlit
streamlit run app/app.py
```

## 🔧 Si vous avez des Problèmes

### Problème: "py -3.12" ne fonctionne pas

Essayez:
```bash
python3.12 -m venv venv312
# ou
C:\Python312\python.exe -m venv venv312
```

### Problème: L'environnement ne s'active pas

Sur Windows, utilisez:
```bash
venv312\Scripts\activate.bat
# ou PowerShell:
venv312\Scripts\Activate.ps1
```

Si vous avez une erreur d'exécution de script dans PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## ✅ Checklist

- [ ] Python 3.12 installé
- [ ] Environnement virtuel créé
- [ ] Environnement activé
- [ ] Dépendances installées
- [ ] Tests passent (`python test_project.py`)

## 🎯 Prochaines Étapes

Une fois tout installé:

1. **Entraîner un modèle**:
   ```bash
   python src/train.py --use_cifar10 --epochs 5
   ```

2. **Explorer les notebooks**:
   ```bash
   jupyter notebook notebooks/
   ```

3. **Lancer l'app Streamlit**:
   ```bash
   streamlit run app/app.py
   ```

4. **Pousser sur GitHub** (quand tout fonctionne)

