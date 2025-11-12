@echo off
REM Script de configuration de l'environnement Python 3.11 (Windows CMD)
REM Exécutez: setup_env.bat

echo ========================================
echo Configuration de l'environnement Python 3.11
echo ========================================
echo.

echo [1/5] Verification de Python 3.11...
py -3.11 --version
if errorlevel 1 (
    echo ERREUR: Python 3.11 non trouve. Installez-le depuis python.org
    pause
    exit /b 1
)
echo OK: Python 3.11 trouve
echo.

echo [2/5] Creation de l'environnement virtuel...
if exist venv311 (
    echo Suppression de l'ancien environnement...
    rmdir /s /q venv311
)
py -3.11 -m venv venv311
if errorlevel 1 (
    echo ERREUR: Impossible de creer l'environnement
    pause
    exit /b 1
)
echo OK: Environnement virtuel cree
echo.

echo [3/5] Activation de l'environnement...
call venv311\Scripts\activate.bat
python --version
echo.

echo [4/5] Mise a jour de pip...
python -m pip install --upgrade pip --quiet
echo.

echo [5/5] Installation des dependances (cela peut prendre quelques minutes)...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERREUR: Probleme lors de l'installation
    echo Essayez: pip install -r requirements-core.txt
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation terminee avec succes!
echo ========================================
echo.
echo Prochaines etapes:
echo 1. Activez l'environnement: venv311\Scripts\activate.bat
echo 2. Testez: python test_project.py
echo 3. Entrainez: python src\train.py --use_cifar10 --epochs 1
echo.
pause

