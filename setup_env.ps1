# Script de configuration de l'environnement Python 3.11
# Exécutez: .\setup_env.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Configuration de l'environnement Python 3.11" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Vérifier Python 3.11
Write-Host "`n[1/5] Vérification de Python 3.11..." -ForegroundColor Yellow
$python311 = py -3.11 --version
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Python 3.11 trouvé: $python311" -ForegroundColor Green
} else {
    Write-Host "✗ Python 3.11 non trouvé. Installez-le depuis python.org" -ForegroundColor Red
    exit 1
}

# Créer l'environnement virtuel
Write-Host "`n[2/5] Création de l'environnement virtuel..." -ForegroundColor Yellow
if (Test-Path "venv311") {
    Write-Host "⚠ Environnement venv311 existe déjà. Suppression..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force venv311
}
py -3.11 -m venv venv311
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Environnement virtuel créé" -ForegroundColor Green
} else {
    Write-Host "✗ Erreur lors de la création de l'environnement" -ForegroundColor Red
    exit 1
}

# Activer l'environnement
Write-Host "`n[3/5] Activation de l'environnement..." -ForegroundColor Yellow
& .\venv311\Scripts\Activate.ps1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Environnement activé" -ForegroundColor Green
    python --version
} else {
    Write-Host "✗ Erreur lors de l'activation" -ForegroundColor Red
    exit 1
}

# Mettre à jour pip
Write-Host "`n[4/5] Mise à jour de pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ pip mis à jour" -ForegroundColor Green
} else {
    Write-Host "✗ Erreur lors de la mise à jour de pip" -ForegroundColor Red
}

# Installer les dépendances
Write-Host "`n[5/5] Installation des dépendances (cela peut prendre quelques minutes)..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓✓✓ Installation terminée avec succès! ✓✓✓" -ForegroundColor Green
    Write-Host "`nProchaines étapes:" -ForegroundColor Cyan
    Write-Host "1. Activez l'environnement: .\venv311\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host "2. Testez: python test_project.py" -ForegroundColor White
    Write-Host "3. Entraînez: python src/train.py --use_cifar10 --epochs 1" -ForegroundColor White
} else {
    Write-Host "`n✗ Erreur lors de l'installation des dépendances" -ForegroundColor Red
    Write-Host "Essayez: pip install -r requirements-core.txt" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Cyan

