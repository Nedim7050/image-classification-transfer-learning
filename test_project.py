"""
Script de test pour vérifier que tous les composants du projet fonctionnent correctement
avant de pousser sur GitHub et déployer sur Streamlit
"""

import sys
import os
from pathlib import Path
import traceback

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Couleurs pour l'affichage
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_success(msg):
    print(f"{GREEN}✅ {msg}{RESET}")

def print_error(msg):
    print(f"{RED}❌ {msg}{RESET}")

def print_warning(msg):
    print(f"{YELLOW}⚠️  {msg}{RESET}")

def print_info(msg):
    print(f"{BLUE}ℹ️  {msg}{RESET}")

def test_imports():
    """Test 1: Vérifier que tous les imports fonctionnent"""
    print("\n" + "="*60)
    print("TEST 1: Vérification des imports")
    print("="*60)
    
    try:
        import torch
        import torchvision
        import cv2
        import numpy as np
        import matplotlib
        import seaborn
        import streamlit
        import sklearn
        import tqdm
        import PIL
        print_success("Tous les packages de base sont installés")
        
        # Test imports locaux
        sys.path.append(str(Path(__file__).parent))
        from src.dataset import load_cifar10_dataset, load_imagefolder_dataset
        from src.train import create_model
        from src.infer import predict_from_array, load_model
        from src.utils import calculate_accuracy, save_checkpoint
        print_success("Tous les modules locaux s'importent correctement")
        return True
    except ImportError as e:
        print_error(f"Erreur d'import: {e}")
        print_warning("Exécutez: pip install -r requirements.txt")
        return False
    except Exception as e:
        print_error(f"Erreur inattendue: {e}")
        traceback.print_exc()
        return False

def test_cuda():
    """Test 2: Vérifier la disponibilité CUDA"""
    print("\n" + "="*60)
    print("TEST 2: Vérification CUDA")
    print("="*60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print_success(f"CUDA disponible: {torch.cuda.get_device_name(0)}")
            print_info(f"Version CUDA: {torch.version.cuda}")
        else:
            print_warning("CUDA non disponible - l'entraînement se fera sur CPU")
        return True
    except Exception as e:
        print_error(f"Erreur lors de la vérification CUDA: {e}")
        return False

def test_dataset_loading():
    """Test 3: Tester le chargement du dataset CIFAR-10"""
    print("\n" + "="*60)
    print("TEST 3: Chargement du dataset CIFAR-10")
    print("="*60)
    
    try:
        sys.path.append(str(Path(__file__).parent))
        from src.dataset import load_cifar10_dataset
        
        print_info("Téléchargement de CIFAR-10 (peut prendre quelques instants)...")
        train_loader, val_loader, class_names = load_cifar10_dataset(
            data_dir='./data',
            img_size=224,
            batch_size=16,  # Plus petit pour le test
            num_workers=0   # Éviter les problèmes de multiprocessing
        )
        
        print_success(f"Dataset chargé: {len(class_names)} classes")
        print_info(f"Classes: {class_names}")
        print_info(f"Train batches: {len(train_loader)}")
        print_info(f"Val batches: {len(val_loader)}")
        
        # Tester un batch
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        print_success(f"Batch test: {images.shape}, labels: {labels.shape}")
        
        return True
    except Exception as e:
        print_error(f"Erreur lors du chargement du dataset: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test 4: Tester la création du modèle"""
    print("\n" + "="*60)
    print("TEST 4: Création du modèle")
    print("="*60)
    
    try:
        import torch
        sys.path.append(str(Path(__file__).parent))
        from src.train import create_model
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print_info(f"Device: {device}")
        
        # Test ResNet50
        model, _ = create_model('resnet50', num_classes=10, pretrained=True, 
                               freeze_backbone=True, unfreeze_last_n=0)
        print_success("Modèle ResNet50 créé avec succès")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        print_success(f"Forward pass réussi: output shape {output.shape}")
        
        return True
    except Exception as e:
        print_error(f"Erreur lors de la création du modèle: {e}")
        traceback.print_exc()
        return False

def test_training_step():
    """Test 5: Tester une étape d'entraînement"""
    print("\n" + "="*60)
    print("TEST 5: Test d'une étape d'entraînement")
    print("="*60)
    
    try:
        import torch
        import torch.nn as nn
        sys.path.append(str(Path(__file__).parent))
        from src.dataset import load_cifar10_dataset
        from src.train import create_model
        from src.utils import calculate_accuracy
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Charger un petit dataset
        train_loader, _, class_names = load_cifar10_dataset(
            data_dir='./data',
            img_size=224,
            batch_size=8,
            num_workers=0
        )
        
        # Créer modèle
        model, _ = create_model('resnet50', len(class_names), pretrained=True,
                               freeze_backbone=True, unfreeze_last_n=0)
        
        # Test une étape
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        acc = calculate_accuracy(outputs, labels)
        print_success(f"Étape d'entraînement réussie: Loss={loss.item():.4f}, Acc={acc:.2f}%")
        
        return True
    except Exception as e:
        print_error(f"Erreur lors de l'étape d'entraînement: {e}")
        traceback.print_exc()
        return False

def test_inference():
    """Test 6: Tester l'inférence"""
    print("\n" + "="*60)
    print("TEST 6: Test d'inférence")
    print("="*60)
    
    try:
        import torch
        import numpy as np
        sys.path.append(str(Path(__file__).parent))
        from src.train import create_model
        from src.infer import predict_from_array
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Créer un modèle et le sauvegarder temporairement
        model, _ = create_model('resnet50', num_classes=10, pretrained=True,
                               freeze_backbone=True, unfreeze_last_n=0)
        
        os.makedirs('./models', exist_ok=True)
        test_model_path = './models/test_model.pth'
        torch.save({
            'state_dict': model.state_dict(),
            'model_name': 'resnet50',
            'num_classes': 10,
            'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        }, test_model_path)
        
        # Test inférence
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        results = predict_from_array(dummy_image, test_model_path, topk=3, device=device)
        
        print_success(f"Inférence réussie: {len(results)} prédictions")
        for class_name, prob in results:
            print_info(f"  {class_name}: {prob*100:.2f}%")
        
        # Nettoyer
        if os.path.exists(test_model_path):
            os.remove(test_model_path)
        
        return True
    except Exception as e:
        print_error(f"Erreur lors de l'inférence: {e}")
        traceback.print_exc()
        return False

def test_streamlit_app():
    """Test 7: Vérifier que l'app Streamlit peut être importée"""
    print("\n" + "="*60)
    print("TEST 7: Vérification de l'application Streamlit")
    print("="*60)
    
    try:
        app_path = Path(__file__).parent / 'app' / 'app.py'
        if not app_path.exists():
            print_error(f"Fichier app.py non trouvé: {app_path}")
            return False
        
        # Vérifier que le fichier peut être lu
        with open(app_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Vérifier les imports essentiels
        if 'import streamlit' in content:
            print_success("Application Streamlit trouvée")
        else:
            print_error("Import streamlit manquant dans app.py")
            return False
        
        if 'from src.infer import' in content:
            print_success("Imports locaux corrects")
        else:
            print_warning("Vérifiez les imports dans app.py")
        
        print_info("Pour tester l'app: streamlit run app/app.py")
        return True
    except Exception as e:
        print_error(f"Erreur lors de la vérification de l'app: {e}")
        return False

def test_file_structure():
    """Test 8: Vérifier la structure des fichiers"""
    print("\n" + "="*60)
    print("TEST 8: Vérification de la structure des fichiers")
    print("="*60)
    
    required_files = [
        'requirements.txt',
        'README.md',
        'LICENSE',
        '.gitignore',
        'src/__init__.py',
        'src/dataset.py',
        'src/train.py',
        'src/infer.py',
        'src/utils.py',
        'app/app.py',
        'scripts/convert_to_onnx.py',
        'notebooks/01_EDA_images.ipynb',
        'notebooks/02_training_transfer_learning.ipynb',
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = Path(__file__).parent / file_path
        if full_path.exists():
            print_success(f"✓ {file_path}")
        else:
            print_error(f"✗ {file_path} manquant")
            missing_files.append(file_path)
    
    if missing_files:
        print_warning(f"{len(missing_files)} fichier(s) manquant(s)")
        return False
    else:
        print_success("Tous les fichiers requis sont présents")
        return True

def test_onnx_conversion():
    """Test 9: Tester la conversion ONNX (optionnel)"""
    print("\n" + "="*60)
    print("TEST 9: Vérification du script ONNX")
    print("="*60)
    
    try:
        script_path = Path(__file__).parent / 'scripts' / 'convert_to_onnx.py'
        if not script_path.exists():
            print_error("Script convert_to_onnx.py non trouvé")
            return False
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'torch.onnx.export' in content:
            print_success("Script ONNX semble correct")
        else:
            print_warning("Vérifiez le script ONNX")
        
        # Vérifier les imports
        try:
            import onnx
            print_success("Package onnx disponible")
        except ImportError:
            print_warning("Package onnx non installé (optionnel)")
        
        return True
    except Exception as e:
        print_error(f"Erreur: {e}")
        return False

def main():
    """Exécuter tous les tests"""
    print("\n" + "="*60)
    print("🧪 TESTS DU PROJET - Image Classification Transfer Learning")
    print("="*60)
    
    results = []
    
    # Tests
    results.append(("Imports", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("Structure fichiers", test_file_structure()))
    results.append(("Chargement dataset", test_dataset_loading()))
    results.append(("Création modèle", test_model_creation()))
    results.append(("Étape entraînement", test_training_step()))
    results.append(("Inférence", test_inference()))
    results.append(("App Streamlit", test_streamlit_app()))
    results.append(("Script ONNX", test_onnx_conversion()))
    
    # Résumé
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSÉ" if result else "❌ ÉCHOUÉ"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*60)
    if passed == total:
        print_success(f"Tous les tests sont passés! ({passed}/{total})")
        print_info("\nProchaines étapes:")
        print_info("1. Vérifiez que .gitignore exclut bien les fichiers volumineux")
        print_info("2. Testez manuellement: streamlit run app/app.py")
        print_info("3. Entraînez un modèle: python src/train.py --use_cifar10 --epochs 1")
        print_info("4. Poussez sur GitHub")
        print_info("5. Déployez sur Streamlit Cloud")
    else:
        print_error(f"Certains tests ont échoué ({passed}/{total})")
        print_warning("Corrigez les erreurs avant de pousser sur GitHub")
    print("="*60 + "\n")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

