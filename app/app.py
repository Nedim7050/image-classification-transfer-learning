"""
Streamlit application for image classification inference
"""

import streamlit as st
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.infer import load_model, predict_from_array, predict

# Configuration
st.set_page_config(
    page_title="Image Classification - Transfer Learning",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🖼️ Image Classification avec Transfer Learning</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Configuration")

# Model selection
model_path = st.sidebar.text_input(
    "Chemin du modèle",
    value="models/best_model.pth",
    help="Chemin vers le fichier de checkpoint du modèle (.pth)"
)

# Device selection
device_options = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
device = st.sidebar.selectbox("Device", device_options, index=0 if 'cpu' in device_options else 1)

# Top-k predictions
topk = st.sidebar.slider("Nombre de prédictions (Top-K)", min_value=1, max_value=10, value=3)

# Image size
img_size = st.sidebar.slider("Taille d'image", min_value=32, max_value=512, value=224, step=32)

# Load model button
load_model_btn = st.sidebar.button("🔄 Charger le Modèle", type="primary")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload d'Image")
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Choisissez une image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
        help="Upload une image pour la classification"
    )
    
    # Example images (if CIFAR-10)
    st.subheader("📷 Images d'Exemple")
    use_example = st.checkbox("Utiliser une image d'exemple CIFAR-10")
    
    if use_example:
        example_class = st.selectbox(
            "Classe d'exemple",
            ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        )
        
        # Load CIFAR-10 example
        try:
            from torchvision.datasets import CIFAR10
            import torchvision.transforms as transforms
            
            # Download and load CIFAR-10
            cifar10 = CIFAR10(root='../data', train=False, download=True, transform=transforms.ToTensor())
            
            # Find an example of the selected class
            class_idx = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                        'dog', 'frog', 'horse', 'ship', 'truck'].index(example_class)
            
            example_image = None
            for img, label in cifar10:
                if label == class_idx:
                    example_image = img
                    break
            
            if example_image is not None:
                # Convert to PIL Image
                img_np = example_image.permute(1, 2, 0).numpy()
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                st.image(img_pil, caption=f"Exemple: {example_class}", use_container_width=True)
                uploaded_file = None  # Clear uploaded file
                image_to_predict = img_np
        except Exception as e:
            st.error(f"Erreur lors du chargement de l'exemple: {e}")
            image_to_predict = None
    else:
        image_to_predict = None
    
    # Display uploaded image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image uploadée", use_container_width=True)
        image_to_predict = np.array(image.convert('RGB'))

with col2:
    st.header("🔮 Prédictions")
    
    # Model status
    model_loaded = False
    model_info = None
    
    if load_model_btn or os.path.exists(model_path):
        try:
            if os.path.exists(model_path):
                with st.spinner("Chargement du modèle..."):
                    model, class_names, model_name = load_model(model_path, device)
                    model_loaded = True
                    model_info = {
                        'model_name': model_name,
                        'num_classes': len(class_names),
                        'class_names': class_names
                    }
                    st.success(f"✅ Modèle chargé: {model_name}")
                    st.info(f"📊 {len(class_names)} classes: {', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''}")
            else:
                st.error(f"❌ Modèle non trouvé: {model_path}")
                st.info("💡 Entraînez d'abord un modèle avec `python src/train.py` ou utilisez le notebook `02_training_transfer_learning.ipynb`")
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du modèle: {e}")
            st.exception(e)
    
    # Predict button
    if model_loaded and image_to_predict is not None:
        predict_btn = st.button("🚀 Prédire", type="primary", use_container_width=True)
        
        if predict_btn:
            with st.spinner("Prédiction en cours..."):
                try:
                    # Make prediction
                    results = predict_from_array(
                        image_to_predict,
                        model_path,
                        topk=topk,
                        device=device,
                        img_size=img_size
                    )
                    
                    # Display results
                    st.markdown("### 📊 Résultats de Classification")
                    
                    # Top prediction
                    top_class, top_prob = results[0]
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>🏆 Prédiction Principale</h3>
                        <h2 style="color: #1f77b4;">{top_class}</h2>
                        <p style="font-size: 1.5rem;">Confiance: <strong>{top_prob*100:.2f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Bar chart
                    classes = [r[0] for r in results]
                    probs = [r[1] * 100 for r in results]
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.barh(classes[::-1], probs[::-1], color=sns.color_palette("husl", len(classes)))
                    ax.set_xlabel('Probabilité (%)', fontsize=12)
                    ax.set_title('Top-K Prédictions', fontsize=14, fontweight='bold')
                    ax.set_xlim(0, 100)
                    
                    # Add value labels on bars
                    for i, (bar, prob) in enumerate(zip(bars, probs[::-1])):
                        ax.text(prob + 1, i, f'{prob:.2f}%', va='center', fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Detailed results table
                    st.markdown("### 📋 Détails des Prédictions")
                    results_data = {
                        'Classe': [r[0] for r in results],
                        'Probabilité (%)': [f"{r[1]*100:.2f}" for r in results]
                    }
                    st.table(results_data)
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de la prédiction: {e}")
                    st.exception(e)
    
    elif not model_loaded:
        st.info("👆 Cliquez sur 'Charger le Modèle' pour commencer")
    
    elif image_to_predict is None:
        st.info("👆 Upload une image ou sélectionnez un exemple pour faire une prédiction")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Image Classification avec Transfer Learning | PyTorch + Streamlit</p>
    <p>Entraînez votre modèle avec <code>python src/train.py</code> ou utilisez les notebooks Jupyter</p>
</div>
""", unsafe_allow_html=True)

