#!/usr/bin/env python3
"""
🔬 VAE Explorer - Interactive Streamlit App

Professional interface for exploring trained VAE models.

Features:
- Load trained checkpoints
- Navigate latent space with sliders
- Real-time activation visualization
- 3D latent space exploration
- Image interpolation
- Random walks
- Channel-by-channel analysis

Usage:
    streamlit run scripts/vae_explorer_streamlit.py
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from sklearn.decomposition import PCA
from PIL import Image
import io

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.vae_equivariant import ConfigurableVAE

# Page config
st.set_page_config(
    page_title="🔬 VAE Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    .stSlider > div > div {
        background-color: #4a4a6a;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    h1, h2, h3 {
        color: #e0e0ff !important;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(checkpoint_path: str):
    """Load model (cached)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('model_config', {})
    latent_dim = config.get('latent_dim', 512)
    
    model = ConfigurableVAE(
        latent_dim=latent_dim,
        base_channels=32,
        num_encoder_layers=6,
        num_decoder_layers=6,
        capture_activations=True
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device, latent_dim, checkpoint


def generate_image(model, z, device):
    """Generate image from latent vector."""
    with torch.no_grad():
        z_tensor = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
        recon = model.decode(z_tensor)
        img = recon[0].cpu().permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


def get_activations(model, z, device):
    """Get layer activations."""
    model.enable_activation_capture()
    
    with torch.no_grad():
        z_tensor = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
        _ = model.decode(z_tensor)
    
    activations = model.get_activations()
    model.disable_activation_capture()
    
    return activations


def create_activation_grid(activations, layer_name, n_channels=64, contrast=1.0, gamma=1.0):
    """Create grid visualization of activations."""
    if layer_name not in activations:
        return None
    
    act = activations[layer_name]
    if act.dim() == 4:
        act = act[0]
    
    act = act.cpu().numpy()
    n_channels = min(n_channels, act.shape[0])
    cols = 8
    rows = (n_channels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 14 * rows / cols), 
                             facecolor='#1a1a2e')
    axes = axes.flatten()
    
    # Custom colormap
    colors = ['#0f0f1a', '#1a1a4e', '#3a3a8e', '#6a6aae', '#9a9ace', '#cacaee']
    cmap = LinearSegmentedColormap.from_list('custom', colors)
    
    for i in range(n_channels):
        ch = act[i]
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        # Apply contrast and gamma
        ch = np.clip(ch * contrast, 0, 1)
        ch = np.power(ch, gamma)
        axes[i].imshow(ch, cmap=cmap)
        axes[i].axis('off')
        axes[i].set_title(f'{i}', color='white', fontsize=8, pad=2)
    
    for i in range(n_channels, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{layer_name}: {act.shape}', color='white', fontsize=14, y=0.98)
    plt.tight_layout()
    
    return fig


def create_multi_layer_view(activations, layer_names, contrast=1.0, gamma=1.0):
    """Create visualization of multiple layers side by side."""
    valid_layers = [l for l in layer_names if l in activations]
    
    if not valid_layers:
        return None
    
    n_layers = len(valid_layers)
    fig, axes = plt.subplots(2, n_layers, figsize=(4 * n_layers, 8), facecolor='#1a1a2e')
    
    if n_layers == 1:
        axes = axes.reshape(2, 1)
    
    for i, layer_name in enumerate(valid_layers):
        act = activations[layer_name]
        if act.dim() == 4:
            act = act[0]
        act = act.cpu().numpy()
        
        # Mean activation (top row)
        act_mean = act.mean(axis=0)
        act_mean = (act_mean - act_mean.min()) / (act_mean.max() - act_mean.min() + 1e-8)
        act_mean = np.clip(act_mean * contrast, 0, 1)
        act_mean = np.power(act_mean, gamma)
        
        axes[0, i].imshow(act_mean, cmap='magma')
        axes[0, i].set_title(f'{layer_name}\nmean', color='white', fontsize=10)
        axes[0, i].axis('off')
        
        # Std activation (bottom row) - shows where features vary
        act_std = act.std(axis=0)
        act_std = (act_std - act_std.min()) / (act_std.max() - act_std.min() + 1e-8)
        act_std = np.clip(act_std * contrast, 0, 1)
        act_std = np.power(act_std, gamma)
        
        axes[1, i].imshow(act_std, cmap='viridis')
        axes[1, i].set_title(f'std\n{act.shape}', color='white', fontsize=9)
        axes[1, i].axis('off')
    
    plt.suptitle('Layer Comparison (mean & std)', color='white', fontsize=14, y=0.98)
    plt.tight_layout()
    
    return fig


def create_all_layers_overview(activations, output_img):
    """Create overview of all layers."""
    n_layers = len(activations)
    cols = min(6, n_layers + 1)
    rows = (n_layers + 1 + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), 
                             facecolor='#1a1a2e')
    axes = np.array(axes).flatten()
    
    # Output image
    axes[0].imshow(output_img)
    axes[0].set_title('Output', color='white', fontsize=10)
    axes[0].axis('off')
    
    # Layer activations
    for i, (name, act) in enumerate(activations.items()):
        if act.dim() == 4:
            act = act[0]
        
        act_mean = act.mean(dim=0).cpu().numpy()
        act_mean = (act_mean - act_mean.min()) / (act_mean.max() - act_mean.min() + 1e-8)
        
        axes[i + 1].imshow(act_mean, cmap='magma')
        axes[i + 1].set_title(f'{name}\n{tuple(act.shape)}', color='white', fontsize=8)
        axes[i + 1].axis('off')
    
    for i in range(n_layers + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def create_latent_3d(model, device, latent_dim, n_samples=300):
    """Create 3D latent space visualization."""
    latents = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            z = torch.randn(1, latent_dim, device=device)
            latents.append(z.cpu().numpy())
    
    latents = np.concatenate(latents, axis=0)
    
    pca = PCA(n_components=3)
    latents_3d = pca.fit_transform(latents)
    
    fig = go.Figure(data=[
        go.Scatter3d(
            x=latents_3d[:, 0],
            y=latents_3d[:, 1],
            z=latents_3d[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=np.arange(n_samples),
                colorscale='Viridis',
                opacity=0.7
            )
        )
    ])
    
    fig.update_layout(
        title=f'Latent Space (PCA: {pca.explained_variance_ratio_.sum():.1%} var)',
        scene=dict(
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
            zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%})',
            bgcolor='#1a1a2e',
            xaxis=dict(gridcolor='#3a3a5e', color='white'),
            yaxis=dict(gridcolor='#3a3a5e', color='white'),
            zaxis=dict(gridcolor='#3a3a5e', color='white'),
        ),
        paper_bgcolor='#1a1a2e',
        font=dict(color='white'),
        margin=dict(l=0, r=0, b=0, t=40),
        height=500
    )
    
    return fig


def interpolate_images(model, z1, z2, device, steps=8):
    """Interpolate between two latent vectors."""
    images = []
    
    with torch.no_grad():
        for alpha in np.linspace(0, 1, steps):
            z = (1 - alpha) * z1 + alpha * z2
            z_tensor = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
            recon = model.decode(z_tensor)
            img = recon[0].cpu().permute(1, 2, 0).numpy()
            images.append(np.clip(img, 0, 1))
    
    return images


def main():
    st.title("🔬 VAE Explorer")
    st.markdown("*Explora tu autoencoder entrenado de forma interactiva*")
    
    # Sidebar
    with st.sidebar:
        st.header("📂 Cargar Modelo")
        
        checkpoint_path = st.text_input(
            "Ruta al checkpoint",
            value="./checkpoints/best_model.pt"
        )
        
        if st.button("🚀 Cargar", type="primary"):
            if Path(checkpoint_path).exists():
                with st.spinner("Cargando modelo..."):
                    try:
                        model, device, latent_dim, checkpoint = load_model(checkpoint_path)
                        st.session_state['model'] = model
                        st.session_state['device'] = device
                        st.session_state['latent_dim'] = latent_dim
                        st.session_state['checkpoint'] = checkpoint
                        st.session_state['current_z'] = np.random.randn(latent_dim).astype(np.float32)
                        st.success("✅ Modelo cargado!")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.error("Archivo no encontrado")
        
        if 'model' in st.session_state:
            st.divider()
            st.markdown("### 📊 Info del Modelo")
            checkpoint = st.session_state['checkpoint']
            n_params = sum(p.numel() for p in st.session_state['model'].parameters())
            
            st.metric("Parámetros", f"{n_params:,}")
            st.metric("Latent Dim", st.session_state['latent_dim'])
            st.metric("Epoch", checkpoint.get('epoch', 'N/A'))
            st.metric("Best Val Loss", f"{checkpoint.get('best_val_loss', 0):.2f}")
            st.metric("Device", st.session_state['device'])
    
    # Main content
    if 'model' not in st.session_state:
        st.info("👈 Carga un modelo desde la barra lateral para comenzar")
        return
    
    model = st.session_state['model']
    device = st.session_state['device']
    latent_dim = st.session_state['latent_dim']
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎚️ Navegación", "🔍 Activaciones", "🌐 Latent 3D", 
        "🔄 Interpolación", "🚶 Random Walk"
    ])
    
    with tab1:
        st.header("🎚️ Navegar Espacio Latente")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Controla las primeras 16 dimensiones")
            
            # Sliders in 2 rows
            z = st.session_state['current_z'].copy()
            
            cols = st.columns(8)
            for i in range(8):
                with cols[i]:
                    z[i] = st.slider(f"z[{i}]", -3.0, 3.0, float(z[i]), 0.1, key=f"z_{i}")
            
            cols = st.columns(8)
            for i in range(8, 16):
                with cols[i-8]:
                    z[i] = st.slider(f"z[{i}]", -3.0, 3.0, float(z[i]), 0.1, key=f"z_{i}")
            
            st.session_state['current_z'] = z
        
        with col2:
            if st.button("🎲 Random", type="secondary"):
                st.session_state['current_z'] = np.random.randn(latent_dim).astype(np.float32)
                st.rerun()
            
            # Generate and display image
            img = generate_image(model, st.session_state['current_z'], device)
            st.image(img, caption="Imagen Generada", use_container_width=True)
    
    with tab2:
        st.header("🔍 Visualización de Activaciones")
        
        # Get actual layer names from activations
        with st.spinner("Detectando capas..."):
            activations = get_activations(model, st.session_state['current_z'], device)
            all_layers = list(activations.keys())
        
        if not all_layers:
            st.warning("No se detectaron activaciones. Asegurate de que el modelo tenga capture_activations=True")
        else:
            st.success(f"✅ {len(all_layers)} capas detectadas")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown("### ⚙️ Controles")
                
                selected_layer = st.selectbox("Capa individual", all_layers)
                n_channels = st.slider("Canales", 16, 128, 64, 16)
                
                st.markdown("---")
                st.markdown("### 🎨 Ajustes visuales")
                contrast = st.slider("Contraste", 0.5, 3.0, 1.0, 0.1)
                gamma = st.slider("Gamma", 0.2, 3.0, 1.0, 0.1)
                
                st.markdown("---")
                st.markdown("### 📊 Multi-capa")
                selected_multi = st.multiselect(
                    "Comparar capas",
                    all_layers,
                    default=all_layers[:min(4, len(all_layers))]
                )
            
            with col2:
                # Tabs for different views
                view_tab1, view_tab2 = st.tabs(["🔬 Capa Individual", "📊 Multi-Capa"])
                
                with view_tab1:
                    if selected_layer in activations:
                        fig = create_activation_grid(activations, selected_layer, n_channels, contrast, gamma)
                        if fig:
                            st.pyplot(fig)
                            plt.close()
                        
                        # Stats
                        act = activations[selected_layer]
                        if act.dim() == 4:
                            act = act[0]
                        st.markdown(f"""
                        **Estadísticas de `{selected_layer}`:**
                        - Shape: `{tuple(act.shape)}`
                        - Mean: `{act.mean().item():.4f}`
                        - Std: `{act.std().item():.4f}`
                        - Min: `{act.min().item():.4f}`
                        - Max: `{act.max().item():.4f}`
                        - Sparsity: `{(act.abs() < 0.01).float().mean().item():.1%}`
                        """)
                    else:
                        st.warning(f"Capa '{selected_layer}' no encontrada")
                
                with view_tab2:
                    if selected_multi:
                        fig = create_multi_layer_view(activations, selected_multi, contrast, gamma)
                        if fig:
                            st.pyplot(fig)
                            plt.close()
                    else:
                        st.info("Seleccioná capas para comparar")
        
        st.divider()
        st.markdown("### 🖼️ Vista General (todas las capas)")
        
        if st.button("Generar Overview Completo"):
            with st.spinner("Procesando..."):
                activations = get_activations(model, st.session_state['current_z'], device)
                img = generate_image(model, st.session_state['current_z'], device)
                fig = create_all_layers_overview(activations, img)
                st.pyplot(fig)
                plt.close()
    
    with tab3:
        st.header("🌐 Espacio Latente 3D")
        
        n_samples = st.slider("Número de muestras", 100, 500, 300, 50)
        
        if st.button("📊 Generar Plot 3D", type="primary"):
            with st.spinner("Generando visualización 3D..."):
                fig = create_latent_3d(model, device, latent_dim, n_samples)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("🔄 Interpolación")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🎲 Generar Z1"):
                st.session_state['z1'] = np.random.randn(latent_dim).astype(np.float32)
            
            if 'z1' in st.session_state:
                img1 = generate_image(model, st.session_state['z1'], device)
                st.image(img1, caption="Imagen 1", use_container_width=True)
        
        with col2:
            if st.button("🎲 Generar Z2"):
                st.session_state['z2'] = np.random.randn(latent_dim).astype(np.float32)
            
            if 'z2' in st.session_state:
                img2 = generate_image(model, st.session_state['z2'], device)
                st.image(img2, caption="Imagen 2", use_container_width=True)
        
        with col3:
            steps = st.slider("Pasos de interpolación", 4, 16, 8)
            
            if st.button("🔄 Interpolar", type="primary"):
                if 'z1' in st.session_state and 'z2' in st.session_state:
                    with st.spinner("Interpolando..."):
                        images = interpolate_images(
                            model, 
                            st.session_state['z1'], 
                            st.session_state['z2'], 
                            device, 
                            steps
                        )
                        
                        # Display as grid
                        cols = st.columns(min(steps, 8))
                        for i, img in enumerate(images[:8]):
                            with cols[i % 8]:
                                st.image(img, caption=f"α={i/(steps-1):.2f}", use_container_width=True)
                        
                        if steps > 8:
                            cols = st.columns(min(steps - 8, 8))
                            for i, img in enumerate(images[8:]):
                                with cols[i]:
                                    st.image(img, caption=f"α={(i+8)/(steps-1):.2f}", use_container_width=True)
    
    with tab5:
        st.header("🚶 Random Walk")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            walk_steps = st.slider("Pasos", 8, 32, 16)
            step_size = st.slider("Tamaño de paso", 0.1, 2.0, 0.5, 0.1)
            
            start_walk = st.button("🚶 Iniciar Walk", type="primary")
        
        with col2:
            if start_walk:
                with st.spinner("Generando random walk..."):
                    z = np.random.randn(latent_dim).astype(np.float32)
                    images = []
                    
                    for _ in range(walk_steps):
                        img = generate_image(model, z, device)
                        images.append(img)
                        z = z + np.random.randn(latent_dim).astype(np.float32) * step_size
                    
                    # Display grid
                    cols_per_row = 8
                    for row_start in range(0, walk_steps, cols_per_row):
                        cols = st.columns(min(cols_per_row, walk_steps - row_start))
                        for i, col in enumerate(cols):
                            idx = row_start + i
                            if idx < walk_steps:
                                with col:
                                    st.image(images[idx], caption=f"Step {idx}", use_container_width=True)


if __name__ == "__main__":
    main()

