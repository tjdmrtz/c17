#!/usr/bin/env python3
"""
Interactive VAE Explorer - Full-featured GUI for analyzing autoencoders.

Features:
- Load any trained VAE checkpoint
- Real-time latent space navigation with sliders
- 3D latent space visualization with rotation
- Layer activation viewer (all encoder/decoder layers)
- Interpolation between images
- Random sampling and walks
- Reconstruction comparison
- Channel-by-channel activation grids

Usage:
    python scripts/interactive_explorer.py
    python scripts/interactive_explorer.py --checkpoint path/to/model.pt

Then open http://localhost:7860 in your browser
"""

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pathlib import Path
import sys
import argparse
from typing import Optional, Dict, List, Tuple
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vae_equivariant import ConfigurableVAE


class VAEExplorer:
    """Interactive VAE exploration backend."""
    
    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.current_z = None
        self.latent_dim = 512
        self.pca = None
        self.latent_cache = []
        self.image_cache = []
        self.label_cache = []
        
    def load_model(self, checkpoint_path: str) -> str:
        """Load a VAE model from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Get model config
            config = checkpoint.get('model_config', {})
            self.latent_dim = config.get('latent_dim', 512)
            
            # Create model
            self.model = ConfigurableVAE(
                input_channels=3,
                latent_dim=self.latent_dim,
                base_channels=32,
                num_encoder_layers=6,
                num_decoder_layers=6,
                capture_activations=True
            )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize random latent
            self.current_z = torch.randn(1, self.latent_dim, device=self.device)
            
            # Get layer info
            layers = self.model.get_layer_names()
            n_params = sum(p.numel() for p in self.model.parameters())
            
            return f"""✅ Modelo cargado exitosamente!
            
📊 Información:
- Parámetros: {n_params:,}
- Dimensión latente: {self.latent_dim}
- Capas encoder: {len(layers.get('encoder', []))}
- Capas decoder: {len(layers.get('decoder', []))}
- Device: {self.device}
- Epoch: {checkpoint.get('epoch', 'N/A')}
- Best val loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}"""
        
        except Exception as e:
            return f"❌ Error cargando modelo: {str(e)}"
    
    def generate_from_sliders(self, *slider_values) -> np.ndarray:
        """Generate image from latent slider values."""
        if self.model is None:
            return np.zeros((512, 512, 3))
        
        # Create latent vector from sliders (first 32 dims controllable)
        z = torch.zeros(1, self.latent_dim, device=self.device)
        for i, val in enumerate(slider_values):
            if i < self.latent_dim:
                z[0, i] = val
        
        self.current_z = z
        
        with torch.no_grad():
            recon = self.model.decode(z)
        
        img = recon[0].cpu().permute(1, 2, 0).numpy()
        return np.clip(img, 0, 1)
    
    def random_sample(self) -> Tuple[np.ndarray, str]:
        """Generate random sample."""
        if self.model is None:
            return np.zeros((512, 512, 3)), "Modelo no cargado"
        
        self.current_z = torch.randn(1, self.latent_dim, device=self.device)
        
        with torch.no_grad():
            recon = self.model.decode(self.current_z)
        
        img = recon[0].cpu().permute(1, 2, 0).numpy()
        
        # Get top 5 latent values
        top_vals = self.current_z[0].cpu().numpy()
        top_indices = np.argsort(np.abs(top_vals))[-5:][::-1]
        info = "Top dims: " + ", ".join([f"z[{i}]={top_vals[i]:.2f}" for i in top_indices])
        
        return np.clip(img, 0, 1), info
    
    def encode_image(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """Encode an image and show reconstruction."""
        if self.model is None or image is None:
            return np.zeros((512, 512, 3)), "Modelo no cargado"
        
        # Preprocess
        if image.shape[-1] == 4:  # RGBA
            image = image[:, :, :3]
        
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        img_tensor = torch.nn.functional.interpolate(img_tensor, size=(512, 512), mode='bilinear')
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            mu, logvar = self.model.encode(img_tensor)
            self.current_z = mu
            recon = self.model.decode(mu)
        
        recon_img = recon[0].cpu().permute(1, 2, 0).numpy()
        
        # Calculate reconstruction error
        mse = ((img_tensor.cpu().numpy() - recon.cpu().numpy()) ** 2).mean()
        
        return np.clip(recon_img, 0, 1), f"MSE: {mse:.6f}, z_mean: {mu.mean().item():.3f}, z_std: {mu.std().item():.3f}"
    
    def get_activations(self, layer_name: str) -> np.ndarray:
        """Get activation visualization for a specific layer."""
        if self.model is None or self.current_z is None:
            return np.zeros((400, 400, 3))
        
        self.model.enable_activation_capture()
        
        with torch.no_grad():
            _ = self.model.decode(self.current_z)
        
        activations = self.model.get_activations()
        self.model.disable_activation_capture()
        
        if layer_name not in activations:
            # Return placeholder
            fig, ax = plt.subplots(figsize=(6, 6), facecolor='#1a1a2e')
            ax.text(0.5, 0.5, f"Layer '{layer_name}' not found\n\nAvailable: {list(activations.keys())}", 
                   ha='center', va='center', color='white', fontsize=10)
            ax.set_facecolor('#1a1a2e')
            ax.axis('off')
            return self._fig_to_array(fig)
        
        act = activations[layer_name]
        if act.dim() == 4:
            act = act[0]
        
        # Create grid of first 64 channels
        n_channels = min(64, act.shape[0])
        cols = 8
        rows = (n_channels + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 12), facecolor='#1a1a2e')
        axes = axes.flatten()
        
        for i in range(n_channels):
            ch = act[i].cpu().numpy()
            ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
            axes[i].imshow(ch, cmap='viridis')
            axes[i].axis('off')
            axes[i].set_title(f'{i}', color='white', fontsize=7)
        
        for i in range(n_channels, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'{layer_name}: {tuple(act.shape)}', color='white', fontsize=12)
        plt.tight_layout()
        
        return self._fig_to_array(fig)
    
    def get_all_activations_overview(self) -> np.ndarray:
        """Get overview of all layer activations."""
        if self.model is None or self.current_z is None:
            return np.zeros((600, 1200, 3))
        
        self.model.enable_activation_capture()
        
        with torch.no_grad():
            recon = self.model.decode(self.current_z)
        
        activations = self.model.get_activations()
        self.model.disable_activation_capture()
        
        n_layers = len(activations)
        if n_layers == 0:
            return np.zeros((600, 1200, 3))
        
        cols = min(8, n_layers + 1)
        rows = (n_layers + 1 + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), facecolor='#1a1a2e')
        axes = np.array(axes).flatten()
        
        # Show output image first
        img = recon[0].cpu().permute(1, 2, 0).numpy()
        axes[0].imshow(np.clip(img, 0, 1))
        axes[0].set_title('Output', color='white', fontsize=10)
        axes[0].axis('off')
        
        # Show each layer's mean activation
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
        return self._fig_to_array(fig)
    
    def interpolate(self, image1: np.ndarray, image2: np.ndarray, steps: int = 8) -> List[np.ndarray]:
        """Interpolate between two images."""
        if self.model is None:
            return [np.zeros((512, 512, 3)) for _ in range(steps)]
        
        results = []
        
        def preprocess(img):
            if img is None:
                return torch.randn(1, 3, 512, 512, device=self.device)
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
            t = torch.nn.functional.interpolate(t, size=(512, 512), mode='bilinear')
            return t.to(self.device)
        
        img1 = preprocess(image1)
        img2 = preprocess(image2)
        
        with torch.no_grad():
            z1, _ = self.model.encode(img1)
            z2, _ = self.model.encode(img2)
            
            for alpha in np.linspace(0, 1, steps):
                z = (1 - alpha) * z1 + alpha * z2
                recon = self.model.decode(z)
                img = recon[0].cpu().permute(1, 2, 0).numpy()
                results.append(np.clip(img, 0, 1))
        
        return results
    
    def create_interpolation_grid(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """Create interpolation visualization."""
        images = self.interpolate(image1, image2, steps=8)
        
        fig, axes = plt.subplots(1, 8, figsize=(20, 3), facecolor='#1a1a2e')
        
        for i, (ax, img) in enumerate(zip(axes, images)):
            ax.imshow(img)
            ax.set_title(f'α={i/7:.2f}', color='white', fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        return self._fig_to_array(fig)
    
    def random_walk(self, steps: int = 16, step_size: float = 0.5) -> np.ndarray:
        """Random walk through latent space."""
        if self.model is None:
            return np.zeros((400, 800, 3))
        
        z = torch.randn(1, self.latent_dim, device=self.device)
        
        cols = min(8, steps)
        rows = (steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows), facecolor='#1a1a2e')
        axes = np.array(axes).flatten()
        
        with torch.no_grad():
            for i in range(steps):
                recon = self.model.decode(z)
                img = recon[0].cpu().permute(1, 2, 0).numpy()
                
                axes[i].imshow(np.clip(img, 0, 1))
                axes[i].set_title(f'Step {i}', color='white', fontsize=8)
                axes[i].axis('off')
                
                # Random step
                z = z + torch.randn_like(z) * step_size
        
        for i in range(steps, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Random Walk in Latent Space', color='white', fontsize=12)
        plt.tight_layout()
        
        return self._fig_to_array(fig)
    
    def create_latent_3d(self, dataloader=None) -> go.Figure:
        """Create 3D latent space visualization."""
        if self.model is None:
            return go.Figure()
        
        # Generate random samples if no dataloader
        n_samples = 200
        
        latents = []
        with torch.no_grad():
            for _ in range(n_samples):
                z = torch.randn(1, self.latent_dim, device=self.device)
                latents.append(z.cpu().numpy())
        
        latents = np.concatenate(latents, axis=0)
        
        # PCA to 3D
        pca = PCA(n_components=3)
        latents_3d = pca.fit_transform(latents)
        
        # Create 3D scatter
        fig = go.Figure(data=[
            go.Scatter3d(
                x=latents_3d[:, 0],
                y=latents_3d[:, 1],
                z=latents_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=np.arange(n_samples),
                    colorscale='Viridis',
                    opacity=0.8
                )
            )
        ])
        
        fig.update_layout(
            title='Latent Space (PCA 3D)',
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3',
                bgcolor='#1a1a2e'
            ),
            paper_bgcolor='#1a1a2e',
            font=dict(color='white'),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        return fig
    
    def _fig_to_array(self, fig) -> np.ndarray:
        """Convert matplotlib figure to numpy array."""
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        return img
    
    def get_layer_names(self) -> List[str]:
        """Get list of available layer names."""
        if self.model is None:
            return []
        
        layers = self.model.get_layer_names()
        all_layers = layers.get('encoder', []) + layers.get('decoder', [])
        return all_layers if all_layers else ['enc_0', 'enc_1', 'dec_0', 'dec_1']


# Global explorer instance
explorer = VAEExplorer()


def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(
        title="🔬 VAE Explorer",
        theme=gr.themes.Base(
            primary_hue="violet",
            secondary_hue="purple",
            neutral_hue="slate",
        ),
        css="""
        .gradio-container { max-width: 1400px !important; }
        .dark { background-color: #0f0f1a !important; }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🔬 Interactive VAE Explorer
        
        Explora tu autoencoder entrenado: navega el espacio latente, visualiza activaciones, genera interpolaciones y más.
        """)
        
        with gr.Tab("📂 Cargar Modelo"):
            with gr.Row():
                checkpoint_input = gr.Textbox(
                    label="Ruta al checkpoint",
                    placeholder="./checkpoints/best_model.pt",
                    value="./checkpoints/best_model.pt"
                )
                load_btn = gr.Button("🚀 Cargar Modelo", variant="primary")
            
            load_status = gr.Textbox(label="Estado", lines=8, interactive=False)
            
            load_btn.click(
                fn=explorer.load_model,
                inputs=[checkpoint_input],
                outputs=[load_status]
            )
        
        with gr.Tab("🎚️ Navegar Espacio Latente"):
            gr.Markdown("### Controla las primeras 16 dimensiones del espacio latente")
            
            with gr.Row():
                with gr.Column(scale=2):
                    sliders = []
                    with gr.Row():
                        for i in range(8):
                            sliders.append(gr.Slider(-3, 3, value=0, step=0.1, label=f"z[{i}]"))
                    with gr.Row():
                        for i in range(8, 16):
                            sliders.append(gr.Slider(-3, 3, value=0, step=0.1, label=f"z[{i}]"))
                
                with gr.Column(scale=1):
                    latent_output = gr.Image(label="Imagen Generada", height=400)
            
            random_btn = gr.Button("🎲 Random Sample", variant="secondary")
            random_info = gr.Textbox(label="Info", interactive=False)
            
            # Update on slider change
            for slider in sliders:
                slider.change(
                    fn=explorer.generate_from_sliders,
                    inputs=sliders,
                    outputs=[latent_output]
                )
            
            random_btn.click(
                fn=explorer.random_sample,
                outputs=[latent_output, random_info]
            )
        
        with gr.Tab("🔍 Activaciones"):
            gr.Markdown("### Visualiza las activaciones de cada capa")
            
            with gr.Row():
                layer_dropdown = gr.Dropdown(
                    label="Seleccionar capa",
                    choices=['dec_0', 'dec_1', 'dec_2', 'dec_3', 'dec_4', 'dec_5',
                            'enc_0', 'enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5'],
                    value='dec_2'
                )
                refresh_btn = gr.Button("🔄 Actualizar", variant="primary")
            
            activation_output = gr.Image(label="Activaciones (64 canales)", height=600)
            
            with gr.Row():
                overview_btn = gr.Button("📊 Vista General (todas las capas)", variant="secondary")
            
            overview_output = gr.Image(label="Overview de Activaciones", height=400)
            
            refresh_btn.click(
                fn=explorer.get_activations,
                inputs=[layer_dropdown],
                outputs=[activation_output]
            )
            
            overview_btn.click(
                fn=explorer.get_all_activations_overview,
                outputs=[overview_output]
            )
        
        with gr.Tab("🔄 Interpolación"):
            gr.Markdown("### Interpola entre dos imágenes en el espacio latente")
            
            with gr.Row():
                interp_img1 = gr.Image(label="Imagen 1", type="numpy", height=256)
                interp_img2 = gr.Image(label="Imagen 2", type="numpy", height=256)
            
            interp_btn = gr.Button("🔄 Interpolar", variant="primary")
            interp_output = gr.Image(label="Interpolación", height=300)
            
            interp_btn.click(
                fn=explorer.create_interpolation_grid,
                inputs=[interp_img1, interp_img2],
                outputs=[interp_output]
            )
        
        with gr.Tab("🚶 Random Walk"):
            gr.Markdown("### Camina aleatoriamente por el espacio latente")
            
            with gr.Row():
                walk_steps = gr.Slider(4, 32, value=16, step=1, label="Pasos")
                walk_size = gr.Slider(0.1, 2.0, value=0.5, step=0.1, label="Tamaño de paso")
            
            walk_btn = gr.Button("🚶 Iniciar Walk", variant="primary")
            walk_output = gr.Image(label="Random Walk", height=500)
            
            walk_btn.click(
                fn=explorer.random_walk,
                inputs=[walk_steps, walk_size],
                outputs=[walk_output]
            )
        
        with gr.Tab("🖼️ Encode/Decode"):
            gr.Markdown("### Codifica una imagen y ve su reconstrucción")
            
            with gr.Row():
                encode_input = gr.Image(label="Imagen de entrada", type="numpy", height=300)
                encode_output = gr.Image(label="Reconstrucción", height=300)
            
            encode_btn = gr.Button("🔄 Encode + Decode", variant="primary")
            encode_info = gr.Textbox(label="Métricas", interactive=False)
            
            encode_btn.click(
                fn=explorer.encode_image,
                inputs=[encode_input],
                outputs=[encode_output, encode_info]
            )
        
        with gr.Tab("🌐 Espacio Latente 3D"):
            gr.Markdown("### Visualización 3D del espacio latente (PCA)")
            
            plot_btn = gr.Button("📊 Generar Plot 3D", variant="primary")
            plot_3d = gr.Plot(label="Latent Space 3D")
            
            plot_btn.click(
                fn=explorer.create_latent_3d,
                outputs=[plot_3d]
            )
        
        gr.Markdown("""
        ---
        ### 💡 Tips:
        - Carga primero tu modelo en la pestaña "Cargar Modelo"
        - Usa los sliders para navegar el espacio latente
        - Las activaciones muestran qué "ve" cada capa del decoder
        - La interpolación muestra transiciones suaves entre patrones
        """)
    
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive VAE Explorer")
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to load automatically')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port to run the server on')
    parser.add_argument('--share', action='store_true',
                       help='Create a public link')
    
    args = parser.parse_args()
    
    # Auto-load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        result = explorer.load_model(args.checkpoint)
        print(result)
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )




