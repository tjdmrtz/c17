"""
Latent Space Explorer for Latent Diffusion Models.

Interactive tool to navigate the latent space and visualize:
- Generated images at different latent positions
- Decoder activations at each layer
- Interpolations between points
- Random walks in latent space

Usage:
    from src.visualization.latent_explorer import LatentExplorer
    
    explorer = LatentExplorer(model, device='cuda')
    
    # Explore random point
    explorer.explore_point(z=None)  # Random
    
    # Interpolate between two images  
    explorer.interpolate(img1, img2, steps=10)
    
    # Random walk
    explorer.random_walk(steps=20, step_size=0.5)
    
    # Interactive exploration (Jupyter)
    explorer.interactive()
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass 
class ExplorationResult:
    """Result of a latent space exploration."""
    latent: torch.Tensor
    image: torch.Tensor
    activations: Dict[str, torch.Tensor]


class LatentExplorer:
    """
    Interactive latent space explorer with activation visualization.
    
    Features:
    - Navigate latent space and see generated images
    - Visualize decoder activations as images
    - Interpolate between any two points
    - Random walks through latent space
    - Export videos/GIFs of explorations
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize explorer.
        
        Args:
            model: Trained model with decode() and enable_decoder_activation_capture() methods
            device: Device for inference
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Determine latent shape
        if hasattr(model, 'config'):
            self.latent_channels = model.config.latent_channels
            self.latent_size = model.config.latent_size
        else:
            # Default for standard VAE
            self.latent_channels = 256
            self.latent_size = 1
        
        # Current position in latent space
        self.current_z = None
        self.history = []
    
    def _get_random_latent(self) -> torch.Tensor:
        """Get a random point in latent space."""
        if self.latent_size > 1:
            # Spatial latent (like LDM)
            return torch.randn(1, self.latent_channels, 
                             self.latent_size, self.latent_size, device=self.device)
        else:
            # Vector latent (like standard VAE)
            return torch.randn(1, self.latent_channels, device=self.device)
    
    def _decode_with_activations(self, z: torch.Tensor) -> ExplorationResult:
        """Decode latent and capture activations."""
        # Enable activation capture
        if hasattr(self.model, 'enable_decoder_activation_capture'):
            self.model.enable_decoder_activation_capture()
        elif hasattr(self.model, 'vae'):
            self.model.vae.enable_decoder_activation_capture()
        
        with torch.no_grad():
            if hasattr(self.model, 'decode'):
                image = self.model.decode(z)
            elif hasattr(self.model, 'decoder'):
                image = self.model.decoder(z)
            else:
                raise ValueError("Model must have decode() or decoder attribute")
        
        # Get activations
        if hasattr(self.model, 'get_decoder_activations'):
            activations = self.model.get_decoder_activations()
        elif hasattr(self.model, 'vae'):
            activations = self.model.vae.get_decoder_activations()
        else:
            activations = {}
        
        # Disable capture
        if hasattr(self.model, 'disable_decoder_activation_capture'):
            self.model.disable_decoder_activation_capture()
        elif hasattr(self.model, 'vae'):
            self.model.vae.disable_decoder_activation_capture()
        
        return ExplorationResult(
            latent=z.cpu(),
            image=image.cpu(),
            activations=activations
        )
    
    def explore_point(self, z: Optional[torch.Tensor] = None,
                      save_path: Optional[str] = None,
                      show_activations: bool = True) -> ExplorationResult:
        """
        Explore a single point in latent space.
        
        Args:
            z: Latent vector (random if None)
            save_path: Optional path to save visualization
            show_activations: Whether to show decoder activations
        
        Returns:
            ExplorationResult with image and activations
        """
        if z is None:
            z = self._get_random_latent()
        else:
            z = z.to(self.device)
        
        result = self._decode_with_activations(z)
        self.current_z = z
        self.history.append(result)
        
        # Visualize
        if show_activations and result.activations:
            self._visualize_with_activations(result, save_path)
        else:
            self._visualize_simple(result, save_path)
        
        return result
    
    def _visualize_simple(self, result: ExplorationResult, 
                          save_path: Optional[str] = None):
        """Simple visualization with just the generated image."""
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0f0f1a')
        
        img = result.image[0].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title('Generated Image', color='white', fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
        plt.show()
        plt.close()
    
    def _visualize_with_activations(self, result: ExplorationResult,
                                     save_path: Optional[str] = None):
        """Visualize generated image with decoder activations."""
        n_activations = len(result.activations)
        
        fig = plt.figure(figsize=(16, 8), facecolor='#0f0f1a')
        
        # Layout: Image on left, activations on right
        gs = GridSpec(2, n_activations + 1, figure=fig, width_ratios=[2] + [1]*n_activations)
        
        # Generated image (left, spans 2 rows)
        ax_img = fig.add_subplot(gs[:, 0])
        img = result.image[0].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax_img.imshow(img)
        ax_img.set_title('Generated Image', color='white', fontsize=12)
        ax_img.axis('off')
        
        # Activations (right side)
        for i, (name, act) in enumerate(result.activations.items()):
            if act.dim() == 4:
                act = act[0]  # Remove batch dim
            
            # Show mean activation and first channel
            ax_mean = fig.add_subplot(gs[0, i + 1])
            ax_ch = fig.add_subplot(gs[1, i + 1])
            
            # Mean across channels
            act_mean = act.mean(dim=0).numpy()
            act_mean = (act_mean - act_mean.min()) / (act_mean.max() - act_mean.min() + 1e-8)
            ax_mean.imshow(act_mean, cmap='viridis')
            ax_mean.set_title(f'{name}\nmean', color='white', fontsize=9)
            ax_mean.axis('off')
            
            # First channel
            act_ch0 = act[0].numpy()
            act_ch0 = (act_ch0 - act_ch0.min()) / (act_ch0.max() - act_ch0.min() + 1e-8)
            ax_ch.imshow(act_ch0, cmap='plasma')
            ax_ch.set_title(f'ch[0]\n{tuple(act.shape)}', color='white', fontsize=8)
            ax_ch.axis('off')
        
        plt.suptitle('Latent Space Exploration', color='white', fontsize=14, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
        plt.show()
        plt.close()
    
    def interpolate(self, z1: torch.Tensor, z2: torch.Tensor,
                    steps: int = 10, 
                    save_path: Optional[str] = None) -> List[ExplorationResult]:
        """
        Interpolate between two latent points.
        
        Args:
            z1: Starting latent
            z2: Ending latent
            steps: Number of interpolation steps
            save_path: Optional path to save visualization
        
        Returns:
            List of ExplorationResults along the path
        """
        z1 = z1.to(self.device)
        z2 = z2.to(self.device)
        
        results = []
        alphas = torch.linspace(0, 1, steps)
        
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            result = self._decode_with_activations(z)
            results.append(result)
        
        # Visualize
        self._visualize_interpolation(results, save_path)
        
        return results
    
    def interpolate_images(self, img1: torch.Tensor, img2: torch.Tensor,
                           steps: int = 10,
                           save_path: Optional[str] = None) -> List[ExplorationResult]:
        """
        Interpolate between two images (encodes them first).
        
        Args:
            img1: First image tensor
            img2: Second image tensor
            steps: Number of interpolation steps
            save_path: Optional path to save
        
        Returns:
            List of ExplorationResults
        """
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        # Encode images to get latents
        with torch.no_grad():
            if hasattr(self.model, 'encode'):
                z1 = self.model.encode(img1)
                z2 = self.model.encode(img2)
            elif hasattr(self.model, 'encoder'):
                mu1, _ = self.model.encoder(img1)
                mu2, _ = self.model.encoder(img2)
                z1, z2 = mu1, mu2
            else:
                raise ValueError("Model must have encode() or encoder attribute")
        
        return self.interpolate(z1, z2, steps, save_path)
    
    def _visualize_interpolation(self, results: List[ExplorationResult],
                                  save_path: Optional[str] = None):
        """Visualize interpolation results."""
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(2 * n, 2), facecolor='#0f0f1a')
        
        for i, result in enumerate(results):
            img = result.image[0].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[i].imshow(img)
            axes[i].axis('off')
        
        plt.suptitle('Latent Space Interpolation', color='white', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
        plt.show()
        plt.close()
    
    def random_walk(self, steps: int = 20, step_size: float = 0.3,
                    start_z: Optional[torch.Tensor] = None,
                    save_path: Optional[str] = None) -> List[ExplorationResult]:
        """
        Random walk through latent space.
        
        Args:
            steps: Number of steps
            step_size: Size of each random step
            start_z: Starting point (random if None)
            save_path: Optional path to save visualization
        
        Returns:
            List of ExplorationResults along the walk
        """
        if start_z is None:
            z = self._get_random_latent()
        else:
            z = start_z.to(self.device)
        
        results = []
        
        for _ in range(steps):
            result = self._decode_with_activations(z)
            results.append(result)
            
            # Random step
            noise = torch.randn_like(z) * step_size
            z = z + noise
        
        self._visualize_walk(results, save_path)
        
        return results
    
    def _visualize_walk(self, results: List[ExplorationResult],
                        save_path: Optional[str] = None):
        """Visualize random walk."""
        n = len(results)
        cols = min(10, n)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows), facecolor='#0f0f1a')
        axes = axes.flatten() if n > 1 else [axes]
        
        for i, result in enumerate(results):
            img = result.image[0].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[i].imshow(img)
            axes[i].set_title(f'Step {i}', color='white', fontsize=8)
            axes[i].axis('off')
        
        # Hide unused
        for i in range(n, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Random Walk in Latent Space', color='white', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
        plt.show()
        plt.close()
    
    def explore_dimension(self, dim: int, 
                          range_vals: Tuple[float, float] = (-3, 3),
                          steps: int = 10,
                          base_z: Optional[torch.Tensor] = None,
                          save_path: Optional[str] = None) -> List[ExplorationResult]:
        """
        Explore a single dimension of the latent space.
        
        Args:
            dim: Dimension index to vary
            range_vals: Range of values to explore
            steps: Number of steps
            base_z: Base latent (zeros if None)
            save_path: Optional path to save
        
        Returns:
            List of ExplorationResults
        """
        if base_z is None:
            base_z = torch.zeros(1, self.latent_channels, device=self.device)
            if self.latent_size > 1:
                base_z = base_z.unsqueeze(-1).unsqueeze(-1).expand(
                    -1, -1, self.latent_size, self.latent_size
                )
        else:
            base_z = base_z.to(self.device)
        
        results = []
        values = torch.linspace(range_vals[0], range_vals[1], steps)
        
        for val in values:
            z = base_z.clone()
            if z.dim() == 2:
                z[0, dim] = val
            else:
                z[0, dim, :, :] = val
            
            result = self._decode_with_activations(z)
            results.append(result)
        
        self._visualize_dimension(results, dim, values, save_path)
        
        return results
    
    def _visualize_dimension(self, results: List[ExplorationResult],
                              dim: int, values: torch.Tensor,
                              save_path: Optional[str] = None):
        """Visualize dimension exploration."""
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.5), facecolor='#0f0f1a')
        
        for i, (result, val) in enumerate(zip(results, values)):
            img = result.image[0].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[i].imshow(img)
            axes[i].set_title(f'{val:.1f}', color='white', fontsize=10)
            axes[i].axis('off')
        
        plt.suptitle(f'Latent Dimension {dim}', color='white', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
        plt.show()
        plt.close()
    
    def activation_grid(self, z: Optional[torch.Tensor] = None,
                        layer: str = 'dec_up2',
                        num_channels: int = 64,
                        save_path: Optional[str] = None) -> torch.Tensor:
        """
        Show grid of activation channels for a specific layer.
        
        Args:
            z: Latent to decode (random if None)
            layer: Layer name to visualize
            num_channels: Number of channels to show
            save_path: Optional save path
        
        Returns:
            Activation tensor
        """
        if z is None:
            z = self._get_random_latent()
        
        result = self._decode_with_activations(z.to(self.device))
        
        if layer not in result.activations:
            print(f"Layer '{layer}' not found. Available: {list(result.activations.keys())}")
            return None
        
        act = result.activations[layer]
        if act.dim() == 4:
            act = act[0]
        
        num_channels = min(num_channels, act.shape[0])
        cols = int(np.ceil(np.sqrt(num_channels)))
        rows = int(np.ceil(num_channels / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5), facecolor='#0f0f1a')
        axes = axes.flatten()
        
        for i in range(num_channels):
            ch = act[i].numpy()
            ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
            axes[i].imshow(ch, cmap='viridis')
            axes[i].axis('off')
        
        for i in range(num_channels, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Activations: {layer} ({act.shape[0]} ch, {act.shape[1]}×{act.shape[2]})',
                    color='white', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
        plt.show()
        plt.close()
        
        return act
    
    def save_video(self, results: List[ExplorationResult], 
                   output_path: str, fps: int = 10):
        """
        Save exploration results as video.
        
        Args:
            results: List of ExplorationResults
            output_path: Path to save video (mp4)
            fps: Frames per second
        """
        try:
            import imageio
        except ImportError:
            print("Install imageio: pip install imageio[ffmpeg]")
            return
        
        frames = []
        for result in results:
            img = result.image[0].permute(1, 2, 0).numpy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            frames.append(img)
        
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Saved video to {output_path}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_exploration_report(model: nn.Module, 
                               output_dir: str,
                               device: str = 'cuda',
                               num_samples: int = 9) -> str:
    """
    Create a comprehensive exploration report.
    
    Args:
        model: Trained model
        output_dir: Output directory
        device: Device
        num_samples: Number of random samples
    
    Returns:
        Path to output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    explorer = LatentExplorer(model, device)
    
    # Random samples
    print("Generating random samples...")
    for i in range(num_samples):
        explorer.explore_point(
            save_path=output_dir / f'random_sample_{i:02d}.png',
            show_activations=True
        )
    
    # Random walk
    print("Generating random walk...")
    walk_results = explorer.random_walk(
        steps=20, step_size=0.3,
        save_path=output_dir / 'random_walk.png'
    )
    
    # Dimension exploration (first 5 dims)
    print("Exploring dimensions...")
    for dim in range(min(5, explorer.latent_channels)):
        explorer.explore_dimension(
            dim=dim, steps=7,
            save_path=output_dir / f'dimension_{dim:02d}.png'
        )
    
    print(f"Report saved to {output_dir}")
    return str(output_dir)


if __name__ == "__main__":
    print("Latent Explorer - Test Mode")
    print("=" * 50)
    print("\nThis module requires a trained model to explore.")
    print("Usage example:")
    print("""
    from src.models.latent_diffusion import LatentDiffusionModel
    from src.visualization.latent_explorer import LatentExplorer
    
    # Load trained model
    model = LatentDiffusionModel()
    model.load_state_dict(torch.load('checkpoint.pt'))
    
    # Create explorer
    explorer = LatentExplorer(model, device='cuda')
    
    # Explore!
    explorer.explore_point()  # Random sample with activations
    explorer.random_walk(steps=20)  # Walk through latent space
    explorer.interpolate_images(img1, img2)  # Interpolate between images
    """)






