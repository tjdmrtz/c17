"""
Activation Visualization Tool for VAE Models.

Allows interactive visualization of intermediate layer activations as images.
Useful for understanding what features the network learns at each layer.

Usage:
    from src.visualization.activation_viewer import ActivationViewer
    
    viewer = ActivationViewer(model, device='cuda')
    viewer.visualize_single_image(image_tensor, save_path='activations.png')
    viewer.create_activation_grid(image_tensor, layer_name='enc_2')
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings


class ActivationViewer:
    """
    Tool for visualizing intermediate activations of VAE models.
    
    Features:
    - Visualize all encoder/decoder layer activations
    - Create grids of feature maps for specific layers
    - Compare activations across different inputs
    - Export activations as images
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize the activation viewer.
        
        Args:
            model: VAE model with activation capture capability
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Check if model supports activation capture
        if not hasattr(model, 'enable_activation_capture'):
            raise ValueError("Model must have activation capture capability. "
                           "Use ConfigurableVAE or RotationEquivariantVAE.")
    
    def _get_activations(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run forward pass and capture activations."""
        self.model.enable_activation_capture()
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            _ = self.model(image)
        
        activations = self.model.get_activations()
        self.model.disable_activation_capture()
        
        return activations
    
    def get_layer_names(self) -> Dict[str, List[str]]:
        """Get available layer names for visualization."""
        return self.model.get_layer_names()
    
    def visualize_single_image(self, 
                                image: torch.Tensor,
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (20, 12),
                                cmap: str = 'viridis') -> plt.Figure:
        """
        Visualize activations for all layers from a single input image.
        
        Args:
            image: Input tensor (C, H, W) or (1, C, H, W)
            save_path: Optional path to save the figure
            figsize: Figure size
            cmap: Colormap for activation visualization
        
        Returns:
            matplotlib Figure object
        """
        activations = self._get_activations(image)
        
        if not activations:
            warnings.warn("No activations captured!")
            return None
        
        # Prepare input image for display
        if image.dim() == 4:
            image = image[0]
        input_img = image.cpu().permute(1, 2, 0).numpy()
        input_img = np.clip(input_img, 0, 1)
        
        # Create figure
        n_layers = len(activations) + 1  # +1 for input
        n_cols = min(6, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=figsize, facecolor='#0f0f1a')
        gs = GridSpec(n_rows, n_cols, figure=fig)
        
        # Plot input image
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(input_img)
        ax.set_title('Input\n(3, 512, 512)', color='white', fontsize=10)
        ax.axis('off')
        
        # Plot activations
        layer_names = list(activations.keys())
        for idx, name in enumerate(layer_names):
            row = (idx + 1) // n_cols
            col = (idx + 1) % n_cols
            
            ax = fig.add_subplot(gs[row, col])
            
            act = activations[name]
            if act.dim() == 4:
                act = act[0]  # Take first sample
            
            # Average over channels for visualization
            act_mean = act.mean(dim=0).numpy()
            
            # Normalize
            act_mean = (act_mean - act_mean.min()) / (act_mean.max() - act_mean.min() + 1e-8)
            
            ax.imshow(act_mean, cmap=cmap)
            shape_str = f"({act.shape[0]}, {act.shape[1]}, {act.shape[2]})"
            ax.set_title(f'{name}\n{shape_str}', color='white', fontsize=9)
            ax.axis('off')
        
        plt.suptitle('Layer Activations', color='white', fontsize=14, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        return fig
    
    def create_activation_grid(self,
                               image: torch.Tensor,
                               layer_name: str,
                               num_channels: int = 64,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (16, 16),
                               cmap: str = 'viridis') -> plt.Figure:
        """
        Create a grid showing individual feature maps from a specific layer.
        
        Args:
            image: Input tensor
            layer_name: Name of the layer to visualize
            num_channels: Number of channels to display
            save_path: Optional path to save the figure
            figsize: Figure size
            cmap: Colormap
        
        Returns:
            matplotlib Figure object
        """
        activations = self._get_activations(image)
        
        if layer_name not in activations:
            available = list(activations.keys())
            raise ValueError(f"Layer '{layer_name}' not found. Available: {available}")
        
        act = activations[layer_name]
        if act.dim() == 4:
            act = act[0]
        
        num_channels = min(num_channels, act.shape[0])
        n_cols = int(np.ceil(np.sqrt(num_channels)))
        n_rows = int(np.ceil(num_channels / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, facecolor='#0f0f1a')
        axes = np.array(axes).flatten()
        
        for i in range(num_channels):
            ax = axes[i]
            
            channel_act = act[i].numpy()
            channel_act = (channel_act - channel_act.min()) / (channel_act.max() - channel_act.min() + 1e-8)
            
            ax.imshow(channel_act, cmap=cmap)
            ax.set_title(f'Ch {i}', color='white', fontsize=8)
            ax.axis('off')
        
        # Hide unused axes
        for i in range(num_channels, len(axes)):
            axes[i].axis('off')
            axes[i].set_visible(False)
        
        plt.suptitle(f'Feature Maps: {layer_name} ({act.shape[0]} channels, {act.shape[1]}×{act.shape[2]})',
                    color='white', fontsize=14, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        return fig
    
    def compare_inputs(self,
                       images: List[torch.Tensor],
                       labels: Optional[List[str]] = None,
                       layer_name: str = 'enc_2',
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (16, 8)) -> plt.Figure:
        """
        Compare activations from multiple input images side by side.
        
        Args:
            images: List of input tensors
            labels: Optional labels for each image
            layer_name: Layer to compare
            save_path: Optional save path
            figsize: Figure size
        
        Returns:
            matplotlib Figure object
        """
        n_images = len(images)
        if labels is None:
            labels = [f'Image {i+1}' for i in range(n_images)]
        
        fig, axes = plt.subplots(2, n_images, figsize=figsize, facecolor='#0f0f1a')
        
        for i, (image, label) in enumerate(zip(images, labels)):
            activations = self._get_activations(image)
            
            # Show input
            if image.dim() == 4:
                image = image[0]
            input_img = image.cpu().permute(1, 2, 0).numpy()
            input_img = np.clip(input_img, 0, 1)
            
            axes[0, i].imshow(input_img)
            axes[0, i].set_title(label, color='white', fontsize=12)
            axes[0, i].axis('off')
            
            # Show activation
            if layer_name in activations:
                act = activations[layer_name]
                if act.dim() == 4:
                    act = act[0]
                act_mean = act.mean(dim=0).numpy()
                act_mean = (act_mean - act_mean.min()) / (act_mean.max() - act_mean.min() + 1e-8)
                
                axes[1, i].imshow(act_mean, cmap='viridis')
                axes[1, i].set_title(f'{layer_name}', color='white', fontsize=10)
            axes[1, i].axis('off')
        
        plt.suptitle(f'Activation Comparison: {layer_name}', color='white', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        return fig
    
    def save_all_activations(self,
                             image: torch.Tensor,
                             output_dir: str,
                             prefix: str = 'activation') -> Dict[str, str]:
        """
        Save each layer's activation as a separate image file.
        
        Args:
            image: Input tensor
            output_dir: Directory to save images
            prefix: Filename prefix
        
        Returns:
            Dictionary mapping layer names to saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        activations = self._get_activations(image)
        saved_files = {}
        
        for name, act in activations.items():
            if act.dim() == 4:
                act = act[0]
            
            # Save mean activation
            act_mean = act.mean(dim=0).numpy()
            act_mean = (act_mean - act_mean.min()) / (act_mean.max() - act_mean.min() + 1e-8)
            
            filepath = output_dir / f'{prefix}_{name}.png'
            
            plt.figure(figsize=(8, 8), facecolor='#0f0f1a')
            plt.imshow(act_mean, cmap='viridis')
            plt.title(f'{name}: {tuple(act.shape)}', color='white')
            plt.axis('off')
            plt.savefig(filepath, dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
            plt.close()
            
            saved_files[name] = str(filepath)
        
        print(f"Saved {len(saved_files)} activation images to {output_dir}")
        return saved_files
    
    def get_activation_stats(self, image: torch.Tensor) -> Dict[str, Dict]:
        """
        Get statistics for each layer's activations.
        
        Returns:
            Dictionary with stats (mean, std, min, max, sparsity) per layer
        """
        activations = self._get_activations(image)
        stats = {}
        
        for name, act in activations.items():
            act_flat = act.flatten().numpy()
            
            stats[name] = {
                'shape': tuple(act.shape),
                'mean': float(np.mean(act_flat)),
                'std': float(np.std(act_flat)),
                'min': float(np.min(act_flat)),
                'max': float(np.max(act_flat)),
                'sparsity': float((np.abs(act_flat) < 0.01).mean()),  # % of near-zero values
            }
        
        return stats


def create_activation_report(model: nn.Module,
                            dataloader: torch.utils.data.DataLoader,
                            output_dir: str,
                            device: str = 'cuda',
                            num_samples: int = 5) -> str:
    """
    Create a comprehensive activation report for a model.
    
    Args:
        model: VAE model
        dataloader: DataLoader with sample images
        output_dir: Directory for output files
        device: Device for inference
        num_samples: Number of samples to analyze
    
    Returns:
        Path to the report HTML file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    viewer = ActivationViewer(model, device)
    
    # Get samples
    samples = []
    labels = []
    for images, lbls in dataloader:
        for i in range(min(len(images), num_samples - len(samples))):
            samples.append(images[i])
            labels.append(f"Sample {len(samples)}")
        if len(samples) >= num_samples:
            break
    
    # Generate visualizations
    for i, sample in enumerate(samples):
        viewer.visualize_single_image(
            sample,
            save_path=output_dir / f'sample_{i}_all_layers.png'
        )
        
        # Get layer names and create grid for middle encoder layer
        layer_names = viewer.get_layer_names()
        enc_layers = layer_names.get('encoder', [])
        if enc_layers:
            mid_layer = enc_layers[len(enc_layers) // 2]
            viewer.create_activation_grid(
                sample,
                layer_name=mid_layer,
                num_channels=32,
                save_path=output_dir / f'sample_{i}_{mid_layer}_grid.png'
            )
    
    # Compare samples
    if len(samples) >= 2:
        viewer.compare_inputs(
            samples[:min(4, len(samples))],
            labels[:min(4, len(samples))],
            save_path=output_dir / 'comparison.png'
        )
    
    print(f"Activation report saved to {output_dir}")
    return str(output_dir)


if __name__ == "__main__":
    # Demo with synthetic data
    print("Activation Viewer Demo")
    print("=" * 50)
    
    # Import model
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.models.vae_equivariant import ConfigurableVAE
    
    # Create model
    model = ConfigurableVAE(
        latent_dim=256,
        base_channels=32,
        num_encoder_layers=5,
        num_decoder_layers=6,
        capture_activations=True
    )
    
    # Create viewer
    viewer = ActivationViewer(model, device='cpu')
    
    print(f"Available layers: {viewer.get_layer_names()}")
    
    # Test with random image
    test_image = torch.rand(1, 3, 512, 512)
    
    # Get stats
    stats = viewer.get_activation_stats(test_image)
    print("\nActivation statistics:")
    for name, s in stats.items():
        print(f"  {name}: shape={s['shape']}, mean={s['mean']:.3f}, sparsity={s['sparsity']:.1%}")
    
    print("\n✅ Activation viewer ready!")
    print("Use viewer.visualize_single_image(image) to visualize")





