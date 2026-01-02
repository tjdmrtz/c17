#!/usr/bin/env python3
"""
Script to evaluate and visualize a trained VAE.

This script loads a trained model and generates various visualizations:
- Reconstructions from test set
- Random samples from prior
- Latent space interpolations
- t-SNE visualization of latent space

Usage:
    python scripts/evaluate_vae.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import CrystallographicDataset
from src.dataset.pattern_generator import WALLPAPER_GROUPS
from src.models import CrystallographicVAE


def load_model(checkpoint_path: str, device: torch.device) -> CrystallographicVAE:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['model_config']
    model = CrystallographicVAE(
        in_channels=1,
        latent_dim=config['latent_dim'],
        input_size=config['input_size'],
        num_classes=config['num_classes']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Latent dim: {config['latent_dim']}")
    print(f"  Input size: {config['input_size']}")
    print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    return model


def generate_grid_samples(model: CrystallographicVAE,
                          device: torch.device,
                          num_samples: int = 64,
                          output_path: str = None):
    """Generate a grid of random samples."""
    with torch.no_grad():
        samples = model.sample(num_samples, device)
    
    rows = int(np.sqrt(num_samples))
    cols = num_samples // rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.patch.set_facecolor('#0f0f1a')
    
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(samples[i, 0].cpu().numpy(), cmap='magma')
        ax.axis('off')
    
    plt.suptitle('Generated Samples from Prior', color='#eaeaea', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, facecolor='#0f0f1a', bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def generate_reconstructions(model: CrystallographicVAE,
                             dataset: CrystallographicDataset,
                             device: torch.device,
                             num_per_group: int = 2,
                             output_path: str = None):
    """Generate reconstructions for each wallpaper group."""
    groups = list(WALLPAPER_GROUPS.keys())
    
    fig, axes = plt.subplots(len(groups), num_per_group * 2, 
                             figsize=(num_per_group * 4, len(groups) * 2))
    fig.patch.set_facecolor('#0f0f1a')
    
    for g_idx, group_name in enumerate(groups):
        # Get samples for this group
        group_samples = [i for i, (_, label) in enumerate(dataset.samples) 
                        if label == g_idx][:num_per_group]
        
        for s_idx, sample_idx in enumerate(group_samples):
            image, _ = dataset[sample_idx]
            image = image.unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(image)
                recon = outputs['reconstruction']
            
            # Original
            ax_orig = axes[g_idx, s_idx * 2]
            ax_orig.imshow(image[0, 0].cpu().numpy(), cmap='inferno')
            ax_orig.axis('off')
            if s_idx == 0:
                ax_orig.set_ylabel(group_name, color='#eaeaea', fontsize=10, rotation=0, 
                                  ha='right', va='center')
            
            # Reconstruction
            ax_recon = axes[g_idx, s_idx * 2 + 1]
            ax_recon.imshow(recon[0, 0].cpu().numpy(), cmap='inferno')
            ax_recon.axis('off')
    
    # Add column headers
    for i in range(num_per_group):
        axes[0, i * 2].set_title('Original', color='#4ECDC4', fontsize=10)
        axes[0, i * 2 + 1].set_title('Reconstructed', color='#FF6B6B', fontsize=10)
    
    plt.suptitle('Reconstructions by Wallpaper Group', color='#eaeaea', fontsize=16, y=1.01)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, facecolor='#0f0f1a', bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def generate_interpolation_grid(model: CrystallographicVAE,
                                dataset: CrystallographicDataset,
                                device: torch.device,
                                groups_to_interpolate: list = None,
                                steps: int = 8,
                                output_path: str = None):
    """Generate interpolations between different wallpaper groups."""
    if groups_to_interpolate is None:
        groups_to_interpolate = [('p1', 'p6m'), ('p4', 'p3'), ('pm', 'cmm')]
    
    group_to_idx = {name: idx for idx, name in enumerate(WALLPAPER_GROUPS.keys())}
    
    fig, axes = plt.subplots(len(groups_to_interpolate), steps,
                             figsize=(steps * 2, len(groups_to_interpolate) * 2))
    fig.patch.set_facecolor('#0f0f1a')
    
    for row, (g1, g2) in enumerate(groups_to_interpolate):
        # Get one sample from each group
        samples_g1 = [i for i, (_, l) in enumerate(dataset.samples) if l == group_to_idx[g1]]
        samples_g2 = [i for i, (_, l) in enumerate(dataset.samples) if l == group_to_idx[g2]]
        
        img1, _ = dataset[samples_g1[0]]
        img2, _ = dataset[samples_g2[0]]
        
        img1 = img1.unsqueeze(0).to(device)
        img2 = img2.unsqueeze(0).to(device)
        
        with torch.no_grad():
            interpolations = model.interpolate(img1, img2, steps=steps)
        
        for col in range(steps):
            ax = axes[row, col] if len(groups_to_interpolate) > 1 else axes[col]
            ax.imshow(interpolations[col, 0].cpu().numpy(), cmap='plasma')
            ax.axis('off')
            
            if col == 0:
                ax.set_title(g1, color='#4ECDC4', fontsize=10)
            elif col == steps - 1:
                ax.set_title(g2, color='#FF6B6B', fontsize=10)
    
    plt.suptitle('Latent Space Interpolation Between Groups', 
                color='#eaeaea', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, facecolor='#0f0f1a', bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def visualize_latent_space_2d(model: CrystallographicVAE,
                              dataset: CrystallographicDataset,
                              device: torch.device,
                              output_path: str = None,
                              method: str = 'tsne'):
    """Visualize latent space with dimensionality reduction."""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    # Encode all samples
    all_latents = []
    all_labels = []
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            mu, _ = model.encode(images)
            all_latents.append(mu.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    latents = np.concatenate(all_latents, axis=0)
    labels = np.array(all_labels)
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        title = 'Latent Space (t-SNE)'
    else:
        reducer = PCA(n_components=2)
        title = 'Latent Space (PCA)'
    
    print(f"Running {method.upper()}...")
    latents_2d = reducer.fit_transform(latents)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    
    # Create colormap for 17 groups
    colors = plt.cm.tab20(np.linspace(0, 1, 17))
    group_names = list(WALLPAPER_GROUPS.keys())
    
    for g_idx in range(17):
        mask = labels == g_idx
        ax.scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                  c=[colors[g_idx]], label=group_names[g_idx],
                  alpha=0.7, s=30, edgecolors='white', linewidth=0.3)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
             facecolor='#1a1a2e', edgecolor='#eaeaea', labelcolor='#eaeaea',
             ncol=1, fontsize=9)
    
    ax.set_title(title, color='#eaeaea', fontsize=16, pad=20)
    ax.tick_params(colors='#eaeaea')
    
    for spine in ax.spines.values():
        spine.set_color('#3d3d5c')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, facecolor='#0f0f1a', bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained VAE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', '-o', type=str, default='./output/evaluation',
                       help='Output directory')
    parser.add_argument('--samples-per-group', '-n', type=int, default=50,
                       help='Samples per group for evaluation')
    parser.add_argument('--resolution', '-r', type=int, default=128,
                       help='Image resolution')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    # Create evaluation dataset
    print("\nCreating evaluation dataset...")
    dataset = CrystallographicDataset(
        num_samples_per_group=args.samples_per_group,
        resolution=args.resolution,
        seed=12345  # Different seed for evaluation
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Random samples
    generate_grid_samples(
        model, device, num_samples=64,
        output_path=str(output_dir / 'random_samples.png')
    )
    
    # Reconstructions
    generate_reconstructions(
        model, dataset, device, num_per_group=2,
        output_path=str(output_dir / 'reconstructions.png')
    )
    
    # Interpolations
    generate_interpolation_grid(
        model, dataset, device,
        groups_to_interpolate=[('p1', 'p6m'), ('p4', 'p3'), ('pm', 'cmm'), ('p2', 'p4m')],
        steps=10,
        output_path=str(output_dir / 'interpolations.png')
    )
    
    # Latent space visualization
    try:
        visualize_latent_space_2d(
            model, dataset, device, method='tsne',
            output_path=str(output_dir / 'latent_space_tsne.png')
        )
        
        visualize_latent_space_2d(
            model, dataset, device, method='pca',
            output_path=str(output_dir / 'latent_space_pca.png')
        )
    except ImportError:
        print("sklearn not available, skipping latent space visualization")
    
    print(f"\nAll evaluations saved to: {output_dir}")


if __name__ == "__main__":
    main()








