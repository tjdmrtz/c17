#!/usr/bin/env python3
"""
Generate Random Samples from Latent Space for All 17 Wallpaper Groups

This script generates a gallery image showing random samples from the latent space
decoded with each of the 17 crystallographic wallpaper groups.

Usage:
    python scripts/generate_all_17_samples.py --checkpoint output/symmetry_vae_17groups_XXXXXX/best_model.pt
    python scripts/generate_all_17_samples.py  # Uses most recent checkpoint
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import datetime

from src.models.symmetry_invariant_vae import (
    SymmetryInvariantVAE,
    SymmetryVAEConfig
)


# All 17 wallpaper groups organized by lattice type
ALL_17_GROUPS = [
    'p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm',
    'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m'
]

# Lattice type info and colors
LATTICE_INFO = {
    'p1': ('Oblique', '#1a5276'), 'p2': ('Oblique', '#2980b9'),
    'pm': ('Rectangular', '#27ae60'), 'pg': ('Rectangular', '#2ecc71'),
    'cm': ('Rectangular', '#58d68d'), 'pmm': ('Rectangular', '#145a32'),
    'pmg': ('Rectangular', '#196f3d'), 'pgg': ('Rectangular', '#1d8348'),
    'cmm': ('Rectangular', '#239b56'),
    'p4': ('Square', '#6c3483'), 'p4m': ('Square', '#8e44ad'),
    'p4g': ('Square', '#a569bd'),
    'p3': ('Hexagonal', '#b9770e'), 'p3m1': ('Hexagonal', '#d68910'),
    'p31m': ('Hexagonal', '#f39c12'), 'p6': ('Hexagonal', '#c0392b'),
    'p6m': ('Hexagonal', '#e74c3c'),
}

# Colormaps by lattice type
LATTICE_CMAPS = {
    'Oblique': 'Blues',
    'Rectangular': 'Greens',
    'Square': 'Purples',
    'Hexagonal': 'YlOrRd'
}


def find_latest_checkpoint() -> Path:
    """Find the most recent checkpoint in output directory."""
    output_dir = Path("output")
    
    # Look for symmetry_vae directories
    vae_dirs = sorted(output_dir.glob("symmetry_vae_17groups_*"), reverse=True)
    
    for vae_dir in vae_dirs:
        checkpoint = vae_dir / "best_model.pt"
        if checkpoint.exists():
            return checkpoint
    
    raise FileNotFoundError("No checkpoint found. Please train a model first or specify --checkpoint")


def load_model(checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint or use default
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = SymmetryVAEConfig(
            resolution=128,
            latent_dim=2,
            hidden_dims=[64, 128, 256, 512]
        )
    
    model = SymmetryInvariantVAE(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded (latent_dim={config.latent_dim}, resolution={config.resolution})")
    return model, config


def generate_gallery_single_z(model, device, save_path, seed=None):
    """
    Generate a gallery showing the same latent point decoded with all 17 groups.
    
    This demonstrates how the same latent code produces different symmetries.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    model.eval()
    
    # Generate a random latent point
    z = torch.randn(1, model.config.latent_dim, device=device) * 0.8
    
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor('#0a0a0a')
    
    # Create grid: 4 rows x 5 cols (17 groups + title info)
    gs = GridSpec(4, 5, figure=fig, hspace=0.25, wspace=0.15,
                  left=0.02, right=0.98, top=0.88, bottom=0.02)
    
    with torch.no_grad():
        for idx, group in enumerate(ALL_17_GROUPS):
            row = idx // 5
            col = idx % 5
            
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor('#0a0a0a')
            
            # Decode with this group's symmetry
            sample = model.decode(z, group)
            
            lattice, color = LATTICE_INFO[group]
            cmap = LATTICE_CMAPS[lattice]
            
            ax.imshow(sample[0, 0].cpu().numpy(), cmap=cmap)
            ax.set_title(f'{group}\n({lattice})', fontsize=11, color='white',
                        fontweight='bold', pad=5)
            ax.axis('off')
            
            # Add colored border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(3)
    
    # Hide remaining axes
    for idx in range(17, 20):
        row = idx // 5
        col = idx % 5
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
    
    # Latent point info
    z_str = ", ".join([f"{v:.2f}" for v in z[0].cpu().numpy()])
    
    fig.suptitle(
        f'All 17 Wallpaper Groups from Single Latent Point\n'
        f'z = [{z_str}]',
        fontsize=18, color='white', fontweight='bold', y=0.96
    )
    
    plt.savefig(save_path, dpi=150, facecolor='#0a0a0a',
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved single-z gallery to {save_path}")


def generate_gallery_multiple_samples(model, device, save_path, n_samples=4, seed=None):
    """
    Generate a gallery with multiple random samples for each of the 17 groups.
    
    Each row shows multiple random samples from the same group.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    model.eval()
    
    fig = plt.figure(figsize=(n_samples * 3 + 2, 17 * 1.5 + 2))
    fig.patch.set_facecolor('#0a0a0a')
    
    # Create grid: 17 rows (groups) x n_samples cols
    gs = GridSpec(17, n_samples + 1, figure=fig, hspace=0.08, wspace=0.05,
                  left=0.08, right=0.98, top=0.95, bottom=0.02,
                  width_ratios=[0.5] + [1] * n_samples)
    
    with torch.no_grad():
        for group_idx, group in enumerate(ALL_17_GROUPS):
            lattice, color = LATTICE_INFO[group]
            cmap = LATTICE_CMAPS[lattice]
            
            # Label axis
            ax_label = fig.add_subplot(gs[group_idx, 0])
            ax_label.set_facecolor('#0a0a0a')
            ax_label.text(0.9, 0.5, group, fontsize=14, color=color,
                         fontweight='bold', ha='right', va='center',
                         transform=ax_label.transAxes)
            ax_label.axis('off')
            
            # Generate random samples
            z = torch.randn(n_samples, model.config.latent_dim, device=device) * 1.0
            samples = model.decode(z, group)
            
            for sample_idx in range(n_samples):
                ax = fig.add_subplot(gs[group_idx, sample_idx + 1])
                ax.set_facecolor('#0a0a0a')
                
                ax.imshow(samples[sample_idx, 0].cpu().numpy(), cmap=cmap)
                ax.axis('off')
                
                # Add subtle border
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(color)
                    spine.set_linewidth(1.5)
    
    fig.suptitle(
        'Random Samples from Latent Space - All 17 Wallpaper Groups',
        fontsize=20, color='white', fontweight='bold', y=0.98
    )
    
    plt.savefig(save_path, dpi=150, facecolor='#0a0a0a',
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved multi-sample gallery to {save_path}")


def generate_gallery_grid(model, device, save_path, samples_per_group=6, seed=None):
    """
    Generate a comprehensive grid gallery: 17 groups x N samples each.
    
    This creates a compact view showing variety within each symmetry group.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    model.eval()
    
    # Calculate grid dimensions
    n_cols_samples = samples_per_group
    n_cols_total = n_cols_samples
    n_rows = 17
    
    cell_size = 1.8
    fig_width = n_cols_total * cell_size + 2
    fig_height = n_rows * cell_size + 2
    
    fig, axes = plt.subplots(n_rows, n_cols_total + 1, 
                              figsize=(fig_width, fig_height),
                              gridspec_kw={'width_ratios': [0.6] + [1] * n_cols_total})
    fig.patch.set_facecolor('#0a0a0a')
    
    with torch.no_grad():
        for group_idx, group in enumerate(ALL_17_GROUPS):
            lattice, color = LATTICE_INFO[group]
            cmap = LATTICE_CMAPS[lattice]
            
            # Group label
            ax_label = axes[group_idx, 0]
            ax_label.set_facecolor('#0a0a0a')
            ax_label.text(0.95, 0.5, f'{group}', fontsize=12, color=color,
                         fontweight='bold', ha='right', va='center',
                         transform=ax_label.transAxes)
            ax_label.axis('off')
            
            # Generate samples
            z = torch.randn(samples_per_group, model.config.latent_dim, device=device)
            samples = model.decode(z, group)
            
            for s_idx in range(samples_per_group):
                ax = axes[group_idx, s_idx + 1]
                ax.set_facecolor('#0a0a0a')
                ax.imshow(samples[s_idx, 0].cpu().numpy(), cmap=cmap)
                ax.axis('off')
                
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(color)
                    spine.set_linewidth(1)
    
    fig.suptitle(
        'Random Samples from Latent Space\nAll 17 Wallpaper Groups',
        fontsize=18, color='white', fontweight='bold', y=0.995
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, left=0.05, right=0.98, bottom=0.01, hspace=0.05, wspace=0.02)
    
    plt.savefig(save_path, dpi=150, facecolor='#0a0a0a',
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved grid gallery to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate samples for all 17 wallpaper groups')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (default: find latest)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: same as checkpoint)')
    parser.add_argument('--samples', type=int, default=6,
                       help='Number of samples per group (default: 6)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['single', 'multi', 'grid', 'all'],
                       help='Gallery mode: single (same z), multi (row per group), grid, or all')
    
    args = parser.parse_args()
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_latest_checkpoint()
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(checkpoint_path, device)
    
    # Generate galleries
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.mode in ['single', 'all']:
        save_path = output_dir / f'samples_single_z_{timestamp}.png'
        generate_gallery_single_z(model, device, save_path, seed=args.seed)
    
    if args.mode in ['multi', 'all']:
        save_path = output_dir / f'samples_multi_{timestamp}.png'
        generate_gallery_multiple_samples(model, device, save_path, 
                                          n_samples=args.samples, seed=args.seed)
    
    if args.mode in ['grid', 'all']:
        save_path = output_dir / f'samples_grid_17groups_{timestamp}.png'
        generate_gallery_grid(model, device, save_path, 
                             samples_per_group=args.samples, seed=args.seed)
    
    print(f"\nDone! Output saved to {output_dir}")


if __name__ == "__main__":
    main()

