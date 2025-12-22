#!/usr/bin/env python3
"""
Train Symmetry-Invariant VAE on Turing Patterns

This script trains a VAE that learns representations invariant to the
symmetry operations of the 17 wallpaper groups.

The training process:
1. Generate Turing patterns with various symmetries
2. Apply data augmentation with random transformations
3. Train with reconstruction + KL + invariance losses
4. Visualize the 2D latent space

Usage:
    python scripts/train_symmetry_vae.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from src.dataset.turing_patterns import TuringPatternGenerator, PatternType
from src.models.symmetry_invariant_vae import (
    SymmetryInvariantVAE, 
    SymmetryVAEConfig,
    SymmetryTransforms,
    SymmetryVAELoss
)


class TuringPatternDataset(Dataset):
    """
    Dataset of Turing patterns with crystallographic symmetries.
    
    Generates patterns on-the-fly or loads from cache.
    """
    
    def __init__(self, 
                 num_samples: int = 1000,
                 resolution: int = 128,
                 groups: list = None,
                 pattern_types: list = None,
                 cache_dir: str = None,
                 seed: int = 42,
                 simulation_steps: int = 3000):
        
        self.num_samples = num_samples
        self.resolution = resolution
        self.groups = groups or ['p4m', 'p6m', 'pmm', 'p4', 'p6', 'cm', 'pgg']
        self.pattern_types = pattern_types or [PatternType.SPOTS, PatternType.STRIPES, PatternType.MAZE]
        self.simulation_steps = simulation_steps
        
        self.rng = np.random.default_rng(seed)
        
        # Cache for patterns
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.patterns = []
        self.labels = []  # (group_name, pattern_type)
        
        self._generate_or_load()
    
    def _generate_or_load(self):
        """Generate patterns or load from cache."""
        
        if self.cache_dir and self.cache_dir.exists():
            cache_file = self.cache_dir / f"turing_dataset_{self.num_samples}_{self.resolution}.npz"
            if cache_file.exists():
                print(f"Loading cached dataset from {cache_file}")
                data = np.load(cache_file, allow_pickle=True)
                self.patterns = list(data['patterns'])
                self.labels = list(data['labels'])
                return
        
        print(f"Generating {self.num_samples} Turing patterns...")
        
        samples_per_combo = max(1, self.num_samples // (len(self.groups) * len(self.pattern_types)))
        
        for group in tqdm(self.groups, desc="Groups"):
            for ptype in self.pattern_types:
                for i in range(samples_per_combo):
                    seed = self.rng.integers(0, 100000)
                    gen = TuringPatternGenerator(self.resolution, seed=seed)
                    
                    pattern = gen.generate(
                        group, ptype, 
                        steps=self.simulation_steps,
                        symmetry_project_interval=50
                    )
                    
                    # Normalize to [0, 1]
                    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)
                    
                    self.patterns.append(pattern.astype(np.float32))
                    self.labels.append((group, ptype.value))
        
        # Shuffle
        indices = self.rng.permutation(len(self.patterns))
        self.patterns = [self.patterns[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        
        # Cache
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"turing_dataset_{self.num_samples}_{self.resolution}.npz"
            np.savez(cache_file, patterns=self.patterns, labels=self.labels)
            print(f"Cached dataset to {cache_file}")
    
    def __len__(self):
        return len(self.patterns)
    
    def __getitem__(self, idx):
        pattern = self.patterns[idx]
        label = self.labels[idx]
        
        # Convert to tensor
        tensor = torch.from_numpy(pattern).unsqueeze(0)  # [1, H, W]
        
        return tensor, label[0], label[1]  # pattern, group, type


class SymmetryAugmentation:
    """
    Data augmentation that applies random symmetry transformations.
    
    This helps the model learn invariance even without explicit symmetry pooling.
    """
    
    def __init__(self, resolution: int, groups: list = None):
        self.transforms = SymmetryTransforms(resolution)
        self.groups = groups or ['p4m']
        self.resolution = resolution
    
    def __call__(self, x: torch.Tensor, group: str) -> torch.Tensor:
        """Apply random transformation from the group."""
        transforms = self.transforms.get_group_transforms(group)
        
        # Random transform
        idx = np.random.randint(len(transforms))
        x_aug = transforms[idx](x)
        
        # Random translation (for translation invariance)
        shift_h = np.random.randint(-self.resolution // 4, self.resolution // 4)
        shift_v = np.random.randint(-self.resolution // 4, self.resolution // 4)
        x_aug = self.transforms.translate(x_aug, shift_h, shift_v)
        
        return x_aug


def train_epoch(model, dataloader, optimizer, loss_fn, device, augment=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_inv = 0
    
    for batch in dataloader:
        patterns, groups, _ = batch
        patterns = patterns.to(device)
        
        # Use first group in batch (simplified)
        group = groups[0]
        
        # Augmentation
        if augment is not None:
            patterns = augment(patterns, group)
        
        optimizer.zero_grad()
        
        outputs = model(patterns, group)
        losses = loss_fn(patterns, outputs, group)
        
        losses['loss'].backward()
        optimizer.step()
        
        total_loss += losses['loss'].item()
        total_recon += losses['recon_loss'].item()
        total_kl += losses['kl_loss'].item()
        total_inv += losses['invariance_loss'].item()
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'recon': total_recon / n,
        'kl': total_kl / n,
        'inv': total_inv / n
    }


def evaluate(model, dataloader, loss_fn, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_recon = 0
    
    all_mu = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            patterns, groups, ptypes = batch
            patterns = patterns.to(device)
            group = groups[0]
            
            outputs = model(patterns, group)
            losses = loss_fn(patterns, outputs, group)
            
            total_loss += losses['loss'].item()
            total_recon += losses['recon_loss'].item()
            
            all_mu.append(outputs['mu'].cpu())
            all_labels.extend([(g, p) for g, p in zip(groups, ptypes)])
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'recon': total_recon / n,
        'mu': torch.cat(all_mu, dim=0),
        'labels': all_labels
    }


def test_invariance(model, device, resolution=128):
    """Test if the model is truly invariant to transformations."""
    model.eval()
    transforms = SymmetryTransforms(resolution)
    
    # Generate a test pattern
    gen = TuringPatternGenerator(resolution, seed=123)
    pattern = gen.generate('p4m', PatternType.SPOTS, steps=2000)
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)
    
    x = torch.from_numpy(pattern.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    
    # Get latent for original
    mu_orig = model.encode(x, 'p4m')
    
    # Test transformations
    results = {}
    test_transforms = [
        ('rot90', lambda t: transforms.rotate_90(t, 1)),
        ('rot180', lambda t: transforms.rotate_90(t, 2)),
        ('flipH', lambda t: transforms.flip_h(t)),
        ('flipV', lambda t: transforms.flip_v(t)),
        ('translate', lambda t: transforms.translate(t, 32, 32)),
    ]
    
    for name, transform in test_transforms:
        x_t = transform(x)
        mu_t = model.encode(x_t, 'p4m')
        diff = (mu_orig - mu_t).abs().mean().item()
        results[name] = diff
    
    return results


def visualize_latent_space(mu, labels, save_path, title="Latent Space"):
    """Visualize the 2D latent space."""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    # Color by group
    groups = list(set(l[0] for l in labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(groups)))
    group_to_color = {g: colors[i] for i, g in enumerate(groups)}
    
    # Marker by pattern type
    types = list(set(l[1] for l in labels))
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    type_to_marker = {t: markers[i % len(markers)] for i, t in enumerate(types)}
    
    for i, (z, label) in enumerate(zip(mu, labels)):
        group, ptype = label
        ax.scatter(z[0], z[1], 
                   c=[group_to_color[group]], 
                   marker=type_to_marker[ptype],
                   s=60, alpha=0.7, edgecolors='white', linewidths=0.5)
    
    # Legend for groups
    for group, color in group_to_color.items():
        ax.scatter([], [], c=[color], label=group, s=100)
    
    ax.legend(loc='upper right', facecolor='#1a1a1a', edgecolor='white',
              labelcolor='white', fontsize=10, title='Group', title_fontsize=12)
    
    ax.set_xlabel('z₁', fontsize=14, color='white')
    ax.set_ylabel('z₂', fontsize=14, color='white')
    ax.set_title(title, fontsize=16, color='white', fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#0a0a0a', 
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved latent space visualization to {save_path}")


def visualize_reconstructions(model, dataloader, device, save_path, n_samples=8):
    """Visualize original vs reconstructed patterns."""
    
    model.eval()
    
    # Get samples
    batch = next(iter(dataloader))
    patterns, groups, _ = batch
    patterns = patterns[:n_samples].to(device)
    group = groups[0]
    
    with torch.no_grad():
        outputs = model(patterns, group)
        recons = outputs['recon']
    
    # Plot
    fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
    fig.patch.set_facecolor('#0a0a0a')
    
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(patterns[i, 0].cpu().numpy(), cmap='viridis')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12, color='white')
        
        # Reconstruction
        axes[1, i].imshow(recons[i, 0].cpu().numpy(), cmap='viridis')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Recon', fontsize=12, color='white')
    
    plt.suptitle(f'Reconstructions ({group} symmetry)', 
                 fontsize=14, color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#0a0a0a',
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved reconstructions to {save_path}")


def visualize_interpolation(model, dataloader, device, save_path, n_steps=10):
    """Visualize interpolation between two patterns."""
    
    model.eval()
    
    # Get two samples
    batch = next(iter(dataloader))
    patterns, groups, _ = batch
    x1 = patterns[0:1].to(device)
    x2 = patterns[1:2].to(device)
    group = groups[0]
    
    with torch.no_grad():
        interpolated = model.interpolate(x1, x2, group, steps=n_steps)
    
    # Plot
    fig, axes = plt.subplots(1, n_steps, figsize=(2*n_steps, 2))
    fig.patch.set_facecolor('#0a0a0a')
    
    for i in range(n_steps):
        axes[i].imshow(interpolated[i, 0].cpu().numpy(), cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'{i/(n_steps-1):.1f}', fontsize=10, color='white')
    
    plt.suptitle('Latent Space Interpolation', 
                 fontsize=14, color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#0a0a0a',
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved interpolation to {save_path}")


def visualize_samples(model, device, save_path, n_samples=16, group='p4m'):
    """Visualize random samples from the latent space."""
    
    model.eval()
    
    with torch.no_grad():
        samples = model.sample(n_samples, group, device)
    
    # Plot
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    fig.patch.set_facecolor('#0a0a0a')
    
    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            ax.imshow(samples[i, 0].cpu().numpy(), cmap='plasma')
        ax.axis('off')
    
    plt.suptitle(f'Random Samples from Latent Space ({group})', 
                 fontsize=14, color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#0a0a0a',
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved samples to {save_path}")


def visualize_latent_space_17(mu, labels, save_path, title="Latent Space"):
    """Visualize the 2D latent space with all 17 groups."""
    
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    # Color by group - use distinct colors for 17 groups
    groups = sorted(list(set(l[0] for l in labels)))
    
    # Custom colors for lattice types
    lattice_colors = {
        # Oblique (blues)
        'p1': '#1a5276', 'p2': '#2980b9',
        # Rectangular (greens)
        'pm': '#1e8449', 'pg': '#27ae60', 'cm': '#2ecc71',
        'pmm': '#145a32', 'pmg': '#196f3d', 'pgg': '#1d8348', 'cmm': '#239b56',
        # Square (purples)
        'p4': '#6c3483', 'p4m': '#8e44ad', 'p4g': '#a569bd',
        # Hexagonal (oranges/reds)
        'p3': '#b9770e', 'p3m1': '#d68910', 'p31m': '#f39c12',
        'p6': '#c0392b', 'p6m': '#e74c3c',
    }
    
    # Markers by pattern type
    types = list(set(l[1] for l in labels))
    markers = {'spots': 'o', 'stripes': 's', 'maze': '^', 'coral': 'D'}
    
    for i, (z, label) in enumerate(zip(mu, labels)):
        group, ptype = label
        color = lattice_colors.get(group, '#ffffff')
        marker = markers.get(ptype, 'o')
        ax.scatter(z[0], z[1], c=color, marker=marker,
                   s=80, alpha=0.8, edgecolors='white', linewidths=0.5)
    
    # Create legend with groups organized by lattice type
    lattice_groups = {
        'Oblique': ['p1', 'p2'],
        'Rectangular': ['pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm'],
        'Square': ['p4', 'p4m', 'p4g'],
        'Hexagonal': ['p3', 'p3m1', 'p31m', 'p6', 'p6m']
    }
    
    # Add legend handles
    handles = []
    for lattice, grps in lattice_groups.items():
        for g in grps:
            if g in groups:
                h = ax.scatter([], [], c=lattice_colors[g], label=g, s=100)
                handles.append(h)
    
    ax.legend(handles=handles, loc='upper right', facecolor='#1a1a1a', 
              edgecolor='white', labelcolor='white', fontsize=9, 
              title='Wallpaper Group', title_fontsize=11, ncol=2)
    
    ax.set_xlabel('z₁', fontsize=14, color='white')
    ax.set_ylabel('z₂', fontsize=14, color='white')
    ax.set_title(title, fontsize=16, color='white', fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#0a0a0a', 
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved latent space visualization to {save_path}")


def visualize_all_17_decoder_outputs(model, device, save_path):
    """Generate and visualize decoder outputs for all 17 wallpaper groups."""
    
    model.eval()
    
    all_17_groups = [
        'p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm',
        'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m'
    ]
    
    # Lattice type colors for borders/backgrounds
    lattice_info = {
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
    lattice_cmaps = {
        'Oblique': 'Blues',
        'Rectangular': 'Greens', 
        'Square': 'Purples',
        'Hexagonal': 'YlOrRd'
    }
    
    # Generate samples from the same latent point for each group
    z_fixed = torch.tensor([[0.0, 0.0]], device=device)  # Center of latent space
    z_random = torch.randn(1, 2, device=device) * 0.5     # Random point
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.patch.set_facecolor('#0a0a0a')
    
    with torch.no_grad():
        for idx, (ax, group) in enumerate(zip(axes.flat[:17], all_17_groups)):
            # Decode with this group's symmetry
            sample = model.decode(z_random, group)
            
            lattice, color = lattice_info[group]
            cmap = lattice_cmaps[lattice]
            
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
    for ax in axes.flat[17:]:
        ax.axis('off')
    
    plt.suptitle('Decoder Outputs for All 17 Wallpaper Groups\n'
                 'Same latent point z → Different crystallographic symmetries',
                 fontsize=16, color='white', fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    plt.savefig(save_path, dpi=150, facecolor='#0a0a0a',
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved all 17 groups decoder outputs to {save_path}")


def main():
    # Configuration - LARGER MODEL
    config = SymmetryVAEConfig(
        resolution=128,
        latent_dim=2,
        hidden_dims=[64, 128, 256, 512],  # Más potente
        use_symmetry_pooling=True,
        beta=1.0,
        invariance_weight=10.0,
    )
    
    # ALL 17 wallpaper groups
    ALL_17_GROUPS = [
        'p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm',
        'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m'
    ]
    
    # Training settings - MUCH LONGER TRAINING
    num_epochs = 500
    batch_size = 32
    learning_rate = 1e-4
    num_train_samples = 1700  # ~100 per group
    num_val_samples = 340     # ~20 per group
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"output/symmetry_vae_17groups_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Dataset
    print("\n" + "="*60)
    print("CREATING DATASETS - ALL 17 WALLPAPER GROUPS")
    print("="*60)
    
    cache_dir = Path("data/turing_cache")
    
    train_dataset = TuringPatternDataset(
        num_samples=num_train_samples,
        resolution=config.resolution,
        groups=ALL_17_GROUPS,
        pattern_types=[PatternType.SPOTS, PatternType.STRIPES],
        cache_dir=cache_dir,
        seed=42,
        simulation_steps=2000
    )
    
    val_dataset = TuringPatternDataset(
        num_samples=num_val_samples,
        resolution=config.resolution,
        groups=ALL_17_GROUPS,
        pattern_types=[PatternType.SPOTS],
        cache_dir=cache_dir,
        seed=123,
        simulation_steps=2000
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Model
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    
    model = SymmetryInvariantVAE(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    loss_fn = SymmetryVAELoss(config)
    
    # Augmentation
    augment = SymmetryAugmentation(config.resolution)
    
    # Training
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    history = {
        'train_loss': [], 'train_recon': [], 'train_kl': [], 'train_inv': [],
        'val_loss': [], 'val_recon': []
    }
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, augment)
        
        # Validate
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_recon'].append(train_metrics['recon'])
        history['train_kl'].append(train_metrics['kl'])
        history['train_inv'].append(train_metrics['inv'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_recon'].append(val_metrics['recon'])
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train: L={train_metrics['loss']:.4f} R={train_metrics['recon']:.4f} "
                  f"KL={train_metrics['kl']:.4f} Inv={train_metrics['inv']:.4f} | "
                  f"Val: L={val_metrics['loss']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_loss': val_metrics['loss'],
            }, output_dir / 'best_model.pt')
    
    # Test invariance
    print("\n" + "="*60)
    print("TESTING INVARIANCE")
    print("="*60)
    
    invariance_results = test_invariance(model, device, config.resolution)
    print("Representation difference under transformations:")
    for name, diff in invariance_results.items():
        status = "✓" if diff < 0.1 else "✗"
        print(f"  {name:12s}: {diff:.4f} {status}")
    
    # Visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Latent space with all 17 groups
    val_metrics = evaluate(model, val_loader, loss_fn, device)
    visualize_latent_space_17(
        val_metrics['mu'].numpy(), 
        val_metrics['labels'],
        output_dir / 'latent_space_17groups.png',
        title=f'2D Latent Space - All 17 Wallpaper Groups'
    )
    
    # Reconstructions
    visualize_reconstructions(model, val_loader, device, output_dir / 'reconstructions.png')
    
    # Interpolation
    visualize_interpolation(model, val_loader, device, output_dir / 'interpolation.png')
    
    # Samples for ALL 17 groups in a single gallery
    visualize_all_17_decoder_outputs(model, device, output_dir / 'decoder_all_17_groups.png')
    
    # Individual samples for selected groups
    for group in ALL_17_GROUPS:
        visualize_samples(model, device, output_dir / f'samples_{group}.png', 
                         n_samples=9, group=group)
    
    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('#0a0a0a')
    
    for ax in axes:
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
    
    axes[0].plot(history['train_loss'], label='Train', color='#4ecdc4')
    axes[0].plot(history['val_loss'], label='Val', color='#ff6b6b')
    axes[0].set_title('Total Loss', color='white')
    axes[0].legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    
    axes[1].plot(history['train_recon'], label='Train', color='#4ecdc4')
    axes[1].plot(history['val_recon'], label='Val', color='#ff6b6b')
    axes[1].set_title('Reconstruction Loss', color='white')
    axes[1].legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    
    axes[2].plot(history['train_kl'], label='KL', color='#ffe66d')
    axes[2].plot(history['train_inv'], label='Invariance', color='#95e1d3')
    axes[2].set_title('KL & Invariance Loss', color='white')
    axes[2].legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, 
                facecolor='#0a0a0a', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()

