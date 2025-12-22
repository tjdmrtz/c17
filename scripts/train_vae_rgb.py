#!/usr/bin/env python3
"""
Train RGB Symmetry-Invariant VAE on Crystallographic Patterns.

Optimized for RTX 4090 with mixed precision and torch.compile.

Usage:
    python scripts/train_vae_rgb.py --epochs 200 --batch-size 64
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.symmetry_invariant_vae_rgb import (
    SymmetryInvariantVAE_RGB,
    SymmetryVAEConfigRGB,
    SymmetryVAELossRGB,
)
from src.dataset.transition_dataset import H5PatternDataset, ALL_17_GROUPS


def setup_cuda_optimizations():
    """Setup CUDA optimizations for RTX 4090."""
    if torch.cuda.is_available():
        # Enable cudnn benchmark for faster convolutions
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Enable TF32 for faster matmul on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        
        return True
    return False


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, scaler, use_amp=True):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False)
    
    for batch_idx, (patterns, labels, group_names) in enumerate(pbar):
        patterns = patterns.to(device, non_blocking=True)
        group_name = group_names[0]
        
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        # Mixed precision forward pass
        with autocast(enabled=use_amp):
            outputs = model(patterns, group_name)
            losses = loss_fn(patterns, outputs, group_name)
            loss = losses['loss']
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_recon += losses['recon_loss'].item()
        total_kl += losses['kl_loss'].item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{losses["recon_loss"].item():.4f}',
            'kl': f'{losses["kl_loss"].item():.4f}',
        })
    
    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches,
        'kl_loss': total_kl / n_batches,
    }


@torch.no_grad()
def validate(model, dataloader, loss_fn, device, use_amp=True):
    """Validate the model with mixed precision."""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    for patterns, labels, group_names in dataloader:
        patterns = patterns.to(device, non_blocking=True)
        group_name = group_names[0]
        
        with autocast(enabled=use_amp):
            outputs = model(patterns, group_name)
            losses = loss_fn(patterns, outputs, group_name)
        
        total_loss += losses['loss'].item()
        total_recon += losses['recon_loss'].item()
        total_kl += losses['kl_loss'].item()
    
    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches,
        'kl_loss': total_kl / n_batches,
    }


@torch.no_grad()
def visualize_reconstructions(model, dataloader, device, save_path, use_amp=True):
    """Visualize reconstructions for each group."""
    model.eval()
    
    samples_by_group = {g: None for g in ALL_17_GROUPS}
    
    for patterns, labels, group_names in dataloader:
        for i, (pattern, label, group_name) in enumerate(zip(patterns, labels, group_names)):
            if samples_by_group[group_name] is None:
                samples_by_group[group_name] = pattern.unsqueeze(0).to(device)
        
        if all(v is not None for v in samples_by_group.values()):
            break
    
    fig, axes = plt.subplots(17, 3, figsize=(9, 51))
    fig.patch.set_facecolor('#0a0a0a')
    
    for idx, group_name in enumerate(ALL_17_GROUPS):
        pattern = samples_by_group[group_name]
        
        if pattern is not None:
            with autocast(enabled=use_amp):
                outputs = model(pattern, group_name)
            recon = outputs['recon']
            
            ax = axes[idx, 0]
            img = pattern[0].cpu().permute(1, 2, 0).numpy()
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(f'{group_name} - Original', color='white', fontsize=10)
            ax.axis('off')
            
            ax = axes[idx, 1]
            img = recon[0].float().cpu().permute(1, 2, 0).numpy()
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title('Reconstructed', color='white', fontsize=10)
            ax.axis('off')
            
            ax = axes[idx, 2]
            error = (pattern[0] - recon[0].float()).abs().mean(dim=0).cpu().numpy()
            ax.imshow(error, cmap='hot', vmin=0, vmax=0.5)
            ax.set_title('Error', color='white', fontsize=10)
            ax.axis('off')
        else:
            for j in range(3):
                axes[idx, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    
    print(f"Saved reconstructions to {save_path}")


@torch.no_grad()
def visualize_latent_space(model, dataloader, device, save_path, use_amp=True):
    """Visualize 2D latent space with all groups."""
    model.eval()
    
    latents = []
    group_labels = []
    
    for patterns, labels, group_names in tqdm(dataloader, desc='Encoding', leave=False):
        patterns = patterns.to(device, non_blocking=True)
        
        for i, (pattern, label, group_name) in enumerate(zip(patterns, labels, group_names)):
            with autocast(enabled=use_amp):
                z = model.encode(pattern.unsqueeze(0), group_name)
            latents.append(z.float().cpu().numpy())
            group_labels.append(label.item())
    
    latents = np.concatenate(latents, axis=0)
    group_labels = np.array(group_labels)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    try:
        from scripts.dashboard.theme import GROUP_COLORS
    except ImportError:
        GROUP_COLORS = {g: f'C{i % 10}' for i, g in enumerate(ALL_17_GROUPS)}
    
    for group_idx, group_name in enumerate(ALL_17_GROUPS):
        mask = group_labels == group_idx
        if mask.sum() > 0:
            color = GROUP_COLORS.get(group_name, '#888888')
            ax.scatter(
                latents[mask, 0], latents[mask, 1],
                c=color, label=group_name, alpha=0.6, s=20
            )
    
    ax.set_xlabel('z₁', fontsize=12, color='white')
    ax.set_ylabel('z₂', fontsize=12, color='white')
    ax.set_title('Latent Space (2D)', fontsize=14, color='white', fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.legend(loc='upper right', facecolor='#1a1a1a', edgecolor='white',
              labelcolor='white', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2, color='white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    
    print(f"Saved latent space to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train RGB Symmetry VAE (RTX 4090 Optimized)')
    parser.add_argument('--data-path', type=str,
                       default='data/colored_crystallographic/crystallographic_patterns_colored.h5',
                       help='Path to H5 dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (64-128 for RTX 4090)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--latent-dim', type=int, default=2,
                       help='Latent dimension')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='KL weight')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of workers (0 is fastest with preloaded data)')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile (PyTorch 2.0+)')
    
    args = parser.parse_args()
    
    # Setup CUDA optimizations
    has_cuda = setup_cuda_optimizations()
    device = torch.device('cuda' if has_cuda else 'cpu')
    print(f"Using device: {device}")
    
    use_amp = has_cuda and not args.no_amp
    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")
    
    # Output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"output/vae_rgb_256_{timestamp}"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = H5PatternDataset(args.data_path, split='train')
    val_dataset = H5PatternDataset(args.data_path, split='val')
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Use appropriate number of workers (0 is best with preloaded data)
    num_workers = args.num_workers
    
    loader_kwargs = {
        'batch_size': args.batch_size,
        'pin_memory': True,
    }
    if num_workers > 0:
        loader_kwargs['num_workers'] = num_workers
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 4
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    
    # Create model
    config = SymmetryVAEConfigRGB(
        resolution=256,
        latent_dim=args.latent_dim,
        beta=args.beta,
    )
    
    model = SymmetryInvariantVAE_RGB(config).to(device)
    
    # Optionally compile model (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    
    # Loss and optimizer
    loss_fn = SymmetryVAELossRGB(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)
    
    # Resume if specified
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
    
    # Training history
    history = {
        'train_loss': [], 'train_recon': [], 'train_kl': [],
        'val_loss': [], 'val_recon': [], 'val_kl': [],
    }
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({
            'resolution': config.resolution,
            'latent_dim': config.latent_dim,
            'hidden_dims': config.hidden_dims,
            'beta': config.beta,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'use_amp': use_amp,
            'num_workers': num_workers,
        }, f, indent=2)
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING VAE (RTX 4090 Optimized)")
    print("="*60)
    
    import time
    train_start = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch + 1, scaler, use_amp
        )
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device, use_amp)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_recon'].append(train_metrics['recon_loss'])
        history['train_kl'].append(train_metrics['kl_loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_recon'].append(val_metrics['recon_loss'])
        history['val_kl'].append(val_metrics['kl_loss'])
        
        # Print progress
        print(f"Epoch {epoch + 1:3d}/{args.epochs} | "
              f"Train: {train_metrics['loss']:.4f} | "
              f"Val: {val_metrics['loss']:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'best_loss': best_loss,
            }, output_dir / 'best_model.pt')
            print(f"  → New best model saved (loss: {best_loss:.4f})")
        
        # Periodic checkpoints and visualizations
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, output_dir / f'checkpoint_epoch_{epoch + 1:03d}.pt')
            
            visualize_reconstructions(
                model, val_loader, device,
                output_dir / f'reconstructions_epoch_{epoch + 1:03d}.png',
                use_amp
            )
            
            if config.latent_dim == 2:
                visualize_latent_space(
                    model, val_loader, device,
                    output_dir / f'latent_space_epoch_{epoch + 1:03d}.png',
                    use_amp
                )
        
        # Save history
        with open(output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    total_time = time.time() - train_start
    
    # Final save
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'config': config,
    }, output_dir / 'final_model.pt')
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('#0a0a0a')
    
    for ax in axes:
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], label='Train', color='#4ecdc4')
    axes[0].plot(epochs, history['val_loss'], label='Val', color='#ff6b6b')
    axes[0].set_xlabel('Epoch', color='white')
    axes[0].set_ylabel('Total Loss', color='white')
    axes[0].legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    axes[0].set_title('Total Loss', color='white', fontweight='bold')
    
    axes[1].plot(epochs, history['train_recon'], label='Train', color='#4ecdc4')
    axes[1].plot(epochs, history['val_recon'], label='Val', color='#ff6b6b')
    axes[1].set_xlabel('Epoch', color='white')
    axes[1].set_ylabel('Reconstruction Loss', color='white')
    axes[1].legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    axes[1].set_title('Reconstruction Loss', color='white', fontweight='bold')
    
    axes[2].plot(epochs, history['train_kl'], label='Train', color='#4ecdc4')
    axes[2].plot(epochs, history['val_kl'], label='Val', color='#ff6b6b')
    axes[2].set_xlabel('Epoch', color='white')
    axes[2].set_ylabel('KL Loss', color='white')
    axes[2].legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
    axes[2].set_title('KL Divergence', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    print(f"Average time per epoch: {total_time/args.epochs:.1f}s")
    print(f"Output directory: {output_dir}")
    print(f"Best validation loss: {best_loss:.4f}")
    
    # Cleanup
    train_dataset.close()
    val_dataset.close()


if __name__ == "__main__":
    main()
