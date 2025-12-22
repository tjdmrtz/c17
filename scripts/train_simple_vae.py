#!/usr/bin/env python3
"""
Train Simple VAE on RGB Crystallographic Patterns.

Optimized for RTX 4090 - fast and effective.
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
from torch.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.vae_simple_rgb import SimpleVAE, SimpleVAEConfig, SimpleVAELoss
from src.dataset.transition_dataset import H5PatternDataset, ALL_17_GROUPS


def setup_cuda():
    """Setup CUDA optimizations."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return True
    return False


def train_epoch(model, dataloader, optimizer, loss_fn, device, scaler, epoch, total_epochs):
    """Train one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    # Beta warmup: start at 0, increase to target over first 50 epochs
    warmup_epochs = 50
    target_beta = 0.001
    if epoch < warmup_epochs:
        current_beta = target_beta * (epoch / warmup_epochs)
    else:
        current_beta = target_beta
    loss_fn.set_beta(current_beta)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{total_epochs}', leave=False)
    
    for patterns, labels, group_names in pbar:
        patterns = patterns.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast('cuda'):
            outputs = model(patterns)
            losses = loss_fn(patterns, outputs)
        
        loss = losses['loss']
        
        # Skip if NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss, skipping batch")
            optimizer.zero_grad(set_to_none=True)
            continue
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += losses['loss'].item()
        total_recon += losses['recon_loss'].item()
        total_kl += losses['kl_loss'].item()
        
        pbar.set_postfix({
            'loss': f"{losses['loss'].item():.4f}",
            'recon': f"{losses['recon_loss'].item():.4f}",
            'β': f"{current_beta:.5f}",
        })
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'recon_loss': total_recon / n,
        'kl_loss': total_kl / n,
        'beta': current_beta,
    }


@torch.no_grad()
def validate(model, dataloader, loss_fn, device):
    """Validate."""
    model.eval()
    total_loss = 0
    total_recon = 0
    
    for patterns, labels, group_names in dataloader:
        patterns = patterns.to(device, non_blocking=True)
        
        with autocast('cuda'):
            outputs = model(patterns)
            losses = loss_fn(patterns, outputs)
        
        total_loss += losses['loss'].item()
        total_recon += losses['recon_loss'].item()
    
    n = len(dataloader)
    return {'loss': total_loss / n, 'recon_loss': total_recon / n}


@torch.no_grad()
def visualize(model, dataloader, device, save_path, n_samples=8):
    """Visualize reconstructions."""
    model.eval()
    
    # Get one batch
    patterns, labels, group_names = next(iter(dataloader))
    patterns = patterns[:n_samples].to(device)
    
    with autocast('cuda'):
        outputs = model(patterns)
    recons = outputs['recon'].float()
    
    # Plot
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
    fig.patch.set_facecolor('#1a1a1a')
    
    for i in range(n_samples):
        # Original
        img = patterns[i].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(np.clip(img, 0, 1))
        axes[0, i].axis('off')
        axes[0, i].set_title(group_names[i], color='white', fontsize=8)
        
        # Reconstruction
        img = recons[i].cpu().permute(1, 2, 0).numpy()
        axes[1, i].imshow(np.clip(img, 0, 1))
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Original', color='white', fontsize=10)
    axes[1, 0].set_ylabel('Recon', color='white', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
    plt.close()


@torch.no_grad()  
def visualize_all_groups(model, dataloader, device, save_path):
    """Visualize one sample from each group."""
    model.eval()
    
    samples = {g: None for g in ALL_17_GROUPS}
    
    for patterns, labels, group_names in dataloader:
        for p, g in zip(patterns, group_names):
            if samples[g] is None:
                samples[g] = p
        if all(v is not None for v in samples.values()):
            break
    
    fig, axes = plt.subplots(17, 3, figsize=(6, 34))
    fig.patch.set_facecolor('#1a1a1a')
    
    for idx, group in enumerate(ALL_17_GROUPS):
        if samples[group] is None:
            continue
            
        pattern = samples[group].unsqueeze(0).to(device)
        
        with autocast('cuda'):
            outputs = model(pattern)
        recon = outputs['recon'][0].float().cpu()
        orig = pattern[0].cpu()
        
        # Original
        axes[idx, 0].imshow(np.clip(orig.permute(1, 2, 0).numpy(), 0, 1))
        axes[idx, 0].axis('off')
        axes[idx, 0].set_title(f'{group}', color='white', fontsize=8, loc='left')
        
        # Recon
        axes[idx, 1].imshow(np.clip(recon.permute(1, 2, 0).numpy(), 0, 1))
        axes[idx, 1].axis('off')
        
        # Error
        error = (orig - recon).abs().mean(dim=0).numpy()
        axes[idx, 2].imshow(error, cmap='hot', vmin=0, vmax=0.3)
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, facecolor='#1a1a1a', bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Simple VAE')
    parser.add_argument('--data-path', type=str,
                       default='data/colored_crystallographic/crystallographic_patterns_colored.h5')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    # Setup
    has_cuda = setup_cuda()
    device = torch.device('cuda' if has_cuda else 'cpu')
    
    # Output dir
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"output/simple_vae_{timestamp}"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    
    # Data
    print("\nLoading data into RAM...")
    train_data = H5PatternDataset(args.data_path, split='train', preload=True)
    val_data = H5PatternDataset(args.data_path, split='val', preload=True)
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, 
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, num_workers=0)
    
    # Model
    config = SimpleVAEConfig(latent_dim=args.latent_dim, beta=0.0001)
    model = SimpleVAE(config).to(device)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Latent dim: {args.latent_dim}")
    
    # Training setup
    loss_fn = SimpleVAELoss(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')
    
    # Resume
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({
            'latent_dim': args.latent_dim,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
        }, f, indent=2)
    
    history = {'train_loss': [], 'val_loss': [], 'recon_loss': [], 'beta': []}
    
    # Train
    print("\n" + "="*50)
    print("TRAINING")
    print("="*50)
    
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, 
                                    device, scaler, epoch, args.epochs)
        val_metrics = validate(model, val_loader, loss_fn, device)
        scheduler.step()
        
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['recon_loss'].append(train_metrics['recon_loss'])
        history['beta'].append(train_metrics['beta'])
        
        print(f"Epoch {epoch+1:3d} | Train: {train_metrics['loss']:.4f} | "
              f"Val: {val_metrics['loss']:.4f} | Recon: {train_metrics['recon_loss']:.4f} | "
              f"β: {train_metrics['beta']:.5f}")
        
        # Save best
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'best_loss': best_loss,
            }, output_dir / 'best_model.pt')
            print(f"  → Best model saved")
        
        # Visualize every 5 epochs
        if (epoch + 1) % 5 == 0:
            visualize(model, val_loader, device, 
                     output_dir / f'recon_epoch_{epoch+1:03d}.png')
            visualize_all_groups(model, val_loader, device,
                                output_dir / f'all_groups_epoch_{epoch+1:03d}.png')
        
        # Save history
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f)
    
    # Final model
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'config': config,
    }, output_dir / 'final_model.pt')
    
    # Plot curves
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['val_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title('Training Loss')
    plt.savefig(output_dir / 'loss_curve.png', dpi=150)
    plt.close()
    
    print("\n" + "="*50)
    print(f"DONE! Output: {output_dir}")
    print(f"Best loss: {best_loss:.4f}")
    print("="*50)
    
    train_data.close()
    val_data.close()


if __name__ == "__main__":
    main()

