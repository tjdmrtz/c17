#!/usr/bin/env python3
"""
Train Latent Diffusion Model for Crystallographic Patterns.

Two-stage training:
1. Stage 1: Train VAE for latent compression (~1-2 hours)
2. Stage 2: Train diffusion model in latent space (~3-5 hours)

Usage:
    # Full training (both stages)
    python scripts/train_latent_diffusion.py --epochs-vae 100 --epochs-diffusion 500
    
    # Only VAE (Stage 1)
    python scripts/train_latent_diffusion.py --stage 1 --epochs-vae 100
    
    # Only Diffusion (Stage 2, requires trained VAE)
    python scripts/train_latent_diffusion.py --stage 2 --epochs-diffusion 500 --vae-checkpoint path/to/vae.pt
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import h5py
from tqdm import tqdm

from src.models.latent_diffusion import (
    LatentDiffusionModel, 
    LatentDiffusionConfig
)
from src.models.kl_scheduler import KLScheduler
from src.visualization.training_logger import TrainingLogger
from torchvision.utils import make_grid, save_image


# =============================================================================
# DATASET
# =============================================================================

class CrystallographicDataset(torch.utils.data.Dataset):
    """Load crystallographic patterns from HDF5."""
    
    def __init__(self, hdf5_path: str, preload: bool = True):
        self.hdf5_path = hdf5_path
        
        with h5py.File(hdf5_path, 'r') as f:
            self.length = len(f['labels'])
            
            if preload:
                print("  Loading dataset into RAM...")
                patterns = f['patterns'][:]
                self.patterns = torch.from_numpy(patterns).float().permute(0, 3, 1, 2)
                self.labels = torch.from_numpy(f['labels'][:]).long()
                print(f"  Loaded {self.length} samples")
            else:
                self.patterns = None
                self.labels = None
                self.file = None
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.patterns is not None:
            return self.patterns[idx], self.labels[idx]
        else:
            if self.file is None:
                self.file = h5py.File(self.hdf5_path, 'r')
            pattern = torch.from_numpy(self.file['patterns'][idx]).float().permute(2, 0, 1)
            label = int(self.file['labels'][idx])
            return pattern, label


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_vae_epoch(model, loader, optimizer, device, beta=0.0001):
    """Train VAE for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    pbar = tqdm(loader, desc="VAE Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model.train_vae_step(images)
        
        # Losses
        recon_loss = F.mse_loss(outputs['reconstruction'], images, reduction='sum') / images.shape[0]
        
        kl_loss = -0.5 * torch.sum(1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp())
        kl_loss = kl_loss / images.shape[0]
        
        # Classification loss
        class_loss = F.cross_entropy(outputs['class_logits'], labels)
        
        loss = recon_loss + beta * kl_loss + 0.1 * class_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.1f}',
            'recon': f'{recon_loss.item():.1f}',
            'kl': f'{kl_loss.item():.1f}'
        })
    
    n = len(loader)
    return {
        'loss': total_loss / n,
        'recon': total_recon / n,
        'kl': total_kl / n
    }


@torch.no_grad()
def validate_vae(model, loader, device, beta=0.0001):
    """Validate VAE."""
    model.eval()
    total_loss = 0
    total_recon = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model.train_vae_step(images)
        
        recon_loss = F.mse_loss(outputs['reconstruction'], images, reduction='sum') / images.shape[0]
        kl_loss = -0.5 * torch.sum(1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp())
        kl_loss = kl_loss / images.shape[0]
        
        loss = recon_loss + beta * kl_loss
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        
        preds = outputs['class_logits'].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.shape[0]
    
    n = len(loader)
    return {
        'loss': total_loss / n,
        'recon': total_recon / n,
        'accuracy': correct / total
    }


def train_diffusion_epoch(model, loader, optimizer, device):
    """Train diffusion for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Diffusion Training")
    for images, _ in pbar:
        images = images.to(device)
        
        optimizer.zero_grad()
        
        loss = model.train_diffusion_step(images)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)


@torch.no_grad()
def validate_diffusion(model, loader, device):
    """Validate diffusion."""
    model.eval()
    total_loss = 0
    
    for images, _ in loader:
        images = images.to(device)
        loss = model.train_diffusion_step(images)
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def generate_samples(model, device, num_samples=4, steps=50):
    """Generate samples from the model."""
    model.eval()
    samples = model.generate(num_samples, device, use_ddim=True, steps=steps)
    return samples


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Latent Diffusion Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Stage selection
    parser.add_argument('--stage', type=int, default=0, choices=[0, 1, 2],
                       help='Training stage: 0=both, 1=VAE only, 2=diffusion only')
    
    # Data
    parser.add_argument('--data-path', type=str,
                       default='data/colored_crystallographic/crystallographic_patterns_colored.h5')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # VAE training
    parser.add_argument('--epochs-vae', type=int, default=100)
    parser.add_argument('--lr-vae', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=0.0001,
                       help='KL weight (low for better reconstruction)')
    
    # KL Annealing
    parser.add_argument('--kl-annealing', type=str, default='warmup',
                       choices=['constant', 'linear', 'warmup', 'cyclical', 'sigmoid'],
                       help='KL annealing schedule')
    parser.add_argument('--kl-warmup-epochs', type=int, default=20)
    parser.add_argument('--kl-anneal-epochs', type=int, default=40)
    
    # Diffusion training
    parser.add_argument('--epochs-diffusion', type=int, default=500)
    parser.add_argument('--lr-diffusion', type=float, default=1e-4)
    
    # Checkpoints
    parser.add_argument('--vae-checkpoint', type=str, default=None,
                       help='Path to VAE checkpoint (for stage 2)')
    parser.add_argument('--output-dir', type=str, default='runs/latent_diffusion')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from latest checkpoint in output-dir')
    parser.add_argument('--resume-dir', type=str, default=None,
                       help='Specific run directory to resume from')
    
    # Model config
    parser.add_argument('--latent-channels', type=int, default=4)
    parser.add_argument('--base-channels', type=int, default=64)
    
    # Other
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Handle resume or create new output directory
    start_epoch_vae = 1
    start_epoch_diffusion = 1
    resume_checkpoint = None
    
    if args.resume or args.resume_dir:
        base_dir = Path(args.output_dir)
        
        if args.resume_dir:
            # Use specific directory
            output_dir = Path(args.resume_dir)
        else:
            # Find latest run
            if base_dir.exists():
                runs = sorted([d for d in base_dir.iterdir() if d.is_dir()], 
                             key=lambda x: x.name, reverse=True)
                if runs:
                    output_dir = runs[0]
                    print(f"📂 Resuming from: {output_dir}")
                else:
                    print("⚠️  No previous runs found, starting fresh")
                    output_dir = base_dir / datetime.now().strftime('%Y%m%d_%H%M%S')
            else:
                output_dir = base_dir / datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Find best/latest checkpoint
        if output_dir.exists():
            # Look for VAE checkpoints
            vae_best = output_dir / 'vae_best.pt'
            vae_checkpoints = sorted(output_dir.glob('vae_epoch_*.pt'), reverse=True)
            
            if vae_best.exists():
                resume_checkpoint = vae_best
                ckpt = torch.load(vae_best, map_location='cpu')
                start_epoch_vae = ckpt.get('epoch', 0) + 1
                print(f"  Found VAE best checkpoint (epoch {start_epoch_vae - 1})")
            elif vae_checkpoints:
                resume_checkpoint = vae_checkpoints[0]
                ckpt = torch.load(resume_checkpoint, map_location='cpu')
                start_epoch_vae = ckpt.get('epoch', 0) + 1
                print(f"  Found VAE checkpoint: {resume_checkpoint.name} (epoch {start_epoch_vae - 1})")
            
            # Look for diffusion checkpoints
            diff_best = output_dir / 'diffusion_best.pt'
            diff_checkpoints = sorted(output_dir.glob('diffusion_epoch_*.pt'), reverse=True)
            
            if diff_best.exists():
                diff_ckpt = torch.load(diff_best, map_location='cpu')
                start_epoch_diffusion = diff_ckpt.get('epoch', 0) + 1
                print(f"  Found Diffusion best checkpoint (epoch {start_epoch_diffusion - 1})")
            elif diff_checkpoints:
                diff_ckpt = torch.load(diff_checkpoints[0], map_location='cpu')
                start_epoch_diffusion = diff_ckpt.get('epoch', 0) + 1
                print(f"  Found Diffusion checkpoint (epoch {start_epoch_diffusion - 1})")
    else:
        # Create new output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(args.output_dir) / timestamp
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # TensorBoard
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    # Load dataset
    print("\n📦 Loading dataset...")
    dataset = CrystallographicDataset(args.data_path, preload=True)
    
    # Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    # Create model
    print("\n🧠 Creating Latent Diffusion Model...")
    config = LatentDiffusionConfig(
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
    )
    model = LatentDiffusionModel(config)
    model = model.to(device)
    
    info = model.get_model_info()
    for k, v in info.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    
    # Load VAE checkpoint if provided (explicit or from resume)
    vae_optimizer_state = None
    if args.vae_checkpoint and Path(args.vae_checkpoint).exists():
        print(f"\n📂 Loading VAE from {args.vae_checkpoint}")
        checkpoint = torch.load(args.vae_checkpoint, map_location=device)
        model.vae.load_state_dict(checkpoint['vae_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            vae_optimizer_state = checkpoint['optimizer_state_dict']
    elif resume_checkpoint is not None:
        print(f"\n📂 Loading VAE from {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.vae.load_state_dict(checkpoint['vae_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            vae_optimizer_state = checkpoint['optimizer_state_dict']
        if 'val_loss' in checkpoint:
            print(f"  Previous best val_loss: {checkpoint['val_loss']:.2f}")
    
    # =========================================================================
    # STAGE 1: Train VAE
    # =========================================================================
    
    if args.stage in [0, 1]:
        print("\n" + "=" * 70)
        print("STAGE 1: TRAINING VAE")
        if start_epoch_vae > 1:
            print(f"  Resuming from epoch {start_epoch_vae}")
        print("=" * 70)
        
        # Setup KL annealing
        kl_scheduler = KLScheduler(
            schedule_type=args.kl_annealing,
            target_beta=args.beta,
            min_beta=0.0,
            warmup_epochs=args.kl_warmup_epochs,
            anneal_epochs=args.kl_anneal_epochs,
            total_epochs=args.epochs_vae
        )
        print(f"  KL Annealing: {kl_scheduler}")
        
        # Setup disk logger
        disk_logger = TrainingLogger(
            output_dir=output_dir / 'vae_logs',
            writer=writer,
            device=str(device)
        )
        
        optimizer = optim.AdamW(model.vae.parameters(), lr=args.lr_vae, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs_vae)
        
        # Load optimizer state if resuming
        if vae_optimizer_state is not None:
            try:
                optimizer.load_state_dict(vae_optimizer_state)
                print("  Loaded optimizer state")
            except:
                print("  Could not load optimizer state, starting fresh")
        
        # Advance scheduler to current epoch
        for _ in range(start_epoch_vae - 1):
            scheduler.step()
        
        best_val_loss = float('inf')
        
        for epoch in range(start_epoch_vae, args.epochs_vae + 1):
            # Get current beta from scheduler
            current_beta = kl_scheduler.get_beta(epoch - 1)
            print(f"\nEpoch {epoch}/{args.epochs_vae} (β={current_beta:.6f})")
            
            train_metrics = train_vae_epoch(model, train_loader, optimizer, device, current_beta)
            val_metrics = validate_vae(model, val_loader, device, current_beta)
            
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            
            # Log to TensorBoard
            writer.add_scalar('VAE/train_loss', train_metrics['loss'], epoch)
            writer.add_scalar('VAE/train_recon', train_metrics['recon'], epoch)
            writer.add_scalar('VAE/train_kl', train_metrics['kl'], epoch)
            writer.add_scalar('VAE/val_loss', val_metrics['loss'], epoch)
            writer.add_scalar('VAE/val_recon', val_metrics['recon'], epoch)
            writer.add_scalar('VAE/val_accuracy', val_metrics['accuracy'], epoch)
            writer.add_scalar('VAE/beta', current_beta, epoch)
            writer.add_scalar('VAE/learning_rate', lr, epoch)
            
            print(f"  Train - Loss: {train_metrics['loss']:.1f} | Recon: {train_metrics['recon']:.1f} | KL: {train_metrics['kl']:.1f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.1f} | Recon: {val_metrics['recon']:.1f} | Acc: {val_metrics['accuracy']:.1%}")
            
            # Save best
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'vae_state_dict': model.vae.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'beta': current_beta,
                }, output_dir / 'vae_best.pt')
                print("  ✓ Saved best VAE!")
            
            # Basic logging every 5 epochs
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    # Get sample batch
                    sample_imgs, _ = next(iter(val_loader))
                    sample_imgs = sample_imgs[:8].to(device)
                    outputs = model.train_vae_step(sample_imgs)
                    
                    # Log reconstructions to TensorBoard
                    comparison = torch.cat([sample_imgs, outputs['reconstruction']], dim=0)
                    writer.add_images('VAE/reconstructions', comparison, epoch)
                    
                    # Save to disk
                    save_image(comparison, output_dir / 'vae_logs' / 'reconstructions' / f'recon_epoch_{epoch:04d}.png', 
                              nrow=8, normalize=True)
                
                # Generate samples from VAE
                with torch.no_grad():
                    z = torch.randn(8, config.latent_channels, config.latent_size, config.latent_size, device=device)
                    samples = model.decode(z)
                    writer.add_images('VAE/samples', samples, epoch)
                    save_image(samples, output_dir / 'vae_logs' / 'samples' / f'samples_epoch_{epoch:04d}.png',
                              nrow=4, normalize=True)
            
            # Extended logging every 20 epochs
            if epoch % 20 == 0:
                disk_logger.log_latent_space_2d(model.vae, val_loader, epoch, method='pca')
                disk_logger.log_latent_space_3d(model.vae, val_loader, epoch)
                disk_logger.log_cluster_analysis(model.vae, val_loader, epoch)
            
            # Save checkpoint
            if epoch % 20 == 0:
                torch.save({
                    'epoch': epoch,
                    'vae_state_dict': model.vae.state_dict(),
                    'beta': current_beta,
                }, output_dir / f'vae_epoch_{epoch:03d}.pt')
        
        print("\n✅ VAE training complete!")
    
    # =========================================================================
    # STAGE 2: Train Diffusion
    # =========================================================================
    
    if args.stage in [0, 2]:
        print("\n" + "=" * 70)
        print("STAGE 2: TRAINING DIFFUSION")
        if start_epoch_diffusion > 1:
            print(f"  Resuming from epoch {start_epoch_diffusion}")
        print("=" * 70)
        
        # Load diffusion checkpoint if resuming
        diff_optimizer_state = None
        if args.resume or args.resume_dir:
            diff_best = output_dir / 'diffusion_best.pt'
            diff_checkpoints = sorted(output_dir.glob('diffusion_epoch_*.pt'), reverse=True)
            
            if diff_best.exists():
                ckpt = torch.load(diff_best, map_location=device)
                if 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'])
                    print(f"  Loaded diffusion model from best checkpoint")
                if 'optimizer_state_dict' in ckpt:
                    diff_optimizer_state = ckpt['optimizer_state_dict']
            elif diff_checkpoints:
                ckpt = torch.load(diff_checkpoints[0], map_location=device)
                if 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'])
                    print(f"  Loaded diffusion model from {diff_checkpoints[0].name}")
        
        # Freeze VAE
        model.freeze_vae()
        print("  VAE frozen")
        
        # Create samples directory
        (output_dir / 'diffusion_samples').mkdir(exist_ok=True)
        
        # Optimizer for diffusion only
        optimizer = optim.AdamW(model.unet.parameters(), lr=args.lr_diffusion, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs_diffusion)
        
        # Load optimizer state if resuming
        if diff_optimizer_state is not None:
            try:
                optimizer.load_state_dict(diff_optimizer_state)
                print("  Loaded optimizer state")
            except:
                print("  Could not load optimizer state, starting fresh")
        
        # Advance scheduler to current epoch
        for _ in range(start_epoch_diffusion - 1):
            scheduler.step()
        
        best_val_loss = float('inf')
        
        for epoch in range(start_epoch_diffusion, args.epochs_diffusion + 1):
            print(f"\nEpoch {epoch}/{args.epochs_diffusion}")
            
            train_loss = train_diffusion_epoch(model, train_loader, optimizer, device)
            val_loss = validate_diffusion(model, val_loader, device)
            
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            
            # Log to TensorBoard
            writer.add_scalar('Diffusion/train_loss', train_loss, epoch)
            writer.add_scalar('Diffusion/val_loss', val_loss, epoch)
            writer.add_scalar('Diffusion/learning_rate', lr, epoch)
            
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.2e}")
            
            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, output_dir / 'diffusion_best.pt')
                print("  ✓ Saved best model!")
            
            # Generate samples every 10 epochs
            if epoch % 10 == 0:
                print("  Generating samples...")
                samples = generate_samples(model, device, num_samples=8, steps=50)
                
                # Log to TensorBoard
                writer.add_images('Diffusion/samples', samples, epoch)
                
                # Save to disk
                save_image(samples, output_dir / 'diffusion_samples' / f'samples_epoch_{epoch:04d}.png',
                          nrow=4, normalize=True)
            
            # Extended generation every 50 epochs
            if epoch % 50 == 0:
                print("  Extended generation (16 samples, 100 steps)...")
                samples_hq = generate_samples(model, device, num_samples=16, steps=100)
                writer.add_images('Diffusion/samples_hq', samples_hq, epoch)
                save_image(samples_hq, output_dir / 'diffusion_samples' / f'samples_hq_epoch_{epoch:04d}.png',
                          nrow=4, normalize=True)
                
                # Interpolation in latent space
                print("  Generating latent interpolations...")
                with torch.no_grad():
                    z1 = torch.randn(1, config.latent_channels, config.latent_size, config.latent_size, device=device)
                    z2 = torch.randn(1, config.latent_channels, config.latent_size, config.latent_size, device=device)
                    
                    interpolations = []
                    for alpha in np.linspace(0, 1, 8):
                        z_interp = (1 - alpha) * z1 + alpha * z2
                        img = model.decode(z_interp)
                        interpolations.append(img)
                    
                    interp_grid = torch.cat(interpolations, dim=0)
                    writer.add_images('Diffusion/interpolations', interp_grid, epoch)
                    save_image(interp_grid, output_dir / 'diffusion_samples' / f'interpolation_epoch_{epoch:04d}.png',
                              nrow=8, normalize=True)
                
                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                }, output_dir / f'diffusion_epoch_{epoch:03d}.pt')
        
        print("\n✅ Diffusion training complete!")
    
    # =========================================================================
    # FINAL
    # =========================================================================
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(config),
    }, output_dir / 'model_final.pt')
    
    print("\n" + "=" * 70)
    print(f"✅ Training complete! Output: {output_dir}")
    print("=" * 70)
    
    writer.close()


if __name__ == "__main__":
    main()

