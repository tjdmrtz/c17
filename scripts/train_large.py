#!/usr/bin/env python3
"""
Train the Large Crystallographic VAE on a big dataset.

This script:
1. Generates a large dataset (2000+ samples per group)
2. Trains the large VAE model with attention and deep residuals
3. Logs everything to TensorBoard
4. Saves the best model

Usage:
    python scripts/train_large.py --epochs 100 --samples-per-group 2000
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from src.dataset import CrystallographicDataset, create_dataloaders
from src.models.vae_large import CrystallographicVAELarge, count_parameters
from src.models import VAELoss, VAETrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train Large Crystallographic VAE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--samples-per-group', '-n', type=int, default=2000,
                       help='Samples per wallpaper group (17 groups)')
    parser.add_argument('--resolution', '-r', type=int, default=128,
                       help='Image resolution')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                       help='Batch size (larger for big dataset)')
    
    # Model arguments
    parser.add_argument('--latent-dim', '-l', type=int, default=256,
                       help='Latent space dimension')
    parser.add_argument('--base-channels', type=int, default=64,
                       help='Base channels (64 for large model)')
    
    # Training arguments
    parser.add_argument('--epochs', '-e', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (lower for large model)')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='KL weight (0.5 is good balance)')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Classification loss weight')
    
    # Logging arguments
    parser.add_argument('--visualize-every', type=int, default=5,
                       help='Epochs between visualizations')
    parser.add_argument('--embedding-every', type=int, default=10,
                       help='Epochs between embedding logging')
    
    # Directory arguments
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='./runs',
                       help='TensorBoard log directory')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--num-workers', type=int, default=4,
                       help='DataLoader workers')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Experiment name
    if args.experiment_name is None:
        args.experiment_name = f"large_vae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Print configuration
    print("=" * 70)
    print("LARGE CRYSTALLOGRAPHIC VAE TRAINING")
    print("=" * 70)
    
    total_samples = args.samples_per_group * 17
    print(f"\n📊 Dataset Configuration:")
    print(f"  Samples per group: {args.samples_per_group}")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Resolution: {args.resolution}x{args.resolution}")
    print(f"  Batch size: {args.batch_size}")
    
    print(f"\n🧠 Model Configuration:")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Base channels: {args.base_channels}")
    
    print(f"\n⚙️ Training Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Beta (KL weight): {args.beta}")
    print(f"  Gamma (class weight): {args.gamma}")
    
    # Create dataset
    print(f"\n📦 Creating large dataset...")
    print(f"  This will generate {total_samples:,} samples...")
    
    train_loader, val_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_samples_per_group=args.samples_per_group,
        resolution=args.resolution,
        train_split=0.9,  # More training data for large dataset
        seed=args.seed
    )
    
    print(f"  Train samples: {len(train_loader.dataset):,}")
    print(f"  Val samples: {len(val_loader.dataset):,}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create model
    print(f"\n🏗️ Creating Large VAE model...")
    model = CrystallographicVAELarge(
        in_channels=1,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        input_size=args.resolution,
        num_classes=17
    )
    
    num_params = count_parameters(model)
    print(f"  Total parameters: {num_params:,}")
    print(f"  Model size: ~{num_params * 4 / 1e6:.1f} MB")
    
    # Create trainer
    print(f"\n🚀 Initializing trainer...")
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        beta=args.beta,
        gamma=args.gamma,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        log_interval=50,
        image_log_interval=200
    )
    
    print(f"\n" + "=" * 70)
    print(f"📈 To monitor training:")
    print(f"   tensorboard --logdir={args.log_dir}")
    print("=" * 70)
    
    # Train
    history = trainer.train(
        num_epochs=args.epochs,
        save_every=10,
        visualize_every=args.visualize_every,
        embedding_every=args.embedding_every
    )
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n📈 Final Results:")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Best val loss: {trainer.best_val_loss:.4f}")
    print(f"  Final recon loss: {history['val_recon_loss'][-1]:.4f}")
    print(f"  Final KL loss: {history['val_kl_loss'][-1]:.4f}")
    
    print(f"\n📁 Output files:")
    print(f"  Best model: {args.checkpoint_dir}/best_model.pt")
    print(f"  TensorBoard: {args.log_dir}/{args.experiment_name}")
    
    print(f"\n🔍 To view results:")
    print(f"   tensorboard --logdir={args.log_dir}")


if __name__ == "__main__":
    main()








