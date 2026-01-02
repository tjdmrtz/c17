#!/usr/bin/env python3
"""
Train Deep VAE on 512×512 colored crystallographic patterns.

Architecture:
- ~42 convolutional layers
- ~8M parameters
- SE attention in every block
- Strong regularization (dropout, weight decay)

Optimized for 8.5k sample dataset without overfitting.

Usage:
    python scripts/train_deep_vae.py --epochs 150 --beta 0.3
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from src.dataset.dataset_colored import ColoredCrystallographicDataset
from src.models.vae_deep import DeepCrystallographicVAE, count_layers
from src.models import VAELoss, VAETrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train Deep VAE for 512×512 patterns",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    parser.add_argument('--data-dir', type=str, 
                       default='./data/colored_crystallographic',
                       help='Dataset directory (with images/ subfolder)')
    parser.add_argument('--samples-per-group', '-n', type=int, default=500)
    parser.add_argument('--resolution', '-r', type=int, default=512)
    parser.add_argument('--batch-size', '-b', type=int, default=8,
                       help='Batch size (small for 512×512)')
    
    # Model
    parser.add_argument('--latent-dim', '-l', type=int, default=256)
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate for regularization')
    
    # Training
    parser.add_argument('--epochs', '-e', type=int, default=150)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='L2 regularization')
    parser.add_argument('--beta', type=float, default=0.3,
                       help='KL weight (lower = better reconstruction)')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Classification loss weight')
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='./runs')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--visualize-every', type=int, default=5)
    parser.add_argument('--embedding-every', type=int, default=10)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # Optimize for 512×512
        torch.backends.cudnn.benchmark = True
    
    exp_name = f"deep_vae_512_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("=" * 70)
    print("🔬 DEEP CRYSTALLOGRAPHIC VAE (512×512)")
    print("=" * 70)
    
    # Create dataset
    print(f"\n📦 Loading dataset from {args.data_dir}...")
    
    dataset = ColoredCrystallographicDataset(
        num_samples_per_group=args.samples_per_group,
        resolution=args.resolution,
        seed=args.seed
    )
    
    total_samples = len(dataset)
    train_size = int(0.85 * total_samples)
    val_size = total_samples - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"  Total samples: {total_samples}")
    print(f"  Train: {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"  Val: {len(val_dataset)} ({len(val_loader)} batches)")
    print(f"  Resolution: {args.resolution}×{args.resolution} RGB")
    
    # Create model
    print(f"\n🧠 Creating Deep VAE...")
    model = DeepCrystallographicVAE(
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        dropout=args.dropout,
        num_classes=17
    )
    
    # Count everything
    total_params = sum(p.numel() for p in model.parameters())
    layer_counts = count_layers(model)
    total_conv = layer_counts['Conv2d']
    
    print(f"  Parameters: {total_params:,}")
    print(f"  Conv layers: {total_conv}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Dropout: {args.dropout}")
    
    # Ratio check
    ratio = total_samples / total_params
    print(f"\n  📊 Samples/Parameters ratio: {ratio:.4f}")
    if ratio < 0.001:
        print(f"  ⚠️  Low ratio - using strong regularization")
    else:
        print(f"  ✅ Good ratio for training")
    
    print(f"\n⚙️ Training Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Beta (KL): {args.beta}")
    print(f"  Batch size: {args.batch_size}")
    
    # Create trainer
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
        experiment_name=exp_name,
        log_interval=20,
        image_log_interval=50
    )
    
    # Apply weight decay manually (since trainer uses AdamW)
    trainer.optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    print(f"\n📈 TensorBoard: tensorboard --logdir={args.log_dir}")
    print("=" * 70)
    
    # Train
    history = trainer.train(
        num_epochs=args.epochs,
        save_every=15,
        visualize_every=args.visualize_every,
        embedding_every=args.embedding_every
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n📈 Final Results:")
    print(f"  Best val loss: {trainer.best_val_loss:.4f}")
    print(f"  Final recon loss: {history['val_recon_loss'][-1]:.4f}")
    print(f"  Final KL loss: {history['val_kl_loss'][-1]:.4f}")
    
    # Check for overfitting
    train_final = history['train_loss'][-1]
    val_final = history['val_loss'][-1]
    gap = val_final - train_final
    
    print(f"\n  Train loss: {train_final:.4f}")
    print(f"  Val loss: {val_final:.4f}")
    print(f"  Gap: {gap:.4f}", end="")
    
    if gap > 0.5:
        print(" ⚠️ (possible overfitting)")
    elif gap < 0.1:
        print(" ✅ (good generalization)")
    else:
        print(" (acceptable)")
    
    print(f"\n📁 Model saved: {args.checkpoint_dir}/best_model.pt")


if __name__ == "__main__":
    main()






