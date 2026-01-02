#!/usr/bin/env python3
"""
Auto-scaling VAE training script.

Automatically selects the appropriate model size based on dataset size:
- < 15k samples → VAE Base (3.5M params)
- 15k-50k samples → VAE Medium (6M params)
- > 50k samples → VAE Large (15M params)

Usage:
    python scripts/train_auto.py --samples-per-group 1000 --epochs 100
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from src.dataset import create_dataloaders
from src.models import (
    CrystallographicVAE, 
    CrystallographicVAEMedium,
    CrystallographicVAELarge,
    VAETrainer,
    get_model_for_dataset_size
)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-scaling VAE Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--samples-per-group', '-n', type=int, default=1000,
                       help='Samples per group (×17 groups)')
    parser.add_argument('--resolution', '-r', type=int, default=128)
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--beta', type=float, default=0.5,
                       help='KL weight')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--log-dir', type=str, default='./runs')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    
    args = parser.parse_args()
    
    # Calculate dataset size
    total_samples = args.samples_per_group * 17
    
    # Get recommended configuration
    config = get_model_for_dataset_size(total_samples, args.resolution)
    
    print("=" * 70)
    print("AUTO-SCALING VAE TRAINING")
    print("=" * 70)
    print(f"\n📊 Dataset:")
    print(f"  Samples per group: {args.samples_per_group}")
    print(f"  Total samples: {total_samples:,}")
    
    print(f"\n🤖 Auto-selected model:")
    print(f"  Model: {config['model_class'].__name__}")
    print(f"  Parameters: {config['params']}")
    print(f"  Latent dim: {config['latent_dim']}")
    print(f"  Base channels: {config['base_channels']}")
    print(f"  Recommended batch size: {config['recommended_batch_size']}")
    print(f"  Recommended LR: {config['recommended_lr']}")
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create dataset
    print(f"\n📦 Creating dataset...")
    train_loader, val_loader = create_dataloaders(
        batch_size=config['recommended_batch_size'],
        num_samples_per_group=args.samples_per_group,
        resolution=args.resolution,
        train_split=0.85,
        seed=args.seed
    )
    print(f"  Train: {len(train_loader.dataset):,} samples")
    print(f"  Val: {len(val_loader.dataset):,} samples")
    
    # Create model
    print(f"\n🏗️ Creating model...")
    model = config['model_class'](
        in_channels=1,
        latent_dim=config['latent_dim'],
        base_channels=config['base_channels'],
        input_size=args.resolution,
        num_classes=17
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Actual parameters: {num_params:,}")
    
    # Experiment name
    exp_name = f"auto_{config['model_class'].__name__}_{total_samples}samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create trainer
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=config['recommended_lr'],
        beta=args.beta,
        gamma=0.1,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        experiment_name=exp_name,
        log_interval=50,
        image_log_interval=200
    )
    
    print(f"\n📈 TensorBoard: tensorboard --logdir={args.log_dir}")
    print("=" * 70)
    
    # Train
    history = trainer.train(
        num_epochs=args.epochs,
        save_every=max(1, args.epochs // 5),
        visualize_every=max(1, args.epochs // 10),
        embedding_every=max(1, args.epochs // 5)
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best val loss: {trainer.best_val_loss:.4f}")
    print(f"  Final recon: {history['val_recon_loss'][-1]:.4f}")
    print(f"  Final KL: {history['val_kl_loss'][-1]:.4f}")
    print(f"\n  Model: {args.checkpoint_dir}/best_model.pt")


if __name__ == "__main__":
    main()








