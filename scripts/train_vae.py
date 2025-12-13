#!/usr/bin/env python3
"""
Script to train the Crystallographic VAE with comprehensive TensorBoard logging.

This script trains a Variational Autoencoder on the 17 wallpaper groups,
learning a continuous latent representation of crystallographic patterns.

All metrics, reconstructions, samples, activations, and embeddings are logged
to TensorBoard for visualization.

Usage:
    python scripts/train_vae.py --epochs 100 --latent-dim 128
    python scripts/train_vae.py --resume checkpoints/best_model.pt
    
    # View TensorBoard:
    tensorboard --logdir=./runs
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from src.dataset import CrystallographicDataset, create_dataloaders
from src.models import CrystallographicVAE, VAELoss, VAETrainer
from src.models.trainer import plot_training_curves


def main():
    parser = argparse.ArgumentParser(
        description="Train the Crystallographic VAE with TensorBoard logging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--samples-per-group', '-n', type=int, default=200,
                       help='Samples per wallpaper group')
    parser.add_argument('--resolution', '-r', type=int, default=128,
                       help='Image resolution')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Batch size')
    
    # Model arguments
    parser.add_argument('--latent-dim', '-l', type=int, default=128,
                       help='Latent space dimension')
    parser.add_argument('--base-channels', type=int, default=32,
                       help='Base channels in conv layers')
    
    # Training arguments
    parser.add_argument('--epochs', '-e', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='KL divergence weight (β-VAE)')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Classification loss weight')
    
    # Logging arguments
    parser.add_argument('--log-interval', type=int, default=20,
                       help='Batches between scalar logging')
    parser.add_argument('--image-log-interval', type=int, default=100,
                       help='Batches between image/activation logging')
    parser.add_argument('--visualize-every', type=int, default=5,
                       help='Epochs between full visualization logging')
    parser.add_argument('--embedding-every', type=int, default=10,
                       help='Epochs between embedding logging')
    
    # Directory arguments
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='./runs',
                       help='TensorBoard log directory')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name (default: timestamp)')
    
    # Other arguments
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to train on')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Experiment name
    if args.experiment_name is None:
        args.experiment_name = f"vae_lat{args.latent_dim}_beta{args.beta}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("=" * 70)
    print("CRYSTALLOGRAPHIC VAE TRAINING")
    print("=" * 70)
    print(f"\n📊 Configuration:")
    print(f"  Samples per group: {args.samples_per_group}")
    print(f"  Resolution: {args.resolution}x{args.resolution}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  β (KL weight): {args.beta}")
    print(f"  γ (Classification weight): {args.gamma}")
    print(f"\n📁 Directories:")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  TensorBoard: {log_dir}/{args.experiment_name}")
    print()
    
    # Create data loaders
    print("📦 Creating dataset...")
    train_loader, val_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_samples_per_group=args.samples_per_group,
        resolution=args.resolution,
        train_split=0.85,
        seed=args.seed
    )
    
    total_samples = args.samples_per_group * 17
    print(f"  Total samples: {total_samples}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create model
    print("\n🧠 Creating model...")
    model = CrystallographicVAE(
        in_channels=1,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        input_size=args.resolution,
        num_classes=17
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create trainer with TensorBoard logging
    print("\n🚀 Initializing trainer...")
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
        log_interval=args.log_interval,
        image_log_interval=args.image_log_interval
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n📂 Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    print("\n" + "=" * 70)
    print("📈 To monitor training, run in another terminal:")
    print(f"   tensorboard --logdir={args.log_dir}")
    print("=" * 70)
    
    # Train
    history = trainer.train(
        num_epochs=args.epochs,
        save_every=10,
        visualize_every=args.visualize_every,
        embedding_every=args.embedding_every
    )
    
    # Plot and save training curves
    output_dir = Path(args.log_dir) / args.experiment_name
    curves_path = output_dir / 'training_curves.png'
    
    import matplotlib
    matplotlib.use('Agg')
    
    fig = plot_training_curves(history, save_path=str(curves_path))
    print(f"\n📊 Training curves saved to: {curves_path}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n📈 Results:")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Best val loss: {trainer.best_val_loss:.4f}")
    print(f"  Final learning rate: {history['lr'][-1]:.2e}")
    
    print(f"\n📁 Output files:")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  TensorBoard logs: {output_dir}")
    print(f"  Best model: {checkpoint_dir}/best_model.pt")
    
    print(f"\n🔍 To view TensorBoard:")
    print(f"   tensorboard --logdir={args.log_dir}")
    print("\n   Then open: http://localhost:6006")


if __name__ == "__main__":
    main()
