#!/usr/bin/env python3
"""
Train VAE on colored crystallographic patterns.

Each wallpaper group has its unique color palette, making the
patterns visually distinctive and easier to learn.

Usage:
    python scripts/train_colored_vae.py --samples-per-group 500 --epochs 100
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np

from src.dataset import create_colored_dataloaders
from src.models import VAELoss, VAETrainer


class ColoredVAE(nn.Module):
    """
    VAE for RGB colored crystallographic patterns.
    
    Same architecture as base VAE but with 3 input/output channels.
    """
    
    def __init__(self,
                 latent_dim: int = 128,
                 base_channels: int = 32,
                 input_size: int = 128,
                 num_classes: int = 17):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.num_classes = num_classes
        
        c = base_channels
        
        # Encoder
        self.encoder = nn.Sequential(
            # 128 -> 64
            nn.Conv2d(3, c, 7, stride=2, padding=3),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            
            # 32 -> 16
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            
            # 16 -> 8
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            
            # 8 -> 4
            nn.Conv2d(c * 4, c * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
        )
        
        self.fc_mu = nn.Linear(c * 8 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(c * 8 * 4 * 4, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, c * 8 * 4 * 4)
        
        self.decoder = nn.Sequential(
            # 4 -> 8
            nn.ConvTranspose2d(c * 8, c * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            
            # 8 -> 16
            nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            
            # 16 -> 32
            nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            
            # 32 -> 64
            nn.ConvTranspose2d(c, c // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(c // 2),
            nn.ReLU(inplace=True),
            
            # 64 -> 128
            nn.ConvTranspose2d(c // 2, c // 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(c // 4),
            nn.ReLU(inplace=True),
            
            # Output: 3 channels (RGB)
            nn.Conv2d(c // 4, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.base_channels = c
        
        # Classifier from latent
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim // 2, num_classes)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, self.base_channels * 8, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        recon = self.decode(z)
        class_logits = self.classifier(mu)
        
        return {
            'reconstruction': recon,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'class_logits': class_logits
        }
    
    def sample(self, num_samples: int, device):
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
    
    def interpolate(self, x1, x2, steps=10):
        mu1, _ = self.encode(x1)
        mu2, _ = self.encode(x2)
        
        alphas = torch.linspace(0, 1, steps, device=x1.device)
        return torch.cat([self.decode((1-a)*mu1 + a*mu2) for a in alphas], dim=0)


def main():
    parser = argparse.ArgumentParser(
        description="Train Colored VAE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--samples-per-group', '-n', type=int, default=500)
    parser.add_argument('--resolution', '-r', type=int, default=128)
    parser.add_argument('--batch-size', '-b', type=int, default=32)
    parser.add_argument('--latent-dim', '-l', type=int, default=128)
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--log-dir', type=str, default='./runs')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    exp_name = f"colored_vae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("=" * 70)
    print("🎨 COLORED CRYSTALLOGRAPHIC VAE TRAINING")
    print("=" * 70)
    print(f"\nSamples per group: {args.samples_per_group}")
    print(f"Total samples: {args.samples_per_group * 17}")
    print(f"Resolution: {args.resolution}x{args.resolution} RGB")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Epochs: {args.epochs}")
    
    # Create colored dataloaders
    print("\n📦 Creating colored dataset...")
    train_loader, val_loader = create_colored_dataloaders(
        batch_size=args.batch_size,
        num_samples_per_group=args.samples_per_group,
        resolution=args.resolution,
        train_split=0.85,
        seed=args.seed
    )
    
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    
    # Create model
    print("\n🧠 Creating Colored VAE...")
    model = ColoredVAE(
        latent_dim=args.latent_dim,
        base_channels=32,
        input_size=args.resolution,
        num_classes=17
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # Create trainer
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        beta=args.beta,
        gamma=0.1,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        experiment_name=exp_name,
        log_interval=50,
        image_log_interval=100
    )
    
    print(f"\n📈 TensorBoard: tensorboard --logdir={args.log_dir}")
    print("=" * 70)
    
    # Train
    history = trainer.train(
        num_epochs=args.epochs,
        save_every=10,
        visualize_every=5,
        embedding_every=10
    )
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best val loss: {trainer.best_val_loss:.4f}")
    print(f"  Model: {args.checkpoint_dir}/best_model.pt")


if __name__ == "__main__":
    main()






