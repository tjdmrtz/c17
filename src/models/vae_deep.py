"""
Deep VAE for high-resolution (512×512) crystallographic patterns.

Architecture optimized for:
- 512×512 RGB input
- ~8.5k samples dataset
- Maximum depth without overfitting
- Strong regularization (dropout, weight decay, batch norm)

Total: ~40 convolutional layers with SE attention and residual connections.
Parameters: ~8M (balanced for 8.5k samples)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math


class SEBlock(nn.Module):
    """Squeeze-and-Excitation for channel attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    """
    Residual block with SE attention and dropout.
    
    Each block has 3 conv layers + SE + residual connection.
    """
    def __init__(self, in_ch: int, out_ch: int, 
                 stride: int = 1, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        
        self.se = SEBlock(out_ch)
        self.dropout = nn.Dropout2d(dropout)
        
        # Shortcut
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out = self.dropout(out)
        out = F.relu(out + self.shortcut(x))
        return out


class ResBlockUp(nn.Module):
    """Residual block with upsampling."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        
        self.se = SEBlock(out_ch)
        self.dropout = nn.Dropout2d(dropout)
        
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
    
    def forward(self, x):
        x_up = self.up(x)
        out = F.relu(self.bn1(self.conv1(x_up)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out = self.dropout(out)
        out = F.relu(out + self.shortcut(x))
        return out


class DeepEncoder(nn.Module):
    """
    Deep encoder for 512×512 RGB input.
    
    Architecture: 512 → 256 → 128 → 64 → 32 → 16 → 8 → 4
    8 stages, each with 1-2 residual blocks.
    Total conv layers: ~20
    """
    def __init__(self, latent_dim: int = 256, base_ch: int = 32, dropout: float = 0.1):
        super().__init__()
        
        c = base_ch  # 32
        
        # Stem: 512 → 256
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        
        # Stage 1: 256 → 128 (c → c*2 = 32 → 64)
        self.stage1 = nn.Sequential(
            ResBlock(c, c*2, stride=2, dropout=dropout),
            ResBlock(c*2, c*2, dropout=dropout),
        )
        
        # Stage 2: 128 → 64 (64 → 96)
        self.stage2 = nn.Sequential(
            ResBlock(c*2, c*3, stride=2, dropout=dropout),
            ResBlock(c*3, c*3, dropout=dropout),
        )
        
        # Stage 3: 64 → 32 (96 → 128)
        self.stage3 = nn.Sequential(
            ResBlock(c*3, c*4, stride=2, dropout=dropout),
            ResBlock(c*4, c*4, dropout=dropout),
        )
        
        # Stage 4: 32 → 16 (128 → 192)
        self.stage4 = nn.Sequential(
            ResBlock(c*4, c*6, stride=2, dropout=dropout),
            ResBlock(c*6, c*6, dropout=dropout),
        )
        
        # Stage 5: 16 → 8 (192 → 256)
        self.stage5 = nn.Sequential(
            ResBlock(c*6, c*8, stride=2, dropout=dropout),
            ResBlock(c*8, c*8, dropout=dropout),
        )
        
        # Stage 6: 8 → 4 (256 → 256)
        self.stage6 = nn.Sequential(
            ResBlock(c*8, c*8, stride=2, dropout=dropout),
        )
        
        # Global pooling + latent projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(c*8, latent_dim)
        self.fc_logvar = nn.Linear(c*8, latent_dim)
        
        self.out_channels = c * 8
    
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)      # 512 → 256
        x = self.stage1(x)    # 256 → 128
        x = self.stage2(x)    # 128 → 64
        x = self.stage3(x)    # 64 → 32
        x = self.stage4(x)    # 32 → 16
        x = self.stage5(x)    # 16 → 8
        x = self.stage6(x)    # 8 → 4
        
        x = self.pool(x).flatten(1)
        
        return self.fc_mu(x), self.fc_logvar(x)


class DeepDecoder(nn.Module):
    """
    Deep decoder for 512×512 RGB output.
    
    Architecture: 4 → 8 → 16 → 32 → 64 → 128 → 256 → 512
    7 upsampling stages with residual blocks.
    Total conv layers: ~20
    """
    def __init__(self, latent_dim: int = 256, base_ch: int = 32, dropout: float = 0.1):
        super().__init__()
        
        c = base_ch  # 32
        self.c = c
        
        # Project latent to spatial
        self.fc = nn.Linear(latent_dim, c*8 * 4 * 4)
        
        # Stage 1: 4 → 8 (256 → 256)
        self.stage1 = ResBlockUp(c*8, c*8, dropout=dropout)
        
        # Stage 2: 8 → 16 (256 → 192)
        self.stage2 = ResBlockUp(c*8, c*6, dropout=dropout)
        
        # Stage 3: 16 → 32 (192 → 128)
        self.stage3 = ResBlockUp(c*6, c*4, dropout=dropout)
        
        # Stage 4: 32 → 64 (128 → 96)
        self.stage4 = ResBlockUp(c*4, c*3, dropout=dropout)
        
        # Stage 5: 64 → 128 (96 → 64)
        self.stage5 = ResBlockUp(c*3, c*2, dropout=dropout)
        
        # Stage 6: 128 → 256 (64 → 32)
        self.stage6 = ResBlockUp(c*2, c, dropout=dropout)
        
        # Stage 7: 256 → 512 (32 → 16)
        self.stage7 = ResBlockUp(c, c//2, dropout=dropout)
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(c//2, c//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c//2, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.c*8, 4, 4)
        
        x = self.stage1(x)   # 4 → 8
        x = self.stage2(x)   # 8 → 16
        x = self.stage3(x)   # 16 → 32
        x = self.stage4(x)   # 32 → 64
        x = self.stage5(x)   # 64 → 128
        x = self.stage6(x)   # 128 → 256
        x = self.stage7(x)   # 256 → 512
        
        return self.output(x)


class DeepCrystallographicVAE(nn.Module):
    """
    Deep VAE for 512×512 RGB crystallographic patterns.
    
    Architecture:
    - Encoder: 7 stages with residual blocks (~20 conv layers)
    - Decoder: 7 stages with residual blocks (~20 conv layers)
    - Total: ~42 convolutional layers
    - SE attention in every block
    - Dropout regularization throughout
    
    Parameters: ~8M (optimized for 8.5k sample dataset)
    
    Recommended hyperparameters:
    - beta: 0.3-0.5 (lower for better reconstruction)
    - lr: 5e-4
    - weight_decay: 1e-4
    - dropout: 0.1
    """
    
    def __init__(self,
                 latent_dim: int = 256,
                 base_channels: int = 32,
                 dropout: float = 0.1,
                 num_classes: int = 17):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.input_size = 512
        self.num_classes = num_classes
        
        self.encoder = DeepEncoder(latent_dim, base_channels, dropout)
        self.decoder = DeepDecoder(latent_dim, base_channels, dropout)
        
        # Classifier with regularization
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim // 2, num_classes)
        )
    
    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        reconstruction = self.decode(z)
        class_logits = self.classifier(mu)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'class_logits': class_logits
        }
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
    
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, 
                    steps: int = 10) -> torch.Tensor:
        mu1, _ = self.encode(x1)
        mu2, _ = self.encode(x2)
        
        alphas = torch.linspace(0, 1, steps, device=x1.device)
        return torch.cat([self.decode((1-a)*mu1 + a*mu2) for a in alphas], dim=0)


def count_layers(model: nn.Module) -> dict:
    """Count different types of layers in the model."""
    counts = {
        'Conv2d': 0,
        'ConvTranspose2d': 0,
        'Linear': 0,
        'BatchNorm2d': 0,
        'Dropout': 0,
    }
    
    for module in model.modules():
        for layer_type in counts.keys():
            if module.__class__.__name__ == layer_type:
                counts[layer_type] += 1
    
    return counts


if __name__ == "__main__":
    # Test the model
    model = DeepCrystallographicVAE(latent_dim=256, base_channels=32, dropout=0.1)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 60)
    print("DEEP CRYSTALLOGRAPHIC VAE")
    print("=" * 60)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1e6:.1f} MB")
    
    # Count layers
    layer_counts = count_layers(model)
    print(f"\nLayer counts:")
    for layer_type, count in layer_counts.items():
        print(f"  {layer_type}: {count}")
    
    total_conv = layer_counts['Conv2d'] + layer_counts['ConvTranspose2d']
    print(f"\n  Total convolutional: {total_conv}")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    x = torch.randn(2, 3, 512, 512)
    
    with torch.no_grad():
        outputs = model(x)
    
    print(f"  Input: {x.shape}")
    print(f"  Reconstruction: {outputs['reconstruction'].shape}")
    print(f"  Latent (mu): {outputs['mu'].shape}")
    print(f"  Class logits: {outputs['class_logits'].shape}")
    
    print("\n✅ Model ready for training!")





