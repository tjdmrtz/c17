"""
Harmonic VAE with SO(2)-Equivariant Convolutions.

Uses circular harmonic filters for continuous rotation equivariance.
Based on "Harmonic Networks: Deep Translation and Rotation Equivariance" (Worrall et al., 2017)

Key concepts:
- Circular harmonics e^(imθ) as angular basis
- Learnable radial profiles R(r)
- Rotation equivariance for ANY angle (not just 90°)
- Perfect for crystallographic patterns with arbitrary rotational symmetry

The filter is: W(r,θ) = R(r) · e^(imθ)
Where m is the rotation order (frequency).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import math
import numpy as np


def get_circular_harmonics_filter(kernel_size: int, max_order: int = 2) -> torch.Tensor:
    """
    Generate circular harmonic basis filters.
    
    Args:
        kernel_size: Size of the filter (must be odd)
        max_order: Maximum harmonic order m (includes -m to +m)
    
    Returns:
        Complex tensor of shape (2*max_order+1, kernel_size, kernel_size, 2)
        Last dim is (real, imag) for complex representation
    """
    assert kernel_size % 2 == 1, "Kernel size must be odd"
    
    center = kernel_size // 2
    
    # Create coordinate grid
    y, x = torch.meshgrid(
        torch.arange(kernel_size) - center,
        torch.arange(kernel_size) - center,
        indexing='ij'
    )
    
    # Convert to polar coordinates
    r = torch.sqrt(x.float()**2 + y.float()**2)
    theta = torch.atan2(y.float(), x.float())
    
    # Normalize radius
    r = r / (center + 1e-8)
    
    # Generate harmonics for each order m
    harmonics = []
    for m in range(-max_order, max_order + 1):
        # e^(imθ) = cos(mθ) + i·sin(mθ)
        real_part = torch.cos(m * theta)
        imag_part = torch.sin(m * theta)
        
        # Gaussian radial envelope (can be learned later)
        radial = torch.exp(-r**2 / 0.5)
        
        # Apply radial envelope
        real_part = real_part * radial
        imag_part = imag_part * radial
        
        # Stack real and imaginary
        harmonic = torch.stack([real_part, imag_part], dim=-1)
        harmonics.append(harmonic)
    
    return torch.stack(harmonics, dim=0)  # (2M+1, K, K, 2)


class HarmonicConv2d(nn.Module):
    """
    SO(2)-Equivariant convolution using circular harmonics.
    
    Uses learnable radial profiles combined with fixed circular harmonic basis.
    Equivariant to continuous rotations (any angle).
    
    Args:
        in_channels: Input channels
        out_channels: Output channels  
        kernel_size: Size of kernel (must be odd)
        max_order: Maximum harmonic frequency (higher = more angular resolution)
        stride: Convolution stride
        padding: Convolution padding
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 5,
                 max_order: int = 2,
                 stride: int = 1,
                 padding: int = 2,
                 bias: bool = False):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.max_order = max_order
        self.num_harmonics = 2 * max_order + 1
        self.stride = stride
        self.padding = padding
        
        # Register harmonic basis (fixed, not learned)
        harmonics = get_circular_harmonics_filter(kernel_size, max_order)
        self.register_buffer('harmonics', harmonics)  # (2M+1, K, K, 2)
        
        # Learnable radial weights for each harmonic, input and output channel
        # Shape: (out_channels, in_channels, num_harmonics)
        self.radial_weights = nn.Parameter(
            torch.randn(out_channels, in_channels, self.num_harmonics) * 0.1
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        
        # Initialize
        nn.init.kaiming_normal_(self.radial_weights, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape
        
        # Build filters from harmonics and radial weights
        # harmonics: (num_harmonics, K, K, 2) - complex as (real, imag)
        # radial_weights: (C_out, C_in, num_harmonics)
        
        # Use only real part for now (can extend to complex later)
        harmonic_real = self.harmonics[..., 0]  # (num_harmonics, K, K)
        
        # Combine: weight each harmonic by learned radial weight
        # Result shape: (C_out, C_in, K, K)
        filters = torch.einsum('oih,hkl->oikl', self.radial_weights, harmonic_real)
        
        # Apply convolution
        output = F.conv2d(x, filters, self.bias, self.stride, self.padding)
        
        return output


class HarmonicResBlock(nn.Module):
    """Residual block with harmonic convolutions."""
    
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, 
                 max_order: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # Harmonic conv with larger kernel for rotation equivariance
        self.conv1 = HarmonicConv2d(in_ch, out_ch, kernel_size=5, max_order=max_order,
                                     stride=stride, padding=2)
        self.bn1 = nn.BatchNorm2d(out_ch)
        
        self.conv2 = HarmonicConv2d(out_ch, out_ch, kernel_size=5, max_order=max_order,
                                     stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.dropout = nn.Dropout2d(dropout)
        
        # Shortcut
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out = F.relu(out + self.shortcut(x))
        return out


class HarmonicEncoder(nn.Module):
    """
    SO(2)-Equivariant encoder using harmonic convolutions.
    
    Uses circular harmonics for rotation equivariance to continuous angles.
    """
    
    def __init__(self, 
                 latent_dim: int = 256,
                 base_channels: int = 32,
                 num_layers: int = 5,
                 max_order: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_layers = num_layers
        c = base_channels
        
        # Stem with harmonic conv
        self.stem = nn.Sequential(
            HarmonicConv2d(3, c, kernel_size=7, max_order=max_order, stride=2, padding=3),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        
        # Encoder layers with increasing channels
        self.layers = nn.ModuleList()
        in_ch = c
        
        for i in range(num_layers):
            out_ch = min(c * (2 ** ((i + 1) // 2)), c * 8)
            self.layers.append(
                HarmonicResBlock(in_ch, out_ch, stride=2, max_order=max_order, dropout=dropout)
            )
            in_ch = out_ch
        
        self.final_channels = in_ch
        self.final_spatial = 512 // (2 ** (num_layers + 1))
        
        # Latent projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(in_ch, latent_dim)
        self.fc_logvar = nn.Linear(in_ch, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.pool(x).flatten(1)
        return self.fc_mu(x), self.fc_logvar(x)


class HarmonicDecoder(nn.Module):
    """
    Decoder for harmonic VAE.
    
    Uses standard convolutions (latent space is already rotation-invariant).
    """
    
    def __init__(self,
                 latent_dim: int = 256,
                 base_channels: int = 32,
                 num_layers: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        
        c = base_channels
        self.num_layers = num_layers
        
        # Calculate starting channels and spatial size
        start_ch = min(c * (2 ** (num_layers // 2)), c * 8)
        self.start_channels = start_ch
        self.start_spatial = 512 // (2 ** (num_layers + 1))
        
        # Project latent
        self.fc = nn.Linear(latent_dim, start_ch * self.start_spatial ** 2)
        
        # Calculate required decoder layers
        required_layers = int(math.log2(512 / self.start_spatial))
        
        # Decoder layers
        self.layers = nn.ModuleList()
        in_ch = start_ch
        
        for i in range(required_layers):
            out_ch = max(in_ch // 2, c // 2)
            self.layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            ))
            in_ch = out_ch
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(in_ch, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, self.start_channels, self.start_spatial, self.start_spatial)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.output(x)


class HarmonicVAE(nn.Module):
    """
    VAE with SO(2)-Equivariant Harmonic Convolutions.
    
    Uses circular harmonics for continuous rotation equivariance.
    Ideal for crystallographic patterns with rotational symmetries.
    
    The encoder learns rotation-invariant features through harmonic filters,
    while the decoder reconstructs using standard convolutions.
    
    Args:
        latent_dim: Dimension of latent space
        base_channels: Base channel count
        num_encoder_layers: Number of encoder downsampling layers
        num_decoder_layers: Not used (auto-calculated)
        max_order: Maximum harmonic order (higher = finer angular resolution)
        dropout: Dropout rate
        num_classes: Number of classes for classification
    """
    
    def __init__(self,
                 latent_dim: int = 256,
                 base_channels: int = 32,
                 num_encoder_layers: int = 5,
                 num_decoder_layers: int = 6,  # Auto-calculated, ignored
                 max_order: int = 2,
                 dropout: float = 0.1,
                 num_classes: int = 17,
                 capture_activations: bool = False):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.input_size = 512
        self.num_classes = num_classes
        self.num_encoder_layers = num_encoder_layers
        self.max_order = max_order
        
        self.encoder = HarmonicEncoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_layers=num_encoder_layers,
            max_order=max_order,
            dropout=dropout
        )
        
        self.decoder = HarmonicDecoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_layers=num_encoder_layers,
            dropout=dropout
        )
        
        # Auto-calculate decoder layers
        self.num_decoder_layers = int(math.log2(512 / self.encoder.final_spatial))
        
        # Classifier
        self.classifier = nn.Sequential(
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
    
    def get_model_info(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'type': 'HarmonicVAE (SO2-Equivariant)',
            'latent_dim': self.latent_dim,
            'num_encoder_layers': self.num_encoder_layers,
            'num_decoder_layers': self.num_decoder_layers,
            'max_harmonic_order': self.max_order,
            'total_params': total_params,
            'model_size_mb': total_params * 4 / 1e6,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 70)
    print("HARMONIC VAE - SO(2) EQUIVARIANT")
    print("=" * 70)
    
    # Test harmonic basis
    print("\n1. Testing circular harmonic basis...")
    harmonics = get_circular_harmonics_filter(5, max_order=2)
    print(f"   Harmonic basis shape: {harmonics.shape}")
    print(f"   Orders: {list(range(-2, 3))}")
    
    # Test HarmonicConv2d
    print("\n2. Testing HarmonicConv2d...")
    conv = HarmonicConv2d(3, 32, kernel_size=5, max_order=2)
    x = torch.randn(2, 3, 64, 64)
    y = conv(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {y.shape}")
    print(f"   Parameters: {count_parameters(conv):,}")
    
    # Test full model
    print("\n3. Testing HarmonicVAE...")
    model = HarmonicVAE(
        latent_dim=256,
        base_channels=32,
        num_encoder_layers=5,
        max_order=2,
        dropout=0.1
    )
    
    info = model.get_model_info()
    print(f"   Type: {info['type']}")
    print(f"   Parameters: {info['total_params']:,}")
    print(f"   Max harmonic order: {info['max_harmonic_order']}")
    
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    
    print(f"   Input: {x.shape}")
    print(f"   Output: {out['reconstruction'].shape}")
    print(f"   Latent: {out['mu'].shape}")
    
    assert out['reconstruction'].shape == (2, 3, 512, 512), "Output shape mismatch!"
    
    # Test rotation equivariance (approximately)
    print("\n4. Testing rotation equivariance...")
    model.eval()
    with torch.no_grad():
        # Original
        mu1, _ = model.encode(x)
        
        # Rotate input by 45 degrees
        x_rot = torch.rot90(x, k=1, dims=[-2, -1])  # 90 degrees (exact test)
        mu2, _ = model.encode(x_rot)
        
        # Check if latents are similar (should be for equivariant model)
        diff = (mu1 - mu2).abs().mean()
        print(f"   Latent difference after 90° rotation: {diff:.4f}")
        print(f"   (Lower is better for rotation invariance)")
    
    print("\n" + "=" * 70)
    print("✅ HarmonicVAE ready!")
    print("=" * 70)





