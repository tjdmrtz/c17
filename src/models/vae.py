"""
Variational Autoencoder (VAE) for Crystallographic Patterns.

Architecture:
- Encoder: CNN that maps images to latent distribution (μ, σ)
- Decoder: Transposed CNN that reconstructs images from latent vectors
- Variational: Reparametrization trick for differentiable sampling

The model learns a continuous latent space where similar symmetry patterns
are clustered together, enabling interpolation and generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class ResidualBlock(nn.Module):
    """Residual block with optional downsampling/upsampling."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 downsample: bool = False, upsample: bool = False):
        super().__init__()
        
        stride = 2 if downsample else 1
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.upsample = upsample
        if upsample:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Shortcut connection
        self.shortcut = nn.Identity()
        if in_channels != out_channels or downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = self.up(x)
        
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        
        return out


class Encoder(nn.Module):
    """
    CNN Encoder for VAE.
    
    Maps input images to latent distribution parameters (μ, log_σ²).
    Uses progressively deeper convolutions with residual connections.
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 latent_dim: int = 128,
                 base_channels: int = 32,
                 input_size: int = 128):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.input_size = input_size
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Residual blocks with progressive downsampling
        # Input: 128 -> 32 after initial
        # Block 1: 32 -> 16
        # Block 2: 16 -> 8
        # Block 3: 8 -> 4
        self.layer1 = ResidualBlock(base_channels, base_channels * 2, downsample=True)
        self.layer2 = ResidualBlock(base_channels * 2, base_channels * 4, downsample=True)
        self.layer3 = ResidualBlock(base_channels * 4, base_channels * 8, downsample=True)
        
        # Calculate flattened size
        # For 128x128 input: 128 -> 32 -> 16 -> 8 -> 4 = 4x4 spatial
        self.flat_size = base_channels * 8 * 4 * 4
        
        # Latent space projections
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution.
        
        Args:
            x: Input images [B, 1, H, W]
            
        Returns:
            mu: Mean of latent distribution [B, latent_dim]
            logvar: Log variance of latent distribution [B, latent_dim]
        """
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class Decoder(nn.Module):
    """
    CNN Decoder for VAE.
    
    Maps latent vectors back to image space using transposed convolutions.
    """
    
    def __init__(self,
                 out_channels: int = 1,
                 latent_dim: int = 128,
                 base_channels: int = 32,
                 output_size: int = 128):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.base_channels = base_channels
        
        # Project latent to spatial feature map
        self.flat_size = base_channels * 8 * 4 * 4
        self.fc = nn.Linear(latent_dim, self.flat_size)
        
        # Upsampling blocks (reverse of encoder)
        # 4 -> 8 -> 16 -> 32 -> 64 -> 128
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(base_channels // 2, base_channels // 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Final output layer
        self.output = nn.Sequential(
            nn.Conv2d(base_channels // 4, out_channels, 3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to image.
        
        Args:
            z: Latent vectors [B, latent_dim]
            
        Returns:
            Reconstructed images [B, 1, H, W]
        """
        x = self.fc(z)
        x = x.view(-1, self.base_channels * 8, 4, 4)
        
        x = self.layer1(x)   # 4 -> 8
        x = self.layer2(x)   # 8 -> 16
        x = self.layer3(x)   # 16 -> 32
        x = self.layer4(x)   # 32 -> 64
        x = self.layer5(x)   # 64 -> 128
        
        x = self.output(x)
        
        return x


class CrystallographicVAE(nn.Module):
    """
    Complete Variational Autoencoder for crystallographic patterns.
    
    The model learns a continuous latent representation where:
    - Similar symmetry patterns cluster together
    - Interpolation between patterns produces valid crystallographic patterns
    - The latent space can be conditioned on symmetry group (optional)
    
    Architecture:
        Input [B, 1, 128, 128] 
        → Encoder → (μ, log σ²) 
        → Reparametrization → z [B, latent_dim]
        → Decoder → Output [B, 1, 128, 128]
    """
    
    def __init__(self,
                 in_channels: int = 1,
                 latent_dim: int = 128,
                 base_channels: int = 32,
                 input_size: int = 128,
                 num_classes: int = 17):
        """
        Initialize the VAE.
        
        Args:
            in_channels: Number of input channels (1 for grayscale)
            latent_dim: Dimension of latent space
            base_channels: Base number of channels in conv layers
            input_size: Input image size (assumed square)
            num_classes: Number of wallpaper groups for optional classification
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.num_classes = num_classes
        
        self.encoder = Encoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            input_size=input_size
        )
        
        self.decoder = Decoder(
            out_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            output_size=input_size
        )
        
        # Optional classifier from latent space
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim // 2, num_classes)
        )
        
    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparametrization trick: z = μ + σ * ε, where ε ~ N(0, 1)
        
        This allows gradients to flow through the sampling operation.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu  # Use mean during inference
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            x: Input images [B, 1, H, W]
            
        Returns:
            Dictionary with:
                - reconstruction: Reconstructed images
                - mu: Latent mean
                - logvar: Latent log variance
                - z: Sampled latent vector
                - class_logits: Classification logits
        """
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        reconstruction = self.decode(z)
        class_logits = self.classifier(mu)  # Use mean for classification
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'class_logits': class_logits
        }
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample new patterns from the prior distribution.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate on
            
        Returns:
            Generated images [num_samples, 1, H, W]
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
    
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, 
                    steps: int = 10) -> torch.Tensor:
        """
        Interpolate between two patterns in latent space.
        
        Args:
            x1: First pattern [1, 1, H, W]
            x2: Second pattern [1, 1, H, W]
            steps: Number of interpolation steps
            
        Returns:
            Interpolated patterns [steps, 1, H, W]
        """
        mu1, _ = self.encode(x1)
        mu2, _ = self.encode(x2)
        
        # Linear interpolation in latent space
        alphas = torch.linspace(0, 1, steps, device=x1.device)
        interpolations = []
        
        for alpha in alphas:
            z = (1 - alpha) * mu1 + alpha * mu2
            interpolations.append(self.decode(z))
        
        return torch.cat(interpolations, dim=0)


class VAELoss(nn.Module):
    """
    Combined loss for VAE training.
    
    Loss = Reconstruction Loss + β * KL Divergence + γ * Classification Loss
    
    The β parameter controls the trade-off between reconstruction quality
    and latent space regularization (β-VAE).
    """
    
    def __init__(self, 
                 beta: float = 1.0,
                 gamma: float = 0.1,
                 reconstruction_loss: str = 'mse'):
        """
        Args:
            beta: Weight for KL divergence (β-VAE)
            gamma: Weight for classification loss
            reconstruction_loss: 'mse' or 'bce'
        """
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.reconstruction_loss = reconstruction_loss
    
    def set_beta(self, beta: float):
        """Update beta value (for KL annealing)."""
        self.beta = beta
        
    def forward(self, 
                outputs: Dict[str, torch.Tensor],
                targets: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss.
        
        Args:
            outputs: VAE output dictionary
            targets: Target images [B, 1, H, W]
            labels: Optional class labels [B]
            
        Returns:
            Dictionary with loss components and total loss
        """
        recon = outputs['reconstruction']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Reconstruction loss
        if self.reconstruction_loss == 'mse':
            recon_loss = F.mse_loss(recon, targets, reduction='sum') / targets.size(0)
        else:  # bce
            recon_loss = F.binary_cross_entropy(recon, targets, reduction='sum') / targets.size(0)
        
        # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / targets.size(0)
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        # Classification loss (optional)
        class_loss = torch.tensor(0.0, device=targets.device)
        if labels is not None and 'class_logits' in outputs:
            class_loss = F.cross_entropy(outputs['class_logits'], labels)
            total_loss = total_loss + self.gamma * class_loss
        
        return {
            'loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'classification_loss': class_loss
        }


def create_vae(latent_dim: int = 128, 
               input_size: int = 128,
               base_channels: int = 32,
               num_classes: int = 17) -> CrystallographicVAE:
    """
    Factory function to create a VAE model.
    
    Args:
        latent_dim: Dimension of latent space
        input_size: Input image size
        base_channels: Base channels for convolutions
        num_classes: Number of wallpaper groups
        
    Returns:
        Configured VAE model
    """
    return CrystallographicVAE(
        in_channels=1,
        latent_dim=latent_dim,
        base_channels=base_channels,
        input_size=input_size,
        num_classes=num_classes
    )



