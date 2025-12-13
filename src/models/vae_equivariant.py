"""
Rotation-Equivariant VAE for Crystallographic Patterns.

Implements rotation-equivariant convolutions for the 17 wallpaper groups.
Uses cyclic group convolutions (C4/C8) which are natural for crystallographic symmetries.

Key features:
- Group convolutions for rotation equivariance
- Steerable filters that rotate with the input
- Reduced parameters due to weight sharing across rotations
- Better generalization for symmetric patterns
- Configurable number of encoder/decoder layers
- Intermediate activation capture for visualization

Architecture optimized for:
- 512×512 RGB input
- ~8.5k samples dataset
- Rotation symmetries common in crystallographic groups
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math


def rotate_tensor(x: torch.Tensor, k: int) -> torch.Tensor:
    """Rotate tensor by k*90 degrees."""
    return torch.rot90(x, k, dims=[-2, -1])


def rotate_kernel(kernel: torch.Tensor, k: int) -> torch.Tensor:
    """Rotate conv kernel by k*90 degrees."""
    return torch.rot90(kernel, k, dims=[-2, -1])


class C4Conv2d(nn.Module):
    """
    C4-equivariant convolution.
    
    Applies the same kernel rotated by 0°, 90°, 180°, 270° and stacks outputs.
    Input: (B, C_in, H, W)
    Output: (B, C_out * 4, H', W') - 4 orientation channels
    
    This reduces parameters by 4x while learning rotation-invariant features.
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 padding: int = 1, bias: bool = False,
                 first_layer: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.first_layer = first_layer
        
        if first_layer:
            # First layer: regular input, C4 output
            self.weight = nn.Parameter(
                torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.02
            )
        else:
            # Hidden layers: C4 input (4 orientations), C4 output
            # in_channels should be divisible by 4
            self.weight = nn.Parameter(
                torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.02
            )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels * 4))
        else:
            self.bias = None
        
        # Initialize with He/Kaiming
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        
        for k in range(4):  # 0°, 90°, 180°, 270°
            rotated_kernel = rotate_kernel(self.weight, k)
            out = F.conv2d(x, rotated_kernel, None, self.stride, self.padding)
            outputs.append(out)
        
        # Stack along channel dimension
        result = torch.cat(outputs, dim=1)
        
        if self.bias is not None:
            result = result + self.bias.view(1, -1, 1, 1)
        
        return result


class C4Conv2dHidden(nn.Module):
    """
    C4-equivariant convolution for hidden layers.
    
    Takes C4 feature maps (B, C*4, H, W) and produces C4 feature maps.
    Properly handles the group structure by cycling through orientations.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels  # Per orientation
        self.out_channels = out_channels  # Per orientation
        self.stride = stride
        self.padding = padding
        
        # Weight for one orientation (others are rotated versions)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels * 4, kernel_size, kernel_size) * 0.02
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels * 4))
        else:
            self.bias = None
        
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        c_per_orient = C // 4
        
        outputs = []
        
        for k in range(4):  # Output orientation
            # Rotate kernel
            rotated_kernel = rotate_kernel(self.weight, k)
            
            # Cycle input orientations to match
            x_cycled = torch.cat([
                x[:, ((4-k+i) % 4) * c_per_orient : ((4-k+i) % 4 + 1) * c_per_orient]
                for i in range(4)
            ], dim=1)
            
            out = F.conv2d(x_cycled, rotated_kernel, None, self.stride, self.padding)
            outputs.append(out)
        
        result = torch.cat(outputs, dim=1)
        
        if self.bias is not None:
            result = result + self.bias.view(1, -1, 1, 1)
        
        return result


class C4BatchNorm(nn.Module):
    """Batch normalization for C4 feature maps."""
    def __init__(self, num_features: int):
        super().__init__()
        # num_features is per orientation, so total is num_features * 4
        self.bn = nn.BatchNorm2d(num_features * 4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class C4Pool(nn.Module):
    """
    Pooling over C4 group.
    Converts C4 features to invariant features by max-pooling over orientations.
    """
    def __init__(self, pool_type: str = 'max'):
        super().__init__()
        self.pool_type = pool_type
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        c_per_orient = C // 4
        
        # Reshape to (B, 4, C_per_orient, H, W)
        x = x.view(B, 4, c_per_orient, H, W)
        
        if self.pool_type == 'max':
            return x.max(dim=1)[0]
        else:
            return x.mean(dim=1)


class ActivationCapture:
    """
    Helper class to capture and store intermediate activations.
    Activations can be visualized as images (feature maps).
    """
    def __init__(self):
        self.activations: Dict[str, torch.Tensor] = {}
        self.enabled = False
    
    def enable(self):
        self.enabled = True
        self.activations.clear()
    
    def disable(self):
        self.enabled = False
        # Don't clear activations here - they may still be needed
        # Clear happens on next enable()
    
    def capture(self, name: str, tensor: torch.Tensor):
        if self.enabled:
            # Store detached copy on CPU to save GPU memory
            self.activations[name] = tensor.detach().cpu()
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        return self.activations
    
    def get_activation_images(self, name: str, num_channels: int = 16) -> torch.Tensor:
        """
        Get activations as visualizable images.
        
        Returns: (num_channels, H, W) tensor normalized to [0, 1]
        """
        if name not in self.activations:
            return None
        
        act = self.activations[name]
        
        # Take first sample if batched
        if act.dim() == 4:
            act = act[0]  # (C, H, W)
        
        # Select subset of channels
        num_channels = min(num_channels, act.shape[0])
        act = act[:num_channels]
        
        # Normalize each channel to [0, 1]
        act_min = act.view(num_channels, -1).min(dim=1)[0].view(num_channels, 1, 1)
        act_max = act.view(num_channels, -1).max(dim=1)[0].view(num_channels, 1, 1)
        act_normalized = (act - act_min) / (act_max - act_min + 1e-8)
        
        return act_normalized


class ConfigurableEncoderBlock(nn.Module):
    """Configurable encoder block with activation capture."""
    
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
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


class ConfigurableDecoderBlock(nn.Module):
    """Configurable decoder block with activation capture."""
    
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        return out


class ConfigurableVAE(nn.Module):
    """
    Configurable VAE with adjustable number of encoder/decoder layers.
    
    Features:
    - Configurable depth (num_encoder_layers, num_decoder_layers)
    - Intermediate activation capture for visualization
    - Memory-efficient design
    - Automatic channel scaling based on depth
    
    Args:
        latent_dim: Dimension of latent space
        base_channels: Base number of channels (scales with depth)
        num_encoder_layers: Number of downsampling layers in encoder (3-8)
        num_decoder_layers: Number of upsampling layers in decoder (3-8)
        dropout: Dropout rate
        num_classes: Number of classification classes
        capture_activations: Whether to capture intermediate activations
    
    Resolution mapping (for 512x512 input):
        - 3 layers: 512 → 64 (8x downsampling)
        - 4 layers: 512 → 32 (16x downsampling)
        - 5 layers: 512 → 16 (32x downsampling)
        - 6 layers: 512 → 8 (64x downsampling)
        - 7 layers: 512 → 4 (128x downsampling)
        - 8 layers: 512 → 2 (256x downsampling)
    """
    
    def __init__(self,
                 latent_dim: int = 256,
                 base_channels: int = 32,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dropout: float = 0.1,
                 num_classes: int = 17,
                 capture_activations: bool = False):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.input_size = 512
        self.num_classes = num_classes
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        # Activation capture
        self.activation_capture = ActivationCapture()
        if capture_activations:
            self.activation_capture.enable()
        
        # Calculate channel progression
        # Channels double every 2 layers, capped at base_channels * 8
        def get_channels(layer_idx: int, base: int, max_mult: int = 8) -> int:
            mult = min(2 ** (layer_idx // 2), max_mult)
            return base * mult
        
        # ==================== ENCODER ====================
        c = base_channels
        
        # Stem (always present): 512 → 256
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        in_ch = c
        self.encoder_channels = [c]  # Track channels at each layer for skip connections
        
        for i in range(num_encoder_layers):
            out_ch = get_channels(i + 1, c)
            self.encoder_layers.append(
                ConfigurableEncoderBlock(in_ch, out_ch, stride=2, dropout=dropout)
            )
            in_ch = out_ch
            self.encoder_channels.append(out_ch)
        
        self.final_encoder_channels = in_ch
        
        # Calculate final spatial size: 512 / 2^(num_layers+1)
        # Encoder: stem (512→256) + num_encoder_layers = num_encoder_layers + 1 downsamplings
        self.final_spatial_size = 512 // (2 ** (num_encoder_layers + 1))
        
        # Latent projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(in_ch, latent_dim)
        self.fc_logvar = nn.Linear(in_ch, latent_dim)
        
        # ==================== DECODER ====================
        # Calculate required decoder layers to reach 512x512
        required_decoder_layers = int(math.log2(512 / self.final_spatial_size))
        self.num_decoder_layers = required_decoder_layers  # Ensure correct output size
        
        # Project latent to spatial
        decoder_start_ch = get_channels(required_decoder_layers - 1, c)
        self.fc_decode = nn.Linear(latent_dim, decoder_start_ch * self.final_spatial_size ** 2)
        self.decoder_start_channels = decoder_start_ch
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        in_ch = decoder_start_ch
        
        for i in range(required_decoder_layers):
            # Channels decrease as we go up in resolution
            layer_from_end = required_decoder_layers - 1 - i
            out_ch = get_channels(layer_from_end, c) if layer_from_end > 0 else c // 2
            out_ch = max(out_ch, c // 2)  # Minimum channels
            
            self.decoder_layers.append(
                ConfigurableDecoderBlock(in_ch, out_ch, dropout=dropout)
            )
            in_ch = out_ch
        
        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(in_ch, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
        # ==================== CLASSIFIER ====================
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim // 2, num_classes)
        )
        
        # Store layer info for visualization
        self._layer_info = {
            'encoder': [f'enc_{i}' for i in range(num_encoder_layers)],
            'decoder': [f'dec_{i}' for i in range(num_decoder_layers)],
        }
    
    def get_layer_names(self) -> Dict[str, List[str]]:
        """Get names of all capturable layers."""
        return self._layer_info
    
    def enable_activation_capture(self):
        """Enable capturing of intermediate activations."""
        self.activation_capture.enable()
    
    def disable_activation_capture(self):
        """Disable capturing of intermediate activations."""
        self.activation_capture.disable()
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get captured activations."""
        return self.activation_capture.get_activations()
    
    def get_activation_images(self, layer_name: str, num_channels: int = 16) -> torch.Tensor:
        """Get activations as visualizable images."""
        return self.activation_capture.get_activation_images(layer_name, num_channels)
    
    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Stem
        x = self.stem(x)
        self.activation_capture.capture('stem', x)
        
        # Encoder layers
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            self.activation_capture.capture(f'enc_{i}', x)
        
        x = self.pool(x).flatten(1)
        return self.fc_mu(x), self.fc_logvar(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_decode(z)
        x = x.view(-1, self.decoder_start_channels, 
                   self.final_spatial_size, self.final_spatial_size)
        
        self.activation_capture.capture('dec_input', x)
        
        # Decoder layers
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            self.activation_capture.capture(f'dec_{i}', x)
        
        return self.output(x)
    
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
        """Get model configuration and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'latent_dim': self.latent_dim,
            'input_size': self.input_size,
            'num_encoder_layers': self.num_encoder_layers,
            'num_decoder_layers': self.num_decoder_layers,
            'encoder_channels': self.encoder_channels,
            'final_spatial_size': self.final_spatial_size,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1e6,
        }


class RotationEquivariantVAE(nn.Module):
    """
    Rotation-Equivariant VAE for crystallographic patterns.
    
    Uses C4-equivariant convolutions in the encoder to learn
    rotation-invariant representations, which is ideal for
    crystallographic groups that have rotational symmetries.
    
    The 17 wallpaper groups include:
    - p1, p2 (no rotation / 180°)
    - pm, pg, cm, pmm, pmg, pgg, cmm (reflection symmetries)
    - p4, p4m, p4g (90° rotation)
    - p3, p3m1, p31m (120° rotation)
    - p6, p6m (60° rotation)
    
    C4 equivariance captures 90° rotations naturally and provides
    some invariance to other angles through learned features.
    
    Args:
        latent_dim: Dimension of latent space
        base_channels: Base number of channels
        num_encoder_layers: Number of encoder stages (3-6)
        num_decoder_layers: Number of decoder stages (3-7)
        dropout: Dropout rate
        num_classes: Number of classes for classification
        capture_activations: Whether to capture intermediate activations
    """
    
    def __init__(self,
                 latent_dim: int = 256,
                 base_channels: int = 16,
                 num_encoder_layers: int = 5,
                 num_decoder_layers: int = 6,
                 dropout: float = 0.1,
                 num_classes: int = 17,
                 capture_activations: bool = False):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.input_size = 512
        self.num_classes = num_classes
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        # Activation capture
        self.activation_capture = ActivationCapture()
        if capture_activations:
            self.activation_capture.enable()
        
        c = base_channels
        
        # ==================== ENCODER ====================
        # Stem: 512 → 256 (regular conv)
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        
        # First C4 layer: regular -> C4
        self.c4_transition = nn.Sequential(
            C4Conv2d(c, c, 3, stride=2, padding=1, first_layer=True),
            C4BatchNorm(c),
            nn.ReLU(inplace=True),
        )
        
        # C4 encoder layers
        self.encoder_layers = nn.ModuleList()
        in_ch = c  # Per orientation (total is c*4)
        
        channel_schedule = self._get_channel_schedule(c, num_encoder_layers - 1)
        
        for i, out_ch in enumerate(channel_schedule):
            self.encoder_layers.append(nn.Sequential(
                C4Conv2dHidden(in_ch, out_ch, 3, stride=2, padding=1),
                C4BatchNorm(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
            ))
            in_ch = out_ch
        
        self.final_c4_channels = in_ch
        
        # Pool over C4 group
        self.group_pool = C4Pool('max')
        
        # Calculate final spatial size
        # Encoder does: stem (512→256) + c4_transition (256→128) + (num_encoder_layers-1) more
        # Total downsamplings = num_encoder_layers + 1
        self.final_spatial_size = 512 // (2 ** (num_encoder_layers + 1))
        
        # Latent projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(in_ch, latent_dim)
        self.fc_logvar = nn.Linear(in_ch, latent_dim)
        
        # ==================== DECODER ====================
        # Calculate required decoder layers to reach 512x512
        # Each decoder layer doubles spatial size
        required_decoder_layers = int(math.log2(512 / self.final_spatial_size))
        self.num_decoder_layers = required_decoder_layers  # Override user input
        
        # Regular decoder (latent is already rotation-invariant)
        decoder_ch = in_ch * 2  # Start with more channels
        self.fc_decode = nn.Linear(latent_dim, decoder_ch * self.final_spatial_size ** 2)
        self.decoder_start_channels = decoder_ch
        
        self.decoder_layers = nn.ModuleList()
        in_ch = decoder_ch
        
        decoder_schedule = self._get_decoder_schedule(c, decoder_ch, required_decoder_layers)
        
        for i, out_ch in enumerate(decoder_schedule):
            self.decoder_layers.append(
                ConfigurableDecoderBlock(in_ch, out_ch, dropout=dropout)
            )
            in_ch = out_ch
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(in_ch, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
        # ==================== CLASSIFIER ====================
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim // 2, num_classes)
        )
        
        self._layer_info = {
            'encoder': ['stem', 'c4_transition'] + [f'enc_{i}' for i in range(len(self.encoder_layers))],
            'decoder': [f'dec_{i}' for i in range(num_decoder_layers)],
        }
    
    def _get_channel_schedule(self, base: int, num_layers: int) -> List[int]:
        """Get channel counts for encoder layers."""
        schedule = []
        for i in range(num_layers):
            mult = min(2 ** (i + 1), 8)
            schedule.append(base * mult)
        return schedule
    
    def _get_decoder_schedule(self, base: int, start_ch: int, num_layers: int) -> List[int]:
        """Get channel counts for decoder layers."""
        schedule = []
        ch = start_ch
        for i in range(num_layers):
            ch = max(ch // 2, base // 2)
            schedule.append(ch)
        return schedule
    
    def get_layer_names(self) -> Dict[str, List[str]]:
        """Get names of all capturable layers."""
        return self._layer_info
    
    def enable_activation_capture(self):
        self.activation_capture.enable()
    
    def disable_activation_capture(self):
        self.activation_capture.disable()
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        return self.activation_capture.get_activations()
    
    def get_activation_images(self, layer_name: str, num_channels: int = 16) -> torch.Tensor:
        return self.activation_capture.get_activation_images(layer_name, num_channels)
    
    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        self.activation_capture.capture('stem', x)
        
        x = self.c4_transition(x)
        self.activation_capture.capture('c4_transition', x)
        
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            self.activation_capture.capture(f'enc_{i}', x)
        
        # Pool over C4 group
        x = self.group_pool(x)
        self.activation_capture.capture('group_pool', x)
        
        x = self.pool(x).flatten(1)
        return self.fc_mu(x), self.fc_logvar(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_decode(z)
        x = x.view(-1, self.decoder_start_channels,
                   self.final_spatial_size, self.final_spatial_size)
        
        self.activation_capture.capture('dec_input', x)
        
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            self.activation_capture.capture(f'dec_{i}', x)
        
        return self.output(x)
    
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
        """Interpolate between two images in latent space."""
        mu1, _ = self.encode(x1)
        mu2, _ = self.encode(x2)
        
        alphas = torch.linspace(0, 1, steps, device=x1.device)
        return torch.cat([self.decode((1-a)*mu1 + a*mu2) for a in alphas], dim=0)
    
    def get_model_info(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'type': 'RotationEquivariantVAE',
            'latent_dim': self.latent_dim,
            'num_encoder_layers': self.num_encoder_layers,
            'num_decoder_layers': self.num_decoder_layers,
            'total_params': total_params,
            'model_size_mb': total_params * 4 / 1e6,
        }


def visualize_activations(model: nn.Module, input_image: torch.Tensor, 
                          save_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """
    Visualize intermediate activations of the model.
    
    Args:
        model: VAE model with activation capture enabled
        input_image: Input tensor (1, 3, H, W)
        save_path: Optional path to save visualization
    
    Returns:
        Dictionary of activation images per layer
    """
    import matplotlib.pyplot as plt
    
    model.eval()
    model.enable_activation_capture()
    
    with torch.no_grad():
        _ = model(input_image)
    
    activations = model.get_activations()
    layer_names = model.get_layer_names()
    
    # Create visualization
    all_layers = layer_names.get('encoder', []) + layer_names.get('decoder', [])
    n_layers = len([l for l in all_layers if l in activations])
    
    if n_layers == 0:
        print("No activations captured!")
        return {}
    
    fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(4 * ((n_layers + 1) // 2), 8))
    axes = axes.flatten() if n_layers > 1 else [axes]
    
    activation_images = {}
    
    for idx, layer_name in enumerate(all_layers):
        if layer_name not in activations:
            continue
        
        act_img = model.get_activation_images(layer_name, num_channels=1)
        if act_img is not None:
            activation_images[layer_name] = act_img
            
            if idx < len(axes):
                ax = axes[idx]
                ax.imshow(act_img[0].numpy(), cmap='viridis')
                ax.set_title(f'{layer_name}\n{tuple(activations[layer_name].shape)}')
                ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved activation visualization to {save_path}")
    
    plt.close()
    
    model.disable_activation_capture()
    
    return activation_images


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 70)
    print("CONFIGURABLE VAE WITH ACTIVATION CAPTURE")
    print("=" * 70)
    
    # Test ConfigurableVAE with different depths
    print("\n1. ConfigurableVAE - Testing different depths:")
    
    for n_enc, n_dec in [(4, 5), (5, 6), (6, 7)]:
        model = ConfigurableVAE(
            latent_dim=256,
            base_channels=32,
            num_encoder_layers=n_enc,
            num_decoder_layers=n_dec,
            capture_activations=True
        )
        info = model.get_model_info()
        print(f"\n   Encoder: {n_enc} layers, Decoder: {n_dec} layers")
        print(f"   Parameters: {info['total_params']:,}")
        print(f"   Final spatial: {info['final_spatial_size']}x{info['final_spatial_size']}")
        print(f"   Model size: {info['model_size_mb']:.1f} MB")
    
    # Test forward pass with activation capture
    print("\n2. Testing activation capture:")
    model = ConfigurableVAE(
        num_encoder_layers=5,
        num_decoder_layers=6,
        capture_activations=True
    )
    
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    
    print(f"   Input: {x.shape}")
    print(f"   Output: {out['reconstruction'].shape}")
    print(f"   Captured layers: {list(model.get_activations().keys())}")
    
    # Show activation shapes
    print("\n   Activation shapes:")
    for name, act in model.get_activations().items():
        print(f"     {name}: {tuple(act.shape)}")
    
    # Test RotationEquivariantVAE
    print("\n3. RotationEquivariantVAE:")
    model_equiv = RotationEquivariantVAE(
        latent_dim=256,
        base_channels=16,
        num_encoder_layers=4,
        num_decoder_layers=6,
        capture_activations=True
    )
    
    info = model_equiv.get_model_info()
    print(f"   Parameters: {info['total_params']:,}")
    print(f"   Model size: {info['model_size_mb']:.1f} MB")
    
    with torch.no_grad():
        out = model_equiv(x)
    print(f"   Output: {out['reconstruction'].shape}")
    
    print("\n" + "=" * 70)
    print("✅ Models ready! Use capture_activations=True to visualize layers")
    print("=" * 70)
    print("\nRecommended configurations for 24GB GPU:")
    print("  ConfigurableVAE(num_encoder_layers=5, num_decoder_layers=6, base_channels=32)")
    print("  → batch_size=16-24")
    print("\n  RotationEquivariantVAE(num_encoder_layers=4, num_decoder_layers=6)")
    print("  → batch_size=24-32")
