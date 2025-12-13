#!/usr/bin/env python3
"""
Train Deep VAE using the pre-generated dataset from disk.

Comprehensive logging to TensorBoard and disk:
- Losses (total, reconstruction, KL, classification)
- Sample images and reconstructions for each symmetry group
- Latent space visualizations (2D and 3D)
- Embeddings saved to TensorBoard projector
- All metrics and visualizations also saved to disk

Usage:
    python scripts/train_from_disk.py --epochs 150 --batch-size 32
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
import h5py
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.models.vae_deep import DeepCrystallographicVAE, count_layers
from src.models.vae_equivariant import ConfigurableVAE, RotationEquivariantVAE
from src.models.vae_harmonic import HarmonicVAE
from src.models.kl_scheduler import KLScheduler
from src.models import VAETrainer
from src.visualization.training_logger import TrainingLogger

# Wallpaper groups
GROUPS = ['p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm',
          'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m']


class HDF5Dataset(Dataset):
    """Load pre-generated colored patterns from HDF5 file.
    
    Preloads entire dataset into RAM for fast GPU training.
    """
    
    def __init__(self, hdf5_path: str, transform=None, preload: bool = True):
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.preloaded = False
        
        # Load metadata and optionally preload data
        with h5py.File(hdf5_path, 'r') as f:
            self.length = len(f['labels'])
            self.resolution = f.attrs.get('resolution', 512)
            
            if preload:
                # Preload entire dataset into RAM (~6.5GB for 8500 512x512 images)
                print(f"  Loading dataset into RAM...")
                # Load as float32 tensors directly [N, 3, H, W]
                patterns_np = f['patterns'][:]  # [N, H, W, 3]
                self.patterns = torch.from_numpy(patterns_np).float().permute(0, 3, 1, 2)
                self.labels = torch.from_numpy(f['labels'][:]).long()
                self.preloaded = True
                
                mem_gb = self.patterns.numel() * 4 / 1e9
                print(f"  Loaded dataset: {self.length} samples at {self.resolution}×{self.resolution}")
                print(f"  RAM usage: {mem_gb:.1f} GB")
            else:
                print(f"  Loaded dataset: {self.length} samples at {self.resolution}×{self.resolution}")
                self.file = None
                self.patterns = None
                self.labels_data = None
    
    def _open_file(self):
        if not self.preloaded and self.file is None:
            self.file = h5py.File(self.hdf5_path, 'r')
            self.patterns = self.file['patterns']
            self.labels_data = self.file['labels']
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.preloaded:
            # Fast path: data already in RAM as tensors
            tensor = self.patterns[idx]
            label = self.labels[idx]
        else:
            # Slow path: read from disk
            self._open_file()
            pattern = self.patterns[idx]
            label = int(self.labels_data[idx])
            tensor = torch.from_numpy(pattern).float().permute(2, 0, 1)
        
        if self.transform:
            tensor = self.transform(tensor)
        
        return tensor, label
    
    def __del__(self):
        if hasattr(self, 'file') and self.file is not None:
            self.file.close()


class ImageFolderDataset(Dataset):
    """Load pre-generated patterns from image files."""
    
    GROUPS = ['p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm',
              'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m']
    
    def __init__(self, images_dir: str, transform=None):
        from PIL import Image
        
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.samples = []
        
        # Scan all images
        for group_idx, group_name in enumerate(self.GROUPS):
            group_dir = self.images_dir / group_name
            if group_dir.exists():
                for img_path in sorted(group_dir.glob("*.png")):
                    self.samples.append((str(img_path), group_idx))
        
        print(f"  Found {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        img_path, label = self.samples[idx]
        
        # Load and convert to tensor
        img = Image.open(img_path).convert('RGB')
        pattern = np.array(img, dtype=np.float32) / 255.0
        
        # [H, W, 3] -> [3, H, W]
        tensor = torch.from_numpy(pattern).permute(2, 0, 1)
        
        if self.transform:
            tensor = self.transform(tensor)
        
        return tensor, label


def save_latent_visualizations(model, dataloader, device, output_dir, epoch, max_samples=1000):
    """Save latent space visualizations to disk."""
    model.eval()
    
    all_latents = []
    all_labels = []
    all_images = []
    
    with torch.no_grad():
        count = 0
        for images, labels in dataloader:
            if count >= max_samples:
                break
            images = images.to(device)
            mu, _ = model.encode(images)
            all_latents.append(mu.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_images.append(images.cpu())
            count += len(images)
    
    latents = np.concatenate(all_latents, axis=0)[:max_samples]
    labels = np.array(all_labels)[:max_samples]
    
    # Save embeddings to disk
    embeddings_dir = output_dir / 'embeddings'
    embeddings_dir.mkdir(exist_ok=True)
    
    np.savez(
        embeddings_dir / f'embeddings_epoch_{epoch:03d}.npz',
        latents=latents,
        labels=labels,
        label_names=[GROUPS[l] for l in labels],
        epoch=epoch
    )
    
    # PCA 2D
    pca_2d = PCA(n_components=2)
    latents_2d = pca_2d.fit_transform(latents)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    
    colors = plt.cm.tab20(np.linspace(0, 1, 17))
    for g_idx in range(17):
        mask = labels == g_idx
        if mask.sum() > 0:
            ax.scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                      c=[colors[g_idx]], label=GROUPS[g_idx],
                      alpha=0.7, s=30, edgecolors='white', linewidth=0.2)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
             facecolor='#1a1a2e', edgecolor='#eaeaea', labelcolor='#eaeaea')
    ax.set_title(f'Latent Space PCA 2D - Epoch {epoch}', color='#eaeaea', fontsize=16)
    ax.tick_params(colors='#eaeaea')
    for spine in ax.spines.values():
        spine.set_color('#3d3d5c')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'latent_pca_2d_epoch_{epoch:03d}.png', 
               dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
    plt.close()
    
    # PCA 3D
    pca_3d = PCA(n_components=3)
    latents_3d = pca_3d.fit_transform(latents)
    
    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor('#0f0f1a')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0f0f1a')
    
    for g_idx in range(17):
        mask = labels == g_idx
        if mask.sum() > 0:
            ax.scatter(latents_3d[mask, 0], latents_3d[mask, 1], latents_3d[mask, 2],
                      c=[colors[g_idx]], label=GROUPS[g_idx],
                      alpha=0.7, s=20)
    
    ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5),
             facecolor='#1a1a2e', edgecolor='#eaeaea', labelcolor='#eaeaea', fontsize=8)
    ax.set_title(f'Latent Space PCA 3D - Epoch {epoch}', color='#eaeaea', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'latent_pca_3d_epoch_{epoch:03d}.png',
               dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
    plt.close()
    
    # t-SNE 2D (slower, do less frequently)
    if epoch % 20 == 0 or epoch == 1:
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents)-1))
            latents_tsne = tsne.fit_transform(latents)
            
            fig, ax = plt.subplots(figsize=(14, 12))
            fig.patch.set_facecolor('#0f0f1a')
            ax.set_facecolor('#0f0f1a')
            
            for g_idx in range(17):
                mask = labels == g_idx
                if mask.sum() > 0:
                    ax.scatter(latents_tsne[mask, 0], latents_tsne[mask, 1],
                              c=[colors[g_idx]], label=GROUPS[g_idx],
                              alpha=0.7, s=30)
            
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                     facecolor='#1a1a2e', edgecolor='#eaeaea', labelcolor='#eaeaea')
            ax.set_title(f'Latent Space t-SNE - Epoch {epoch}', color='#eaeaea', fontsize=16)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'latent_tsne_epoch_{epoch:03d}.png',
                       dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  t-SNE failed: {e}")
    
    return pca_2d.explained_variance_ratio_.sum()


def save_reconstructions_per_group(model, dataloader, device, output_dir, epoch):
    """Save original vs reconstructed images for each group."""
    from torchvision.utils import make_grid, save_image
    
    model.eval()
    
    # Collect one sample per group
    group_samples = {}
    
    with torch.no_grad():
        for images, labels in dataloader:
            for i, label in enumerate(labels):
                g_idx = label.item()
                if g_idx not in group_samples:
                    group_samples[g_idx] = images[i:i+1].to(device)
                if len(group_samples) == 17:
                    break
            if len(group_samples) == 17:
                break
        
        # Reconstruct
        originals = []
        reconstructions = []
        
        for g_idx in range(17):
            if g_idx in group_samples:
                img = group_samples[g_idx]
                outputs = model(img)
                originals.append(img.cpu())
                reconstructions.append(outputs['reconstruction'].cpu())
        
        if originals:
            originals = torch.cat(originals, dim=0)
            reconstructions = torch.cat(reconstructions, dim=0)
            
            # Save grids
            recon_dir = output_dir / 'reconstructions'
            recon_dir.mkdir(exist_ok=True)
            
            # Individual comparisons
            comparison = torch.cat([originals, reconstructions], dim=0)
            grid = make_grid(comparison, nrow=17, padding=2, normalize=True)
            save_image(grid, recon_dir / f'comparison_epoch_{epoch:03d}.png')
            
            # Also create matplotlib version with labels
            fig, axes = plt.subplots(2, 17, figsize=(34, 4))
            fig.patch.set_facecolor('#0f0f1a')
            
            for i in range(17):
                # Original
                axes[0, i].imshow(originals[i].permute(1, 2, 0).numpy())
                axes[0, i].axis('off')
                axes[0, i].set_title(GROUPS[i], fontsize=8, color='#4ECDC4')
                
                # Reconstruction
                axes[1, i].imshow(reconstructions[i].permute(1, 2, 0).numpy())
                axes[1, i].axis('off')
            
            axes[0, 0].set_ylabel('Original', color='#eaeaea', fontsize=10)
            axes[1, 0].set_ylabel('Recon', color='#eaeaea', fontsize=10)
            
            plt.suptitle(f'Reconstructions - Epoch {epoch}', color='#eaeaea', fontsize=14)
            plt.tight_layout()
            plt.savefig(recon_dir / f'comparison_labeled_epoch_{epoch:03d}.png',
                       dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
            plt.close()


def save_generated_samples(model, device, output_dir, epoch, num_samples=16):
    """Save randomly generated samples."""
    from torchvision.utils import make_grid, save_image
    
    model.eval()
    
    with torch.no_grad():
        samples = model.sample(num_samples, device)
    
    samples_dir = output_dir / 'samples'
    samples_dir.mkdir(exist_ok=True)
    
    grid = make_grid(samples, nrow=4, padding=2, normalize=True)
    save_image(grid, samples_dir / f'samples_epoch_{epoch:03d}.png')
    
    # Matplotlib version
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.patch.set_facecolor('#0f0f1a')
    
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(samples[i].cpu().permute(1, 2, 0).numpy())
        ax.axis('off')
    
    plt.suptitle(f'Generated Samples - Epoch {epoch}', color='#eaeaea', fontsize=16)
    plt.tight_layout()
    plt.savefig(samples_dir / f'samples_labeled_epoch_{epoch:03d}.png',
               dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
    plt.close()


def save_training_curves(history, output_dir):
    """Save training curves to disk."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor('#0f0f1a')
    
    for ax in axes.flat:
        ax.set_facecolor('#0f0f1a')
        ax.tick_params(colors='#eaeaea')
        for spine in ax.spines.values():
            spine.set_color('#3d3d5c')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Total loss
    axes[0, 0].plot(epochs, history['train_loss'], 'c-', label='Train', linewidth=2)
    if history.get('val_loss'):
        axes[0, 0].plot(epochs, history['val_loss'], 'm-', label='Val', linewidth=2)
    axes[0, 0].set_title('Total Loss', color='#eaeaea', fontsize=14)
    axes[0, 0].set_xlabel('Epoch', color='#eaeaea')
    axes[0, 0].legend(facecolor='#1a1a2e', edgecolor='#eaeaea', labelcolor='#eaeaea')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[0, 1].plot(epochs, history['train_recon_loss'], 'c-', label='Train', linewidth=2)
    if history.get('val_recon_loss'):
        axes[0, 1].plot(epochs, history['val_recon_loss'], 'm-', label='Val', linewidth=2)
    axes[0, 1].set_title('Reconstruction Loss', color='#eaeaea', fontsize=14)
    axes[0, 1].set_xlabel('Epoch', color='#eaeaea')
    axes[0, 1].legend(facecolor='#1a1a2e', edgecolor='#eaeaea', labelcolor='#eaeaea')
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL loss
    axes[1, 0].plot(epochs, history['train_kl_loss'], 'c-', label='Train', linewidth=2)
    if history.get('val_kl_loss'):
        axes[1, 0].plot(epochs, history['val_kl_loss'], 'm-', label='Val', linewidth=2)
    axes[1, 0].set_title('KL Divergence', color='#eaeaea', fontsize=14)
    axes[1, 0].set_xlabel('Epoch', color='#eaeaea')
    axes[1, 0].legend(facecolor='#1a1a2e', edgecolor='#eaeaea', labelcolor='#eaeaea')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    if history.get('lr'):
        axes[1, 1].plot(epochs, history['lr'], 'y-', linewidth=2)
        axes[1, 1].set_title('Learning Rate', color='#eaeaea', fontsize=14)
        axes[1, 1].set_xlabel('Epoch', color='#eaeaea')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training Progress', color='#eaeaea', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=200, 
               facecolor='#0f0f1a', bbox_inches='tight')
    plt.close()
    
    # Also save as JSON
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Train Deep VAE from pre-generated dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    parser.add_argument('--data-dir', type=str, 
                       default='./data/colored_crystallographic',
                       help='Dataset directory')
    parser.add_argument('--use-hdf5', action='store_true', default=True,
                       help='Use HDF5 file (faster) or images folder')
    parser.add_argument('--batch-size', '-b', type=int, default=32)
    
    # Model
    parser.add_argument('--model-type', type=str, default='configurable',
                       choices=['deep', 'configurable', 'equivariant', 'harmonic'],
                       help='Model architecture: deep (original), configurable (CNN), equivariant (C4), harmonic (SO2)')
    parser.add_argument('--latent-dim', '-l', type=int, default=256)
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--num-encoder-layers', type=int, default=6,
                       help='Number of encoder downsampling layers (3-8)')
    parser.add_argument('--num-decoder-layers', type=int, default=7,
                       help='Number of decoder upsampling layers (3-8)')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--capture-activations', action='store_true',
                       help='Enable intermediate activation capture for visualization')
    
    # Training
    parser.add_argument('--epochs', '-e', type=int, default=150)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=0.3,
                       help='Target KL weight (final value if using annealing)')
    parser.add_argument('--gamma', type=float, default=0.1)
    
    # KL Annealing
    parser.add_argument('--kl-annealing', type=str, default=None,
                       choices=['linear', 'warmup', 'cyclical', 'sigmoid'],
                       help='KL annealing schedule type')
    parser.add_argument('--kl-warmup-epochs', type=int, default=20,
                       help='Epochs to keep beta=0 before annealing')
    parser.add_argument('--kl-anneal-epochs', type=int, default=30,
                       help='Epochs to anneal from 0 to target beta')
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='./runs')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Resume training
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--resume-dir', type=str, default=None, help='Directory with checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    
    exp_name = f"deep_vae_disk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("=" * 70)
    print("🔬 DEEP VAE TRAINING (FROM DISK)")
    print("=" * 70)
    
    # Load dataset
    data_path = Path(args.data_dir)
    print(f"\n📦 Loading dataset from {data_path}...")
    
    hdf5_path = data_path / "crystallographic_patterns_colored.h5"
    images_path = data_path / "images"
    
    if args.use_hdf5 and hdf5_path.exists():
        dataset = HDF5Dataset(str(hdf5_path))
    elif images_path.exists():
        dataset = ImageFolderDataset(str(images_path))
    else:
        raise FileNotFoundError(f"No dataset found in {data_path}")
    
    # Split
    total = len(dataset)
    train_size = int(0.85 * total)
    val_size = total - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Optimized DataLoader settings for preloaded dataset
    # With data in RAM, we need fewer workers but want fast GPU transfer
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': min(args.num_workers, 4),  # Less workers needed with RAM data
        'pin_memory': True,
        'persistent_workers': True if args.num_workers > 0 else False,
        'prefetch_factor': 2 if args.num_workers > 0 else None,
    }
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs
    )
    
    print(f"  Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"  Val: {len(val_dataset)} samples ({len(val_loader)} batches)")
    
    # Determine resolution from first sample
    sample, _ = dataset[0]
    resolution = sample.shape[-1]
    print(f"  Resolution: {resolution}×{resolution} RGB")
    
    # Create model
    print(f"\n🧠 Creating {args.model_type.upper()} VAE...")
    
    if args.model_type == 'deep':
        model = DeepCrystallographicVAE(
            latent_dim=args.latent_dim,
            base_channels=args.base_channels,
            dropout=args.dropout,
            num_classes=17
        )
    elif args.model_type == 'configurable':
        model = ConfigurableVAE(
            latent_dim=args.latent_dim,
            base_channels=args.base_channels,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dropout=args.dropout,
            num_classes=17,
            capture_activations=args.capture_activations
        )
    elif args.model_type == 'equivariant':
        # Equivariant model uses smaller base channels due to C4 expansion
        model = RotationEquivariantVAE(
            latent_dim=args.latent_dim,
            base_channels=args.base_channels // 2,  # Half because C4 expands 4x
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dropout=args.dropout,
            num_classes=17,
            capture_activations=args.capture_activations
        )
    elif args.model_type == 'harmonic':
        # SO(2)-equivariant with circular harmonic filters
        model = HarmonicVAE(
            latent_dim=args.latent_dim,
            base_channels=args.base_channels,
            num_encoder_layers=args.num_encoder_layers,
            max_order=2,  # Harmonic order (higher = finer angular resolution)
            dropout=args.dropout,
            num_classes=17,
            capture_activations=args.capture_activations
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    total_params = sum(p.numel() for p in model.parameters())
    
    # Count conv layers
    if hasattr(model, 'get_model_info'):
        model_info = model.get_model_info()
        print(f"  Model type: {args.model_type}")
        print(f"  Encoder layers: {args.num_encoder_layers}")
        print(f"  Decoder layers: {args.num_decoder_layers}")
    else:
        layer_counts = count_layers(model)
        print(f"  Conv layers: {layer_counts['Conv2d']}")
    
    print(f"  Parameters: {total_params:,}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Activation capture: {'enabled' if args.capture_activations else 'disabled'}")
    
    print(f"\n⚙️ Training:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}, Weight decay: {args.weight_decay}")
    print(f"  Beta: {args.beta}")
    
    # Output directory for disk logging
    output_dir = Path(args.log_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args).copy()
    config['total_samples'] = total
    config['total_params'] = total_params
    config['experiment_name'] = exp_name
    if hasattr(model, 'get_model_info'):
        config['model_info'] = model.get_model_info()
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Trainer
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
    
    # Custom optimizer with weight decay
    trainer.optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Setup KL annealing if requested
    if args.kl_annealing:
        kl_scheduler = KLScheduler(
            schedule_type=args.kl_annealing,
            target_beta=args.beta,
            min_beta=0.0,
            warmup_epochs=args.kl_warmup_epochs,
            anneal_epochs=args.kl_anneal_epochs,
            total_epochs=args.epochs
        )
        trainer.set_kl_scheduler(kl_scheduler)
    
    # Setup disk logger for comprehensive logging
    disk_logger = TrainingLogger(
        output_dir=trainer.log_dir / 'disk_logs',
        writer=trainer.writer,
        device=str(trainer.device)
    )
    trainer.set_disk_logger(disk_logger)
    
    print(f"\n📈 TensorBoard: tensorboard --logdir={args.log_dir}")
    print(f"📁 Disk output: {output_dir}")
    print("=" * 70)
    
    # Custom visualization callback
    def visualization_callback(model, epoch):
        print(f"  💾 Saving visualizations to disk...")
        
        # Save reconstructions for each group
        save_reconstructions_per_group(model, val_loader, trainer.device, output_dir, epoch)
        
        # Save generated samples
        save_generated_samples(model, trainer.device, output_dir, epoch)
        
        # Save latent space visualizations (2D, 3D, embeddings)
        pca_var = save_latent_visualizations(model, val_loader, trainer.device, output_dir, epoch)
        print(f"    PCA explained variance: {pca_var:.1%}")
    
    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume:
        resume_dir = Path(args.resume_dir) if args.resume_dir else output_dir
        checkpoint_path = resume_dir / 'checkpoints' / 'best_model.pt'
        if not checkpoint_path.exists():
            # Try latest checkpoint
            checkpoint_dir = resume_dir / 'checkpoints'
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
                if checkpoints:
                    checkpoint_path = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        
        if checkpoint_path.exists():
            print(f"\n🔄 Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            trainer.best_val_loss = checkpoint.get('val_loss', float('inf'))
            print(f"  Resuming from epoch {start_epoch}, best val loss: {trainer.best_val_loss:.4f}")
        else:
            print(f"⚠️ No checkpoint found at {checkpoint_path}, starting fresh")
    
    # Initial visualization (only if not resuming)
    if not args.resume:
        print("\n📸 Initial visualization (before training)...")
        visualization_callback(model, 0)
    
    # Train with callbacks
    remaining_epochs = args.epochs - start_epoch
    if remaining_epochs <= 0:
        print(f"✓ Training already completed ({start_epoch}/{args.epochs} epochs)")
        history = trainer.history
    else:
        trainer.current_epoch = start_epoch
        history = trainer.train(
            num_epochs=remaining_epochs,
            save_every=10,
            visualize_every=2,  # Log reconstructions every 2 epochs
            embedding_every=5
        )
    
    # Save training curves
    save_training_curves(history, output_dir)
    
    # Final visualizations
    print("\n📸 Final visualization...")
    visualization_callback(model, args.epochs)
    
    # Final t-SNE
    print("  Computing final t-SNE...")
    save_latent_visualizations(model, val_loader, trainer.device, output_dir, args.epochs, max_samples=2000)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n📊 Results:")
    print(f"  Best val loss: {trainer.best_val_loss:.4f}")
    print(f"  Final recon loss: {history['val_recon_loss'][-1]:.4f}")
    print(f"  Final KL loss: {history['val_kl_loss'][-1]:.4f}")
    
    print(f"\n📁 Output files:")
    print(f"  Model: {args.checkpoint_dir}/best_model.pt")
    print(f"  TensorBoard: {args.log_dir}/{exp_name}")
    print(f"  Visualizations: {output_dir}")
    print(f"    - training_curves.png")
    print(f"    - reconstructions/")
    print(f"    - samples/")
    print(f"    - embeddings/")
    print(f"    - latent_pca_2d_*.png")
    print(f"    - latent_pca_3d_*.png")
    print(f"    - latent_tsne_*.png")


if __name__ == "__main__":
    main()

