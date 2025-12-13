"""
Training utilities for the Crystallographic VAE.

Provides:
- VAETrainer: Complete training loop with TensorBoard logging and checkpointing
- Learning rate scheduling
- Visualization callbacks during training
- Embedding projections (2D/3D)
- Activation logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Callable, List, Tuple
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from collections import defaultdict

from .vae import CrystallographicVAE, VAELoss


# Wallpaper group names for labeling
WALLPAPER_GROUPS = [
    'p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm',
    'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m'
]


class ActivationHook:
    """Hook to capture layer activations."""
    
    def __init__(self):
        self.activations = {}
        self.handles = []
    
    def register(self, model: nn.Module, layer_names: List[str] = None):
        """Register hooks on specified layers."""
        self.clear()
        
        for name, module in model.named_modules():
            if layer_names is None or name in layer_names:
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    handle = module.register_forward_hook(
                        self._get_hook(name)
                    )
                    self.handles.append(handle)
    
    def _get_hook(self, name: str):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def clear(self):
        """Clear all hooks and activations."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.activations = {}
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        return self.activations


class VAETrainer:
    """
    Trainer for the Crystallographic VAE with comprehensive TensorBoard logging.
    
    Logs:
    - Scalar metrics (losses, learning rate)
    - Images (reconstructions, samples per group)
    - Histograms (weights, gradients, activations)
    - Embeddings (latent space projections in 2D/3D)
    - Activation statistics per symmetry group
    """
    
    def __init__(self,
                 model: CrystallographicVAE,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 lr: float = 1e-3,
                 beta: float = 1.0,
                 gamma: float = 0.1,
                 device: str = 'auto',
                 checkpoint_dir: str = './checkpoints',
                 log_dir: str = './runs',
                 experiment_name: str = None,
                 log_interval: int = 50,
                 image_log_interval: int = 100):
        """
        Initialize the trainer.
        
        Args:
            model: VAE model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            lr: Learning rate
            beta: KL divergence weight (β-VAE)
            gamma: Classification loss weight
            device: Device to train on ('auto', 'cuda', 'cpu')
            checkpoint_dir: Directory for saving checkpoints
            log_dir: Base directory for TensorBoard logs
            experiment_name: Name for this experiment run
            log_interval: Batches between scalar logging
            image_log_interval: Batches between image logging
        """
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Training on: {self.device}")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_interval = log_interval
        self.image_log_interval = image_log_interval
        
        # Loss function
        self.criterion = VAELoss(beta=beta, gamma=gamma, reconstruction_loss='mse')
        self.beta = beta
        self.gamma = gamma
        self.kl_scheduler = None  # Optional KL annealing scheduler
        self.disk_logger = None  # Optional disk logger
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard setup
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        print(f"TensorBoard logs: {self.log_dir}")
        print(f"  Run: tensorboard --logdir={log_dir}")
        
        # Activation hook for layer activations
        self.activation_hook = ActivationHook()
        self._register_activation_hooks()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_recon_loss': [],
            'train_kl_loss': [],
            'val_loss': [],
            'val_recon_loss': [],
            'val_kl_loss': [],
            'lr': []
        }
        
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self.global_step = 0
        
        # Log model graph
        self._log_model_graph()
        
        # Save hyperparameters
        self._log_hyperparameters(lr, beta, gamma)
    
    def set_disk_logger(self, disk_logger):
        """
        Set disk logger for comprehensive logging.
        
        Args:
            disk_logger: TrainingLogger instance
        """
        self.disk_logger = disk_logger
        print(f"Disk logging enabled: {disk_logger.output_dir}")
    
    def set_kl_scheduler(self, scheduler):
        """
        Set KL annealing scheduler.
        
        Args:
            scheduler: KLScheduler instance for beta annealing
            
        Example:
            from src.models.kl_scheduler import KLScheduler
            scheduler = KLScheduler('warmup', target_beta=0.5, warmup_epochs=30, anneal_epochs=50)
            trainer.set_kl_scheduler(scheduler)
        """
        self.kl_scheduler = scheduler
        print(f"KL Annealing enabled: {scheduler}")
    
    def _register_activation_hooks(self):
        """Register hooks on key layers for activation logging."""
        # Key layers to monitor
        layer_names = [
            'encoder.initial.0',  # First conv
            'encoder.layer1.conv1',
            'encoder.layer2.conv1', 
            'encoder.layer3.conv1',
            'decoder.layer1.0',  # First deconv
            'decoder.layer3.0',
            'decoder.layer5.0',
        ]
        self.activation_hook.register(self.model, layer_names)
    
    def _log_model_graph(self):
        """Log model architecture to TensorBoard."""
        try:
            dummy_input = torch.randn(1, 1, self.model.input_size, 
                                     self.model.input_size).to(self.device)
            self.writer.add_graph(self.model, dummy_input)
        except Exception as e:
            print(f"Could not log model graph: {e}")
    
    def _log_hyperparameters(self, lr: float, beta: float, gamma: float):
        """Log hyperparameters."""
        hparams = {
            'lr': lr,
            'beta': beta,
            'gamma': gamma,
            'latent_dim': self.model.latent_dim,
            'input_size': self.model.input_size,
            'batch_size': self.train_loader.batch_size,
            'device': str(self.device)
        }
        
        # Save to file
        with open(self.log_dir / 'hparams.json', 'w') as f:
            json.dump(hparams, f, indent=2)
        
        # Log to TensorBoard
        self.writer.add_text('hyperparameters', json.dumps(hparams, indent=2))
    
    def _log_scalars(self, losses: Dict[str, torch.Tensor], prefix: str = 'train'):
        """Log scalar metrics."""
        self.writer.add_scalar(f'{prefix}/total_loss', losses['loss'].item(), self.global_step)
        self.writer.add_scalar(f'{prefix}/reconstruction_loss', 
                              losses['reconstruction_loss'].item(), self.global_step)
        self.writer.add_scalar(f'{prefix}/kl_divergence', 
                              losses['kl_loss'].item(), self.global_step)
        
        if losses['classification_loss'].item() > 0:
            self.writer.add_scalar(f'{prefix}/classification_loss',
                                  losses['classification_loss'].item(), self.global_step)
    
    def _log_learning_rate(self):
        """Log current learning rate."""
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('train/learning_rate', lr, self.global_step)
    
    def _log_weight_histograms(self):
        """Log weight histograms."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'weights/{name}', param.data, self.global_step)
                if param.grad is not None:
                    self.writer.add_histogram(f'gradients/{name}', param.grad, self.global_step)
    
    def _log_activation_histograms(self):
        """Log activation histograms."""
        activations = self.activation_hook.get_activations()
        for name, activation in activations.items():
            self.writer.add_histogram(f'activations/{name}', activation, self.global_step)
    
    def _log_activation_stats_per_group(self, dataloader: DataLoader):
        """Log activation statistics averaged per symmetry group."""
        self.model.eval()
        
        # Collect activations per group
        group_activations = defaultdict(lambda: defaultdict(list))
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                _ = self.model(images)
                
                activations = self.activation_hook.get_activations()
                
                for i, label in enumerate(labels):
                    group_name = WALLPAPER_GROUPS[label.item()]
                    for layer_name, activation in activations.items():
                        # Mean activation per sample
                        mean_act = activation[i].mean().item()
                        std_act = activation[i].std().item()
                        group_activations[group_name][layer_name].append((mean_act, std_act))
        
        # Average and log per group
        for group_name, layers in group_activations.items():
            for layer_name, stats in layers.items():
                means = [s[0] for s in stats]
                stds = [s[1] for s in stats]
                
                avg_mean = np.mean(means)
                avg_std = np.mean(stds)
                
                safe_layer_name = layer_name.replace('.', '_')
                self.writer.add_scalar(
                    f'activations_by_group/{group_name}/{safe_layer_name}_mean',
                    avg_mean, self.global_step
                )
                self.writer.add_scalar(
                    f'activations_by_group/{group_name}/{safe_layer_name}_std',
                    avg_std, self.global_step
                )
    
    @torch.no_grad()
    def _log_reconstructions_per_group(self, dataloader: DataLoader, tag: str = 'val'):
        """Log original and reconstructed images for each symmetry group."""
        self.model.eval()
        
        # Collect one sample per group
        group_samples = {}
        
        for images, labels in dataloader:
            for i, label in enumerate(labels):
                group_idx = label.item()
                if group_idx not in group_samples:
                    group_samples[group_idx] = images[i:i+1]
                
                if len(group_samples) == 17:
                    break
            if len(group_samples) == 17:
                break
        
        # Create grid: originals on top, reconstructions on bottom
        originals = []
        reconstructions = []
        
        for group_idx in range(17):
            if group_idx in group_samples:
                img = group_samples[group_idx].to(self.device)
                outputs = self.model(img)
                recon = outputs['reconstruction']
                
                originals.append(img)
                reconstructions.append(recon)
        
        if originals:
            originals = torch.cat(originals, dim=0)
            reconstructions = torch.cat(reconstructions, dim=0)
            
            # Log as image grid
            from torchvision.utils import make_grid
            
            orig_grid = make_grid(originals, nrow=17, normalize=True, padding=2)
            recon_grid = make_grid(reconstructions, nrow=17, normalize=True, padding=2)
            
            # Combined grid
            combined = torch.cat([originals, reconstructions], dim=0)
            combined_grid = make_grid(combined, nrow=17, normalize=True, padding=2)
            
            self.writer.add_image(f'{tag}/originals', orig_grid, self.global_step)
            self.writer.add_image(f'{tag}/reconstructions', recon_grid, self.global_step)
            self.writer.add_image(f'{tag}/comparison', combined_grid, self.global_step)
            
            # Log individual group reconstructions
            for idx, (orig, recon) in enumerate(zip(originals, reconstructions)):
                group_name = WALLPAPER_GROUPS[idx]
                pair = torch.stack([orig, recon], dim=0)
                pair_grid = make_grid(pair, nrow=2, normalize=True, padding=1)
                self.writer.add_image(f'{tag}_by_group/{group_name}', pair_grid, self.global_step)
    
    @torch.no_grad()
    def _log_samples_per_group(self, num_samples: int = 4):
        """Log generated samples from prior."""
        self.model.eval()
        
        samples = self.model.sample(num_samples * 17, self.device)
        
        from torchvision.utils import make_grid
        grid = make_grid(samples, nrow=num_samples, normalize=True, padding=2)
        self.writer.add_image('samples/from_prior', grid, self.global_step)
    
    @torch.no_grad()
    def _log_latent_embeddings(self, dataloader: DataLoader, max_samples: int = 1000):
        """Log latent space embeddings for TensorBoard projector."""
        from sklearn.decomposition import PCA
        
        self.model.eval()
        
        all_latents = []
        all_labels = []
        all_images = []
        
        count = 0
        for images, labels in dataloader:
            if count >= max_samples:
                break
                
            images = images.to(self.device)
            mu, _ = self.model.encode(images)
            
            all_latents.append(mu.cpu())
            all_labels.extend(labels.numpy())
            all_images.append(images.cpu())
            
            count += len(images)
        
        latents = torch.cat(all_latents, dim=0)[:max_samples]
        labels = np.array(all_labels)[:max_samples]
        images = torch.cat(all_images, dim=0)[:max_samples]
        
        # Create label names
        label_names = [WALLPAPER_GROUPS[l] for l in labels]
        
        # Resize images for sprite (smaller thumbnails)
        from torch.nn.functional import interpolate
        thumbnails = interpolate(images, size=(32, 32), mode='bilinear')
        # Ensure correct format: [N, 3, H, W] for color or repeat for grayscale
        if thumbnails.shape[1] == 1:
            thumbnails = thumbnails.repeat(1, 3, 1, 1)
        # Clamp to valid range
        thumbnails = torch.clamp(thumbnails, 0, 1)
        
        # Log embeddings with metadata and sprite
        try:
            self.writer.add_embedding(
                latents,
                metadata=label_names,
                label_img=thumbnails,
                global_step=self.current_epoch,
                tag=f'latent_space_epoch_{self.current_epoch}'
            )
            self.writer.flush()
        except Exception as e:
            print(f"  Warning: Could not log embedding to TensorBoard: {e}")
        
        # Also save embeddings to disk for manual visualization
        embeddings_dir = self.log_dir / 'embeddings'
        embeddings_dir.mkdir(exist_ok=True)
        
        embedding_data = {
            'latents': latents.numpy(),
            'labels': labels,
            'label_names': label_names,
            'epoch': self.current_epoch
        }
        
        np.savez(
            embeddings_dir / f'embeddings_epoch_{self.current_epoch:03d}.npz',
            **embedding_data
        )
        
        # Also log 2D and 3D PCA projections as scatter plots
        self._log_latent_projections(latents.numpy(), labels)
    
    def _log_latent_projections(self, latents: np.ndarray, labels: np.ndarray):
        """Log 2D and 3D projections of latent space."""
        from sklearn.decomposition import PCA
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Color map for 17 groups
        colors = plt.cm.tab20(np.linspace(0, 1, 17))
        
        # 2D PCA Projection
        pca_2d = PCA(n_components=2)
        latents_2d = pca_2d.fit_transform(latents)
        
        fig_2d, ax_2d = plt.subplots(figsize=(12, 10))
        fig_2d.patch.set_facecolor('#1a1a2e')
        ax_2d.set_facecolor('#1a1a2e')
        
        for g_idx in range(17):
            mask = labels == g_idx
            ax_2d.scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                         c=[colors[g_idx]], label=WALLPAPER_GROUPS[g_idx],
                         alpha=0.7, s=30, edgecolors='white', linewidth=0.3)
        
        ax_2d.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                    facecolor='#2a2a4e', edgecolor='#eaeaea', labelcolor='#eaeaea',
                    ncol=1, fontsize=8)
        ax_2d.set_title(f'Latent Space (PCA 2D) - Epoch {self.current_epoch}', 
                       color='#eaeaea', fontsize=14)
        ax_2d.tick_params(colors='#eaeaea')
        ax_2d.set_xlabel('PC1', color='#eaeaea')
        ax_2d.set_ylabel('PC2', color='#eaeaea')
        
        for spine in ax_2d.spines.values():
            spine.set_color('#3d3d5c')
        
        plt.tight_layout()
        
        # Convert to tensor for TensorBoard
        fig_2d.canvas.draw()
        img_2d = np.frombuffer(fig_2d.canvas.tostring_rgb(), dtype=np.uint8)
        img_2d = img_2d.reshape(fig_2d.canvas.get_width_height()[::-1] + (3,))
        img_2d = torch.from_numpy(img_2d).permute(2, 0, 1)
        
        self.writer.add_image('latent_projection/pca_2d', img_2d, self.global_step)
        
        # Save to disk
        fig_2d.savefig(self.log_dir / f'latent_pca_2d_epoch_{self.current_epoch:03d}.png',
                      dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
        plt.close(fig_2d)
        
        # 3D PCA Projection
        pca_3d = PCA(n_components=3)
        latents_3d = pca_3d.fit_transform(latents)
        
        fig_3d = plt.figure(figsize=(14, 12))
        fig_3d.patch.set_facecolor('#1a1a2e')
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        ax_3d.set_facecolor('#1a1a2e')
        
        for g_idx in range(17):
            mask = labels == g_idx
            ax_3d.scatter(latents_3d[mask, 0], latents_3d[mask, 1], latents_3d[mask, 2],
                         c=[colors[g_idx]], label=WALLPAPER_GROUPS[g_idx],
                         alpha=0.7, s=30, edgecolors='white', linewidth=0.3)
        
        ax_3d.legend(loc='center left', bbox_to_anchor=(1.1, 0.5),
                    facecolor='#2a2a4e', edgecolor='#eaeaea', labelcolor='#eaeaea',
                    ncol=1, fontsize=8)
        ax_3d.set_title(f'Latent Space (PCA 3D) - Epoch {self.current_epoch}',
                       color='#eaeaea', fontsize=14, pad=20)
        ax_3d.set_xlabel('PC1', color='#eaeaea')
        ax_3d.set_ylabel('PC2', color='#eaeaea')
        ax_3d.set_zlabel('PC3', color='#eaeaea')
        ax_3d.tick_params(colors='#eaeaea')
        
        # Style 3D axes
        ax_3d.xaxis.pane.fill = False
        ax_3d.yaxis.pane.fill = False
        ax_3d.zaxis.pane.fill = False
        ax_3d.xaxis.pane.set_edgecolor('#3d3d5c')
        ax_3d.yaxis.pane.set_edgecolor('#3d3d5c')
        ax_3d.zaxis.pane.set_edgecolor('#3d3d5c')
        
        plt.tight_layout()
        
        # Convert to tensor
        fig_3d.canvas.draw()
        img_3d = np.frombuffer(fig_3d.canvas.tostring_rgb(), dtype=np.uint8)
        img_3d = img_3d.reshape(fig_3d.canvas.get_width_height()[::-1] + (3,))
        img_3d = torch.from_numpy(img_3d).permute(2, 0, 1)
        
        self.writer.add_image('latent_projection/pca_3d', img_3d, self.global_step)
        
        # Save to disk
        fig_3d.savefig(self.log_dir / f'latent_pca_3d_epoch_{self.current_epoch:03d}.png',
                      dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
        plt.close(fig_3d)
        
        # Log explained variance
        self.writer.add_scalar('latent_analysis/pca_2d_explained_var', 
                              pca_2d.explained_variance_ratio_.sum(), self.global_step)
        self.writer.add_scalar('latent_analysis/pca_3d_explained_var',
                              pca_3d.explained_variance_ratio_.sum(), self.global_step)
    
    @torch.no_grad()
    def _log_interpolations(self, dataloader: DataLoader):
        """Log latent space interpolations between groups."""
        self.model.eval()
        
        from torchvision.utils import make_grid
        
        # Get samples from different groups
        group_samples = {}
        for images, labels in dataloader:
            for i, label in enumerate(labels):
                group_idx = label.item()
                if group_idx not in group_samples:
                    group_samples[group_idx] = images[i:i+1].to(self.device)
                if len(group_samples) >= 6:
                    break
            if len(group_samples) >= 6:
                break
        
        # Interpolate between pairs
        pairs = [(0, 16), (9, 15), (5, 11)]  # p1-p6m, p4-p6, pmm-p4g
        
        all_interpolations = []
        for g1, g2 in pairs:
            if g1 in group_samples and g2 in group_samples:
                interp = self.model.interpolate(group_samples[g1], group_samples[g2], steps=8)
                all_interpolations.append(interp)
        
        if all_interpolations:
            interp_tensor = torch.cat(all_interpolations, dim=0)
            grid = make_grid(interp_tensor, nrow=8, normalize=True, padding=2)
            self.writer.add_image('interpolations/between_groups', grid, self.global_step)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_class_loss = 0
        num_batches = 0
        
        for batch_idx, (data, labels) in enumerate(self.train_loader):
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            
            # Compute loss
            losses = self.criterion(outputs, data, labels)
            
            # Backward pass
            losses['loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += losses['loss'].item()
            total_recon_loss += losses['reconstruction_loss'].item()
            total_kl_loss += losses['kl_loss'].item()
            total_class_loss += losses['classification_loss'].item()
            num_batches += 1
            
            # Log to TensorBoard
            if batch_idx % self.log_interval == 0:
                self._log_scalars(losses, prefix='train')
                self._log_learning_rate()
                
                print(f"  Batch {batch_idx}/{len(self.train_loader)} | "
                      f"Loss: {losses['loss'].item():.4f} | "
                      f"Recon: {losses['reconstruction_loss'].item():.4f} | "
                      f"KL: {losses['kl_loss'].item():.4f}")
            
            # Log images periodically
            if batch_idx % self.image_log_interval == 0 and batch_idx > 0:
                self._log_activation_histograms()
            
            self.global_step += 1
        
        return {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches,
            'class_loss': total_class_loss / num_batches
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        
        for data, labels in self.val_loader:
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(data)
            losses = self.criterion(outputs, data, labels)
            
            total_loss += losses['loss'].item()
            total_recon_loss += losses['reconstruction_loss'].item()
            total_kl_loss += losses['kl_loss'].item()
            num_batches += 1
            
            # Classification accuracy
            if 'class_logits' in outputs:
                pred = outputs['class_logits'].argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        metrics = {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches,
        }
        
        if total > 0:
            metrics['accuracy'] = correct / total
        
        # Log validation metrics
        self.writer.add_scalar('val/total_loss', metrics['loss'], self.global_step)
        self.writer.add_scalar('val/reconstruction_loss', metrics['recon_loss'], self.global_step)
        self.writer.add_scalar('val/kl_divergence', metrics['kl_loss'], self.global_step)
        
        if 'accuracy' in metrics:
            self.writer.add_scalar('val/classification_accuracy', metrics['accuracy'], self.global_step)
        
        return metrics
    
    def train(self, 
              num_epochs: int,
              save_every: int = 5,
              visualize_every: int = 5,
              embedding_every: int = 10) -> Dict[str, List[float]]:
        """
        Full training loop with comprehensive logging.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            visualize_every: Generate visualizations every N epochs
            embedding_every: Log embeddings every N epochs
            
        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print("STARTING TRAINING WITH TENSORBOARD LOGGING")
        print(f"{'='*60}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"TensorBoard logs: {self.log_dir}")
        print(f"Run: tensorboard --logdir={self.log_dir.parent}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # Update beta if using KL annealing
            if self.kl_scheduler is not None:
                current_beta = self.kl_scheduler.get_beta(epoch)
                self.criterion.set_beta(current_beta)
                self.beta = current_beta
                print(f"\nEpoch {self.current_epoch}/{num_epochs} (β={current_beta:.4f})")
            else:
                print(f"\nEpoch {self.current_epoch}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            if val_metrics:
                self.scheduler.step(val_metrics['loss'])
            
            # Log epoch metrics
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], self.current_epoch)
            self.writer.add_scalar('epoch/val_loss', val_metrics.get('loss', 0), self.current_epoch)
            self.writer.add_scalar('epoch/beta', self.beta, self.current_epoch)
            
            # Update history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon_loss'].append(train_metrics['recon_loss'])
            self.history['train_kl_loss'].append(train_metrics['kl_loss'])
            self.history['lr'].append(current_lr)
            
            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_recon_loss'].append(val_metrics['recon_loss'])
                self.history['val_kl_loss'].append(val_metrics['kl_loss'])
            
            # Print summary
            print(f"  Train Loss: {train_metrics['loss']:.4f} | "
                  f"Recon: {train_metrics['recon_loss']:.4f} | "
                  f"KL: {train_metrics['kl_loss']:.4f}")
            
            if val_metrics:
                acc_str = f" | Acc: {val_metrics.get('accuracy', 0)*100:.1f}%" if 'accuracy' in val_metrics else ""
                print(f"  Val Loss: {val_metrics['loss']:.4f} | "
                      f"Recon: {val_metrics['recon_loss']:.4f} | "
                      f"KL: {val_metrics['kl_loss']:.4f}{acc_str}")
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pt')
                    print("  ✓ New best model saved!")
            
            # Periodic logging
            if self.current_epoch % visualize_every == 0:
                print("  Logging visualizations...")
                self._log_reconstructions_per_group(self.val_loader or self.train_loader)
                self._log_samples_per_group()
                self._log_interpolations(self.val_loader or self.train_loader)
                self._log_weight_histograms()
                self._log_activation_stats_per_group(self.val_loader or self.train_loader)
                
                # Disk logging with TrainingLogger
                if self.disk_logger is not None:
                    self.disk_logger.log_basic(self.model, self.val_loader or self.train_loader, 
                                              self.current_epoch)
            
            # Extended logging every 20 epochs
            if self.current_epoch % 20 == 0 and self.disk_logger is not None:
                self.disk_logger.log_extended(self.model, self.val_loader or self.train_loader,
                                             self.current_epoch)
            
            # Log embeddings
            if self.current_epoch % embedding_every == 0:
                print("  Logging latent embeddings...")
                self._log_latent_embeddings(self.val_loader or self.train_loader)
            
            # Periodic checkpoint
            if self.current_epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{self.current_epoch}.pt')
        
        # Final save
        self.save_checkpoint('final_model.pt')
        self.save_history()
        
        # Close TensorBoard writer
        self.writer.close()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"TensorBoard logs: {self.log_dir}")
        print(f"\nTo view logs, run:")
        print(f"  tensorboard --logdir={self.log_dir.parent}")
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'model_config': {
                'latent_dim': self.model.latent_dim,
                'input_size': self.model.input_size,
                'num_classes': self.model.num_classes
            }
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Also save to log dir
        with open(self.log_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)


def plot_training_curves(history: Dict[str, List[float]], 
                        save_path: Optional[str] = None):
    """Plot training curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor('#1a1a2e')
    
    for ax in axes.flat:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#eaeaea')
        ax.spines['bottom'].set_color('#eaeaea')
        ax.spines['left'].set_color('#eaeaea')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Total loss
    axes[0, 0].plot(epochs, history['train_loss'], 'c-', label='Train', linewidth=2)
    if history['val_loss']:
        axes[0, 0].plot(epochs, history['val_loss'], 'm-', label='Val', linewidth=2)
    axes[0, 0].set_title('Total Loss', color='#eaeaea', fontsize=12)
    axes[0, 0].set_xlabel('Epoch', color='#eaeaea')
    axes[0, 0].legend(facecolor='#2a2a4e', edgecolor='#eaeaea', labelcolor='#eaeaea')
    
    # Reconstruction loss
    axes[0, 1].plot(epochs, history['train_recon_loss'], 'c-', label='Train', linewidth=2)
    if history['val_recon_loss']:
        axes[0, 1].plot(epochs, history['val_recon_loss'], 'm-', label='Val', linewidth=2)
    axes[0, 1].set_title('Reconstruction Loss', color='#eaeaea', fontsize=12)
    axes[0, 1].set_xlabel('Epoch', color='#eaeaea')
    axes[0, 1].legend(facecolor='#2a2a4e', edgecolor='#eaeaea', labelcolor='#eaeaea')
    
    # KL loss
    axes[1, 0].plot(epochs, history['train_kl_loss'], 'c-', label='Train', linewidth=2)
    if history['val_kl_loss']:
        axes[1, 0].plot(epochs, history['val_kl_loss'], 'm-', label='Val', linewidth=2)
    axes[1, 0].set_title('KL Divergence', color='#eaeaea', fontsize=12)
    axes[1, 0].set_xlabel('Epoch', color='#eaeaea')
    axes[1, 0].legend(facecolor='#2a2a4e', edgecolor='#eaeaea', labelcolor='#eaeaea')
    
    # Learning rate
    axes[1, 1].plot(epochs, history['lr'], 'y-', linewidth=2)
    axes[1, 1].set_title('Learning Rate', color='#eaeaea', fontsize=12)
    axes[1, 1].set_xlabel('Epoch', color='#eaeaea')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    
    return fig
