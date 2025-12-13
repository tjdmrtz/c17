"""
Comprehensive Training Logger for VAE.

Logs to both TensorBoard and disk:
- Reconstructions (truth vs generated) every N epochs
- Latent space visualizations (2D, 3D PCA/t-SNE)
- Interpolations between samples
- Cluster analysis (silhouette, composition)
- Generated samples
- All metrics and embeddings

Usage:
    logger = TrainingLogger(output_dir, writer, device)
    
    # Every 5 epochs
    logger.log_reconstructions(model, dataloader, epoch)
    logger.log_samples(model, epoch)
    
    # Every 20 epochs
    logger.log_full_analysis(model, dataloader, epoch)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
from torchvision.utils import make_grid, save_image


# Wallpaper groups
GROUPS = ['p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm',
          'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m']

# Colors for groups
GROUP_COLORS = plt.cm.tab20(np.linspace(0, 1, 17))


class TrainingLogger:
    """
    Comprehensive logger for VAE training.
    
    Saves visualizations and metrics to disk and TensorBoard.
    """
    
    def __init__(self, 
                 output_dir: str,
                 writer=None,  # TensorBoard SummaryWriter
                 device: str = 'cuda'):
        """
        Initialize logger.
        
        Args:
            output_dir: Base directory for saving files
            writer: TensorBoard SummaryWriter (optional)
            device: Device for model inference
        """
        self.output_dir = Path(output_dir)
        self.writer = writer
        self.device = device
        
        # Create subdirectories
        self.dirs = {
            'reconstructions': self.output_dir / 'reconstructions',
            'samples': self.output_dir / 'samples',
            'latent_space': self.output_dir / 'latent_space',
            'interpolations': self.output_dir / 'interpolations',
            'clusters': self.output_dir / 'clusters',
            'embeddings': self.output_dir / 'embeddings',
            'metrics': self.output_dir / 'metrics',
        }
        
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        
        # Track metrics history
        self.metrics_history = {
            'epochs': [],
            'silhouette_scores': [],
            'cluster_purities': [],
        }
    
    def _get_latents_and_labels(self, model: nn.Module, dataloader, 
                                 max_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, List]:
        """Extract latent representations from dataloader."""
        model.eval()
        
        all_latents = []
        all_labels = []
        all_images = []
        
        with torch.no_grad():
            count = 0
            for images, labels in dataloader:
                if count >= max_samples:
                    break
                
                images = images.to(self.device)
                
                if hasattr(model, 'encode'):
                    mu, _ = model.encode(images)
                else:
                    outputs = model(images)
                    mu = outputs['mu']
                
                # Flatten if spatial latent
                if mu.dim() > 2:
                    mu = mu.view(mu.size(0), -1)
                
                all_latents.append(mu.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_images.append(images.cpu())
                count += len(images)
        
        latents = np.concatenate(all_latents, axis=0)[:max_samples]
        labels = np.array(all_labels)[:max_samples]
        images = torch.cat(all_images, dim=0)[:max_samples]
        
        return latents, labels, images
    
    def log_reconstructions(self, model: nn.Module, dataloader, epoch: int,
                            num_samples: int = 8):
        """
        Log original vs reconstructed images.
        
        Saves:
        - Grid comparison image
        - Per-group reconstructions
        """
        model.eval()
        
        # Get samples from each group
        group_samples = {}
        group_labels = {}
        
        with torch.no_grad():
            for images, labels in dataloader:
                for i, label in enumerate(labels):
                    g_idx = label.item()
                    if g_idx not in group_samples and len(group_samples) < 17:
                        group_samples[g_idx] = images[i:i+1].to(self.device)
                        group_labels[g_idx] = g_idx
                
                if len(group_samples) >= 17:
                    break
            
            # Reconstruct
            originals = []
            reconstructions = []
            labels_list = []
            
            for g_idx in sorted(group_samples.keys()):
                img = group_samples[g_idx]
                outputs = model(img)
                originals.append(img.cpu())
                reconstructions.append(outputs['reconstruction'].cpu())
                labels_list.append(g_idx)
        
        if not originals:
            return
        
        originals = torch.cat(originals, dim=0)
        reconstructions = torch.cat(reconstructions, dim=0)
        
        # Save grid comparison
        n = len(originals)
        fig, axes = plt.subplots(2, n, figsize=(2*n, 4), facecolor='#0f0f1a')
        
        for i in range(n):
            # Original
            img_orig = originals[i].permute(1, 2, 0).numpy()
            img_orig = np.clip(img_orig, 0, 1)
            axes[0, i].imshow(img_orig)
            axes[0, i].set_title(GROUPS[labels_list[i]], color='white', fontsize=8)
            axes[0, i].axis('off')
            
            # Reconstruction
            img_recon = reconstructions[i].permute(1, 2, 0).numpy()
            img_recon = np.clip(img_recon, 0, 1)
            axes[1, i].imshow(img_recon)
            axes[1, i].axis('off')
        
        axes[0, 0].set_ylabel('Original', color='white', fontsize=10)
        axes[1, 0].set_ylabel('Reconstructed', color='white', fontsize=10)
        
        plt.suptitle(f'Reconstructions - Epoch {epoch}', color='white', fontsize=12)
        plt.tight_layout()
        
        save_path = self.dirs['reconstructions'] / f'comparison_epoch_{epoch:04d}.png'
        plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
        plt.close()
        
        # Also save to TensorBoard
        if self.writer:
            comparison = torch.cat([originals, reconstructions], dim=0)
            grid = make_grid(comparison, nrow=n, normalize=True, padding=2)
            self.writer.add_image('reconstructions/comparison', grid, epoch)
        
        print(f"    💾 Saved reconstructions to {save_path}")
    
    def log_samples(self, model: nn.Module, epoch: int, num_samples: int = 16):
        """Log randomly generated samples."""
        model.eval()
        
        with torch.no_grad():
            samples = model.sample(num_samples, self.device)
        
        # Save grid
        nrow = int(np.ceil(np.sqrt(num_samples)))
        
        fig, axes = plt.subplots(nrow, nrow, figsize=(12, 12), facecolor='#0f0f1a')
        axes = axes.flatten()
        
        for i in range(num_samples):
            img = samples[i].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[i].imshow(img)
            axes[i].axis('off')
        
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Generated Samples - Epoch {epoch}', color='white', fontsize=14)
        plt.tight_layout()
        
        save_path = self.dirs['samples'] / f'samples_epoch_{epoch:04d}.png'
        plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
        plt.close()
        
        # TensorBoard
        if self.writer:
            grid = make_grid(samples, nrow=nrow, normalize=True)
            self.writer.add_image('samples/generated', grid, epoch)
        
        print(f"    💾 Saved samples to {save_path}")
    
    def log_latent_space_2d(self, model: nn.Module, dataloader, epoch: int,
                            method: str = 'pca', max_samples: int = 1000):
        """
        Log 2D latent space visualization.
        
        Args:
            method: 'pca' or 'tsne'
        """
        latents, labels, _ = self._get_latents_and_labels(model, dataloader, max_samples)
        
        # Dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(latents)
            explained_var = reducer.explained_variance_ratio_.sum()
            title_suffix = f'(explained var: {explained_var:.1%})'
        else:  # tsne
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents)-1))
            coords = reducer.fit_transform(latents)
            title_suffix = ''
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 12), facecolor='#0f0f1a')
        ax.set_facecolor('#0f0f1a')
        
        for g_idx in range(17):
            mask = labels == g_idx
            if mask.sum() > 0:
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=[GROUP_COLORS[g_idx]], label=GROUPS[g_idx],
                          alpha=0.7, s=30, edgecolors='white', linewidth=0.2)
        
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                 facecolor='#1a1a2e', edgecolor='#eaeaea', labelcolor='#eaeaea')
        ax.set_title(f'Latent Space {method.upper()} 2D - Epoch {epoch} {title_suffix}', 
                    color='#eaeaea', fontsize=14)
        ax.tick_params(colors='#eaeaea')
        for spine in ax.spines.values():
            spine.set_color('#3d3d5c')
        
        plt.tight_layout()
        
        save_path = self.dirs['latent_space'] / f'latent_{method}_2d_epoch_{epoch:04d}.png'
        plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
        plt.close()
        
        # TensorBoard
        if self.writer:
            fig2, ax2 = plt.subplots(figsize=(10, 10))
            for g_idx in range(17):
                mask = labels == g_idx
                if mask.sum() > 0:
                    ax2.scatter(coords[mask, 0], coords[mask, 1],
                               c=[GROUP_COLORS[g_idx]], label=GROUPS[g_idx], alpha=0.7, s=20)
            ax2.legend(fontsize=6)
            self.writer.add_figure(f'latent_space/{method}_2d', fig2, epoch)
            plt.close(fig2)
        
        print(f"    💾 Saved latent space 2D ({method}) to {save_path}")
        
        return coords, labels
    
    def log_latent_space_3d(self, model: nn.Module, dataloader, epoch: int,
                            max_samples: int = 1000):
        """Log 3D PCA latent space visualization."""
        latents, labels, _ = self._get_latents_and_labels(model, dataloader, max_samples)
        
        # PCA 3D
        pca = PCA(n_components=3)
        coords = pca.fit_transform(latents)
        
        # Plot
        fig = plt.figure(figsize=(14, 12), facecolor='#0f0f1a')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#0f0f1a')
        
        for g_idx in range(17):
            mask = labels == g_idx
            if mask.sum() > 0:
                ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2],
                          c=[GROUP_COLORS[g_idx]], label=GROUPS[g_idx],
                          alpha=0.7, s=20)
        
        ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5),
                 facecolor='#1a1a2e', edgecolor='#eaeaea', labelcolor='#eaeaea', fontsize=8)
        ax.set_title(f'Latent Space PCA 3D - Epoch {epoch}', color='#eaeaea', fontsize=14)
        
        plt.tight_layout()
        
        save_path = self.dirs['latent_space'] / f'latent_pca_3d_epoch_{epoch:04d}.png'
        plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
        plt.close()
        
        print(f"    💾 Saved latent space 3D to {save_path}")
    
    def log_cluster_analysis(self, model: nn.Module, dataloader, epoch: int,
                             max_samples: int = 1000):
        """
        Analyze clustering quality in latent space.
        
        Computes:
        - Silhouette score (overall and per-sample)
        - Cluster purity (how well clusters match true labels)
        - Saves silhouette map and cluster composition
        """
        latents, labels, _ = self._get_latents_and_labels(model, dataloader, max_samples)
        
        # Silhouette score (using true labels)
        try:
            silhouette_avg = silhouette_score(latents, labels)
            silhouette_per_sample = silhouette_samples(latents, labels)
        except:
            silhouette_avg = 0
            silhouette_per_sample = np.zeros(len(labels))
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=17, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(latents)
        
        # Cluster purity
        purity = self._compute_cluster_purity(labels, cluster_labels)
        
        # Save metrics (convert to Python float for JSON serialization)
        self.metrics_history['epochs'].append(epoch)
        self.metrics_history['silhouette_scores'].append(float(silhouette_avg))
        self.metrics_history['cluster_purities'].append(float(purity))
        
        # Save to JSON
        metrics_path = self.dirs['metrics'] / 'cluster_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # === Silhouette Map ===
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='#0f0f1a')
        
        # Left: Silhouette plot
        ax = axes[0]
        ax.set_facecolor('#0f0f1a')
        
        y_lower = 10
        for g_idx in range(17):
            mask = labels == g_idx
            if mask.sum() == 0:
                continue
            
            cluster_silhouette = silhouette_per_sample[mask]
            cluster_silhouette.sort()
            
            y_upper = y_lower + mask.sum()
            
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette,
                            facecolor=GROUP_COLORS[g_idx], edgecolor=GROUP_COLORS[g_idx],
                            alpha=0.7)
            
            ax.text(-0.05, y_lower + 0.5 * mask.sum(), GROUPS[g_idx],
                   color='white', fontsize=8)
            
            y_lower = y_upper + 10
        
        ax.axvline(x=silhouette_avg, color='red', linestyle='--', linewidth=2,
                  label=f'Avg: {silhouette_avg:.3f}')
        ax.set_xlabel('Silhouette Coefficient', color='white')
        ax.set_ylabel('Samples (by group)', color='white')
        ax.set_title(f'Silhouette Map - Epoch {epoch}', color='white', fontsize=12)
        ax.legend(facecolor='#1a1a2e', labelcolor='white')
        ax.tick_params(colors='white')
        
        # Right: Cluster composition heatmap
        ax2 = axes[1]
        ax2.set_facecolor('#0f0f1a')
        
        # Confusion matrix: true labels vs cluster assignments
        confusion = np.zeros((17, 17))
        for true_label, cluster in zip(labels, cluster_labels):
            confusion[true_label, cluster] += 1
        
        # Normalize by rows
        confusion_norm = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-8)
        
        im = ax2.imshow(confusion_norm, cmap='YlOrRd', aspect='auto')
        ax2.set_xlabel('Predicted Cluster', color='white')
        ax2.set_ylabel('True Group', color='white')
        ax2.set_title(f'Cluster Composition (Purity: {purity:.1%})', color='white', fontsize=12)
        ax2.set_xticks(range(17))
        ax2.set_yticks(range(17))
        ax2.set_xticklabels(range(17), fontsize=7, color='white')
        ax2.set_yticklabels(GROUPS, fontsize=7, color='white')
        
        plt.colorbar(im, ax=ax2, label='Proportion')
        
        plt.tight_layout()
        
        save_path = self.dirs['clusters'] / f'cluster_analysis_epoch_{epoch:04d}.png'
        plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
        plt.close()
        
        # TensorBoard
        if self.writer:
            self.writer.add_scalar('clusters/silhouette_score', silhouette_avg, epoch)
            self.writer.add_scalar('clusters/purity', purity, epoch)
        
        print(f"    📊 Silhouette: {silhouette_avg:.3f}, Purity: {purity:.1%}")
        print(f"    💾 Saved cluster analysis to {save_path}")
        
        return silhouette_avg, purity
    
    def _compute_cluster_purity(self, true_labels, cluster_labels) -> float:
        """Compute cluster purity."""
        contingency = np.zeros((17, 17))
        for true, pred in zip(true_labels, cluster_labels):
            contingency[pred, true] += 1
        
        purity = np.sum(np.max(contingency, axis=1)) / len(true_labels)
        return purity
    
    def log_interpolations(self, model: nn.Module, dataloader, epoch: int,
                           num_pairs: int = 4, steps: int = 8):
        """
        Log interpolations between samples in latent space.
        
        Shows smooth transitions between different crystallographic patterns.
        """
        model.eval()
        
        # Get samples from different groups
        group_samples = {}
        for images, labels in dataloader:
            for i, label in enumerate(labels):
                g_idx = label.item()
                if g_idx not in group_samples:
                    group_samples[g_idx] = images[i:i+1].to(self.device)
                if len(group_samples) >= 17:
                    break
            if len(group_samples) >= 17:
                break
        
        if len(group_samples) < 2:
            return
        
        # Select pairs for interpolation
        groups = list(group_samples.keys())
        np.random.seed(epoch)
        pairs = []
        for _ in range(num_pairs):
            g1, g2 = np.random.choice(groups, 2, replace=False)
            pairs.append((g1, g2))
        
        # Create interpolations
        fig, axes = plt.subplots(num_pairs, steps + 2, 
                                figsize=(2*(steps+2), 2*num_pairs), 
                                facecolor='#0f0f1a')
        
        with torch.no_grad():
            for row, (g1, g2) in enumerate(pairs):
                img1 = group_samples[g1]
                img2 = group_samples[g2]
                
                # Encode
                mu1, _ = model.encode(img1)
                mu2, _ = model.encode(img2)
                
                # Interpolate
                alphas = torch.linspace(0, 1, steps)
                
                for col, alpha in enumerate(alphas):
                    z = (1 - alpha) * mu1 + alpha * mu2
                    recon = model.decode(z)
                    
                    img = recon[0].cpu().permute(1, 2, 0).numpy()
                    img = np.clip(img, 0, 1)
                    
                    ax = axes[row, col + 1] if num_pairs > 1 else axes[col + 1]
                    ax.imshow(img)
                    ax.axis('off')
                    
                    if row == 0:
                        ax.set_title(f'α={alpha:.1f}', color='white', fontsize=8)
                
                # Original images at ends
                ax_left = axes[row, 0] if num_pairs > 1 else axes[0]
                ax_right = axes[row, -1] if num_pairs > 1 else axes[-1]
                
                img1_np = img1[0].cpu().permute(1, 2, 0).numpy()
                img2_np = img2[0].cpu().permute(1, 2, 0).numpy()
                
                ax_left.imshow(np.clip(img1_np, 0, 1))
                ax_left.set_title(f'{GROUPS[g1]}', color='cyan', fontsize=9)
                ax_left.axis('off')
                
                ax_right.imshow(np.clip(img2_np, 0, 1))
                ax_right.set_title(f'{GROUPS[g2]}', color='cyan', fontsize=9)
                ax_right.axis('off')
        
        plt.suptitle(f'Latent Space Interpolations - Epoch {epoch}', color='white', fontsize=12)
        plt.tight_layout()
        
        save_path = self.dirs['interpolations'] / f'interpolations_epoch_{epoch:04d}.png'
        plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
        plt.close()
        
        print(f"    💾 Saved interpolations to {save_path}")
    
    def log_embeddings(self, model: nn.Module, dataloader, epoch: int,
                       max_samples: int = 500):
        """Save embeddings for later analysis."""
        latents, labels, images = self._get_latents_and_labels(model, dataloader, max_samples)
        
        # Save to disk
        save_path = self.dirs['embeddings'] / f'embeddings_epoch_{epoch:04d}.npz'
        np.savez(save_path,
                 latents=latents,
                 labels=labels,
                 label_names=[GROUPS[l] for l in labels],
                 epoch=epoch)
        
        # TensorBoard projector
        if self.writer:
            # Subsample for TensorBoard
            n = min(200, len(latents))
            indices = np.random.choice(len(latents), n, replace=False)
            
            self.writer.add_embedding(
                torch.from_numpy(latents[indices]),
                metadata=[GROUPS[l] for l in labels[indices]],
                global_step=epoch,
                tag='latent_space'
            )
        
        print(f"    💾 Saved embeddings to {save_path}")
    
    def log_basic(self, model: nn.Module, dataloader, epoch: int):
        """Basic logging every 5 epochs."""
        print(f"  📸 Logging epoch {epoch}...")
        self.log_reconstructions(model, dataloader, epoch)
        self.log_samples(model, epoch)
    
    def log_extended(self, model: nn.Module, dataloader, epoch: int):
        """Extended logging every 20 epochs."""
        print(f"  📊 Extended logging epoch {epoch}...")
        self.log_basic(model, dataloader, epoch)
        self.log_latent_space_2d(model, dataloader, epoch, method='pca')
        self.log_latent_space_3d(model, dataloader, epoch)
        self.log_interpolations(model, dataloader, epoch)
        self.log_cluster_analysis(model, dataloader, epoch)
        self.log_embeddings(model, dataloader, epoch)
    
    def log_full(self, model: nn.Module, dataloader, epoch: int):
        """Full logging with t-SNE (slower, every 50 epochs)."""
        print(f"  🔬 Full analysis epoch {epoch}...")
        self.log_extended(model, dataloader, epoch)
        self.log_latent_space_2d(model, dataloader, epoch, method='tsne')


if __name__ == "__main__":
    print("Training Logger - Ready to use")
    print("=" * 50)
    print("""
Usage in training script:

    from src.visualization.training_logger import TrainingLogger
    
    logger = TrainingLogger(output_dir, writer, device)
    
    for epoch in range(num_epochs):
        # ... training code ...
        
        if epoch % 5 == 0:
            logger.log_basic(model, val_loader, epoch)
        
        if epoch % 20 == 0:
            logger.log_extended(model, val_loader, epoch)
    """)

