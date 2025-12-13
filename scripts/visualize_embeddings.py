#!/usr/bin/env python3
"""
Visualize latent space embeddings from saved .npz files.

Creates interactive 2D and 3D visualizations of the latent space
with colors indicating wallpaper group.

Usage:
    python scripts/visualize_embeddings.py --embeddings-dir runs/experiment_name/embeddings
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Wallpaper group names
WALLPAPER_GROUPS = [
    'p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm',
    'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m'
]


def load_embeddings(npz_path: Path) -> dict:
    """Load embeddings from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        'latents': data['latents'],
        'labels': data['labels'],
        'label_names': data['label_names'],
        'epoch': int(data['epoch'])
    }


def plot_embeddings_2d(latents: np.ndarray, labels: np.ndarray, 
                       method: str = 'pca', title: str = '',
                       save_path: str = None):
    """Plot 2D projection of latent space."""
    
    # Reduce dimensionality
    if method == 'pca':
        reducer = PCA(n_components=2)
        title_suffix = 'PCA'
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents)-1))
        title_suffix = 't-SNE'
    
    latents_2d = reducer.fit_transform(latents)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    
    # Colors for each group
    colors = plt.cm.tab20(np.linspace(0, 1, 17))
    
    for g_idx in range(17):
        mask = labels == g_idx
        if mask.sum() > 0:
            ax.scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                      c=[colors[g_idx]], label=WALLPAPER_GROUPS[g_idx],
                      alpha=0.7, s=40, edgecolors='white', linewidth=0.3)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
             facecolor='#1a1a2e', edgecolor='#eaeaea', labelcolor='#eaeaea',
             ncol=1, fontsize=10)
    
    ax.set_title(f'{title} - {title_suffix}', color='#eaeaea', fontsize=16, pad=20)
    ax.set_xlabel('Component 1', color='#eaeaea', fontsize=12)
    ax.set_ylabel('Component 2', color='#eaeaea', fontsize=12)
    ax.tick_params(colors='#eaeaea')
    
    for spine in ax.spines.values():
        spine.set_color('#3d3d5c')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, facecolor='#0f0f1a', bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_embeddings_3d(latents: np.ndarray, labels: np.ndarray,
                       method: str = 'pca', title: str = '',
                       save_path: str = None):
    """Plot 3D projection of latent space."""
    
    # Reduce dimensionality
    if method == 'pca':
        reducer = PCA(n_components=3)
        title_suffix = 'PCA 3D'
    else:
        reducer = TSNE(n_components=3, random_state=42, perplexity=min(30, len(latents)-1))
        title_suffix = 't-SNE 3D'
    
    latents_3d = reducer.fit_transform(latents)
    
    # Plot
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor('#0f0f1a')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0f0f1a')
    
    colors = plt.cm.tab20(np.linspace(0, 1, 17))
    
    for g_idx in range(17):
        mask = labels == g_idx
        if mask.sum() > 0:
            ax.scatter(latents_3d[mask, 0], latents_3d[mask, 1], latents_3d[mask, 2],
                      c=[colors[g_idx]], label=WALLPAPER_GROUPS[g_idx],
                      alpha=0.7, s=30, edgecolors='white', linewidth=0.2)
    
    ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5),
             facecolor='#1a1a2e', edgecolor='#eaeaea', labelcolor='#eaeaea',
             ncol=1, fontsize=9)
    
    ax.set_title(f'{title} - {title_suffix}', color='#eaeaea', fontsize=16, pad=20)
    ax.set_xlabel('PC1', color='#eaeaea', fontsize=10)
    ax.set_ylabel('PC2', color='#eaeaea', fontsize=10)
    ax.set_zlabel('PC3', color='#eaeaea', fontsize=10)
    ax.tick_params(colors='#eaeaea')
    
    # Style panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#3d3d5c')
    ax.yaxis.pane.set_edgecolor('#3d3d5c')
    ax.zaxis.pane.set_edgecolor('#3d3d5c')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, facecolor='#0f0f1a', bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_cluster_analysis(latents: np.ndarray, labels: np.ndarray,
                          title: str = '', save_path: str = None):
    """Analyze cluster quality."""
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.neighbors import KNeighborsClassifier
    
    # Compute metrics
    silhouette_avg = silhouette_score(latents, labels)
    silhouette_vals = silhouette_samples(latents, labels)
    
    # KNN accuracy
    knn = KNeighborsClassifier(n_neighbors=5)
    n = len(labels)
    idx = np.random.permutation(n)
    train_idx, test_idx = idx[:int(0.8*n)], idx[int(0.8*n):]
    knn.fit(latents[train_idx], labels[train_idx])
    knn_acc = knn.score(latents[test_idx], labels[test_idx])
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0f0f1a')
    
    for ax in axes:
        ax.set_facecolor('#0f0f1a')
        ax.tick_params(colors='#eaeaea')
        for spine in ax.spines.values():
            spine.set_color('#3d3d5c')
    
    # Silhouette plot
    y_lower = 10
    colors = plt.cm.tab20(np.linspace(0, 1, 17))
    
    for g_idx in range(17):
        mask = labels == g_idx
        if mask.sum() == 0:
            continue
            
        cluster_silhouette = silhouette_vals[mask]
        cluster_silhouette.sort()
        
        size = cluster_silhouette.shape[0]
        y_upper = y_lower + size
        
        axes[0].fill_betweenx(np.arange(y_lower, y_upper),
                              0, cluster_silhouette,
                              facecolor=colors[g_idx], alpha=0.7)
        axes[0].text(-0.05, y_lower + 0.5 * size, WALLPAPER_GROUPS[g_idx],
                    color='#eaeaea', fontsize=8)
        
        y_lower = y_upper + 10
    
    axes[0].axvline(x=silhouette_avg, color='#FF6B6B', linestyle='--', linewidth=2,
                    label=f'Avg: {silhouette_avg:.3f}')
    axes[0].set_xlabel('Silhouette Coefficient', color='#eaeaea')
    axes[0].set_ylabel('Cluster (Group)', color='#eaeaea')
    axes[0].set_title('Silhouette Analysis', color='#eaeaea', fontsize=14)
    axes[0].legend(facecolor='#1a1a2e', edgecolor='#eaeaea', labelcolor='#eaeaea')
    
    # Per-group silhouette
    group_silhouettes = []
    for g_idx in range(17):
        mask = labels == g_idx
        if mask.sum() > 0:
            group_silhouettes.append(silhouette_vals[mask].mean())
        else:
            group_silhouettes.append(0)
    
    bars = axes[1].barh(range(17), group_silhouettes, color=colors)
    axes[1].set_yticks(range(17))
    axes[1].set_yticklabels(WALLPAPER_GROUPS)
    axes[1].set_xlabel('Mean Silhouette', color='#eaeaea')
    axes[1].set_title(f'Per-Group Quality (KNN Acc: {knn_acc:.1%})', 
                     color='#eaeaea', fontsize=14)
    axes[1].axvline(x=0, color='#eaeaea', linestyle='-', linewidth=0.5)
    
    plt.suptitle(title, color='#eaeaea', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, facecolor='#0f0f1a', bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, {'silhouette': silhouette_avg, 'knn_accuracy': knn_acc}


def main():
    parser = argparse.ArgumentParser(description="Visualize latent embeddings")
    
    parser.add_argument('--embeddings-dir', '-e', type=str, required=True,
                       help='Directory containing embedding .npz files')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Output directory (default: same as embeddings)')
    parser.add_argument('--epoch', type=int, default=None,
                       help='Specific epoch to visualize (default: latest)')
    parser.add_argument('--method', type=str, default='both',
                       choices=['pca', 'tsne', 'both'])
    
    args = parser.parse_args()
    
    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir) if args.output_dir else embeddings_dir.parent / 'embedding_plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find embedding files
    npz_files = sorted(embeddings_dir.glob('embeddings_epoch_*.npz'))
    
    if not npz_files:
        print(f"No embedding files found in {embeddings_dir}")
        return
    
    print(f"Found {len(npz_files)} embedding files")
    
    # Select epoch
    if args.epoch is not None:
        target_file = embeddings_dir / f'embeddings_epoch_{args.epoch:03d}.npz'
        if not target_file.exists():
            print(f"Epoch {args.epoch} not found")
            return
        npz_files = [target_file]
    else:
        # Use latest
        npz_files = [npz_files[-1]]
    
    for npz_path in npz_files:
        print(f"\nProcessing: {npz_path.name}")
        
        data = load_embeddings(npz_path)
        latents = data['latents']
        labels = data['labels']
        epoch = data['epoch']
        
        print(f"  Samples: {len(latents)}, Latent dim: {latents.shape[1]}")
        
        title = f'Latent Space - Epoch {epoch}'
        
        # 2D plots
        if args.method in ['pca', 'both']:
            plot_embeddings_2d(latents, labels, 'pca', title,
                              str(output_dir / f'latent_2d_pca_epoch_{epoch:03d}.png'))
        
        if args.method in ['tsne', 'both']:
            print("  Computing t-SNE (may take a moment)...")
            plot_embeddings_2d(latents, labels, 'tsne', title,
                              str(output_dir / f'latent_2d_tsne_epoch_{epoch:03d}.png'))
        
        # 3D plot
        plot_embeddings_3d(latents, labels, 'pca', title,
                          str(output_dir / f'latent_3d_pca_epoch_{epoch:03d}.png'))
        
        # Cluster analysis
        fig, metrics = plot_cluster_analysis(latents, labels, title,
                                             str(output_dir / f'cluster_analysis_epoch_{epoch:03d}.png'))
        
        print(f"\n  📊 Metrics:")
        print(f"    Silhouette Score: {metrics['silhouette']:.4f}")
        print(f"    KNN Accuracy: {metrics['knn_accuracy']:.1%}")
        
        plt.close('all')
    
    print(f"\n✅ Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()







