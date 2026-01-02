#!/usr/bin/env python3
"""
Hyperparameter search for the Crystallographic VAE.

Runs multiple experiments varying key hyperparameters:
- Beta (KL weight): Controls reconstruction vs latent space organization
- Latent dimension: Size of the latent space
- Learning rate: Training speed and convergence
- Base channels: Model capacity

All experiments are logged to TensorBoard for comparison.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
import itertools

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.dataset import CrystallographicDataset, create_dataloaders
from src.models import CrystallographicVAE, VAELoss, VAETrainer


def run_single_experiment(config: dict, 
                          train_loader: DataLoader,
                          val_loader: DataLoader,
                          base_log_dir: Path,
                          base_checkpoint_dir: Path,
                          device: str = 'auto') -> dict:
    """Run a single experiment with given configuration."""
    
    exp_name = (f"lat{config['latent_dim']}_"
                f"beta{config['beta']}_"
                f"lr{config['lr']:.0e}_"
                f"ch{config['base_channels']}")
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*70}")
    print(f"  Latent dim: {config['latent_dim']}")
    print(f"  Beta: {config['beta']}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Base channels: {config['base_channels']}")
    print(f"  Epochs: {config['epochs']}")
    
    # Create model
    model = CrystallographicVAE(
        in_channels=1,
        latent_dim=config['latent_dim'],
        base_channels=config['base_channels'],
        input_size=config['resolution'],
        num_classes=17
    )
    
    # Create trainer
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=config['lr'],
        beta=config['beta'],
        gamma=config['gamma'],
        device=device,
        checkpoint_dir=str(base_checkpoint_dir / exp_name),
        log_dir=str(base_log_dir),
        experiment_name=exp_name,
        log_interval=50,
        image_log_interval=200
    )
    
    # Train
    history = trainer.train(
        num_epochs=config['epochs'],
        save_every=config['epochs'],  # Only save at end
        visualize_every=max(1, config['epochs'] // 5),
        embedding_every=max(1, config['epochs'] // 3)
    )
    
    # Collect results
    results = {
        'config': config,
        'exp_name': exp_name,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_recon_loss': history['val_recon_loss'][-1],
        'final_kl_loss': history['val_kl_loss'][-1],
        'best_val_loss': trainer.best_val_loss,
        'history': history
    }
    
    # Compute additional metrics
    results['latent_quality'] = evaluate_latent_quality(trainer, val_loader)
    
    print(f"\n  Results:")
    print(f"    Best val loss: {results['best_val_loss']:.4f}")
    print(f"    Final recon loss: {results['final_recon_loss']:.4f}")
    print(f"    Final KL loss: {results['final_kl_loss']:.4f}")
    print(f"    Latent quality score: {results['latent_quality']:.4f}")
    
    return results


@torch.no_grad()
def evaluate_latent_quality(trainer: VAETrainer, val_loader: DataLoader) -> float:
    """
    Evaluate quality of latent space organization.
    
    Measures:
    1. Cluster separation (samples from same group should be close)
    2. Classification accuracy from latent space
    
    Returns a quality score (higher is better).
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import silhouette_score
    
    trainer.model.eval()
    
    all_latents = []
    all_labels = []
    
    for images, labels in val_loader:
        images = images.to(trainer.device)
        mu, _ = trainer.model.encode(images)
        all_latents.append(mu.cpu().numpy())
        all_labels.extend(labels.numpy())
    
    latents = np.concatenate(all_latents, axis=0)
    labels = np.array(all_labels)
    
    # 1. KNN classification accuracy (how well separated are the groups?)
    knn = KNeighborsClassifier(n_neighbors=5)
    
    # Use 80% for "train", 20% for "test"
    n = len(labels)
    idx = np.random.permutation(n)
    train_idx, test_idx = idx[:int(0.8*n)], idx[int(0.8*n):]
    
    knn.fit(latents[train_idx], labels[train_idx])
    knn_accuracy = knn.score(latents[test_idx], labels[test_idx])
    
    # 2. Silhouette score (cluster quality)
    try:
        silhouette = silhouette_score(latents, labels)
    except:
        silhouette = 0.0
    
    # Combined score (weighted average)
    quality_score = 0.7 * knn_accuracy + 0.3 * (silhouette + 1) / 2  # Normalize silhouette to [0,1]
    
    return quality_score


def run_hyperparameter_search(args):
    """Run full hyperparameter search."""
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_log_dir = Path(args.log_dir) / f"hp_search_{timestamp}"
    base_checkpoint_dir = Path(args.checkpoint_dir) / f"hp_search_{timestamp}"
    base_log_dir.mkdir(parents=True, exist_ok=True)
    base_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("HYPERPARAMETER SEARCH FOR CRYSTALLOGRAPHIC VAE")
    print("=" * 70)
    print(f"\nOutput directories:")
    print(f"  TensorBoard: {base_log_dir}")
    print(f"  Checkpoints: {base_checkpoint_dir}")
    
    # Create data loaders (shared across experiments)
    print(f"\nCreating dataset...")
    train_loader, val_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_samples_per_group=args.samples_per_group,
        resolution=args.resolution,
        train_split=0.85,
        seed=args.seed
    )
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    
    # Define hyperparameter grid
    param_grid = {
        'latent_dim': [64, 128, 256],
        'beta': [0.1, 0.5, 1.0, 4.0],  # Key parameter for VAE
        'lr': [1e-3, 5e-4],
        'base_channels': [32, 48],
    }
    
    # Fixed parameters
    fixed_params = {
        'epochs': args.epochs,
        'resolution': args.resolution,
        'gamma': 0.1,
    }
    
    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))
    
    print(f"\nHyperparameter grid:")
    for k, v in param_grid.items():
        print(f"  {k}: {v}")
    print(f"\nTotal experiments: {len(combinations)}")
    
    # Limit if requested
    if args.max_experiments and args.max_experiments < len(combinations):
        # Prioritize: vary beta first, then latent_dim
        np.random.seed(args.seed)
        idx = np.random.choice(len(combinations), args.max_experiments, replace=False)
        combinations = [combinations[i] for i in sorted(idx)]
        print(f"Limited to {args.max_experiments} experiments")
    
    # Run experiments
    all_results = []
    
    for i, combo in enumerate(combinations):
        config = dict(zip(keys, combo))
        config.update(fixed_params)
        
        print(f"\n[Experiment {i+1}/{len(combinations)}]")
        
        try:
            results = run_single_experiment(
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                base_log_dir=base_log_dir,
                base_checkpoint_dir=base_checkpoint_dir,
                device=args.device
            )
            all_results.append(results)
            
            # Save intermediate results
            save_results(all_results, base_log_dir / "results.json")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Analyze and report results
    print("\n" + "=" * 70)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 70)
    
    analyze_results(all_results, base_log_dir)
    
    print(f"\nTo compare experiments in TensorBoard:")
    print(f"  tensorboard --logdir={base_log_dir}")
    
    return all_results


def save_results(results: list, path: Path):
    """Save results to JSON."""
    # Convert to serializable format
    serializable = []
    for r in results:
        sr = {
            'config': r['config'],
            'exp_name': r['exp_name'],
            'final_train_loss': float(r['final_train_loss']),
            'final_val_loss': float(r['final_val_loss']),
            'final_recon_loss': float(r['final_recon_loss']),
            'final_kl_loss': float(r['final_kl_loss']),
            'best_val_loss': float(r['best_val_loss']),
            'latent_quality': float(r['latent_quality']),
        }
        serializable.append(sr)
    
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)


def analyze_results(results: list, output_dir: Path):
    """Analyze and report results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    if not results:
        print("No results to analyze")
        return
    
    # Sort by different criteria
    by_val_loss = sorted(results, key=lambda x: x['best_val_loss'])
    by_recon = sorted(results, key=lambda x: x['final_recon_loss'])
    by_latent = sorted(results, key=lambda x: -x['latent_quality'])
    
    # Combined score (balance reconstruction and latent quality)
    for r in results:
        # Normalize losses (lower is better -> invert)
        max_val = max(x['best_val_loss'] for x in results)
        max_recon = max(x['final_recon_loss'] for x in results)
        
        norm_val = 1 - (r['best_val_loss'] / max_val)
        norm_recon = 1 - (r['final_recon_loss'] / max_recon)
        
        # Combined: 50% reconstruction, 30% latent quality, 20% val loss
        r['combined_score'] = 0.5 * norm_recon + 0.3 * r['latent_quality'] + 0.2 * norm_val
    
    by_combined = sorted(results, key=lambda x: -x['combined_score'])
    
    # Print rankings
    print("\n📊 TOP 5 BY VALIDATION LOSS:")
    print("-" * 70)
    for i, r in enumerate(by_val_loss[:5]):
        print(f"  {i+1}. {r['exp_name']}: val_loss={r['best_val_loss']:.4f}")
    
    print("\n📊 TOP 5 BY RECONSTRUCTION:")
    print("-" * 70)
    for i, r in enumerate(by_recon[:5]):
        print(f"  {i+1}. {r['exp_name']}: recon_loss={r['final_recon_loss']:.4f}")
    
    print("\n📊 TOP 5 BY LATENT QUALITY:")
    print("-" * 70)
    for i, r in enumerate(by_latent[:5]):
        print(f"  {i+1}. {r['exp_name']}: latent_quality={r['latent_quality']:.4f}")
    
    print("\n🏆 TOP 5 OVERALL (COMBINED SCORE):")
    print("-" * 70)
    for i, r in enumerate(by_combined[:5]):
        print(f"  {i+1}. {r['exp_name']}")
        print(f"      recon={r['final_recon_loss']:.4f}, "
              f"latent_q={r['latent_quality']:.4f}, "
              f"combined={r['combined_score']:.4f}")
    
    # Best overall
    best = by_combined[0]
    print(f"\n🥇 BEST MODEL: {best['exp_name']}")
    print(f"   Configuration:")
    for k, v in best['config'].items():
        print(f"     {k}: {v}")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#1a1a2e')
    
    for ax in axes.flat:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#eaeaea')
        for spine in ax.spines.values():
            spine.set_color('#3d3d5c')
    
    # Extract data for plotting
    names = [r['exp_name'] for r in results]
    betas = [r['config']['beta'] for r in results]
    latent_dims = [r['config']['latent_dim'] for r in results]
    recon_losses = [r['final_recon_loss'] for r in results]
    kl_losses = [r['final_kl_loss'] for r in results]
    latent_qualities = [r['latent_quality'] for r in results]
    combined_scores = [r['combined_score'] for r in results]
    
    # 1. Reconstruction vs KL by beta
    colors = plt.cm.viridis(np.array(betas) / max(betas))
    axes[0, 0].scatter(recon_losses, kl_losses, c=betas, cmap='viridis', s=100, alpha=0.7)
    axes[0, 0].set_xlabel('Reconstruction Loss', color='#eaeaea')
    axes[0, 0].set_ylabel('KL Loss', color='#eaeaea')
    axes[0, 0].set_title('Recon vs KL (color=β)', color='#eaeaea')
    
    # 2. Latent quality vs beta
    for ld in set(latent_dims):
        mask = [l == ld for l in latent_dims]
        b = [betas[i] for i in range(len(betas)) if mask[i]]
        q = [latent_qualities[i] for i in range(len(latent_qualities)) if mask[i]]
        axes[0, 1].scatter(b, q, label=f'dim={ld}', s=80, alpha=0.7)
    axes[0, 1].set_xlabel('Beta (β)', color='#eaeaea')
    axes[0, 1].set_ylabel('Latent Quality', color='#eaeaea')
    axes[0, 1].set_title('Latent Quality vs β', color='#eaeaea')
    axes[0, 1].legend(facecolor='#2a2a4e', edgecolor='#eaeaea', labelcolor='#eaeaea')
    
    # 3. Reconstruction vs Latent Quality
    axes[1, 0].scatter(recon_losses, latent_qualities, c=betas, cmap='plasma', s=100, alpha=0.7)
    axes[1, 0].set_xlabel('Reconstruction Loss', color='#eaeaea')
    axes[1, 0].set_ylabel('Latent Quality', color='#eaeaea')
    axes[1, 0].set_title('Recon vs Latent Quality (color=β)', color='#eaeaea')
    
    # 4. Combined scores
    sorted_idx = np.argsort(combined_scores)[::-1][:10]  # Top 10
    top_names = [names[i].replace('_', '\n') for i in sorted_idx]
    top_scores = [combined_scores[i] for i in sorted_idx]
    bars = axes[1, 1].barh(range(len(top_names)), top_scores, color='#4ECDC4')
    axes[1, 1].set_yticks(range(len(top_names)))
    axes[1, 1].set_yticklabels(top_names, fontsize=8)
    axes[1, 1].set_xlabel('Combined Score', color='#eaeaea')
    axes[1, 1].set_title('Top 10 Models', color='#eaeaea')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'experiment_analysis.png', dpi=150, 
                facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    
    print(f"\n📈 Analysis plot saved to: {output_dir / 'experiment_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for Crystallographic VAE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--samples-per-group', '-n', type=int, default=100,
                       help='Samples per wallpaper group')
    parser.add_argument('--resolution', '-r', type=int, default=128,
                       help='Image resolution')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Batch size')
    
    # Experiment arguments
    parser.add_argument('--epochs', '-e', type=int, default=30,
                       help='Epochs per experiment')
    parser.add_argument('--max-experiments', type=int, default=None,
                       help='Limit number of experiments (None = all)')
    
    # Directory arguments
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='./runs',
                       help='TensorBoard log directory')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'])
    
    args = parser.parse_args()
    
    run_hyperparameter_search(args)


if __name__ == "__main__":
    main()








