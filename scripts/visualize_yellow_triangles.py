#!/usr/bin/env python3
"""
Script to visualize crystallographic patterns with yellow triangles.

Generates visualizations with triangular motifs in yellow colormap.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')  # For non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from src.dataset.pattern_generator import WallpaperGroupGenerator, WALLPAPER_GROUPS


def create_yellow_cmap():
    """Create a custom yellow colormap (black to bright yellow)."""
    colors = ['#000000', '#2d2800', '#5c5000', '#8a7800', '#b8a000', '#FFD700', '#FFFF00']
    return mcolors.LinearSegmentedColormap.from_list('yellow', colors)


class TriangleGenerator(WallpaperGroupGenerator):
    """Generator that creates only triangle-based motifs."""
    
    def _create_motif(self, size: int, complexity: int = 3, **kwargs) -> np.ndarray:
        """Create a motif using only triangles."""
        motif = np.zeros((size, size))
        
        for _ in range(complexity):
            # Create random triangle vertices
            cx = self.rng.random() * size
            cy = self.rng.random() * size
            radius = self.rng.random() * size / 3 + size / 8
            
            # Create triangle points
            angles = self.rng.random() * 2 * np.pi  # Random rotation
            p1 = (cx + radius * np.cos(angles), cy + radius * np.sin(angles))
            p2 = (cx + radius * np.cos(angles + 2*np.pi/3), cy + radius * np.sin(angles + 2*np.pi/3))
            p3 = (cx + radius * np.cos(angles + 4*np.pi/3), cy + radius * np.sin(angles + 4*np.pi/3))
            
            # Fill triangle using barycentric coordinates
            y, x = np.ogrid[:size, :size]
            
            # Calculate barycentric coordinates
            def sign(p1, p2, p3):
                return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
            
            def point_in_triangle(px, py, v1, v2, v3):
                d1 = sign((px, py), v1, v2)
                d2 = sign((px, py), v2, v3)
                d3 = sign((px, py), v3, v1)
                
                has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
                has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
                
                return ~(has_neg & has_pos)
            
            mask = point_in_triangle(x, y, p1, p2, p3)
            amplitude = self.rng.random() * 0.5 + 0.5
            motif[mask] += amplitude
        
        # Normalize
        if motif.max() > 0:
            motif = motif / motif.max()
            
        return motif


def plot_single_pattern_yellow(pattern, group_name, cmap, save_path=None):
    """Plot a single pattern with yellow colormap."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor='black')
    
    ax.imshow(pattern, cmap=cmap, interpolation='bilinear')
    ax.set_title(f'Grupo {group_name.upper()}', fontsize=20, color='yellow', fontweight='bold', pad=10)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black', edgecolor='none')
    
    return fig


def plot_all_groups_yellow(generator, cmap, resolution=256, save_path=None):
    """Plot all 17 wallpaper groups with yellow triangles."""
    fig, axes = plt.subplots(4, 5, figsize=(20, 16), facecolor='black')
    axes = axes.flatten()
    
    group_names = list(WALLPAPER_GROUPS.keys())
    
    for idx, group_name in enumerate(group_names):
        pattern = generator.generate(group_name, motif_size=resolution // 4)
        
        axes[idx].imshow(pattern, cmap=cmap, interpolation='bilinear')
        axes[idx].set_title(group_name.upper(), fontsize=14, color='yellow', fontweight='bold')
        axes[idx].axis('off')
    
    # Hide unused axes
    for idx in range(len(group_names), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Los 17 Grupos Cristalográficos - Triángulos Amarillos', 
                 fontsize=24, color='#FFD700', fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black', edgecolor='none')
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize crystallographic patterns with yellow triangles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./output/yellow_triangles",
        help="Output directory for visualizations"
    )
    
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        default=256,
        help="Pattern resolution"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--groups",
        nargs='+',
        default=None,
        help="Specific groups to visualize (default: all)"
    )
    
    parser.add_argument(
        "--complexity",
        type=int,
        default=4,
        help="Number of triangles in each motif"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("VISUALIZACIÓN - TRIÁNGULOS AMARILLOS")
    print("=" * 60)
    print(f"Directorio de salida: {output_path}")
    print(f"Resolución: {args.resolution}x{args.resolution}")
    print(f"Complejidad: {args.complexity} triángulos por motif")
    print("-" * 60)
    
    # Create generator and colormap
    generator = TriangleGenerator(resolution=args.resolution, seed=args.seed)
    cmap = create_yellow_cmap()
    
    # Generate all 17 groups overview
    print("\nGenerando vista de los 17 grupos...")
    fig = plot_all_groups_yellow(
        generator, cmap, 
        resolution=args.resolution,
        save_path=str(output_path / "all_17_groups_yellow.png")
    )
    plt.close(fig)
    print("  ✓ Guardado: all_17_groups_yellow.png")
    
    # Individual patterns
    groups_to_generate = args.groups if args.groups else list(WALLPAPER_GROUPS.keys())
    
    print(f"\nGenerando patrones individuales...")
    for group_name in groups_to_generate:
        if group_name not in WALLPAPER_GROUPS:
            print(f"  ⚠ Grupo desconocido: {group_name}, saltando")
            continue
        
        pattern = generator.generate(group_name, motif_size=args.resolution // 4, complexity=args.complexity)
        fig = plot_single_pattern_yellow(
            pattern, group_name, cmap,
            save_path=str(output_path / f"pattern_{group_name}_yellow.png")
        )
        plt.close(fig)
        print(f"  ✓ Guardado: pattern_{group_name}_yellow.png")
    
    print("\n" + "=" * 60)
    print("¡VISUALIZACIÓN COMPLETA!")
    print("=" * 60)
    print(f"Todas las imágenes guardadas en: {output_path}")


if __name__ == "__main__":
    main()

