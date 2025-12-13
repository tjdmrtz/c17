#!/usr/bin/env python3
"""
Generate beautiful colored visualizations of the 17 wallpaper groups.

Each group has a unique color palette inspired by nature and art.

Usage:
    python scripts/generate_colored_patterns.py
    python scripts/generate_colored_patterns.py --resolution 512 --output-dir ./my_patterns
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.dataset.color_patterns import (
    ColorPatternGenerator, 
    GROUP_PALETTES,
    visualize_all_colored_groups
)
from src.dataset.pattern_generator import WALLPAPER_GROUPS


def generate_group_showcase(generator: ColorPatternGenerator,
                            group_name: str,
                            output_dir: Path,
                            num_variations: int = 4):
    """Generate a showcase of variations for a single group."""
    
    palette = GROUP_PALETTES[group_name]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), facecolor='#0a0a0f')
    fig.suptitle(f"{group_name} - {palette['name']}", 
                fontsize=20, color='#f0f0f0', fontweight='bold', y=0.98)
    
    for i, ax in enumerate(axes.flat):
        # Vary parameters
        seed = 42 + i * 100
        gen = ColorPatternGenerator(resolution=256, seed=seed)
        
        motif_sizes = [48, 64, 80, 96]
        pattern = gen.generate(
            group_name, 
            motif_size=motif_sizes[i],
            complexity=(i % 4) + 3,
            motif_type=['gaussian', 'geometric', 'mixed', 'gaussian'][i]
        )
        
        ax.imshow(pattern, interpolation='bilinear')
        ax.set_title(f"Variation {i+1}", color='#aaaaaa', fontsize=11)
        
        for spine in ax.spines.values():
            spine.set_edgecolor(palette['accent'])
            spine.set_linewidth(2)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = output_dir / f"showcase_{group_name}.png"
    fig.savefig(save_path, dpi=150, facecolor='#0a0a0f', bbox_inches='tight')
    plt.close(fig)
    
    return save_path


def generate_comparison_by_lattice(generator: ColorPatternGenerator,
                                   output_dir: Path):
    """Generate comparison grouped by lattice type."""
    
    lattice_groups = {
        'Oblique': ['p1', 'p2'],
        'Rectangular': ['pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm'],
        'Square': ['p4', 'p4m', 'p4g'],
        'Hexagonal': ['p3', 'p3m1', 'p31m', 'p6', 'p6m'],
    }
    
    for lattice_name, groups in lattice_groups.items():
        n = len(groups)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), 
                                 facecolor='#0a0a0f')
        fig.suptitle(f'{lattice_name} Lattice', fontsize=18, 
                    color='#f0f0f0', fontweight='bold', y=1.02)
        
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        if rows > 1:
            axes = axes.flatten()
        
        for i, group_name in enumerate(groups):
            ax = axes[i] if isinstance(axes, (list, np.ndarray)) else axes
            
            pattern = generator.generate(group_name, motif_size=64)
            ax.imshow(pattern, interpolation='bilinear')
            
            palette = GROUP_PALETTES[group_name]
            ax.set_title(f"{group_name}\n{palette['name']}", 
                        fontsize=10, color='#f0f0f0', fontweight='bold')
            
            for spine in ax.spines.values():
                spine.set_edgecolor(palette['accent'])
                spine.set_linewidth(2)
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide unused axes
        for i in range(n, rows * cols):
            if isinstance(axes, (list, np.ndarray)) and i < len(axes):
                axes[i].axis('off')
        
        plt.tight_layout()
        
        save_path = output_dir / f"lattice_{lattice_name.lower()}.png"
        fig.savefig(save_path, dpi=150, facecolor='#0a0a0f', bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {save_path.name}")


def generate_palette_reference(output_dir: Path):
    """Generate a color palette reference card."""
    
    fig, axes = plt.subplots(4, 5, figsize=(18, 14), facecolor='#0a0a0f')
    axes = axes.flatten()
    
    fig.suptitle('Color Palettes Reference', fontsize=24, 
                color='#f0f0f0', fontweight='bold', y=0.98)
    
    groups = list(GROUP_PALETTES.keys())[:17]
    
    for i, group_name in enumerate(groups):
        ax = axes[i]
        palette = GROUP_PALETTES[group_name]
        
        # Create color swatches
        colors = [palette['bg'], palette['primary'], 
                  palette['secondary'], palette['accent']]
        labels = ['BG', 'Primary', 'Secondary', 'Accent']
        
        for j, (color, label) in enumerate(zip(colors, labels)):
            rect = plt.Rectangle((j * 0.25, 0.2), 0.23, 0.6, 
                                 facecolor=color, transform=ax.transAxes)
            ax.add_patch(rect)
            ax.text(j * 0.25 + 0.115, 0.1, label, 
                   ha='center', fontsize=7, color='#888888',
                   transform=ax.transAxes)
        
        ax.set_title(f"{group_name}: {palette['name']}", 
                    fontsize=11, color='#f0f0f0', fontweight='bold', pad=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # Hide unused axes
    for i in range(17, 20):
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = output_dir / "palette_reference.png"
    fig.savefig(save_path, dpi=150, facecolor='#0a0a0f', bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {save_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate colored crystallographic patterns"
    )
    
    parser.add_argument('--output-dir', '-o', type=str, 
                       default='./output/colored_patterns',
                       help='Output directory')
    parser.add_argument('--resolution', '-r', type=int, default=256,
                       help='Pattern resolution')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GENERATING COLORED CRYSTALLOGRAPHIC PATTERNS")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Resolution: {args.resolution}")
    print()
    
    # Create generator
    generator = ColorPatternGenerator(resolution=args.resolution, seed=args.seed)
    
    # 1. Main visualization with all 17 groups
    print("📊 Generating main visualization...")
    visualize_all_colored_groups(
        resolution=args.resolution,
        save_path=str(output_dir / "all_17_groups_colored.png")
    )
    
    # 2. Individual high-res patterns
    print("\n🎨 Generating individual patterns...")
    for group_name in WALLPAPER_GROUPS.keys():
        gen_hires = ColorPatternGenerator(resolution=512, seed=args.seed)
        pattern = gen_hires.generate(group_name, motif_size=128)
        
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0a0a0f')
        ax.imshow(pattern, interpolation='bilinear')
        ax.axis('off')
        
        palette = GROUP_PALETTES[group_name]
        ax.set_title(f"{group_name} - {palette['name']}", 
                    fontsize=18, color='#f0f0f0', pad=15, fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(output_dir / f"pattern_{group_name}.png",
                   dpi=150, facecolor='#0a0a0f', bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {group_name}")
    
    # 3. Showcases with variations
    print("\n🖼️ Generating showcases...")
    showcase_dir = output_dir / "showcases"
    showcase_dir.mkdir(exist_ok=True)
    
    for group_name in ['p4m', 'p6m', 'cmm', 'pm']:
        generate_group_showcase(generator, group_name, showcase_dir)
        print(f"  ✓ {group_name}")
    
    # 4. Comparison by lattice type
    print("\n📐 Generating lattice comparisons...")
    generate_comparison_by_lattice(generator, output_dir)
    
    # 5. Palette reference
    print("\n🎨 Generating palette reference...")
    generate_palette_reference(output_dir)
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE!")
    print("=" * 60)
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"\nFiles generated:")
    print(f"  • all_17_groups_colored.png - Main overview")
    print(f"  • pattern_{{group}}.png - Individual patterns (17 files)")
    print(f"  • showcases/ - Variation showcases")
    print(f"  • lattice_*.png - By lattice type")
    print(f"  • palette_reference.png - Color palettes")


if __name__ == "__main__":
    main()







