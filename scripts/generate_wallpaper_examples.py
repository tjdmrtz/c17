#!/usr/bin/env python3
"""
Generate example images for all 17 wallpaper groups.
Creates high-quality visualizations for documentation.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc, Circle, Polygon
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from src.dataset.pattern_generator_fixed import FixedWallpaperGenerator


def add_symmetry_annotations(ax, group_name, size):
    """Add symmetry operation annotations to the plot."""
    center = size / 2
    
    # Colors for different symmetry elements
    rotation_colors = {2: '#FF6B6B', 3: '#4ECDC4', 4: '#45B7D1', 6: '#96CEB4'}
    mirror_color = '#FFE66D'
    glide_color = '#DDA0DD'
    
    annotations = {
        'p1': [],  # No point symmetry
        'p2': [('rot2', center, center)],
        'pm': [('mirror_v', center, 0, center, size)],
        'pg': [('glide_v', center, 0, center, size)],
        'cm': [('mirror_v', center, 0, center, size), ('glide_v', center*1.5, 0, center*1.5, size)],
        'pmm': [('mirror_v', center, 0, center, size), ('mirror_h', 0, center, size, center), ('rot2', center, center)],
        'pmg': [('mirror_v', center, 0, center, size), ('glide_h', 0, center, size, center), ('rot2', center, center)],
        'pgg': [('glide_v', center, 0, center, size), ('glide_h', 0, center, size, center), ('rot2', center, center)],
        'cmm': [('mirror_v', center, 0, center, size), ('mirror_h', 0, center, size, center), ('rot2', center, center)],
        'p4': [('rot4', center, center)],
        'p4m': [('rot4', center, center), ('mirror_v', center, 0, center, size), ('mirror_d', 0, 0, size, size)],
        'p4g': [('rot4', center, center), ('glide_d', 0, 0, size, size)],
        'p3': [('rot3', center, center)],
        'p3m1': [('rot3', center, center), ('mirror_v', center, 0, center, size)],
        'p31m': [('rot3', center, center), ('mirror_h', 0, center, size, center)],
        'p6': [('rot6', center, center)],
        'p6m': [('rot6', center, center), ('mirror_v', center, 0, center, size), ('mirror_h', 0, center, size, center)],
    }
    
    for ann in annotations.get(group_name, []):
        if ann[0].startswith('rot'):
            order = int(ann[0][3])
            x, y = ann[1], ann[2]
            color = rotation_colors[order]
            # Draw rotation symbol
            if order == 2:
                ax.plot(x, y, 'o', markersize=8, color=color, markeredgecolor='white', markeredgewidth=1.5)
            elif order == 3:
                triangle = plt.Polygon([(x, y+6), (x-5, y-3), (x+5, y-3)], 
                                       fill=True, color=color, edgecolor='white', linewidth=1.5)
                ax.add_patch(triangle)
            elif order == 4:
                square = plt.Rectangle((x-4, y-4), 8, 8, fill=True, color=color, 
                                       edgecolor='white', linewidth=1.5)
                ax.add_patch(square)
            elif order == 6:
                hexagon = plt.Polygon([(x+5*np.cos(a), y+5*np.sin(a)) for a in np.linspace(0, 2*np.pi, 7)[:-1]],
                                     fill=True, color=color, edgecolor='white', linewidth=1.5)
                ax.add_patch(hexagon)
        elif 'mirror' in ann[0]:
            x1, y1, x2, y2 = ann[1], ann[2], ann[3], ann[4]
            ax.plot([x1, x2], [y1, y2], color=mirror_color, linewidth=2, linestyle='-')
        elif 'glide' in ann[0]:
            x1, y1, x2, y2 = ann[1], ann[2], ann[3], ann[4]
            ax.plot([x1, x2], [y1, y2], color=glide_color, linewidth=2, linestyle='--')


def generate_all_examples(output_dir: Path, resolution: int = 256, seed: int = 42):
    """Generate example images for all 17 wallpaper groups."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = FixedWallpaperGenerator(resolution=resolution, seed=seed)
    
    # Group information
    group_info = {
        'p1': {'name': 'p1', 'lattice': 'Oblique', 'rotation': 1, 'reflection': False, 'glide': False,
               'description': 'Solo traslaciones, sin simetría puntual'},
        'p2': {'name': 'p2', 'lattice': 'Oblique', 'rotation': 2, 'reflection': False, 'glide': False,
               'description': 'Centros de rotación de 180°'},
        'pm': {'name': 'pm', 'lattice': 'Rectangular', 'rotation': 1, 'reflection': True, 'glide': False,
               'description': 'Ejes de reflexión paralelos'},
        'pg': {'name': 'pg', 'lattice': 'Rectangular', 'rotation': 1, 'reflection': False, 'glide': True,
               'description': 'Reflexiones con deslizamiento paralelas'},
        'cm': {'name': 'cm', 'lattice': 'Rectangular', 'rotation': 1, 'reflection': True, 'glide': True,
               'description': 'Reflexión con celda centrada'},
        'pmm': {'name': 'pmm', 'lattice': 'Rectangular', 'rotation': 2, 'reflection': True, 'glide': False,
                'description': 'Ejes de reflexión perpendiculares'},
        'pmg': {'name': 'pmg', 'lattice': 'Rectangular', 'rotation': 2, 'reflection': True, 'glide': True,
                'description': 'Reflexión + deslizamiento perpendicular'},
        'pgg': {'name': 'pgg', 'lattice': 'Rectangular', 'rotation': 2, 'reflection': False, 'glide': True,
                'description': 'Deslizamientos perpendiculares'},
        'cmm': {'name': 'cmm', 'lattice': 'Rectangular', 'rotation': 2, 'reflection': True, 'glide': True,
                'description': 'Celda centrada con reflexiones'},
        'p4': {'name': 'p4', 'lattice': 'Square', 'rotation': 4, 'reflection': False, 'glide': False,
               'description': 'Centros de rotación de 90°'},
        'p4m': {'name': 'p4m', 'lattice': 'Square', 'rotation': 4, 'reflection': True, 'glide': True,
                'description': 'Cuadrado con reflexiones en todos los ejes'},
        'p4g': {'name': 'p4g', 'lattice': 'Square', 'rotation': 4, 'reflection': True, 'glide': True,
                'description': 'Cuadrado con deslizamientos y rotaciones'},
        'p3': {'name': 'p3', 'lattice': 'Hexagonal', 'rotation': 3, 'reflection': False, 'glide': False,
               'description': 'Centros de rotación de 120°'},
        'p3m1': {'name': 'p3m1', 'lattice': 'Hexagonal', 'rotation': 3, 'reflection': True, 'glide': False,
                 'description': 'Rotación 120° con reflexiones por los centros'},
        'p31m': {'name': 'p31m', 'lattice': 'Hexagonal', 'rotation': 3, 'reflection': True, 'glide': False,
                 'description': 'Rotación 120° con reflexiones entre centros'},
        'p6': {'name': 'p6', 'lattice': 'Hexagonal', 'rotation': 6, 'reflection': False, 'glide': False,
               'description': 'Centros de rotación de 60°'},
        'p6m': {'name': 'p6m', 'lattice': 'Hexagonal', 'rotation': 6, 'reflection': True, 'glide': True,
                'description': 'Hexagonal con todas las simetrías'},
    }
    
    groups = ['p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm', 
              'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m']
    
    print("="*60)
    print("GENERATING WALLPAPER GROUP EXAMPLES")
    print("="*60)
    
    # Generate individual images
    for group_name in groups:
        print(f"  Generating {group_name}...", end=" ")
        
        pattern = generator.generate(group_name, motif_size=64, complexity=4)
        
        # Create figure with pattern
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='#1a1a2e')
        ax.imshow(pattern, cmap='magma', vmin=0, vmax=1)
        
        info = group_info[group_name]
        title = f"{group_name} - {info['description']}\n"
        title += f"Lattice: {info['lattice']} | Rotation: C{info['rotation']}"
        if info['reflection']:
            title += " | σ"
        if info['glide']:
            title += " | g"
        
        ax.set_title(title, fontsize=12, color='white', pad=10)
        ax.axis('off')
        
        plt.tight_layout()
        fig.savefig(output_dir / f"{group_name}.png", dpi=150, 
                   facecolor='#1a1a2e', bbox_inches='tight')
        plt.close(fig)
        print("✓")
    
    # Generate overview grid
    print("\n  Generating overview grid...", end=" ")
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16), facecolor='#1a1a2e')
    axes = axes.flatten()
    
    for idx, group_name in enumerate(groups):
        pattern = generator.generate(group_name, motif_size=64, complexity=4)
        
        ax = axes[idx]
        ax.imshow(pattern, cmap='magma', vmin=0, vmax=1)
        
        info = group_info[group_name]
        ax.set_title(f"{group_name}\n{info['lattice']}, C{info['rotation']}", 
                    fontsize=10, color='white')
        ax.axis('off')
    
    # Hide unused axes
    for idx in range(len(groups), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Los 17 Grupos de Papel Tapiz (Wallpaper Groups)', 
                fontsize=16, color='white', y=1.02)
    
    plt.tight_layout()
    fig.savefig(output_dir / "all_17_groups_overview.png", dpi=200,
               facecolor='#1a1a2e', bbox_inches='tight')
    plt.close(fig)
    print("✓")
    
    # Generate lattice type comparison
    print("  Generating lattice comparison...", end=" ")
    
    lattice_groups = {
        'Oblique': ['p1', 'p2'],
        'Rectangular': ['pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm'],
        'Square': ['p4', 'p4m', 'p4g'],
        'Hexagonal': ['p3', 'p3m1', 'p31m', 'p6', 'p6m'],
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), facecolor='#1a1a2e')
    
    for ax_idx, (lattice_name, lattice_grps) in enumerate(lattice_groups.items()):
        ax = axes.flatten()[ax_idx]
        
        # Create composite for this lattice type
        n = len(lattice_grps)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        
        composite = np.zeros((rows * resolution, cols * resolution))
        
        for i, grp in enumerate(lattice_grps):
            pattern = generator.generate(grp, motif_size=64, complexity=4)
            r, c = i // cols, i % cols
            composite[r*resolution:(r+1)*resolution, c*resolution:(c+1)*resolution] = pattern
        
        ax.imshow(composite, cmap='magma', vmin=0, vmax=1)
        ax.set_title(f'{lattice_name} Lattice\n{", ".join(lattice_grps)}', 
                    fontsize=12, color='white')
        ax.axis('off')
    
    plt.tight_layout()
    fig.savefig(output_dir / "lattice_comparison.png", dpi=150,
               facecolor='#1a1a2e', bbox_inches='tight')
    plt.close(fig)
    print("✓")
    
    print(f"\n✅ All images saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    output_dir = Path("docs/images/wallpaper_groups")
    generate_all_examples(output_dir)

