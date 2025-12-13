"""
Visualization utilities for crystallographic patterns.

Provides functions to visualize:
- Individual patterns
- All 17 wallpaper groups side by side
- Symmetry annotations
- Latent space representations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Circle, Arrow
import matplotlib.colors as mcolors
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dataset.pattern_generator import WallpaperGroupGenerator, WALLPAPER_GROUPS, LatticeType


# Color scheme for lattice types
LATTICE_COLORS = {
    LatticeType.OBLIQUE: "#FF6B6B",      # Coral red
    LatticeType.RECTANGULAR: "#4ECDC4",   # Teal
    LatticeType.SQUARE: "#95E1D3",        # Mint
    LatticeType.HEXAGONAL: "#F38181",     # Salmon
}

# Beautiful colormap for patterns
PATTERN_CMAP = 'magma'


class PatternVisualizer:
    """Visualizer for crystallographic patterns."""
    
    def __init__(self, 
                 figsize: Tuple[int, int] = (12, 12),
                 dpi: int = 150,
                 style: str = 'dark'):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size
            dpi: Resolution for saved figures
            style: 'dark' or 'light' theme
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        
        if style == 'dark':
            plt.style.use('dark_background')
            self.bg_color = '#1a1a2e'
            self.text_color = '#eaeaea'
            self.accent_color = '#e94560'
        else:
            plt.style.use('default')
            self.bg_color = '#ffffff'
            self.text_color = '#2d3436'
            self.accent_color = '#6c5ce7'
    
    def plot_single_pattern(self,
                           pattern: np.ndarray,
                           group_name: str,
                           title: Optional[str] = None,
                           show_info: bool = True,
                           save_path: Optional[str] = None,
                           cmap: str = PATTERN_CMAP) -> plt.Figure:
        """
        Plot a single crystallographic pattern.
        
        Args:
            pattern: 2D numpy array with the pattern
            group_name: Name of the wallpaper group
            title: Optional custom title
            show_info: Whether to show group information
            save_path: Path to save the figure
            cmap: Colormap to use
        """
        fig, ax = plt.subplots(figsize=(8, 8), facecolor=self.bg_color)
        ax.set_facecolor(self.bg_color)
        
        # Plot pattern
        im = ax.imshow(pattern, cmap=cmap, interpolation='bilinear')
        
        # Title
        if title is None:
            group_info = WALLPAPER_GROUPS[group_name]
            title = f"Wallpaper Group: {group_name}"
        
        ax.set_title(title, fontsize=16, color=self.text_color, 
                    fontweight='bold', pad=20)
        
        # Add info box
        if show_info:
            group_info = WALLPAPER_GROUPS[group_name]
            info_text = (
                f"Lattice: {group_info.lattice_type.value}\n"
                f"Rotation: {group_info.rotation_order}-fold\n"
                f"Reflection: {'Yes' if group_info.has_reflection else 'No'}\n"
                f"Glide: {'Yes' if group_info.has_glide else 'No'}"
            )
            
            props = dict(boxstyle='round,pad=0.5', 
                        facecolor=LATTICE_COLORS[group_info.lattice_type],
                        alpha=0.8)
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=props, color='#1a1a2e', fontweight='bold')
        
        ax.axis('off')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=self.text_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.text_color)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor=self.bg_color)
            print(f"Saved to {save_path}")
        
        return fig
    
    def plot_symmetry_annotations(self,
                                  pattern: np.ndarray,
                                  group_name: str,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot pattern with symmetry element annotations.
        
        Shows rotation centers, reflection axes, and glide lines.
        """
        fig, ax = plt.subplots(figsize=(10, 10), facecolor=self.bg_color)
        ax.set_facecolor(self.bg_color)
        
        # Plot pattern
        ax.imshow(pattern, cmap='gray', alpha=0.6, interpolation='bilinear')
        
        group_info = WALLPAPER_GROUPS[group_name]
        h, w = pattern.shape
        
        # Draw rotation centers
        if group_info.rotation_order > 1:
            n_centers = group_info.rotation_order
            colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3']
            
            # Place rotation centers at unit cell corners and center
            centers = [(w//4, h//4), (3*w//4, h//4), (w//4, 3*h//4), (3*w//4, 3*h//4)]
            if group_info.rotation_order >= 3:
                centers.append((w//2, h//2))
            
            for i, (cx, cy) in enumerate(centers[:4]):
                circle = Circle((cx, cy), w//20, 
                               color=colors[i % len(colors)], 
                               alpha=0.7, linewidth=2)
                ax.add_patch(circle)
                ax.text(cx, cy, f'{group_info.rotation_order}', 
                       ha='center', va='center', 
                       fontsize=10, fontweight='bold', color='white')
        
        # Draw reflection axes
        if group_info.has_reflection:
            # Vertical axes
            for x in [w//4, w//2, 3*w//4]:
                ax.axvline(x, color='#FF6B6B', linestyle='--', 
                          linewidth=2, alpha=0.8)
            
            # Horizontal axes for pmm, cmm, p4m, p6m
            if group_name in ['pmm', 'cmm', 'p4m', 'p4g', 'p6m']:
                for y in [h//4, h//2, 3*h//4]:
                    ax.axhline(y, color='#4ECDC4', linestyle='--', 
                              linewidth=2, alpha=0.8)
        
        # Draw glide lines
        if group_info.has_glide:
            # Diagonal glide representation
            ax.plot([0, w], [0, h], color='#FFE66D', 
                   linestyle=':', linewidth=3, alpha=0.8)
            ax.plot([0, w], [h, 0], color='#FFE66D', 
                   linestyle=':', linewidth=3, alpha=0.8)
        
        # Title with group info
        title = (f"Symmetry of {group_name}\n"
                f"{group_info.rotation_order}-fold rotation | "
                f"{'Mirror' if group_info.has_reflection else 'No mirror'} | "
                f"{'Glide' if group_info.has_glide else 'No glide'}")
        ax.set_title(title, fontsize=14, color=self.text_color, pad=20)
        
        # Legend
        legend_elements = []
        if group_info.rotation_order > 1:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor='#FF6B6B', 
                                             markersize=15,
                                             label=f'{group_info.rotation_order}-fold rotation'))
        if group_info.has_reflection:
            legend_elements.append(plt.Line2D([0], [0], color='#FF6B6B', 
                                             linestyle='--', linewidth=2,
                                             label='Reflection axis'))
        if group_info.has_glide:
            legend_elements.append(plt.Line2D([0], [0], color='#FFE66D',
                                             linestyle=':', linewidth=2,
                                             label='Glide line'))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='lower right',
                     facecolor=self.bg_color, edgecolor=self.text_color,
                     labelcolor=self.text_color)
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor=self.bg_color)
        
        return fig


def plot_all_groups(resolution: int = 256,
                    motif_size: int = 64,
                    seed: int = 42,
                    save_path: Optional[str] = None,
                    style: str = 'dark') -> plt.Figure:
    """
    Generate and plot all 17 wallpaper groups in a grid.
    
    Args:
        resolution: Pattern resolution
        motif_size: Size of fundamental motif
        seed: Random seed
        save_path: Path to save the figure
        style: 'dark' or 'light' theme
    
    Returns:
        Matplotlib figure
    """
    if style == 'dark':
        plt.style.use('dark_background')
        bg_color = '#0f0f1a'
        text_color = '#eaeaea'
        border_color = '#3d3d5c'
    else:
        plt.style.use('default')
        bg_color = '#f5f5f5'
        text_color = '#2d3436'
        border_color = '#dfe6e9'
    
    # Create figure with 4x5 grid (17 patterns + 3 empty for balance)
    fig = plt.figure(figsize=(20, 18), facecolor=bg_color)
    
    # Title
    fig.suptitle('The 17 Wallpaper Groups\n(Plane Crystallographic Groups)', 
                fontsize=24, fontweight='bold', color=text_color, y=0.98)
    
    # Create grid
    gs = GridSpec(4, 5, figure=fig, hspace=0.35, wspace=0.15,
                  left=0.02, right=0.98, top=0.92, bottom=0.02)
    
    generator = WallpaperGroupGenerator(resolution=resolution, seed=seed)
    groups = list(WALLPAPER_GROUPS.keys())
    
    # Organize by lattice type for visual grouping
    lattice_order = [
        # Oblique
        'p1', 'p2',
        # Rectangular
        'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm',
        # Square
        'p4', 'p4m', 'p4g',
        # Hexagonal
        'p3', 'p3m1', 'p31m', 'p6', 'p6m'
    ]
    
    for idx, group_name in enumerate(lattice_order):
        row = idx // 5
        col = idx % 5
        
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(bg_color)
        
        # Generate pattern
        pattern = generator.generate(group_name, motif_size=motif_size,
                                    complexity=4, motif_type='mixed')
        
        # Get group info for coloring
        group_info = WALLPAPER_GROUPS[group_name]
        lattice_color = LATTICE_COLORS[group_info.lattice_type]
        
        # Plot with beautiful colormap
        im = ax.imshow(pattern, cmap='inferno', interpolation='bilinear')
        
        # Add border color based on lattice type
        for spine in ax.spines.values():
            spine.set_edgecolor(lattice_color)
            spine.set_linewidth(3)
        
        # Title with group name and properties
        symbols = []
        if group_info.rotation_order > 1:
            symbols.append(f"◊{group_info.rotation_order}")
        if group_info.has_reflection:
            symbols.append("⬌")
        if group_info.has_glide:
            symbols.append("⤡")
        
        symbol_str = " ".join(symbols) if symbols else "→"
        
        ax.set_title(f"{group_name}\n{symbol_str}", 
                    fontsize=12, fontweight='bold', 
                    color=text_color, pad=8)
        
        # Lattice type label
        ax.text(0.5, -0.08, group_info.lattice_type.value,
               transform=ax.transAxes, ha='center', fontsize=9,
               color=lattice_color, fontweight='bold')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add legend for lattice types
    legend_ax = fig.add_axes([0.75, 0.01, 0.23, 0.06])
    legend_ax.set_facecolor(bg_color)
    legend_ax.axis('off')
    
    for i, (lattice_type, color) in enumerate(LATTICE_COLORS.items()):
        legend_ax.add_patch(FancyBboxPatch(
            (i * 0.25, 0.3), 0.15, 0.5,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor='none'
        ))
        legend_ax.text(i * 0.25 + 0.075, 0.1, lattice_type.value,
                      ha='center', va='top', fontsize=9,
                      color=text_color, fontweight='bold')
    
    legend_ax.set_xlim(-0.05, 1.05)
    legend_ax.set_ylim(-0.2, 1.2)
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight',
                   facecolor=bg_color, edgecolor='none')
        print(f"Saved all groups visualization to {save_path}")
    
    return fig


def plot_group_comparison(group_names: List[str],
                         num_samples: int = 4,
                         resolution: int = 256,
                         seed: int = 42,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare multiple samples from selected wallpaper groups.
    
    Args:
        group_names: List of group names to compare
        num_samples: Number of samples per group
        resolution: Pattern resolution
        seed: Random seed
        save_path: Path to save figure
    """
    plt.style.use('dark_background')
    bg_color = '#0f0f1a'
    
    n_groups = len(group_names)
    fig, axes = plt.subplots(n_groups, num_samples, 
                            figsize=(4*num_samples, 4*n_groups),
                            facecolor=bg_color)
    
    if n_groups == 1:
        axes = axes.reshape(1, -1)
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for row, group_name in enumerate(group_names):
        group_info = WALLPAPER_GROUPS[group_name]
        
        for col in range(num_samples):
            ax = axes[row, col]
            ax.set_facecolor(bg_color)
            
            # Generate with different seed for variety
            generator = WallpaperGroupGenerator(
                resolution=resolution, 
                seed=seed + row * 100 + col
            )
            
            pattern = generator.generate(group_name, 
                                         motif_size=resolution//4,
                                         complexity=(col % 4) + 2,
                                         motif_type=['gaussian', 'geometric', 'mixed'][col % 3])
            
            ax.imshow(pattern, cmap='plasma', interpolation='bilinear')
            
            # Add border
            color = LATTICE_COLORS[group_info.lattice_type]
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            if col == 0:
                ax.set_ylabel(f"{group_name}\n({group_info.lattice_type.value})",
                            fontsize=12, fontweight='bold', color=color)
    
    fig.suptitle('Wallpaper Group Pattern Variations', 
                fontsize=18, fontweight='bold', color='#eaeaea', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight',
                   facecolor=bg_color)
    
    return fig


def plot_lattice_types(resolution: int = 256,
                      seed: int = 42,
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot representative patterns organized by lattice type.
    """
    plt.style.use('dark_background')
    bg_color = '#0d0d1a'
    
    lattice_groups = {
        LatticeType.OBLIQUE: ['p1', 'p2'],
        LatticeType.RECTANGULAR: ['pm', 'pmm', 'cmm'],
        LatticeType.SQUARE: ['p4', 'p4m'],
        LatticeType.HEXAGONAL: ['p3', 'p6', 'p6m'],
    }
    
    fig = plt.figure(figsize=(16, 12), facecolor=bg_color)
    gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.15)
    
    for idx, (lattice_type, groups) in enumerate(lattice_groups.items()):
        inner_gs = gs[idx // 2, idx % 2].subgridspec(1, len(groups), wspace=0.1)
        
        color = LATTICE_COLORS[lattice_type]
        
        for g_idx, group_name in enumerate(groups):
            ax = fig.add_subplot(inner_gs[0, g_idx])
            ax.set_facecolor(bg_color)
            
            generator = WallpaperGroupGenerator(resolution=resolution, seed=seed)
            pattern = generator.generate(group_name, motif_size=resolution//4)
            
            ax.imshow(pattern, cmap='viridis', interpolation='bilinear')
            
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            
            ax.set_title(group_name, fontsize=14, fontweight='bold',
                        color=color, pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            
            if g_idx == 0:
                ax.text(-0.2, 0.5, lattice_type.value.upper(),
                       transform=ax.transAxes, rotation=90,
                       va='center', ha='center', fontsize=16,
                       fontweight='bold', color=color)
    
    fig.suptitle('Wallpaper Groups by Lattice Type',
                fontsize=20, fontweight='bold', color='#eaeaea', y=0.98)
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight',
                   facecolor=bg_color)
    
    return fig


if __name__ == "__main__":
    # Demo visualization
    import os
    
    output_dir = Path(__file__).parent.parent.parent / "output" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating visualizations...")
    
    # All 17 groups
    fig = plot_all_groups(save_path=str(output_dir / "all_17_groups.png"))
    plt.close(fig)
    
    # Lattice types
    fig = plot_lattice_types(save_path=str(output_dir / "lattice_types.png"))
    plt.close(fig)
    
    # Comparison of selected groups
    fig = plot_group_comparison(
        ['p1', 'p4', 'p6m'],
        num_samples=5,
        save_path=str(output_dir / "group_comparison.png")
    )
    plt.close(fig)
    
    # Individual patterns with symmetry
    visualizer = PatternVisualizer()
    generator = WallpaperGroupGenerator(resolution=256, seed=42)
    
    for group in ['p4m', 'p6m', 'cmm']:
        pattern = generator.generate(group, motif_size=64)
        fig = visualizer.plot_symmetry_annotations(
            pattern, group,
            save_path=str(output_dir / f"symmetry_{group}.png")
        )
        plt.close(fig)
    
    print(f"Visualizations saved to {output_dir}")







