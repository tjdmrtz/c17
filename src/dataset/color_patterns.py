"""
Colorful Crystallographic Pattern Generator.

Each wallpaper group gets a unique, beautiful color palette that makes
the patterns visually distinctive and aesthetically pleasing.

Color palettes inspired by:
- Nature (minerals, crystals, flowers)
- Art movements (Art Deco, Japanese prints)
- Modern design trends
"""

import numpy as np
from typing import Tuple, Optional, Dict
from .pattern_generator import WallpaperGroupGenerator, WALLPAPER_GROUPS, LatticeType


# Beautiful color palettes for each wallpaper group
# Format: (background RGB, primary RGB, secondary RGB, accent RGB)
GROUP_PALETTES = {
    # Oblique - warm earth tones
    'p1': {
        'name': 'Desert Sand',
        'bg': (0.96, 0.91, 0.82),      # Cream
        'primary': (0.80, 0.52, 0.25),  # Terracotta
        'secondary': (0.55, 0.35, 0.20), # Brown
        'accent': (0.95, 0.75, 0.45),   # Gold
    },
    'p2': {
        'name': 'Volcanic',
        'bg': (0.15, 0.12, 0.18),       # Dark purple
        'primary': (0.90, 0.30, 0.25),  # Lava red
        'secondary': (0.95, 0.55, 0.20), # Orange
        'accent': (1.0, 0.85, 0.30),    # Yellow
    },
    
    # Rectangular - ocean and forest
    'pm': {
        'name': 'Ocean Waves',
        'bg': (0.05, 0.15, 0.25),       # Deep blue
        'primary': (0.20, 0.60, 0.80),  # Cyan
        'secondary': (0.40, 0.80, 0.90), # Light cyan
        'accent': (0.95, 0.95, 1.0),    # White foam
    },
    'pg': {
        'name': 'Northern Lights',
        'bg': (0.08, 0.10, 0.18),       # Night sky
        'primary': (0.30, 0.85, 0.60),  # Aurora green
        'secondary': (0.50, 0.30, 0.80), # Purple
        'accent': (0.95, 0.40, 0.60),   # Pink
    },
    'cm': {
        'name': 'Cherry Blossom',
        'bg': (0.98, 0.94, 0.95),       # Soft pink white
        'primary': (0.95, 0.55, 0.65),  # Sakura pink
        'secondary': (0.85, 0.35, 0.45), # Deep pink
        'accent': (0.40, 0.55, 0.35),   # Leaf green
    },
    'pmm': {
        'name': 'Art Deco Gold',
        'bg': (0.12, 0.14, 0.18),       # Charcoal
        'primary': (0.85, 0.70, 0.35),  # Gold
        'secondary': (0.55, 0.45, 0.25), # Bronze
        'accent': (0.95, 0.90, 0.80),   # Cream
    },
    'pmg': {
        'name': 'Tropical',
        'bg': (0.15, 0.30, 0.25),       # Deep jungle
        'primary': (0.30, 0.75, 0.45),  # Tropical green
        'secondary': (0.95, 0.85, 0.20), # Yellow
        'accent': (0.95, 0.40, 0.35),   # Hibiscus red
    },
    'pgg': {
        'name': 'Amethyst',
        'bg': (0.20, 0.15, 0.28),       # Deep purple
        'primary': (0.60, 0.40, 0.80),  # Amethyst
        'secondary': (0.80, 0.60, 0.90), # Lavender
        'accent': (0.95, 0.85, 0.95),   # Light pink
    },
    'cmm': {
        'name': 'Copper Patina',
        'bg': (0.12, 0.18, 0.20),       # Dark teal
        'primary': (0.45, 0.75, 0.70),  # Patina green
        'secondary': (0.75, 0.45, 0.30), # Copper
        'accent': (0.90, 0.75, 0.55),   # Brass
    },
    
    # Square - geometric and bold
    'p4': {
        'name': 'Neon City',
        'bg': (0.08, 0.08, 0.12),       # Near black
        'primary': (0.00, 0.90, 0.95),  # Cyan neon
        'secondary': (0.95, 0.20, 0.60), # Magenta
        'accent': (0.95, 0.95, 0.20),   # Yellow
    },
    'p4m': {
        'name': 'Royal Blue',
        'bg': (0.95, 0.95, 0.98),       # Off-white
        'primary': (0.20, 0.30, 0.65),  # Royal blue
        'secondary': (0.35, 0.45, 0.80), # Lighter blue
        'accent': (0.85, 0.70, 0.25),   # Gold trim
    },
    'p4g': {
        'name': 'Jade Temple',
        'bg': (0.10, 0.12, 0.10),       # Dark
        'primary': (0.30, 0.70, 0.50),  # Jade green
        'secondary': (0.50, 0.85, 0.65), # Light jade
        'accent': (0.90, 0.80, 0.60),   # Ivory
    },
    
    # Hexagonal - natural and crystalline
    'p3': {
        'name': 'Sunset',
        'bg': (0.15, 0.10, 0.20),       # Dusk purple
        'primary': (0.95, 0.50, 0.30),  # Orange
        'secondary': (0.95, 0.30, 0.40), # Coral
        'accent': (0.95, 0.80, 0.40),   # Yellow
    },
    'p3m1': {
        'name': 'Emerald',
        'bg': (0.05, 0.12, 0.10),       # Dark green
        'primary': (0.20, 0.75, 0.45),  # Emerald
        'secondary': (0.40, 0.90, 0.60), # Light emerald
        'accent': (0.95, 0.95, 0.90),   # Diamond white
    },
    'p31m': {
        'name': 'Rose Quartz',
        'bg': (0.28, 0.22, 0.25),       # Dusty mauve
        'primary': (0.90, 0.70, 0.75),  # Rose
        'secondary': (0.95, 0.85, 0.88), # Light pink
        'accent': (0.70, 0.55, 0.60),   # Mauve
    },
    'p6': {
        'name': 'Sapphire',
        'bg': (0.08, 0.10, 0.20),       # Deep blue
        'primary': (0.25, 0.45, 0.85),  # Sapphire
        'secondary': (0.45, 0.65, 0.95), # Light sapphire
        'accent': (0.85, 0.85, 0.95),   # Ice blue
    },
    'p6m': {
        'name': 'Rainbow Crystal',
        'bg': (0.12, 0.12, 0.15),       # Dark
        'primary': (0.40, 0.80, 0.90),  # Cyan
        'secondary': (0.90, 0.50, 0.70), # Pink
        'accent': (0.95, 0.90, 0.50),   # Yellow
    },
}


class ColorPatternGenerator:
    """
    Generates colorful crystallographic patterns.
    
    Each wallpaper group has a unique color palette that makes
    the patterns visually distinctive.
    """
    
    def __init__(self, resolution: int = 256, seed: Optional[int] = None):
        self.resolution = resolution
        self.base_generator = WallpaperGroupGenerator(resolution=resolution, seed=seed)
        self.rng = np.random.default_rng(seed)
    
    def _apply_colormap(self, 
                        pattern: np.ndarray, 
                        palette: Dict) -> np.ndarray:
        """
        Apply a color palette to a grayscale pattern.
        
        Creates a smooth gradient between palette colors based on
        the pattern intensity values.
        """
        h, w = pattern.shape
        colored = np.zeros((h, w, 3))
        
        # Normalize pattern
        p_min, p_max = pattern.min(), pattern.max()
        if p_max > p_min:
            pattern = (pattern - p_min) / (p_max - p_min)
        
        # Get colors
        bg = np.array(palette['bg'])
        primary = np.array(palette['primary'])
        secondary = np.array(palette['secondary'])
        accent = np.array(palette['accent'])
        
        # Create color zones based on intensity
        # 0.0-0.3: bg to primary
        # 0.3-0.6: primary to secondary  
        # 0.6-0.9: secondary to accent
        # 0.9-1.0: accent (brightest)
        
        for i in range(3):  # RGB channels
            zone1 = pattern < 0.3
            zone2 = (pattern >= 0.3) & (pattern < 0.6)
            zone3 = (pattern >= 0.6) & (pattern < 0.9)
            zone4 = pattern >= 0.9
            
            # Interpolate in each zone
            t1 = pattern / 0.3
            t2 = (pattern - 0.3) / 0.3
            t3 = (pattern - 0.6) / 0.3
            
            colored[:, :, i] = np.where(zone1,
                bg[i] + t1 * (primary[i] - bg[i]),
                np.where(zone2,
                    primary[i] + t2 * (secondary[i] - primary[i]),
                    np.where(zone3,
                        secondary[i] + t3 * (accent[i] - secondary[i]),
                        accent[i])))
        
        return np.clip(colored, 0, 1)
    
    def _apply_gradient_colormap(self,
                                  pattern: np.ndarray,
                                  palette: Dict) -> np.ndarray:
        """Alternative: smooth multi-color gradient."""
        from scipy.ndimage import gaussian_filter
        
        h, w = pattern.shape
        
        # Normalize
        p_min, p_max = pattern.min(), pattern.max()
        if p_max > p_min:
            pattern = (pattern - p_min) / (p_max - p_min)
        
        # Create RGB channels with slight offsets for color variation
        r_pattern = gaussian_filter(pattern, sigma=1)
        g_pattern = np.roll(pattern, 5, axis=0)
        b_pattern = np.roll(pattern, 5, axis=1)
        
        bg = np.array(palette['bg'])
        primary = np.array(palette['primary'])
        secondary = np.array(palette['secondary'])
        accent = np.array(palette['accent'])
        
        # Blend based on pattern values
        colored = np.zeros((h, w, 3))
        
        # Use pattern as blend weight
        w1 = (1 - pattern) ** 2  # Background weight
        w2 = 2 * pattern * (1 - pattern)  # Mid tones
        w3 = pattern ** 2  # Highlights
        
        for i in range(3):
            colored[:, :, i] = (w1 * bg[i] + 
                               w2 * (0.5 * primary[i] + 0.5 * secondary[i]) +
                               w3 * accent[i])
        
        return np.clip(colored, 0, 1)
    
    def generate(self,
                 group_name: str,
                 motif_size: int = 64,
                 style: str = 'gradient',
                 palette: Optional[Dict] = None,
                 **kwargs) -> np.ndarray:
        """
        Generate a colorful pattern for a wallpaper group.
        
        Args:
            group_name: Name of the wallpaper group
            motif_size: Size of the fundamental motif
            style: 'zones' or 'gradient'
            palette: Optional custom palette dict. If None, uses group's default palette.
            **kwargs: Additional args for pattern generation
            
        Returns:
            RGB array of shape (H, W, 3) with values in [0, 1]
        """
        # Generate base grayscale pattern
        pattern = self.base_generator.generate(group_name, motif_size, **kwargs)
        
        # Get palette (custom or group default)
        if palette is None:
            palette = GROUP_PALETTES.get(group_name, GROUP_PALETTES['p1'])
        
        # Apply colormap
        if style == 'zones':
            colored = self._apply_colormap(pattern, palette)
        else:
            colored = self._apply_gradient_colormap(pattern, palette)
        
        return colored
    
    def generate_random_palette(self) -> Dict:
        """Generate a random color palette."""
        # Generate random hue for primary color
        hue = self.rng.random()
        
        # Choose palette style
        style = self.rng.choice(['complementary', 'analogous', 'triadic', 'split'])
        
        def hsv_to_rgb(h, s, v):
            """Convert HSV to RGB."""
            import colorsys
            return colorsys.hsv_to_rgb(h, s, v)
        
        if style == 'complementary':
            # Complementary colors
            bg_light = self.rng.random() > 0.5
            if bg_light:
                bg = hsv_to_rgb(hue, 0.05 + self.rng.random() * 0.1, 0.9 + self.rng.random() * 0.1)
            else:
                bg = hsv_to_rgb(hue, 0.1 + self.rng.random() * 0.2, 0.08 + self.rng.random() * 0.12)
            
            primary = hsv_to_rgb(hue, 0.6 + self.rng.random() * 0.3, 0.6 + self.rng.random() * 0.3)
            secondary = hsv_to_rgb((hue + 0.5) % 1.0, 0.5 + self.rng.random() * 0.3, 0.5 + self.rng.random() * 0.3)
            accent = hsv_to_rgb((hue + 0.5) % 1.0, 0.7 + self.rng.random() * 0.2, 0.8 + self.rng.random() * 0.2)
            
        elif style == 'analogous':
            # Analogous colors (nearby on color wheel)
            bg_light = self.rng.random() > 0.5
            if bg_light:
                bg = hsv_to_rgb(hue, 0.05, 0.95)
            else:
                bg = hsv_to_rgb(hue, 0.15, 0.12)
            
            primary = hsv_to_rgb(hue, 0.7, 0.7)
            secondary = hsv_to_rgb((hue + 0.08) % 1.0, 0.6, 0.6)
            accent = hsv_to_rgb((hue - 0.08) % 1.0, 0.8, 0.9)
            
        elif style == 'triadic':
            # Triadic colors (120° apart)
            bg = hsv_to_rgb(hue, 0.1, 0.1 + self.rng.random() * 0.1)
            primary = hsv_to_rgb(hue, 0.7, 0.75)
            secondary = hsv_to_rgb((hue + 0.33) % 1.0, 0.6, 0.65)
            accent = hsv_to_rgb((hue + 0.66) % 1.0, 0.8, 0.9)
            
        else:  # split
            # Split complementary
            bg_light = self.rng.random() > 0.6
            if bg_light:
                bg = hsv_to_rgb(hue, 0.03, 0.96)
            else:
                bg = hsv_to_rgb(hue, 0.2, 0.15)
            
            primary = hsv_to_rgb(hue, 0.75, 0.7)
            secondary = hsv_to_rgb((hue + 0.42) % 1.0, 0.55, 0.6)
            accent = hsv_to_rgb((hue + 0.58) % 1.0, 0.7, 0.85)
        
        return {
            'name': f'Random_{style}',
            'bg': bg,
            'primary': primary,
            'secondary': secondary,
            'accent': accent
        }
    
    def generate_all(self, 
                     motif_size: int = 64,
                     **kwargs) -> Dict[str, np.ndarray]:
        """Generate colored patterns for all 17 groups."""
        return {name: self.generate(name, motif_size, **kwargs)
                for name in WALLPAPER_GROUPS.keys()}
    
    @staticmethod
    def get_palette(group_name: str) -> Dict:
        """Get the color palette for a group."""
        return GROUP_PALETTES.get(group_name, GROUP_PALETTES['p1'])
    
    @staticmethod
    def get_palette_name(group_name: str) -> str:
        """Get the name of a group's color palette."""
        return GROUP_PALETTES.get(group_name, {}).get('name', 'Unknown')


def visualize_all_colored_groups(resolution: int = 256,
                                  motif_size: int = 64,
                                  seed: int = 42,
                                  save_path: Optional[str] = None):
    """
    Create a beautiful visualization of all 17 colored wallpaper groups.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    generator = ColorPatternGenerator(resolution=resolution, seed=seed)
    
    # Dark background
    fig = plt.figure(figsize=(22, 18), facecolor='#0a0a0f')
    
    # Title
    fig.suptitle('The 17 Wallpaper Groups', 
                fontsize=28, fontweight='bold', color='#f0f0f0', 
                y=0.98, fontfamily='serif')
    
    # Subtitle
    fig.text(0.5, 0.945, 'Crystallographic Symmetry Patterns', 
            ha='center', fontsize=14, color='#888888', fontfamily='serif')
    
    # Grid layout
    gs = GridSpec(4, 5, figure=fig, hspace=0.25, wspace=0.12,
                  left=0.03, right=0.97, top=0.90, bottom=0.03)
    
    # Organize groups
    groups_order = [
        'p1', 'p2', 'pm', 'pg', 'cm',
        'pmm', 'pmg', 'pgg', 'cmm', 'p4',
        'p4m', 'p4g', 'p3', 'p3m1', 'p31m',
        'p6', 'p6m'
    ]
    
    for idx, group_name in enumerate(groups_order):
        row = idx // 5
        col = idx % 5
        
        ax = fig.add_subplot(gs[row, col])
        
        # Generate colored pattern
        pattern = generator.generate(group_name, motif_size=motif_size,
                                    complexity=4, motif_type='mixed')
        
        # Display
        ax.imshow(pattern, interpolation='bilinear')
        
        # Get palette info
        palette = GROUP_PALETTES[group_name]
        
        # Title with group name and palette
        ax.set_title(f"{group_name}\n{palette['name']}", 
                    fontsize=11, fontweight='bold', 
                    color='#f0f0f0', pad=8, fontfamily='serif')
        
        # Add subtle border with accent color
        accent = palette['accent']
        for spine in ax.spines.values():
            spine.set_edgecolor(accent)
            spine.set_linewidth(2)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Fill empty cells with info
    for idx in range(17, 20):
        row = idx // 5
        col = idx % 5
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#0a0a0f')
        ax.axis('off')
        
        if idx == 17:
            # Legend for lattice types
            lattice_info = [
                ("Oblique", "#e07040", "p1, p2"),
                ("Rectangular", "#40a0e0", "pm, pg, cm, pmm, pmg, pgg, cmm"),
                ("Square", "#60e080", "p4, p4m, p4g"),
                ("Hexagonal", "#e060a0", "p3, p3m1, p31m, p6, p6m"),
            ]
            
            y_pos = 0.85
            ax.text(0.1, y_pos + 0.1, "Lattice Types:", fontsize=11, 
                   color='#f0f0f0', fontweight='bold', transform=ax.transAxes)
            
            for lattice, color, groups in lattice_info:
                ax.scatter([0.15], [y_pos], c=[color], s=100, transform=ax.transAxes)
                ax.text(0.25, y_pos, f"{lattice}", fontsize=9, 
                       color='#f0f0f0', va='center', transform=ax.transAxes)
                y_pos -= 0.2
    
    if save_path:
        fig.savefig(save_path, dpi=200, facecolor='#0a0a0f', 
                   bbox_inches='tight', edgecolor='none')
        print(f"Saved: {save_path}")
    
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    output_dir = Path(__file__).parent.parent.parent / "output" / "colored_patterns"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate the main visualization
    fig = visualize_all_colored_groups(
        save_path=str(output_dir / "all_17_groups_colored.png")
    )
    plt.close(fig)
    
    # Generate individual patterns
    generator = ColorPatternGenerator(resolution=512, seed=42)
    
    for group_name in WALLPAPER_GROUPS.keys():
        pattern = generator.generate(group_name, motif_size=128)
        
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0a0a0f')
        ax.imshow(pattern, interpolation='bilinear')
        ax.axis('off')
        
        palette = GROUP_PALETTES[group_name]
        ax.set_title(f"{group_name} - {palette['name']}", 
                    fontsize=16, color='#f0f0f0', pad=15, fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(output_dir / f"pattern_{group_name}_colored.png",
                   dpi=150, facecolor='#0a0a0f', bbox_inches='tight')
        plt.close(fig)
    
    print(f"Saved all colored patterns to: {output_dir}")




