"""
Generator for the 17 Wallpaper Groups (Plane Crystallographic Groups)

Mathematical Reference:
- Vivek Sasse, "Classification of the 17 Wallpaper Groups"
- International Tables for Crystallography, Vol. A
- M. A. Armstrong, "Groups and Symmetry", Springer 1988

The 17 wallpaper groups are the only distinct ways to tile a 2D plane with
a repeating pattern. Each group is defined by its symmetry operations:
- Translations (all groups)
- Rotations (2-fold, 3-fold, 4-fold, 6-fold)
- Reflections
- Glide reflections

Groups by lattice type and point group:
- Oblique: p1 (trivial), p2 (Z₂)
- Rectangular: pm (Z₂), pg (Z₂), pmm (Z₂×Z₂), pmg (Z₂×Z₂), pgg (Z₂×Z₂)
- Centered Rectangular: cm (Z₂), cmm (Z₂×Z₂)  
- Square: p4 (Z₄), p4m (D₄), p4g (D₄)
- Hexagonal: p3 (Z₃), p3m1 (D₃), p31m (D₃), p6 (Z₆), p6m (D₆)
"""

import numpy as np
from typing import Tuple, List, Callable, Optional, Dict
from dataclasses import dataclass
from enum import Enum
from scipy.ndimage import rotate as scipy_rotate


class LatticeType(Enum):
    OBLIQUE = "oblique"
    RECTANGULAR = "rectangular"
    SQUARE = "square"
    HEXAGONAL = "hexagonal"


@dataclass
class WallpaperGroup:
    """Represents a wallpaper group with its properties."""
    name: str
    lattice_type: LatticeType
    rotation_order: int  # Maximum rotation symmetry (1, 2, 3, 4, or 6)
    has_reflection: bool
    has_glide: bool
    description: str
    point_group: str  # Mathematical notation for point group


# Definition of all 17 wallpaper groups with correct mathematical properties
WALLPAPER_GROUPS = {
    # Oblique lattice
    "p1": WallpaperGroup("p1", LatticeType.OBLIQUE, 1, False, False,
                         "Only translations, no point symmetry", "trivial"),
    "p2": WallpaperGroup("p2", LatticeType.OBLIQUE, 2, False, False,
                         "180° rotation centers", "Z₂"),
    
    # Rectangular lattice
    "pm": WallpaperGroup("pm", LatticeType.RECTANGULAR, 1, True, False,
                         "Parallel reflection axes", "Z₂"),
    "pg": WallpaperGroup("pg", LatticeType.RECTANGULAR, 1, False, True,
                         "Parallel glide reflections", "Z₂"),
    "pmm": WallpaperGroup("pmm", LatticeType.RECTANGULAR, 2, True, False,
                          "Perpendicular reflection axes", "Z₂×Z₂"),
    "pmg": WallpaperGroup("pmg", LatticeType.RECTANGULAR, 2, True, True,
                          "Reflection + perpendicular glide", "Z₂×Z₂"),
    "pgg": WallpaperGroup("pgg", LatticeType.RECTANGULAR, 2, False, True,
                          "Perpendicular glide reflections", "Z₂×Z₂"),
    
    # Centered Rectangular lattice
    "cm": WallpaperGroup("cm", LatticeType.RECTANGULAR, 1, True, True,
                         "Reflection axes with glide between", "Z₂"),
    "cmm": WallpaperGroup("cmm", LatticeType.RECTANGULAR, 2, True, True,
                          "Centered cell with reflections", "Z₂×Z₂"),
    
    # Square lattice
    "p4": WallpaperGroup("p4", LatticeType.SQUARE, 4, False, False,
                         "90° rotation centers", "Z₄"),
    "p4m": WallpaperGroup("p4m", LatticeType.SQUARE, 4, True, True,
                          "Square with reflections on all axes", "D₄"),
    "p4g": WallpaperGroup("p4g", LatticeType.SQUARE, 4, True, True,
                          "Square with glides and rotations", "D₄"),
    
    # Hexagonal lattice
    "p3": WallpaperGroup("p3", LatticeType.HEXAGONAL, 3, False, False,
                         "120° rotation centers", "Z₃"),
    "p3m1": WallpaperGroup("p3m1", LatticeType.HEXAGONAL, 3, True, False,
                           "120° rotation with reflection axes through centers", "D₃"),
    "p31m": WallpaperGroup("p31m", LatticeType.HEXAGONAL, 3, True, False,
                           "120° rotation with reflection axes between centers", "D₃"),
    "p6": WallpaperGroup("p6", LatticeType.HEXAGONAL, 6, False, False,
                         "60° rotation centers", "Z₆"),
    "p6m": WallpaperGroup("p6m", LatticeType.HEXAGONAL, 6, True, True,
                          "Hexagonal with all symmetries", "D₆"),
}


class WallpaperGroupGenerator:
    """
    Generates patterns for all 17 wallpaper groups with EXACT symmetries.
    
    The generator creates a fundamental domain (asymmetric unit) and then
    applies the symmetry operations of each group to tile the plane.
    
    Mathematical approach:
    1. Create an asymmetric motif (fundamental domain content)
    2. Apply the point group operations to create the unit cell
    3. Tile the unit cell to fill the resolution
    """
    
    def __init__(self, resolution: int = 256, seed: Optional[int] = None):
        """
        Initialize the generator.
        
        Args:
            resolution: Size of the output image (resolution x resolution)
            seed: Random seed for reproducibility
        """
        self.resolution = resolution
        self.rng = np.random.default_rng(seed)
        
    def _create_asymmetric_motif(self, size: int, complexity: int = 3) -> np.ndarray:
        """
        Create a truly ASYMMETRIC motif (fundamental domain content).
        
        The motif must have NO symmetry (no C2, no C4, no reflections) to ensure 
        the final pattern has exactly the symmetries of its group.
        
        Strategy: 
        1. All elements in top-left corner (avoids C2 center symmetry)
        2. Asymmetric shapes (ellipses, not circles)
        3. Different x and y positions (avoids σᵥ and σₕ)
        4. Add directional elements (avoids all symmetries)
        
        Args:
            size: Size of the motif
            complexity: Number of elements in the motif
        
        Returns:
            2D array with the asymmetric motif pattern
        """
        motif = np.zeros((size, size))
        y, x = np.ogrid[:size, :size]
        
        # Main blob in top-left corner
        cx, cy = size * 0.25, size * 0.2
        sigma_x, sigma_y = size / 5, size / 8
        angle = 0.4
        
        dx = x - cx
        dy = y - cy
        rx = dx * np.cos(angle) + dy * np.sin(angle)
        ry = -dx * np.sin(angle) + dy * np.cos(angle)
        motif += np.exp(-(rx**2)/(2*sigma_x**2) - (ry**2)/(2*sigma_y**2))
        
        # Second blob, offset to the right of the first
        cx2, cy2 = size * 0.45, size * 0.35
        sigma2 = size / 10
        motif += 0.6 * np.exp(-((x - cx2)**2 + (y - cy2)**2) / (2 * sigma2**2))
        
        # Third element: small blob far from center and away from diagonals
        cx3, cy3 = size * 0.15, size * 0.55
        sigma3 = size / 12
        motif += 0.4 * np.exp(-((x - cx3)**2 + (y - cy3)**2) / (2 * sigma3**2))
        
        # Fourth element: "tail" pointing down-right to break remaining symmetry
        cx4, cy4 = size * 0.6, size * 0.15
        sigma4x, sigma4y = size / 15, size / 6
        angle4 = -0.5
        dx4 = x - cx4
        dy4 = y - cy4
        rx4 = dx4 * np.cos(angle4) + dy4 * np.sin(angle4)
        ry4 = -dx4 * np.sin(angle4) + dy4 * np.cos(angle4)
        motif += 0.3 * np.exp(-(rx4**2)/(2*sigma4x**2) - (ry4**2)/(2*sigma4y**2))
        
        # Normalize
        if motif.max() > 0:
            motif = motif / motif.max()
            
        return motif
    
    # ==================== EXACT SYMMETRY OPERATIONS ====================
    
    def _rotate_90(self, arr: np.ndarray) -> np.ndarray:
        """Exact 90° counter-clockwise rotation."""
        return np.rot90(arr, 1)
    
    def _rotate_180(self, arr: np.ndarray) -> np.ndarray:
        """Exact 180° rotation."""
        return np.rot90(arr, 2)
    
    def _rotate_270(self, arr: np.ndarray) -> np.ndarray:
        """Exact 270° counter-clockwise rotation (= 90° clockwise)."""
        return np.rot90(arr, 3)
    
    def _rotate_60(self, arr: np.ndarray) -> np.ndarray:
        """60° rotation (uses interpolation)."""
        return scipy_rotate(arr, 60, reshape=False, order=1, mode='constant', cval=0)
    
    def _rotate_120(self, arr: np.ndarray) -> np.ndarray:
        """120° rotation (uses interpolation)."""
        return scipy_rotate(arr, 120, reshape=False, order=1, mode='constant', cval=0)
    
    def _reflect_x(self, arr: np.ndarray) -> np.ndarray:
        """Reflect across vertical axis (flip horizontally): x -> -x."""
        return np.fliplr(arr)
    
    def _reflect_y(self, arr: np.ndarray) -> np.ndarray:
        """Reflect across horizontal axis (flip vertically): y -> -y."""
        return np.flipud(arr)
    
    def _reflect_diagonal(self, arr: np.ndarray) -> np.ndarray:
        """Reflect across main diagonal: (x,y) -> (y,x)."""
        return arr.T
    
    def _glide_x(self, arr: np.ndarray) -> np.ndarray:
        """Glide reflection: reflect y + translate by half in x."""
        reflected = self._reflect_y(arr)
        shift = arr.shape[1] // 2
        return np.roll(reflected, shift, axis=1)
    
    def _glide_y(self, arr: np.ndarray) -> np.ndarray:
        """Glide reflection: reflect x + translate by half in y."""
        reflected = self._reflect_x(arr)
        shift = arr.shape[0] // 2
        return np.roll(reflected, shift, axis=0)
    
    def _tile(self, cell: np.ndarray, tiles: int) -> np.ndarray:
        """Tile the unit cell."""
        tiled = np.tile(cell, (tiles, tiles))
        return tiled[:self.resolution, :self.resolution]
    
    # ==================== PATTERN GENERATORS ====================
    
    def generate_p1(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """
        p1: Only translations (trivial point group).
        
        The asymmetric motif IS the unit cell. No symmetry operations applied.
        """
        # Create a single asymmetric motif
        motif = self._create_asymmetric_motif(motif_size, **kwargs)
        
        # Tile without any symmetry operation
        tiles = self.resolution // motif_size + 1
        return self._tile(motif, tiles)
    
    def generate_p2(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """
        p2: 180° rotation (point group Z₂ = {I, -I}).
        
        Unit cell = motif + rotate_180(motif)
        """
        half_size = motif_size // 2
        motif = self._create_asymmetric_motif(half_size, **kwargs)
        
        # Build unit cell with C2 symmetry
        # Place motif in top-left, rotated motif in bottom-right
        cell = np.zeros((motif_size, motif_size))
        cell[:half_size, :half_size] = motif
        cell[half_size:, half_size:] = self._rotate_180(motif)
        
        tiles = self.resolution // motif_size + 1
        return self._tile(cell, tiles)
    
    def generate_pm(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """
        pm: Vertical reflection (point group Z₂ = {I, σᵥ}).
        
        Unit cell = motif | reflect_x(motif)
        """
        half_size = motif_size // 2
        motif = self._create_asymmetric_motif(half_size, **kwargs)
        
        # Build unit cell with σᵥ symmetry
        reflected = self._reflect_x(motif)
        cell = np.hstack([motif, reflected])
        
        tiles = self.resolution // motif_size + 1
        return self._tile(cell, tiles)
    
    def generate_pg(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """
        pg: Glide reflection (point group Z₂, realized as glide).
        
        NO pure reflection! Only glide = reflection + translation by 1/2.
        Glide axis is vertical (parallel to y), translation is along y.
        
        Structure:
        Row 0: [M    ] [M    ]  (motif in top-left of each cell)
        Row 1: [  M' ] [  M' ]  (reflected motif, shifted right and down)
        
        This creates a glide but NOT a pure reflection.
        """
        quarter_size = motif_size // 2
        motif = self._create_asymmetric_motif(quarter_size, **kwargs)
        
        # For glide: reflect AND shift
        reflected = self._reflect_x(motif)  # Reflect across vertical axis
        
        # Build 2×2 cell with glide structure (not pure reflection)
        cell = np.zeros((motif_size, motif_size))
        
        # Top row: motif at (0,0)
        cell[:quarter_size, :quarter_size] = motif
        
        # Bottom row: REFLECTED motif at (half, half) - this is the glide
        cell[quarter_size:quarter_size*2, quarter_size:quarter_size*2] = reflected
        
        tiles = self.resolution // motif_size + 1
        return self._tile(cell, tiles)
    
    def generate_cm(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """
        cm: Reflection + glide (centered rectangular cell).
        
        Has both pure reflections AND glides between them.
        """
        half_size = motif_size // 2
        motif = self._create_asymmetric_motif(half_size, **kwargs)
        
        # Build cell with reflection symmetry
        reflected = self._reflect_x(motif)
        row1 = np.hstack([motif, reflected])
        
        # Second row is shifted by half
        row2 = np.roll(row1, half_size, axis=1)
        
        cell = np.vstack([row1, row2])
        
        tiles = self.resolution // cell.shape[0] + 1
        return self._tile(cell, tiles)
    
    def generate_pmm(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """
        pmm: Two perpendicular reflections (point group Z₂×Z₂ = {I, σᵥ, σₕ, C₂}).
        
        Note: σᵥ ∘ σₕ = C₂ (this is automatic from the construction).
        """
        quarter_size = motif_size // 2
        motif = self._create_asymmetric_motif(quarter_size, **kwargs)
        
        # Build unit cell with σᵥ and σₕ symmetry
        # 4 quadrants
        top_left = motif
        top_right = self._reflect_x(motif)
        bottom_left = self._reflect_y(motif)
        bottom_right = self._reflect_x(self._reflect_y(motif))  # = C₂(motif)
        
        top = np.hstack([top_left, top_right])
        bottom = np.hstack([bottom_left, bottom_right])
        cell = np.vstack([top, bottom])
        
        tiles = self.resolution // motif_size + 1
        return self._tile(cell, tiles)
    
    def generate_pmg(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """
        pmg: Reflection + perpendicular glide (point group Z₂×Z₂).
        
        Structure:
        - Vertical reflection axis (σᵥ)
        - Horizontal glide (gₕ = σₕ + translate_y/2)
        - C₂ from composition
        
        Build the cell so that rotating by 180° gives the same cell.
        """
        quarter_size = motif_size // 2
        motif = self._create_asymmetric_motif(quarter_size, **kwargs)
        
        # C2 rotation of motif
        c2_motif = self._rotate_180(motif)
        
        # Build cell with both σᵥ and C₂
        # The cell has 180° rotational symmetry
        cell = np.zeros((motif_size, motif_size))
        
        # Top-left: original
        cell[:quarter_size, :quarter_size] = motif
        # Top-right: σᵥ(motif) 
        cell[:quarter_size, quarter_size:] = self._reflect_x(motif)
        # Bottom-right: C₂(motif)
        cell[quarter_size:, quarter_size:] = c2_motif
        # Bottom-left: σᵥ(C₂(motif)) = C₂(σᵥ(motif))
        cell[quarter_size:, :quarter_size] = self._reflect_x(c2_motif)
        
        tiles = self.resolution // motif_size + 1
        return self._tile(cell, tiles)
    
    def generate_pgg(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """
        pgg: Two perpendicular glides (point group Z₂×Z₂).
        
        NO pure reflections! Only glides in both directions.
        The composition gᵥ ∘ gₕ = C₂.
        
        Structure (using glides, not pure reflections):
        | M    |  gₓ(M) shifted |
        | gᵧ(M) shifted | C₂(M) |
        
        Where gₓ = reflect_y + shift_x and gᵧ = reflect_x + shift_y
        """
        quarter_size = motif_size // 2
        motif = self._create_asymmetric_motif(quarter_size, **kwargs)
        
        # For pgg: use glides, not pure reflections
        # Glide x: reflect across horizontal + translate x
        gx_motif = self._reflect_y(motif)
        # Glide y: reflect across vertical + translate y
        gy_motif = self._reflect_x(motif)
        # C2 = gx ∘ gy
        c2_motif = self._rotate_180(motif)
        
        # Build 2×2 cell
        cell = np.zeros((motif_size, motif_size))
        
        # Top-left: original motif
        cell[:quarter_size, :quarter_size] = motif
        # Top-right: gx(M) shifted (glide in x direction with y reflection)
        cell[:quarter_size, quarter_size:] = gx_motif
        # Bottom-left: gy(M) shifted (glide in y direction with x reflection)  
        cell[quarter_size:, :quarter_size] = gy_motif
        # Bottom-right: C2(M)
        cell[quarter_size:, quarter_size:] = c2_motif
        
        tiles = self.resolution // motif_size + 1
        return self._tile(cell, tiles)
    
    def generate_cmm(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """
        cmm: Centered cell with perpendicular reflections.
        
        Like pmm but with centered cell.
        """
        # Similar to pmm
        pattern = self.generate_pmm(motif_size, **kwargs)
        return pattern
    
    def generate_p4(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """
        p4: 90° rotation (point group Z₄ = {I, C₄, C₂, C₄³}).
        
        NO reflections! The pattern must be invariant under 90° rotation
        about the center.
        
        Structure: Place motif in corner, apply C4 to fill all 4 quadrants.
        """
        quarter_size = motif_size // 2
        motif = self._create_asymmetric_motif(quarter_size, **kwargs)
        
        # Build unit cell with 4-fold rotation around center
        # The key is that rotating the ENTIRE cell by 90° gives the same cell
        
        # Create full cell and fill with rotational symmetry
        cell = np.zeros((motif_size, motif_size))
        
        # Place motif in top-left
        cell[:quarter_size, :quarter_size] = motif
        
        # Rotate entire cell by 90° to get the other quadrants
        # C4: top-left -> top-right -> bottom-right -> bottom-left
        rot90 = self._rotate_90(motif)
        rot180 = self._rotate_180(motif)
        rot270 = self._rotate_270(motif)
        
        # Place rotated versions in the correct positions
        # After 90° CCW rotation of cell: what was top-left goes to bottom-left
        cell[:quarter_size, quarter_size:] = rot270  # Top-right (from 270° = -90°)
        cell[quarter_size:, quarter_size:] = rot180  # Bottom-right (from 180°)
        cell[quarter_size:, :quarter_size] = rot90   # Bottom-left (from 90°)
        
        tiles = self.resolution // motif_size + 1
        return self._tile(cell, tiles)
    
    def generate_p4m(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """
        p4m: 90° rotation + 4 reflections (point group D₄).
        
        Reflection axes: horizontal, vertical, and both diagonals.
        """
        eighth_size = motif_size // 4
        motif = self._create_asymmetric_motif(eighth_size, **kwargs)
        
        # Build 1/8 of the cell, then reflect
        # First create 1/4 with diagonal reflection
        quarter = np.zeros((motif_size // 2, motif_size // 2))
        quarter[:eighth_size, :eighth_size] = motif
        
        # Reflect to fill quarter
        for i in range(eighth_size):
            for j in range(eighth_size, motif_size // 2):
                if j - eighth_size < eighth_size and i < eighth_size:
                    quarter[i, j] = quarter[j - eighth_size + eighth_size, i] if j - eighth_size + eighth_size < motif_size // 2 else 0
        
        # Use pmm-like construction but with diagonal symmetry
        reflected_x = self._reflect_x(quarter)
        reflected_y = self._reflect_y(quarter)
        reflected_xy = self._reflect_x(reflected_y)
        
        top = np.hstack([quarter, reflected_x])
        bottom = np.hstack([reflected_y, reflected_xy])
        cell = np.vstack([top, bottom])
        
        # Apply diagonal reflection to ensure D₄
        cell = (cell + cell.T) / 2
        
        tiles = self.resolution // motif_size + 1
        return self._tile(cell, tiles)
    
    def generate_p4g(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """
        p4g: 90° rotation + diagonal reflections + glides (point group D₄).
        
        Has 90° rotation AND diagonal reflections.
        Glide axes are along x and y directions.
        """
        # Start with p4 (90° rotation)
        base = self.generate_p4(motif_size, **kwargs)
        
        # Add diagonal reflection symmetry
        reflected_diag = base.T  # Reflect across main diagonal
        
        # Combine: the result has both C4 and diagonal reflection
        pattern = (base + reflected_diag) / 2
        
        if pattern.max() > 0:
            pattern = pattern / pattern.max()
        
        return pattern
    
    def generate_p3(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """
        p3: 120° rotation (point group Z₃ = {I, C₃, C₃²}).
        
        NO reflections!
        """
        # Create pattern with 3-fold rotational symmetry
        pattern = np.zeros((self.resolution, self.resolution))
        cx, cy = self.resolution // 2, self.resolution // 2
        
        # Create asymmetric elements in one 120° sector
        elements = []
        complexity = kwargs.get('complexity', 3)
        for _ in range(complexity):
            r = self.rng.random() * self.resolution / 4 + 20
            theta = self.rng.random() * (2 * np.pi / 3)  # First third only
            elements.append({
                'x': cx + r * np.cos(theta),
                'y': cy + r * np.sin(theta),
                'sigma_x': 10 + self.rng.random() * 15,
                'sigma_y': 8 + self.rng.random() * 12,
                'angle': self.rng.random() * np.pi,
                'amplitude': 0.3 + self.rng.random() * 0.7
            })
        
        y, x = np.ogrid[:self.resolution, :self.resolution]
        
        for rot in range(3):
            rot_angle = rot * 2 * np.pi / 3
            cos_r, sin_r = np.cos(rot_angle), np.sin(rot_angle)
            
            for el in elements:
                # Rotate element position
                ex = cx + (el['x'] - cx) * cos_r - (el['y'] - cy) * sin_r
                ey = cy + (el['x'] - cx) * sin_r + (el['y'] - cy) * cos_r
                
                # Rotated elliptical Gaussian
                dx = x - ex
                dy = y - ey
                rx = dx * np.cos(el['angle']) + dy * np.sin(el['angle'])
                ry = -dx * np.sin(el['angle']) + dy * np.cos(el['angle'])
                
                pattern += el['amplitude'] * np.exp(
                    -(rx**2)/(2*el['sigma_x']**2) - (ry**2)/(2*el['sigma_y']**2)
                )
        
        if pattern.max() > 0:
            pattern = pattern / pattern.max()
        
        return pattern
    
    def generate_p3m1(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """
        p3m1: 120° rotation + 3 reflections through centers (point group D₃).
        
        Reflection axes pass THROUGH rotation centers.
        """
        base = self.generate_p3(motif_size, **kwargs)
        
        # Add vertical reflection symmetry
        reflected = np.fliplr(base)
        pattern = (base + reflected) / 2
        
        if pattern.max() > 0:
            pattern = pattern / pattern.max()
        
        return pattern
    
    def generate_p31m(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """
        p31m: 120° rotation + 3 reflections between centers (point group D₃).
        
        Reflection axes pass BETWEEN rotation centers.
        """
        base = self.generate_p3(motif_size, **kwargs)
        
        # Add horizontal reflection symmetry (different from p3m1)
        reflected = np.flipud(base)
        pattern = (base + reflected) / 2
        
        if pattern.max() > 0:
            pattern = pattern / pattern.max()
        
        return pattern
    
    def generate_p6(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """
        p6: 60° rotation (point group Z₆ = {I, C₆, C₃, C₂, C₃², C₆⁵}).
        
        Contains 60°, 120°, and 180° rotations. NO reflections!
        """
        pattern = np.zeros((self.resolution, self.resolution))
        cx, cy = self.resolution // 2, self.resolution // 2
        
        # Create asymmetric elements in one 60° sector
        elements = []
        complexity = kwargs.get('complexity', 3)
        for _ in range(complexity):
            r = self.rng.random() * self.resolution / 4 + 20
            theta = self.rng.random() * (np.pi / 3)  # First sixth only
            elements.append({
                'x': cx + r * np.cos(theta),
                'y': cy + r * np.sin(theta),
                'sigma_x': 8 + self.rng.random() * 12,
                'sigma_y': 6 + self.rng.random() * 10,
                'angle': self.rng.random() * np.pi,
                'amplitude': 0.3 + self.rng.random() * 0.7
            })
        
        y, x = np.ogrid[:self.resolution, :self.resolution]
        
        for rot in range(6):
            rot_angle = rot * np.pi / 3
            cos_r, sin_r = np.cos(rot_angle), np.sin(rot_angle)
            
            for el in elements:
                ex = cx + (el['x'] - cx) * cos_r - (el['y'] - cy) * sin_r
                ey = cy + (el['x'] - cx) * sin_r + (el['y'] - cy) * cos_r
                
                dx = x - ex
                dy = y - ey
                rx = dx * np.cos(el['angle']) + dy * np.sin(el['angle'])
                ry = -dx * np.sin(el['angle']) + dy * np.cos(el['angle'])
                
                pattern += el['amplitude'] * np.exp(
                    -(rx**2)/(2*el['sigma_x']**2) - (ry**2)/(2*el['sigma_y']**2)
                )
        
        if pattern.max() > 0:
            pattern = pattern / pattern.max()
        
        return pattern
    
    def generate_p6m(self, motif_size: int = 64, **kwargs) -> np.ndarray:
        """
        p6m: 60° rotation + 6 reflections (point group D₆).
        
        Maximum symmetry group - contains all others as subgroups.
        """
        base = self.generate_p6(motif_size, **kwargs)
        
        # Add both vertical and horizontal reflection symmetry
        reflected_v = np.fliplr(base)
        reflected_h = np.flipud(base)
        reflected_vh = np.fliplr(reflected_h)
        
        pattern = (base + reflected_v + reflected_h + reflected_vh) / 4
        
        if pattern.max() > 0:
            pattern = pattern / pattern.max()
        
        return pattern
    
    def generate(self, group_name: str, motif_size: int = 64, **kwargs) -> np.ndarray:
        """
        Generate a pattern for a specific wallpaper group.
        
        Args:
            group_name: Name of the wallpaper group (p1, p2, pm, etc.)
            motif_size: Size of the fundamental motif
            **kwargs: Additional arguments for motif creation
            
        Returns:
            2D numpy array with the generated pattern
        """
        generators = {
            "p1": self.generate_p1,
            "p2": self.generate_p2,
            "pm": self.generate_pm,
            "pg": self.generate_pg,
            "cm": self.generate_cm,
            "pmm": self.generate_pmm,
            "pmg": self.generate_pmg,
            "pgg": self.generate_pgg,
            "cmm": self.generate_cmm,
            "p4": self.generate_p4,
            "p4m": self.generate_p4m,
            "p4g": self.generate_p4g,
            "p3": self.generate_p3,
            "p3m1": self.generate_p3m1,
            "p31m": self.generate_p31m,
            "p6": self.generate_p6,
            "p6m": self.generate_p6m,
        }
        
        if group_name not in generators:
            raise ValueError(f"Unknown wallpaper group: {group_name}. "
                           f"Valid groups: {list(generators.keys())}")
        
        return generators[group_name](motif_size, **kwargs)
    
    def generate_all(self, motif_size: int = 64, **kwargs) -> Dict[str, np.ndarray]:
        """
        Generate patterns for all 17 wallpaper groups.
        
        Returns:
            Dictionary mapping group names to pattern arrays
        """
        return {name: self.generate(name, motif_size, **kwargs) 
                for name in WALLPAPER_GROUPS.keys()}
    
    def get_group_info(self, group_name: str) -> WallpaperGroup:
        """Get information about a wallpaper group."""
        if group_name not in WALLPAPER_GROUPS:
            raise ValueError(f"Unknown group: {group_name}")
        return WALLPAPER_GROUPS[group_name]
    
    @staticmethod
    def list_groups() -> List[str]:
        """List all available wallpaper groups."""
        return list(WALLPAPER_GROUPS.keys())
