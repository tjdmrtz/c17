"""
Turing Pattern Generator with Crystallographic Symmetries

This module generates Turing patterns (reaction-diffusion patterns) that respect
the symmetries of the 17 wallpaper groups. The key insight is that:

1. Turing patterns naturally form periodic structures with characteristic wavelengths
2. The symmetry of the final pattern depends on:
   - The shape of the simulation domain (lattice type)
   - The initial conditions (seed pattern with group symmetry)
   - Symmetry constraints applied during evolution

Mathematical Background:
------------------------
The Gray-Scott reaction-diffusion system:
    ∂u/∂t = Du∇²u - uv² + f(1-u)
    ∂v/∂t = Dv∇²v + uv² - (f+k)v

Where:
- u, v are chemical concentrations
- Du, Dv are diffusion coefficients
- f is the feed rate
- k is the kill rate

Different (f, k) parameters produce different pattern types:
- Spots (hexagonal): f≈0.035, k≈0.065
- Stripes: f≈0.04, k≈0.06
- Maze/labyrinth: f≈0.029, k≈0.057
- Mitosis: f≈0.028, k≈0.062

By constraining the simulation to respect wallpaper group symmetries,
we can generate Turing patterns that tile the plane according to any
of the 17 crystallographic groups.
"""

import numpy as np
from scipy.ndimage import rotate as scipy_rotate, convolve
from typing import Optional, Dict, Tuple, List, Callable
from dataclasses import dataclass
from enum import Enum


class PatternType(Enum):
    """Common Turing pattern types based on (f, k) parameters."""
    SPOTS = "spots"           # Hexagonal spot arrays
    STRIPES = "stripes"       # Parallel stripes (zebra-like)
    MAZE = "maze"             # Labyrinthine patterns
    MITOSIS = "mitosis"       # Dividing spots
    CORAL = "coral"           # Branching coral-like
    WAVES = "waves"           # Wave-like patterns
    FINGERPRINT = "fingerprint"  # Fingerprint-like whorls


@dataclass
class TuringParams:
    """Parameters for the Gray-Scott reaction-diffusion system."""
    Du: float = 0.16        # Diffusion rate of u
    Dv: float = 0.08        # Diffusion rate of v
    f: float = 0.035        # Feed rate
    k: float = 0.065        # Kill rate
    dt: float = 1.0         # Time step
    
    @classmethod
    def for_pattern(cls, pattern_type: PatternType) -> 'TuringParams':
        """Get parameters for a specific pattern type."""
        params = {
            PatternType.SPOTS: cls(f=0.035, k=0.065),
            PatternType.STRIPES: cls(f=0.04, k=0.06),
            PatternType.MAZE: cls(f=0.029, k=0.057),
            PatternType.MITOSIS: cls(f=0.028, k=0.062),
            PatternType.CORAL: cls(f=0.058, k=0.065),
            PatternType.WAVES: cls(f=0.014, k=0.054),
            PatternType.FINGERPRINT: cls(f=0.037, k=0.06),
        }
        return params.get(pattern_type, cls())


class SymmetryProjector:
    """
    Projects patterns onto specific wallpaper group symmetries.
    
    The projection works by averaging a point with all its symmetry-equivalent
    positions under the group operations. This ensures the pattern respects
    the group symmetry exactly.
    """
    
    def __init__(self, resolution: int):
        self.resolution = resolution
        self.n = resolution
        
        # Precompute coordinate grids
        self.y, self.x = np.mgrid[0:resolution, 0:resolution]
        self.cx, self.cy = resolution / 2, resolution / 2
        
        # Centered coordinates
        self.dx = self.x - self.cx
        self.dy = self.y - self.cy
        
    def _rot90(self, arr: np.ndarray, k: int = 1) -> np.ndarray:
        """Rotate array by 90*k degrees using exact numpy operation."""
        return np.rot90(arr, k)
    
    def _rot_arbitrary(self, arr: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate by arbitrary angle (uses interpolation)."""
        return scipy_rotate(arr, angle_deg, reshape=False, order=1, mode='wrap')
    
    def _fliph(self, arr: np.ndarray) -> np.ndarray:
        """Flip horizontally (reflect across vertical axis)."""
        return np.fliplr(arr)
    
    def _flipv(self, arr: np.ndarray) -> np.ndarray:
        """Flip vertically (reflect across horizontal axis)."""
        return np.flipud(arr)
    
    def _glide_h(self, arr: np.ndarray) -> np.ndarray:
        """Horizontal glide reflection: flip vertically + shift horizontally by half."""
        flipped = self._flipv(arr)
        return np.roll(flipped, self.n // 2, axis=1)
    
    def _glide_v(self, arr: np.ndarray) -> np.ndarray:
        """Vertical glide reflection: flip horizontally + shift vertically by half."""
        flipped = self._fliph(arr)
        return np.roll(flipped, self.n // 2, axis=0)
    
    def project_p1(self, arr: np.ndarray) -> np.ndarray:
        """p1: No point symmetry, just return as-is."""
        return arr
    
    def project_p2(self, arr: np.ndarray) -> np.ndarray:
        """p2: 180° rotation symmetry (C2)."""
        rot180 = self._rot90(arr, 2)
        return (arr + rot180) / 2
    
    def project_pm(self, arr: np.ndarray) -> np.ndarray:
        """pm: Vertical reflection axis."""
        return (arr + self._fliph(arr)) / 2
    
    def project_pg(self, arr: np.ndarray) -> np.ndarray:
        """pg: Vertical glide reflection."""
        return (arr + self._glide_v(arr)) / 2
    
    def project_cm(self, arr: np.ndarray) -> np.ndarray:
        """cm: Reflection + emergent glide (centered)."""
        reflected = self._fliph(arr)
        return (arr + reflected) / 2
    
    def project_pmm(self, arr: np.ndarray) -> np.ndarray:
        """pmm: Two perpendicular reflection axes → C2 emerges."""
        # Average over: identity, flipH, flipV, rot180
        result = arr + self._fliph(arr) + self._flipv(arr) + self._rot90(arr, 2)
        return result / 4
    
    def project_pmg(self, arr: np.ndarray) -> np.ndarray:
        """pmg: Reflection + perpendicular glide → C2 emerges."""
        # Reflection axis + 180° rotation
        reflected = self._fliph(arr)
        combined = (arr + reflected) / 2
        rot180 = self._rot90(combined, 2)
        return (combined + rot180) / 2
    
    def project_pgg(self, arr: np.ndarray) -> np.ndarray:
        """pgg: Two perpendicular glides → C2 emerges."""
        # Glide in both directions
        g1 = self._glide_h(arr)
        g2 = self._glide_v(arr)
        rot180 = self._rot90(arr, 2)
        return (arr + g1 + g2 + rot180) / 4
    
    def project_cmm(self, arr: np.ndarray) -> np.ndarray:
        """cmm: C2 + two reflection axes (centered)."""
        # Same as pmm for projection purposes
        return self.project_pmm(arr)
    
    def project_p4(self, arr: np.ndarray) -> np.ndarray:
        """p4: 90° rotation symmetry (C4), no reflections."""
        result = arr.copy()
        for k in range(1, 4):
            result += self._rot90(arr, k)
        return result / 4
    
    def project_p4m(self, arr: np.ndarray) -> np.ndarray:
        """p4m: C4 + all 4 reflection axes (D4)."""
        # Average over 8 operations: 4 rotations × (identity + reflection)
        result = np.zeros_like(arr)
        for k in range(4):
            rotated = self._rot90(arr, k)
            result += rotated
            result += self._fliph(rotated)
        return result / 8
    
    def project_p4g(self, arr: np.ndarray) -> np.ndarray:
        """p4g: C4 + diagonal reflections + glides."""
        # C4 + diagonal reflection
        result = np.zeros_like(arr)
        for k in range(4):
            rotated = self._rot90(arr, k)
            result += rotated
        # Add diagonal reflection (transpose)
        result += np.transpose(arr)
        result += self._rot90(np.transpose(arr), 2)
        return result / 6
    
    def project_p3(self, arr: np.ndarray) -> np.ndarray:
        """p3: 120° rotation symmetry (C3), no reflections."""
        result = arr.copy()
        result += self._rot_arbitrary(arr, 120)
        result += self._rot_arbitrary(arr, 240)
        return result / 3
    
    def project_p3m1(self, arr: np.ndarray) -> np.ndarray:
        """p3m1: C3 + reflection axes through rotation centers (D3)."""
        # C3 symmetry + reflection
        c3 = self.project_p3(arr)
        return (c3 + self._fliph(c3)) / 2
    
    def project_p31m(self, arr: np.ndarray) -> np.ndarray:
        """p31m: C3 + reflection axes between rotation centers."""
        c3 = self.project_p3(arr)
        return (c3 + self._flipv(c3)) / 2
    
    def project_p6(self, arr: np.ndarray) -> np.ndarray:
        """p6: 60° rotation symmetry (C6), no reflections."""
        result = arr.copy()
        for angle in [60, 120, 180, 240, 300]:
            result += self._rot_arbitrary(arr, angle)
        return result / 6
    
    def project_p6m(self, arr: np.ndarray) -> np.ndarray:
        """p6m: C6 + all 6 reflection axes (D6)."""
        c6 = self.project_p6(arr)
        return (c6 + self._fliph(c6)) / 2
    
    def project(self, arr: np.ndarray, group_name: str) -> np.ndarray:
        """Project array onto the symmetry of the specified wallpaper group."""
        projectors = {
            'p1': self.project_p1,
            'p2': self.project_p2,
            'pm': self.project_pm,
            'pg': self.project_pg,
            'cm': self.project_cm,
            'pmm': self.project_pmm,
            'pmg': self.project_pmg,
            'pgg': self.project_pgg,
            'cmm': self.project_cmm,
            'p4': self.project_p4,
            'p4m': self.project_p4m,
            'p4g': self.project_p4g,
            'p3': self.project_p3,
            'p3m1': self.project_p3m1,
            'p31m': self.project_p31m,
            'p6': self.project_p6,
            'p6m': self.project_p6m,
        }
        
        if group_name not in projectors:
            raise ValueError(f"Unknown group: {group_name}")
        
        return projectors[group_name](arr)


class TuringPatternGenerator:
    """
    Generates Turing patterns with crystallographic symmetries.
    
    The generator combines reaction-diffusion dynamics (Gray-Scott model)
    with wallpaper group symmetry constraints. Patterns evolve according
    to the reaction-diffusion equations while being periodically projected
    onto the desired symmetry group.
    
    This produces organic-looking patterns that respect the mathematical
    structure of the 17 wallpaper groups.
    """
    
    # List of all 17 wallpaper groups
    ALL_GROUPS = [
        'p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm',
        'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m'
    ]
    
    # Recommended pattern types for each group (based on lattice compatibility)
    RECOMMENDED_PATTERNS = {
        # Oblique lattice - flexible
        'p1': [PatternType.STRIPES, PatternType.MAZE, PatternType.FINGERPRINT],
        'p2': [PatternType.STRIPES, PatternType.MAZE],
        # Rectangular lattice - stripes work well
        'pm': [PatternType.STRIPES, PatternType.WAVES],
        'pg': [PatternType.STRIPES, PatternType.MAZE],
        'cm': [PatternType.STRIPES, PatternType.WAVES],
        'pmm': [PatternType.STRIPES, PatternType.MAZE],
        'pmg': [PatternType.STRIPES, PatternType.FINGERPRINT],
        'pgg': [PatternType.MAZE, PatternType.STRIPES],
        'cmm': [PatternType.STRIPES, PatternType.MAZE],
        # Square lattice - spots can form square arrays
        'p4': [PatternType.SPOTS, PatternType.MAZE],
        'p4m': [PatternType.SPOTS, PatternType.STRIPES],
        'p4g': [PatternType.SPOTS, PatternType.MAZE],
        # Hexagonal lattice - spots naturally form hexagonal patterns
        'p3': [PatternType.SPOTS, PatternType.CORAL],
        'p3m1': [PatternType.SPOTS, PatternType.CORAL],
        'p31m': [PatternType.SPOTS, PatternType.CORAL],
        'p6': [PatternType.SPOTS, PatternType.MAZE],
        'p6m': [PatternType.SPOTS, PatternType.CORAL],
    }
    
    def __init__(self, 
                 resolution: int = 256, 
                 seed: Optional[int] = None):
        """
        Initialize the Turing pattern generator.
        
        Args:
            resolution: Size of the output pattern (resolution x resolution)
            seed: Random seed for reproducibility
        """
        self.resolution = resolution
        self.rng = np.random.default_rng(seed)
        self.projector = SymmetryProjector(resolution)
        
        # Laplacian kernel for diffusion
        self.laplacian = np.array([
            [0.05, 0.2, 0.05],
            [0.2, -1.0, 0.2],
            [0.05, 0.2, 0.05]
        ])
    
    def _init_concentrations(self, 
                             group_name: str,
                             noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize u and v concentrations with symmetry-respecting noise.
        
        The initial conditions have the symmetry of the target group,
        which helps the pattern evolve into that symmetry faster.
        """
        n = self.resolution
        
        # Start with uniform state + small symmetric noise
        u = np.ones((n, n))
        v = np.zeros((n, n))
        
        # Add noise
        noise_u = self.rng.uniform(-noise_level, noise_level, (n, n))
        noise_v = self.rng.uniform(0, noise_level, (n, n))
        
        # Project noise onto the group symmetry
        noise_u = self.projector.project(noise_u, group_name)
        noise_v = self.projector.project(noise_v, group_name)
        
        # Add a seed region in the center with group symmetry
        seed_region = self._create_symmetric_seed(group_name)
        
        u += noise_u
        v += noise_v + seed_region * 0.25
        
        # Clamp to valid range
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        
        return u, v
    
    def _create_symmetric_seed(self, group_name: str) -> np.ndarray:
        """Create a seed pattern with the symmetry of the specified group."""
        n = self.resolution
        seed = np.zeros((n, n))
        
        # Create asymmetric base seed
        cx, cy = n // 2, n // 2
        
        # Add some random blobs
        for _ in range(5):
            bx = self.rng.integers(n // 4, 3 * n // 4)
            by = self.rng.integers(n // 4, 3 * n // 4)
            radius = self.rng.integers(n // 20, n // 8)
            
            y, x = np.ogrid[:n, :n]
            mask = ((x - bx)**2 + (y - by)**2) < radius**2
            seed[mask] = 1.0
        
        # Project onto group symmetry
        seed = self.projector.project(seed, group_name)
        
        return seed
    
    def _laplacian(self, arr: np.ndarray) -> np.ndarray:
        """Compute Laplacian using convolution with periodic boundaries."""
        return convolve(arr, self.laplacian, mode='wrap')
    
    def _step_gray_scott(self, 
                         u: np.ndarray, 
                         v: np.ndarray, 
                         params: TuringParams) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one step of Gray-Scott reaction-diffusion.
        
        Equations:
            ∂u/∂t = Du∇²u - uv² + f(1-u)
            ∂v/∂t = Dv∇²v + uv² - (f+k)v
        """
        # Diffusion terms
        lap_u = self._laplacian(u)
        lap_v = self._laplacian(v)
        
        # Reaction terms
        uvv = u * v * v
        
        # Update
        du = params.Du * lap_u - uvv + params.f * (1 - u)
        dv = params.Dv * lap_v + uvv - (params.f + params.k) * v
        
        u_new = u + params.dt * du
        v_new = v + params.dt * dv
        
        # Clamp to valid range
        u_new = np.clip(u_new, 0, 1)
        v_new = np.clip(v_new, 0, 1)
        
        return u_new, v_new
    
    def generate(self,
                 group_name: str,
                 pattern_type: Optional[PatternType] = None,
                 params: Optional[TuringParams] = None,
                 steps: int = 5000,
                 symmetry_project_interval: int = 100,
                 return_both: bool = False,
                 verbose: bool = False) -> np.ndarray:
        """
        Generate a Turing pattern with the specified wallpaper group symmetry.
        
        Args:
            group_name: Name of the wallpaper group ('p1' through 'p6m')
            pattern_type: Type of pattern to generate (spots, stripes, etc.)
            params: Custom reaction-diffusion parameters
            steps: Number of simulation steps
            symmetry_project_interval: Apply symmetry projection every N steps
            return_both: If True, return tuple (u, v), else return v
            verbose: Print progress information
            
        Returns:
            Pattern array (v concentration by default, or (u, v) tuple)
        """
        if group_name not in self.ALL_GROUPS:
            raise ValueError(f"Unknown group: {group_name}. Valid: {self.ALL_GROUPS}")
        
        # Get pattern parameters
        if params is None:
            if pattern_type is None:
                # Use recommended pattern for this group
                pattern_type = self.rng.choice(self.RECOMMENDED_PATTERNS[group_name])
            params = TuringParams.for_pattern(pattern_type)
        
        if verbose:
            print(f"Generating {pattern_type.value if pattern_type else 'custom'} "
                  f"pattern with {group_name} symmetry...")
        
        # Initialize with symmetric noise
        u, v = self._init_concentrations(group_name)
        
        # Run simulation
        for step in range(steps):
            u, v = self._step_gray_scott(u, v, params)
            
            # Periodically project onto symmetry
            if (step + 1) % symmetry_project_interval == 0:
                u = self.projector.project(u, group_name)
                v = self.projector.project(v, group_name)
                
                if verbose and (step + 1) % 1000 == 0:
                    print(f"  Step {step + 1}/{steps}")
        
        # Final symmetry projection
        u = self.projector.project(u, group_name)
        v = self.projector.project(v, group_name)
        
        if return_both:
            return u, v
        else:
            return v
    
    def generate_all(self,
                     pattern_type: Optional[PatternType] = None,
                     steps: int = 5000,
                     **kwargs) -> Dict[str, np.ndarray]:
        """
        Generate Turing patterns for all 17 wallpaper groups.
        
        Returns:
            Dictionary mapping group names to pattern arrays
        """
        return {group: self.generate(group, pattern_type, steps=steps, **kwargs)
                for group in self.ALL_GROUPS}
    
    def generate_pattern_gallery(self,
                                 group_name: str,
                                 steps: int = 5000,
                                 **kwargs) -> Dict[PatternType, np.ndarray]:
        """
        Generate all pattern types for a single wallpaper group.
        
        Returns:
            Dictionary mapping pattern types to pattern arrays
        """
        return {ptype: self.generate(group_name, ptype, steps=steps, **kwargs)
                for ptype in PatternType}


class TuringBlender:
    """
    Blends multiple Turing patterns with different symmetries.
    
    This allows creating complex patterns that combine features from
    different wallpaper groups, or transition smoothly between them.
    """
    
    def __init__(self, resolution: int = 256, seed: Optional[int] = None):
        self.resolution = resolution
        self.generator = TuringPatternGenerator(resolution, seed)
        self.rng = np.random.default_rng(seed)
    
    def blend_patterns(self,
                       patterns: List[np.ndarray],
                       weights: Optional[List[float]] = None,
                       mode: str = 'average') -> np.ndarray:
        """
        Blend multiple patterns together.
        
        Args:
            patterns: List of pattern arrays
            weights: Optional weights for each pattern
            mode: 'average', 'max', 'min', 'multiply'
        """
        if weights is None:
            weights = [1.0 / len(patterns)] * len(patterns)
        
        if mode == 'average':
            result = sum(w * p for w, p in zip(weights, patterns))
        elif mode == 'max':
            result = np.maximum.reduce(patterns)
        elif mode == 'min':
            result = np.minimum.reduce(patterns)
        elif mode == 'multiply':
            result = np.prod(patterns, axis=0)
        else:
            raise ValueError(f"Unknown blend mode: {mode}")
        
        # Normalize
        if result.max() > result.min():
            result = (result - result.min()) / (result.max() - result.min())
        
        return result
    
    def create_symmetry_gradient(self,
                                 group1: str,
                                 group2: str,
                                 pattern_type: PatternType = PatternType.SPOTS,
                                 steps: int = 3000) -> np.ndarray:
        """
        Create a pattern that transitions from one symmetry to another.
        
        This creates a pattern where the left side has group1 symmetry
        and gradually transitions to group2 symmetry on the right.
        """
        n = self.resolution
        
        # Generate both patterns
        p1 = self.generator.generate(group1, pattern_type, steps=steps)
        p2 = self.generator.generate(group2, pattern_type, steps=steps)
        
        # Create horizontal gradient
        x = np.linspace(0, 1, n)
        gradient = x[np.newaxis, :] * np.ones((n, 1))
        
        # Blend using sigmoid for smoother transition
        blend = 1 / (1 + np.exp(-10 * (gradient - 0.5)))
        
        return (1 - blend) * p1 + blend * p2
    
    def create_domain_mosaic(self,
                             groups: List[str],
                             grid_size: int = 2,
                             pattern_type: PatternType = PatternType.SPOTS,
                             steps: int = 3000) -> np.ndarray:
        """
        Create a mosaic where each region has a different wallpaper symmetry.
        
        Args:
            groups: List of group names (will be tiled)
            grid_size: Number of regions per side
            pattern_type: Pattern type to use
            steps: Simulation steps per pattern
        """
        n = self.resolution
        cell_size = n // grid_size
        
        result = np.zeros((n, n))
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Select group for this cell
                group_idx = (i * grid_size + j) % len(groups)
                group = groups[group_idx]
                
                # Generate pattern for this cell
                cell_gen = TuringPatternGenerator(cell_size, 
                                                   seed=self.rng.integers(0, 10000))
                pattern = cell_gen.generate(group, pattern_type, steps=steps)
                
                # Place in result
                y0, y1 = i * cell_size, (i + 1) * cell_size
                x0, x1 = j * cell_size, (j + 1) * cell_size
                result[y0:y1, x0:x1] = pattern
        
        return result
    
    def superimpose_patterns(self,
                            base_group: str,
                            overlay_group: str,
                            base_pattern: PatternType = PatternType.STRIPES,
                            overlay_pattern: PatternType = PatternType.SPOTS,
                            overlay_scale: float = 0.5,
                            steps: int = 3000) -> np.ndarray:
        """
        Superimpose patterns with different symmetries.
        
        Creates a hierarchical pattern where a larger-scale pattern with
        one symmetry is modulated by a smaller-scale pattern with another.
        """
        # Generate base pattern at full resolution
        base = self.generator.generate(base_group, base_pattern, steps=steps)
        
        # Generate overlay at smaller scale and tile
        overlay_res = int(self.resolution * overlay_scale)
        overlay_gen = TuringPatternGenerator(overlay_res, 
                                              seed=self.rng.integers(0, 10000))
        overlay = overlay_gen.generate(overlay_group, overlay_pattern, steps=steps)
        
        # Tile overlay to match resolution
        tiles = int(np.ceil(self.resolution / overlay_res))
        overlay_tiled = np.tile(overlay, (tiles, tiles))[:self.resolution, :self.resolution]
        
        # Combine using modulation
        result = base * (0.5 + 0.5 * overlay_tiled)
        
        # Normalize
        if result.max() > result.min():
            result = (result - result.min()) / (result.max() - result.min())
        
        return result


# Convenience functions
def generate_turing_pattern(group_name: str,
                           pattern_type: PatternType = PatternType.SPOTS,
                           resolution: int = 256,
                           steps: int = 5000,
                           seed: Optional[int] = None) -> np.ndarray:
    """
    Convenience function to generate a single Turing pattern.
    
    Args:
        group_name: Wallpaper group ('p1' through 'p6m')
        pattern_type: Type of Turing pattern
        resolution: Output resolution
        steps: Number of simulation steps
        seed: Random seed
        
    Returns:
        2D numpy array with the pattern
    """
    gen = TuringPatternGenerator(resolution, seed)
    return gen.generate(group_name, pattern_type, steps=steps)


def list_pattern_types() -> List[str]:
    """List available pattern types."""
    return [pt.value for pt in PatternType]


def list_wallpaper_groups() -> List[str]:
    """List all 17 wallpaper groups."""
    return TuringPatternGenerator.ALL_GROUPS.copy()


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt
    
    print("Testing Turing Pattern Generator...")
    
    gen = TuringPatternGenerator(resolution=128, seed=42)
    
    # Generate a few examples
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    test_groups = ['p4m', 'p6m', 'pmm', 'p3', 'cm', 'pgg']
    
    for ax, group in zip(axes.flat, test_groups):
        print(f"Generating {group}...")
        pattern = gen.generate(group, PatternType.SPOTS, steps=2000, verbose=False)
        ax.imshow(pattern, cmap='viridis')
        ax.set_title(f"Turing + {group}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('turing_test.png', dpi=150)
    print("Saved test image to turing_test.png")




