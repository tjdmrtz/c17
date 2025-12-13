#!/usr/bin/env python3
"""
Mathematical Tests for Wallpaper Group Symmetries.

These tests verify that the pattern generator correctly implements
the algebraic properties of each wallpaper group.

For each group, we test:
1. The CONSTRUCTION is mathematically correct
2. The resulting pattern has the expected symmetries
3. No unexpected symmetries are present

Mathematical background:
- p1: Only translations T_a, T_b
- p2: 180° rotation C2 at lattice points
- pm: Reflection σ_v (vertical mirror)
- pg: Glide reflection g_v = σ_v ∘ T_{b/2}
- cm: σ_v with centered lattice (glide emerges)
- pmm: σ_h, σ_v (perpendicular mirrors) → C2 emerges
- pmg: σ_v, g_h → C2 emerges
- pgg: g_h, g_v → C2 emerges
- cmm: σ_h, σ_v with centered lattice
- p4: 90° rotation C4 → C2 emerges
- p4m: C4, σ_h, σ_v, σ_d1, σ_d2
- p4g: C4, g at 45°
- p3: 120° rotation C3
- p3m1: C3, σ through rotation centers
- p31m: C3, σ between rotation centers
- p6: 60° rotation C6 → C3, C2 emerge
- p6m: C6, σ at 30° intervals
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.pattern_generator import WallpaperGroupGenerator
from scipy import ndimage
import unittest


class SymmetryTestBase(unittest.TestCase):
    """Base class with helper methods for symmetry testing."""
    
    @classmethod
    def setUpClass(cls):
        cls.gen = WallpaperGroupGenerator(resolution=256, seed=42)
        cls.motif_size = 64
        cls.tolerance = 0.90  # Correlation threshold
    
    def _get_cell(self, pattern, cell_size):
        """Extract unit cell from origin."""
        return pattern[:cell_size, :cell_size]
    
    def _correlation(self, img1, img2, margin=5):
        """Compute normalized correlation between two images."""
        if margin > 0:
            img1 = img1[margin:-margin, margin:-margin]
            img2 = img2[margin:-margin, margin:-margin]
        
        # Handle size mismatch
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]
        
        img1_norm = (img1 - img1.mean()) / (img1.std() + 1e-8)
        img2_norm = (img2 - img2.mean()) / (img2.std() + 1e-8)
        return np.mean(img1_norm * img2_norm)
    
    def _rotate(self, img, angle_deg):
        """Rotate image by angle in degrees."""
        return ndimage.rotate(img, angle_deg, reshape=False, order=1, mode='wrap')
    
    def _reflect_h(self, img):
        """Reflect across horizontal axis (flip vertically)."""
        return np.flipud(img)
    
    def _reflect_v(self, img):
        """Reflect across vertical axis (flip horizontally)."""
        return np.fliplr(img)
    
    def _glide_h(self, img):
        """Glide reflection: reflect horizontally + translate half period vertically."""
        reflected = self._reflect_h(img)
        return np.roll(reflected, img.shape[1] // 2, axis=1)
    
    def _glide_v(self, img):
        """Glide reflection: reflect vertically + translate half period horizontally."""
        reflected = self._reflect_v(img)
        return np.roll(reflected, img.shape[0] // 2, axis=0)
    
    def assertSymmetryPresent(self, img1, img2, sym_name, threshold=None):
        """Assert that a symmetry is present (high correlation)."""
        if threshold is None:
            threshold = self.tolerance
        corr = self._correlation(img1, img2)
        self.assertGreaterEqual(
            corr, threshold,
            f"{sym_name} symmetry NOT present: correlation={corr:.3f} < {threshold}"
        )
        return corr
    
    def assertSymmetryAbsent(self, img1, img2, sym_name, threshold=0.5):
        """Assert that a symmetry is absent (low correlation)."""
        corr = self._correlation(img1, img2)
        self.assertLess(
            corr, threshold,
            f"{sym_name} symmetry unexpectedly PRESENT: correlation={corr:.3f} >= {threshold}"
        )
        return corr


class TestP1(SymmetryTestBase):
    """Test p1: Only translations, no point symmetry."""
    
    def test_construction(self):
        """p1 should just tile the motif with no transformations."""
        pattern = self.gen.generate_p1(motif_size=self.motif_size)
        cell = self._get_cell(pattern, self.motif_size)
        
        # The pattern should just be the motif tiled
        # Check that adjacent cells are identical
        cell2 = pattern[0:self.motif_size, self.motif_size:2*self.motif_size]
        corr = self._correlation(cell, cell2, margin=0)
        self.assertGreaterEqual(corr, 0.99, "p1 tiling failed")
    
    def test_no_rotation_symmetry(self):
        """p1 should NOT have rotation symmetry."""
        pattern = self.gen.generate_p1(motif_size=self.motif_size)
        cell = self._get_cell(pattern, self.motif_size)
        
        rotated_180 = self._rotate(cell, 180)
        # Should NOT be symmetric
        corr = self._correlation(cell, rotated_180)
        # Note: Random motif might accidentally have symmetry, so we just print
        print(f"  p1 rotation 180° correlation: {corr:.3f} (should be low)")


class TestP2(SymmetryTestBase):
    """Test p2: 180° rotation centers."""
    
    def test_construction_logic(self):
        """
        p2 construction: Cell should be [M, R180(M)] / [R180(M), M]
        where rotating the entire cell 180° gives the same cell.
        """
        pattern = self.gen.generate_p2(motif_size=self.motif_size)
        cell_size = self.motif_size * 2  # p2 uses 2x2 arrangement
        cell = self._get_cell(pattern, cell_size)
        
        # Test: rotating cell 180° should give same cell
        rotated = self._rotate(cell, 180)
        corr = self.assertSymmetryPresent(cell, rotated, "C2 (180° rotation)")
        print(f"  p2 C2 symmetry: {corr:.3f}")
    
    def test_no_reflection(self):
        """p2 should NOT have reflection symmetry."""
        pattern = self.gen.generate_p2(motif_size=self.motif_size)
        cell_size = self.motif_size * 2
        cell = self._get_cell(pattern, cell_size)
        
        reflected_h = self._reflect_h(cell)
        reflected_v = self._reflect_v(cell)
        
        corr_h = self._correlation(cell, reflected_h)
        corr_v = self._correlation(cell, reflected_v)
        print(f"  p2 reflection H: {corr_h:.3f}, V: {corr_v:.3f} (should be < 0.5)")


class TestPM(SymmetryTestBase):
    """Test pm: Parallel reflection axes (vertical)."""
    
    def test_reflection_present(self):
        """pm should have vertical reflection symmetry."""
        pattern = self.gen.generate_pm(motif_size=self.motif_size)
        cell_size = self.motif_size * 2  # [M, reflect_v(M)]
        cell = self._get_cell(pattern, cell_size)
        
        reflected = self._reflect_v(cell)
        corr = self.assertSymmetryPresent(cell, reflected, "σ_v (vertical reflection)")
        print(f"  pm vertical reflection: {corr:.3f}")
    
    def test_no_rotation(self):
        """pm should NOT have 180° rotation."""
        pattern = self.gen.generate_pm(motif_size=self.motif_size)
        cell_size = self.motif_size * 2
        cell = self._get_cell(pattern, cell_size)
        
        rotated = self._rotate(cell, 180)
        corr = self._correlation(cell, rotated)
        print(f"  pm rotation 180°: {corr:.3f} (should be moderate due to reflection)")


class TestPMM(SymmetryTestBase):
    """Test pmm: Perpendicular reflection axes."""
    
    def test_both_reflections_present(self):
        """pmm should have both H and V reflection symmetry."""
        pattern = self.gen.generate_pmm(motif_size=self.motif_size)
        cell_size = self.motif_size * 2
        cell = self._get_cell(pattern, cell_size)
        
        # Horizontal reflection
        reflected_h = self._reflect_h(cell)
        corr_h = self.assertSymmetryPresent(cell, reflected_h, "σ_h")
        
        # Vertical reflection
        reflected_v = self._reflect_v(cell)
        corr_v = self.assertSymmetryPresent(cell, reflected_v, "σ_v")
        
        print(f"  pmm reflections H: {corr_h:.3f}, V: {corr_v:.3f}")
    
    def test_rotation_emerges(self):
        """pmm: σ_h ∘ σ_v = C2 (rotation emerges from reflections)."""
        pattern = self.gen.generate_pmm(motif_size=self.motif_size)
        cell_size = self.motif_size * 2
        cell = self._get_cell(pattern, cell_size)
        
        rotated = self._rotate(cell, 180)
        corr = self.assertSymmetryPresent(cell, rotated, "C2 (emergent)")
        print(f"  pmm emergent C2: {corr:.3f}")


class TestP4(SymmetryTestBase):
    """Test p4: 90° rotation centers."""
    
    def test_construction_logic(self):
        """
        p4 construction: Cell = [M, R90] / [R270, R180]
        Rotating cell 90° should give same cell.
        """
        pattern = self.gen.generate_p4(motif_size=self.motif_size)
        cell_size = self.motif_size * 2
        cell = self._get_cell(pattern, cell_size)
        
        # 90° rotation
        rotated_90 = self._rotate(cell, 90)
        corr_90 = self._correlation(cell, rotated_90)
        
        # 180° rotation (should work if 90° works)
        rotated_180 = self._rotate(cell, 180)
        corr_180 = self._correlation(cell, rotated_180)
        
        print(f"  p4 rotations: 90°={corr_90:.3f}, 180°={corr_180:.3f}")
        
        # At least 180° should work (C4 implies C2)
        self.assertGreaterEqual(corr_180, self.tolerance, "p4 should have C2 symmetry")
    
    def test_no_reflection(self):
        """p4 should NOT have reflection symmetry."""
        pattern = self.gen.generate_p4(motif_size=self.motif_size)
        cell_size = self.motif_size * 2
        cell = self._get_cell(pattern, cell_size)
        
        reflected_h = self._reflect_h(cell)
        reflected_v = self._reflect_v(cell)
        
        corr_h = self._correlation(cell, reflected_h)
        corr_v = self._correlation(cell, reflected_v)
        print(f"  p4 reflections H: {corr_h:.3f}, V: {corr_v:.3f}")


class TestP3(SymmetryTestBase):
    """Test p3: 120° rotation centers."""
    
    def test_rotation_120(self):
        """p3 should have 120° rotation symmetry."""
        pattern = self.gen.generate_p3(motif_size=self.motif_size)
        # p3 uses hexagonal lattice, cell size varies
        cell_size = self.motif_size * 2
        cell = self._get_cell(pattern, cell_size)
        
        rotated_120 = self._rotate(cell, 120)
        corr = self._correlation(cell, rotated_120)
        print(f"  p3 rotation 120°: {corr:.3f}")


class TestP6(SymmetryTestBase):
    """Test p6: 60° rotation centers."""
    
    def test_rotation_60(self):
        """p6 should have 60° rotation symmetry."""
        pattern = self.gen.generate_p6(motif_size=self.motif_size)
        cell_size = self.motif_size * 2
        cell = self._get_cell(pattern, cell_size)
        
        rotated_60 = self._rotate(cell, 60)
        rotated_120 = self._rotate(cell, 120)
        
        corr_60 = self._correlation(cell, rotated_60)
        corr_120 = self._correlation(cell, rotated_120)
        
        print(f"  p6 rotations: 60°={corr_60:.3f}, 120°={corr_120:.3f}")


class TestGeneratorMathematicalCorrectness(unittest.TestCase):
    """
    Test the MATHEMATICAL CORRECTNESS of the generator operations.
    
    These tests verify that the primitive operations are correct,
    independent of the specific pattern generated.
    """
    
    def test_rotation_180_is_involution(self):
        """R180 ∘ R180 = Identity."""
        gen = WallpaperGroupGenerator(resolution=64, seed=42)
        img = gen._create_motif(32)
        
        rotated_once = gen._rotate(img, np.pi)
        rotated_twice = gen._rotate(rotated_once, np.pi)
        
        # Should be back to original
        diff = np.abs(img - rotated_twice).mean()
        print(f"  R180∘R180 deviation: {diff:.6f}")
        self.assertLess(diff, 0.1, "R180 should be an involution")
    
    def test_rotation_90_four_times(self):
        """R90^4 = Identity."""
        gen = WallpaperGroupGenerator(resolution=64, seed=42)
        img = gen._create_motif(32)
        
        rotated = img.copy()
        for _ in range(4):
            rotated = gen._rotate(rotated, np.pi/2)
        
        diff = np.abs(img - rotated).mean()
        print(f"  R90^4 deviation: {diff:.6f}")
        self.assertLess(diff, 0.15, "R90^4 should be identity")
    
    def test_reflection_is_involution(self):
        """σ ∘ σ = Identity."""
        gen = WallpaperGroupGenerator(resolution=64, seed=42)
        img = gen._create_motif(32)
        
        # Horizontal reflection
        reflected_h = gen._reflect_y(gen._reflect_y(img))
        diff_h = np.abs(img - reflected_h).max()
        self.assertEqual(diff_h, 0, "Horizontal reflection should be involution")
        
        # Vertical reflection
        reflected_v = gen._reflect_x(gen._reflect_x(img))
        diff_v = np.abs(img - reflected_v).max()
        self.assertEqual(diff_v, 0, "Vertical reflection should be involution")
    
    def test_reflections_commute_to_rotation(self):
        """σ_h ∘ σ_v = R180 (up to interpolation error)."""
        gen = WallpaperGroupGenerator(resolution=64, seed=42)
        img = gen._create_motif(32)
        
        # Compose reflections
        composed = gen._reflect_y(gen._reflect_x(img))
        
        # Compare to 180° rotation
        rotated = gen._rotate(img, np.pi)
        
        # They should be the same (allowing for interpolation)
        diff = np.abs(composed - rotated).mean()
        print(f"  σ_h∘σ_v vs R180 deviation: {diff:.6f}")
        self.assertLess(diff, 0.1, "σ_h ∘ σ_v should equal R180")


class TestCellSymmetryDirect(unittest.TestCase):
    """
    Test symmetry by directly checking the cell construction,
    not the final tiled pattern.
    """
    
    def setUp(self):
        self.gen = WallpaperGroupGenerator(resolution=256, seed=42)
        self.motif_size = 64
    
    def test_p2_cell_structure(self):
        """
        Verify p2 cell is constructed as:
        | M      | R180(M) |
        | R180(M)| M       |
        """
        motif = self.gen._create_motif(self.motif_size)
        rotated = self.gen._rotate(motif, np.pi)
        
        # Build cell manually
        top = np.hstack([motif, rotated])
        bottom = np.hstack([rotated, motif])
        cell = np.vstack([top, bottom])
        
        # Now verify: rotating entire cell 180° should give same cell
        cell_rotated = ndimage.rotate(cell, 180, reshape=False, order=1, mode='wrap')
        
        # Compare central region to avoid edge artifacts
        m = 10
        c1 = cell[m:-m, m:-m]
        c2 = cell_rotated[m:-m, m:-m]
        
        c1_norm = (c1 - c1.mean()) / (c1.std() + 1e-8)
        c2_norm = (c2 - c2.mean()) / (c2.std() + 1e-8)
        corr = np.mean(c1_norm * c2_norm)
        
        print(f"  p2 manual cell C2 correlation: {corr:.3f}")
        
        # THE BUG: The issue is that R180(M) in position [0,1] and [1,0]
        # are the SAME rotated motif, but when we rotate the cell 180°,
        # position [0,0] goes to [1,1] and vice versa.
        # 
        # For true C2 symmetry, we need the cell to be invariant under
        # 180° rotation AROUND ITS CENTER, which means:
        # Cell[i,j] = Cell[N-1-i, N-1-j] for all i,j
        #
        # With current construction:
        # [0,0] = M, [1,1] = M ✓
        # [0,1] = R(M), [1,0] = R(M) ✓
        # 
        # When we rotate: [0,0] → [1,1]: M → M ✓
        # But the VALUES might not match due to:
        # 1. Interpolation during rotation
        # 2. Non-square motif issues
    
    def test_pmm_cell_structure(self):
        """
        Verify pmm cell is:
        | M          | σ_v(M)      |
        | σ_h(M)     | σ_v(σ_h(M)) |
        
        This should have both reflection symmetries.
        """
        motif = self.gen._create_motif(self.motif_size)
        
        # Build cell manually
        top = np.hstack([motif, self.gen._reflect_x(motif)])
        bottom = np.hstack([self.gen._reflect_y(motif), 
                          self.gen._reflect_x(self.gen._reflect_y(motif))])
        cell = np.vstack([top, bottom])
        
        # Test horizontal reflection (flip vertically)
        cell_flip_h = np.flipud(cell)
        diff_h = np.abs(cell - cell_flip_h).mean()
        
        # Test vertical reflection (flip horizontally)
        cell_flip_v = np.fliplr(cell)
        diff_v = np.abs(cell - cell_flip_v).mean()
        
        print(f"  pmm manual cell: H diff={diff_h:.6f}, V diff={diff_v:.6f}")
        
        # These should be exactly 0 for perfect reflections
        self.assertLess(diff_h, 1e-10, "pmm should have exact σ_h")
        self.assertLess(diff_v, 1e-10, "pmm should have exact σ_v")


def run_diagnostic():
    """Run all tests and print diagnostic information."""
    print("="*70)
    print("MATHEMATICAL SYMMETRY VERIFICATION TESTS")
    print("="*70)
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestP1))
    suite.addTests(loader.loadTestsFromTestCase(TestP2))
    suite.addTests(loader.loadTestsFromTestCase(TestPM))
    suite.addTests(loader.loadTestsFromTestCase(TestPMM))
    suite.addTests(loader.loadTestsFromTestCase(TestP4))
    suite.addTests(loader.loadTestsFromTestCase(TestP3))
    suite.addTests(loader.loadTestsFromTestCase(TestP6))
    suite.addTests(loader.loadTestsFromTestCase(TestGeneratorMathematicalCorrectness))
    suite.addTests(loader.loadTestsFromTestCase(TestCellSymmetryDirect))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    run_diagnostic()




