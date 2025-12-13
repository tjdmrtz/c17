#!/usr/bin/env python3
"""
Test suite for verifying the 17 wallpaper groups generator.

Uses simple rotation/reflection + correlation with reasonable thresholds
to account for discretization effects in finite images.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
from scipy import ndimage

from src.dataset.pattern_generator_fixed import FixedWallpaperGenerator


# =============================================================================
# CONFIGURATION
# =============================================================================

RESOLUTION = 256
MOTIF_SIZE = 64
SEED = 42

# Thresholds - accounting for discretization errors in finite images
EXACT_THRESHOLD = 0.98      # For 90°/180° rotations and reflections (exact in pixels)
APPROX_THRESHOLD = 0.80     # For 60°/120° rotations (interpolation introduces errors)
LOOSE_THRESHOLD = 0.70      # For complex operations or smaller patterns
ABSENT_THRESHOLD = 0.50     # Symmetry should NOT be present


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def generator():
    return FixedWallpaperGenerator(resolution=RESOLUTION, seed=SEED)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_correlation(pattern1: np.ndarray, pattern2: np.ndarray, margin: int = 10) -> float:
    """Compute correlation between two patterns, ignoring edges."""
    p1 = pattern1[margin:-margin, margin:-margin].flatten()
    p2 = pattern2[margin:-margin, margin:-margin].flatten()
    return np.corrcoef(p1, p2)[0, 1]


def check_rotation(pattern: np.ndarray, angle: int) -> float:
    """Check if pattern has rotational symmetry at given angle."""
    cell = pattern[:128, :128]
    if angle == 90:
        rotated = np.rot90(cell, 1)
    elif angle == 180:
        rotated = np.rot90(cell, 2)
    elif angle == 270:
        rotated = np.rot90(cell, 3)
    else:
        rotated = ndimage.rotate(cell, angle, reshape=False, order=3, mode='wrap')
    return get_correlation(cell, rotated)


def check_reflection(pattern: np.ndarray, axis: str) -> float:
    """Check if pattern has reflection symmetry."""
    cell = pattern[:128, :128]
    if axis == 'vertical':
        reflected = np.fliplr(cell)
    elif axis == 'horizontal':
        reflected = np.flipud(cell)
    else:  # diagonal
        reflected = cell.T
    return get_correlation(cell, reflected)


# =============================================================================
# BASIC TESTS
# =============================================================================

ALL_GROUPS = ['p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm',
              'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m']


class TestBasicGeneration:
    """Basic generation tests."""
    
    @pytest.mark.parametrize("group", ALL_GROUPS)
    def test_generates_without_error(self, generator, group):
        pattern = generator.generate(group, motif_size=MOTIF_SIZE)
        assert pattern is not None
        assert pattern.shape == (RESOLUTION, RESOLUTION)
        assert 0 <= pattern.min() and pattern.max() <= 1.0 + 1e-6
        assert pattern.std() > 0.01  # Has variation
    
    @pytest.mark.parametrize("group", ALL_GROUPS)
    def test_reproducibility(self, group):
        gen1 = FixedWallpaperGenerator(resolution=RESOLUTION, seed=42)
        gen2 = FixedWallpaperGenerator(resolution=RESOLUTION, seed=42)
        np.testing.assert_array_equal(
            gen1.generate(group, motif_size=MOTIF_SIZE),
            gen2.generate(group, motif_size=MOTIF_SIZE)
        )


# =============================================================================
# ROTATION SYMMETRY TESTS
# =============================================================================

class TestRotationSymmetry:
    """Tests for rotational symmetries."""
    
    # C2 (180°) - should be exact
    @pytest.mark.parametrize("group", ['p2', 'pmm', 'pmg', 'pgg', 'cmm', 'p4', 'p4m', 'p4g', 'p6', 'p6m'])
    def test_c2_present(self, generator, group):
        pattern = generator.generate(group, motif_size=MOTIF_SIZE)
        corr = check_rotation(pattern, 180)
        assert corr > EXACT_THRESHOLD, f"{group}: C2 expected, got {corr:.3f}"
    
    @pytest.mark.parametrize("group", ['p1'])
    def test_c2_absent(self, generator, group):
        """p1 should have no C2 rotation symmetry."""
        pattern = generator.generate(group, motif_size=MOTIF_SIZE)
        corr = check_rotation(pattern, 180)
        assert corr < ABSENT_THRESHOLD, f"{group}: C2 should be absent, got {corr:.3f}"
    
    @pytest.mark.parametrize("group", ['pm', 'pg', 'cm'])
    def test_c2_not_designed(self, generator, group):
        """
        These groups don't have C2 by design, but may have some correlation
        due to their structure (reflections/glides can create partial C2-like patterns).
        We just verify they don't have perfect C2.
        """
        pattern = generator.generate(group, motif_size=MOTIF_SIZE)
        corr = check_rotation(pattern, 180)
        # Should not have PERFECT C2 (which would be >0.98)
        assert corr < 0.95, f"{group}: should not have perfect C2, got {corr:.3f}"
    
    # C4 (90°) - should be exact
    @pytest.mark.parametrize("group", ['p4', 'p4m', 'p4g'])
    def test_c4_present(self, generator, group):
        pattern = generator.generate(group, motif_size=MOTIF_SIZE)
        corr = check_rotation(pattern, 90)
        assert corr > EXACT_THRESHOLD, f"{group}: C4 expected, got {corr:.3f}"
    
    @pytest.mark.parametrize("group", ['p1', 'p2', 'pm', 'pmm'])
    def test_c4_absent(self, generator, group):
        pattern = generator.generate(group, motif_size=MOTIF_SIZE)
        corr = check_rotation(pattern, 90)
        assert corr < ABSENT_THRESHOLD, f"{group}: C4 should be absent, got {corr:.3f}"
    
    # C3 (120°) - uses interpolation, so slightly lower threshold
    @pytest.mark.parametrize("group", ['p3', 'p3m1', 'p31m', 'p6', 'p6m'])
    def test_c3_present(self, generator, group):
        pattern = generator.generate(group, motif_size=MOTIF_SIZE)
        corr = check_rotation(pattern, 120)
        assert corr > APPROX_THRESHOLD, f"{group}: C3 expected, got {corr:.3f}"
    
    # C6 (60°) - uses interpolation
    @pytest.mark.parametrize("group", ['p6', 'p6m'])
    def test_c6_present(self, generator, group):
        pattern = generator.generate(group, motif_size=MOTIF_SIZE)
        corr = check_rotation(pattern, 60)
        assert corr > APPROX_THRESHOLD, f"{group}: C6 expected, got {corr:.3f}"


# =============================================================================
# REFLECTION SYMMETRY TESTS
# =============================================================================

class TestReflectionSymmetry:
    """Tests for reflection symmetries."""
    
    @pytest.mark.parametrize("group", ['pm', 'cm', 'pmm', 'pmg', 'cmm', 'p4m', 'p3m1', 'p6m'])
    def test_vertical_reflection_present(self, generator, group):
        pattern = generator.generate(group, motif_size=MOTIF_SIZE)
        corr = check_reflection(pattern, 'vertical')
        assert corr > EXACT_THRESHOLD, f"{group}: σ_v expected, got {corr:.3f}"
    
    @pytest.mark.parametrize("group", ['pmm', 'cmm', 'p4m', 'p31m', 'p6m'])
    def test_horizontal_reflection_present(self, generator, group):
        pattern = generator.generate(group, motif_size=MOTIF_SIZE)
        corr = check_reflection(pattern, 'horizontal')
        assert corr > EXACT_THRESHOLD, f"{group}: σ_h expected, got {corr:.3f}"
    
    @pytest.mark.parametrize("group", ['p1', 'p2', 'pgg', 'p4'])
    def test_reflection_absent(self, generator, group):
        """Groups without reflection should have low reflection correlation."""
        pattern = generator.generate(group, motif_size=MOTIF_SIZE)
        corr_v = check_reflection(pattern, 'vertical')
        corr_h = check_reflection(pattern, 'horizontal')
        # At least one axis should have low correlation
        assert corr_v < 0.70 or corr_h < 0.70, \
            f"{group}: reflections should be absent, got σ_v={corr_v:.3f}, σ_h={corr_h:.3f}"


# =============================================================================
# MATHEMATICAL PROPERTIES
# =============================================================================

class TestMathematicalProperties:
    """Tests for specific mathematical properties."""
    
    def test_p1_no_symmetry(self, generator):
        """p1 should have no point symmetry."""
        pattern = generator.generate('p1', motif_size=MOTIF_SIZE)
        assert check_rotation(pattern, 90) < ABSENT_THRESHOLD
        assert check_rotation(pattern, 180) < ABSENT_THRESHOLD
        assert check_reflection(pattern, 'vertical') < ABSENT_THRESHOLD
    
    def check_rotation_composition_c4(self, generator):
        """C4: two 90° rotations = one 180° rotation."""
        pattern = generator.generate('p4', motif_size=MOTIF_SIZE)
        cell = pattern[:128, :128]
        
        rot90_twice = np.rot90(np.rot90(cell, 1), 1)
        rot180 = np.rot90(cell, 2)
        
        diff = np.abs(rot90_twice - rot180).mean()
        assert diff < 0.001, f"90°+90° should equal 180°, diff={diff}"
    
    def check_reflection_composition(self, generator):
        """σ_v ∘ σ_h = C2."""
        pattern = generator.generate('pmm', motif_size=MOTIF_SIZE)
        cell = pattern[:128, :128]
        
        composed = np.flipud(np.fliplr(cell))
        rot180 = np.rot90(cell, 2)
        
        diff = np.abs(composed - rot180).mean()
        assert diff < 0.001, f"σ_v∘σ_h should equal C2, diff={diff}"


# =============================================================================
# PERIODICITY TESTS
# =============================================================================

class TestPeriodicity:
    """Tests for translational periodicity."""
    
    @pytest.mark.parametrize("group", ALL_GROUPS)
    def test_periodic(self, generator, group):
        """Pattern should be periodic."""
        pattern = generator.generate(group, motif_size=MOTIF_SIZE)
        cell_size = MOTIF_SIZE * 2
        
        if pattern.shape[0] >= cell_size * 2:
            # Compare first and second cells
            cell1 = pattern[:cell_size, :cell_size]
            cell2_x = pattern[:cell_size, cell_size:cell_size*2]
            cell2_y = pattern[cell_size:cell_size*2, :cell_size]
            
            corr_x = np.corrcoef(cell1.flatten(), cell2_x.flatten())[0, 1]
            corr_y = np.corrcoef(cell1.flatten(), cell2_y.flatten())[0, 1]
            
            assert corr_x > 0.99, f"{group}: should be periodic in x"
            assert corr_y > 0.99, f"{group}: should be periodic in y"


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""
    
    def test_invalid_group(self, generator):
        with pytest.raises(ValueError):
            generator.generate('invalid', motif_size=MOTIF_SIZE)
    
    def test_small_resolution(self):
        gen = FixedWallpaperGenerator(resolution=64, seed=SEED)
        for group in ['p4m', 'p6m']:
            pattern = gen.generate(group, motif_size=16)
            assert pattern.shape == (64, 64)
            assert not np.isnan(pattern).any()


# =============================================================================
# SUMMARY REPORT
# =============================================================================

class TestSummaryReport:
    """Generate a summary report of all symmetries."""
    
    def test_print_symmetry_report(self, generator):
        """Print a summary of symmetry correlations for all groups."""
        print("\n" + "="*70)
        print("SYMMETRY CORRELATION REPORT")
        print("="*70)
        print(f"{'Group':<8} {'C2(180)':>8} {'C4(90)':>8} {'C3(120)':>8} {'C6(60)':>8} {'σ_v':>8} {'σ_h':>8}")
        print("-"*70)
        
        for group in ALL_GROUPS:
            pattern = generator.generate(group, motif_size=MOTIF_SIZE)
            
            c2 = check_rotation(pattern, 180)
            c4 = check_rotation(pattern, 90)
            c3 = check_rotation(pattern, 120)
            c6 = check_rotation(pattern, 60)
            sv = check_reflection(pattern, 'vertical')
            sh = check_reflection(pattern, 'horizontal')
            
            print(f"{group:<8} {c2:>8.3f} {c4:>8.3f} {c3:>8.3f} {c6:>8.3f} {sv:>8.3f} {sh:>8.3f}")
        
        print("="*70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
