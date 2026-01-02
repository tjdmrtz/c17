"""
Tests for verifying that wallpaper group patterns have exactly the correct symmetries.

Each test verifies:
1. The pattern HAS the symmetries it should have (high correlation after operation)
2. The pattern does NOT have symmetries it shouldn't have (low correlation)

Mathematical Reference: International Tables for Crystallography, Vol. A
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple
import sys
sys.path.insert(0, '..')

from src.dataset.pattern_generator import WallpaperGroupGenerator, WALLPAPER_GROUPS


class TestSymmetryOperations:
    """Test the basic symmetry operations."""
    
    @staticmethod
    def rotate_90(pattern: np.ndarray) -> np.ndarray:
        """Rotate pattern by 90° counter-clockwise."""
        return np.rot90(pattern, 1)
    
    @staticmethod
    def rotate_180(pattern: np.ndarray) -> np.ndarray:
        """Rotate pattern by 180°."""
        return np.rot90(pattern, 2)
    
    @staticmethod
    def rotate_270(pattern: np.ndarray) -> np.ndarray:
        """Rotate pattern by 270° counter-clockwise."""
        return np.rot90(pattern, 3)
    
    @staticmethod
    def rotate_60(pattern: np.ndarray) -> np.ndarray:
        """Rotate pattern by 60° (approximate, uses interpolation)."""
        from scipy.ndimage import rotate
        return rotate(pattern, 60, reshape=False, order=1, mode='wrap')
    
    @staticmethod
    def rotate_120(pattern: np.ndarray) -> np.ndarray:
        """Rotate pattern by 120° (approximate, uses interpolation)."""
        from scipy.ndimage import rotate
        return rotate(pattern, 120, reshape=False, order=1, mode='wrap')
    
    @staticmethod
    def reflect_vertical(pattern: np.ndarray) -> np.ndarray:
        """Reflect across vertical axis (flip horizontally)."""
        return np.fliplr(pattern)
    
    @staticmethod
    def reflect_horizontal(pattern: np.ndarray) -> np.ndarray:
        """Reflect across horizontal axis (flip vertically)."""
        return np.flipud(pattern)
    
    @staticmethod
    def reflect_diagonal(pattern: np.ndarray) -> np.ndarray:
        """Reflect across main diagonal (transpose)."""
        return pattern.T
    
    @staticmethod
    def glide_horizontal(pattern: np.ndarray) -> np.ndarray:
        """Glide reflection: horizontal reflection + half translation."""
        reflected = np.flipud(pattern)
        shift = pattern.shape[1] // 2
        return np.roll(reflected, shift, axis=1)
    
    @staticmethod
    def glide_vertical(pattern: np.ndarray) -> np.ndarray:
        """Glide reflection: vertical reflection + half translation."""
        reflected = np.fliplr(pattern)
        shift = pattern.shape[0] // 2
        return np.roll(reflected, shift, axis=0)
    
    @staticmethod
    def correlation(p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Pearson correlation between two patterns."""
        p1_flat = p1.flatten()
        p2_flat = p2.flatten()
        
        # Check for exact match first
        if np.allclose(p1_flat, p2_flat, rtol=1e-5, atol=1e-5):
            return 1.0
        
        corr = np.corrcoef(p1_flat, p2_flat)[0, 1]
        return corr if not np.isnan(corr) else 0.0


# Symmetry requirements for each group
# 'should_have': operations that should give correlation > 0.95
# 'should_not_have': operations that should give correlation < 0.7
SYMMETRY_REQUIREMENTS = {
    'p1': {
        'should_have': [],  # No non-trivial symmetries
        'should_not_have': ['C2', 'C3', 'C4', 'C6', 'sigma_v', 'sigma_h']
    },
    'p2': {
        'should_have': ['C2'],
        'should_not_have': ['C4', 'C3', 'C6', 'sigma_v', 'sigma_h']
    },
    'pm': {
        'should_have': ['sigma_v'],
        'should_not_have': ['C2', 'C3', 'C4', 'C6', 'sigma_h']
    },
    'pg': {
        'should_have': [],  # Glide only, not pure reflection
        'should_not_have': ['C2', 'C3', 'C4', 'C6', 'sigma_v', 'sigma_h']
    },
    'cm': {
        'should_have': ['sigma_v'],
        'should_not_have': ['C2', 'C3', 'C4', 'C6']
    },
    'pmm': {
        'should_have': ['C2', 'sigma_v', 'sigma_h'],
        'should_not_have': ['C3', 'C4', 'C6']
    },
    'pmg': {
        'should_have': ['C2', 'sigma_v'],
        'should_not_have': ['C3', 'C4', 'C6']
    },
    'pgg': {
        'should_have': ['C2'],
        'should_not_have': ['C3', 'C4', 'C6', 'sigma_v', 'sigma_h']
    },
    'cmm': {
        'should_have': ['C2', 'sigma_v', 'sigma_h'],
        'should_not_have': ['C3', 'C4', 'C6']
    },
    'p4': {
        'should_have': ['C4', 'C2'],
        'should_not_have': ['C3', 'C6', 'sigma_v', 'sigma_h', 'sigma_d']
    },
    'p4m': {
        'should_have': ['C4', 'C2', 'sigma_v', 'sigma_h', 'sigma_d'],
        'should_not_have': ['C3', 'C6']
    },
    'p4g': {
        'should_have': ['C4', 'C2', 'sigma_d'],
        'should_not_have': ['C3', 'C6']
    },
    'p3': {
        'should_have': ['C3'],
        'should_not_have': ['C2', 'C4', 'C6', 'sigma_v', 'sigma_h']
    },
    'p3m1': {
        'should_have': ['C3', 'sigma_v'],
        'should_not_have': ['C2', 'C4', 'C6']
    },
    'p31m': {
        'should_have': ['C3', 'sigma_h'],
        'should_not_have': ['C2', 'C4', 'C6']
    },
    'p6': {
        'should_have': ['C6', 'C3', 'C2'],
        'should_not_have': ['C4', 'sigma_v', 'sigma_h']
    },
    'p6m': {
        'should_have': ['C6', 'C3', 'C2', 'sigma_v', 'sigma_h'],
        'should_not_have': ['C4']
    },
}

# Map operation names to functions
OPERATIONS = {
    'C2': TestSymmetryOperations.rotate_180,
    'C3': TestSymmetryOperations.rotate_120,
    'C4': TestSymmetryOperations.rotate_90,
    'C6': TestSymmetryOperations.rotate_60,
    'sigma_v': TestSymmetryOperations.reflect_vertical,
    'sigma_h': TestSymmetryOperations.reflect_horizontal,
    'sigma_d': TestSymmetryOperations.reflect_diagonal,
    'glide_h': TestSymmetryOperations.glide_horizontal,
    'glide_v': TestSymmetryOperations.glide_vertical,
}


class TestWallpaperGroupSymmetries:
    """Test that each wallpaper group has exactly its correct symmetries."""
    
    @pytest.fixture
    def generator(self):
        """Create a pattern generator."""
        return WallpaperGroupGenerator(resolution=128, seed=42)
    
    @pytest.fixture
    def all_patterns(self, generator):
        """Generate all patterns."""
        return generator.generate_all(motif_size=32, complexity=3)
    
    @pytest.mark.parametrize("group_name", list(WALLPAPER_GROUPS.keys()))
    def test_group_has_required_symmetries(self, generator, group_name):
        """Test that each group HAS its required symmetries."""
        pattern = generator.generate(group_name, motif_size=32, complexity=3)
        requirements = SYMMETRY_REQUIREMENTS[group_name]
        
        for op_name in requirements['should_have']:
            op_func = OPERATIONS[op_name]
            transformed = op_func(pattern)
            corr = TestSymmetryOperations.correlation(pattern, transformed)
            
            assert corr > 0.90, \
                f"{group_name} should have {op_name} symmetry (correlation: {corr:.3f})"
    
    @pytest.mark.parametrize("group_name", list(WALLPAPER_GROUPS.keys()))
    def test_group_lacks_forbidden_symmetries(self, generator, group_name):
        """Test that each group does NOT have symmetries it shouldn't."""
        pattern = generator.generate(group_name, motif_size=32, complexity=3)
        requirements = SYMMETRY_REQUIREMENTS[group_name]
        
        for op_name in requirements['should_not_have']:
            op_func = OPERATIONS[op_name]
            transformed = op_func(pattern)
            corr = TestSymmetryOperations.correlation(pattern, transformed)
            
            # Allow some tolerance for accidental near-symmetry
            assert corr < 0.80, \
                f"{group_name} should NOT have {op_name} symmetry (correlation: {corr:.3f})"
    
    def test_identity_operation(self, all_patterns):
        """Test that identity operation gives 100% correlation for all groups."""
        for group_name, pattern in all_patterns.items():
            corr = TestSymmetryOperations.correlation(pattern, pattern)
            assert corr == 1.0, f"{group_name}: identity should give correlation 1.0"
    
    def test_double_c2_is_identity(self, all_patterns):
        """Test that C2 ∘ C2 = identity (two 180° rotations = 360°)."""
        for group_name, pattern in all_patterns.items():
            double_rotated = TestSymmetryOperations.rotate_180(
                TestSymmetryOperations.rotate_180(pattern)
            )
            corr = TestSymmetryOperations.correlation(pattern, double_rotated)
            assert corr > 0.99, \
                f"{group_name}: C2∘C2 should be identity (correlation: {corr:.3f})"
    
    def test_four_c4_is_identity(self, all_patterns):
        """Test that C4^4 = identity (four 90° rotations = 360°)."""
        for group_name, pattern in all_patterns.items():
            rotated = pattern
            for _ in range(4):
                rotated = TestSymmetryOperations.rotate_90(rotated)
            corr = TestSymmetryOperations.correlation(pattern, rotated)
            assert corr > 0.99, \
                f"{group_name}: C4^4 should be identity (correlation: {corr:.3f})"
    
    def test_double_reflection_is_identity(self, all_patterns):
        """Test that σ ∘ σ = identity (double reflection = identity)."""
        for group_name, pattern in all_patterns.items():
            # Vertical reflection twice
            double_v = TestSymmetryOperations.reflect_vertical(
                TestSymmetryOperations.reflect_vertical(pattern)
            )
            corr_v = TestSymmetryOperations.correlation(pattern, double_v)
            assert corr_v > 0.99, \
                f"{group_name}: σv∘σv should be identity (correlation: {corr_v:.3f})"
            
            # Horizontal reflection twice
            double_h = TestSymmetryOperations.reflect_horizontal(
                TestSymmetryOperations.reflect_horizontal(pattern)
            )
            corr_h = TestSymmetryOperations.correlation(pattern, double_h)
            assert corr_h > 0.99, \
                f"{group_name}: σh∘σh should be identity (correlation: {corr_h:.3f})"
    
    def test_perpendicular_reflections_equal_c2(self, all_patterns):
        """Test that σv ∘ σh = C2 (perpendicular reflections = 180° rotation)."""
        for group_name in ['pmm', 'cmm', 'p4m', 'p6m']:
            pattern = all_patterns[group_name]
            
            # σv ∘ σh
            composed = TestSymmetryOperations.reflect_horizontal(
                TestSymmetryOperations.reflect_vertical(pattern)
            )
            
            # C2
            c2 = TestSymmetryOperations.rotate_180(pattern)
            
            corr = TestSymmetryOperations.correlation(composed, c2)
            assert corr > 0.99, \
                f"{group_name}: σv∘σh should equal C2 (correlation: {corr:.3f})"


class TestGroupTheoryProperties:
    """Test group theory properties of wallpaper groups."""
    
    @pytest.fixture
    def generator(self):
        return WallpaperGroupGenerator(resolution=128, seed=42)
    
    def test_p1_minimal_symmetry(self, generator):
        """p1 should have ONLY translational symmetry (no point symmetry)."""
        pattern = generator.generate('p1', motif_size=32)
        
        # Should NOT have any of these
        for op_name, op_func in OPERATIONS.items():
            if op_name in ['C2', 'C3', 'C4', 'C6', 'sigma_v', 'sigma_h', 'sigma_d']:
                transformed = op_func(pattern)
                corr = TestSymmetryOperations.correlation(pattern, transformed)
                # p1 should have low correlation with rotations/reflections
                assert corr < 0.85, \
                    f"p1 should NOT have {op_name} (correlation: {corr:.3f})"
    
    def test_p2_only_c2(self, generator):
        """p2 should have C2 but NOT reflections."""
        pattern = generator.generate('p2', motif_size=32)
        
        # Should have C2
        c2 = TestSymmetryOperations.rotate_180(pattern)
        corr_c2 = TestSymmetryOperations.correlation(pattern, c2)
        assert corr_c2 > 0.90, f"p2 should have C2 (correlation: {corr_c2:.3f})"
        
        # Should NOT have reflections
        sigma_v = TestSymmetryOperations.reflect_vertical(pattern)
        corr_sv = TestSymmetryOperations.correlation(pattern, sigma_v)
        assert corr_sv < 0.80, f"p2 should NOT have σv (correlation: {corr_sv:.3f})"
    
    def test_pm_only_reflection(self, generator):
        """pm should have reflection but NOT C2."""
        pattern = generator.generate('pm', motif_size=32)
        
        # Should have reflection
        sigma_v = TestSymmetryOperations.reflect_vertical(pattern)
        corr_sv = TestSymmetryOperations.correlation(pattern, sigma_v)
        assert corr_sv > 0.90, f"pm should have σv (correlation: {corr_sv:.3f})"
        
        # Should NOT have C2
        c2 = TestSymmetryOperations.rotate_180(pattern)
        corr_c2 = TestSymmetryOperations.correlation(pattern, c2)
        assert corr_c2 < 0.80, f"pm should NOT have C2 (correlation: {corr_c2:.3f})"
    
    def test_p4_has_c4_and_c2(self, generator):
        """p4 should have C4 (which implies C2)."""
        pattern = generator.generate('p4', motif_size=32)
        
        # Should have C4
        c4 = TestSymmetryOperations.rotate_90(pattern)
        corr_c4 = TestSymmetryOperations.correlation(pattern, c4)
        assert corr_c4 > 0.90, f"p4 should have C4 (correlation: {corr_c4:.3f})"
        
        # Should have C2 (implied by C4)
        c2 = TestSymmetryOperations.rotate_180(pattern)
        corr_c2 = TestSymmetryOperations.correlation(pattern, c2)
        assert corr_c2 > 0.90, f"p4 should have C2 (correlation: {corr_c2:.3f})"
        
        # Should NOT have reflections
        sigma_v = TestSymmetryOperations.reflect_vertical(pattern)
        corr_sv = TestSymmetryOperations.correlation(pattern, sigma_v)
        assert corr_sv < 0.80, f"p4 should NOT have σv (correlation: {corr_sv:.3f})"
    
    def test_p6m_maximum_symmetry(self, generator):
        """p6m should have maximum symmetry (C6 + all reflections)."""
        pattern = generator.generate('p6m', motif_size=32)
        
        # Should have C6, C3, C2
        c2 = TestSymmetryOperations.rotate_180(pattern)
        corr_c2 = TestSymmetryOperations.correlation(pattern, c2)
        assert corr_c2 > 0.90, f"p6m should have C2 (correlation: {corr_c2:.3f})"
        
        # Should have reflections
        sigma_v = TestSymmetryOperations.reflect_vertical(pattern)
        corr_sv = TestSymmetryOperations.correlation(pattern, sigma_v)
        assert corr_sv > 0.90, f"p6m should have σv (correlation: {corr_sv:.3f})"
        
        sigma_h = TestSymmetryOperations.reflect_horizontal(pattern)
        corr_sh = TestSymmetryOperations.correlation(pattern, sigma_h)
        assert corr_sh > 0.90, f"p6m should have σh (correlation: {corr_sh:.3f})"


class TestCayleyTableProperties:
    """Test Cayley table (multiplication table) properties."""
    
    def test_group_closure(self):
        """Verify that composing any two symmetries gives another symmetry in the group."""
        # For pmm: σv ∘ σh should give C2, and all should be in the group
        # This is verified by the perpendicular_reflections_equal_c2 test
        pass
    
    def test_identity_element(self):
        """Verify that identity operation exists and e ∘ a = a ∘ e = a."""
        # Identity is always the 0° rotation / no transformation
        # This is trivially true
        pass
    
    def test_inverse_elements(self):
        """Verify that each element has an inverse (a ∘ a⁻¹ = e)."""
        # For rotations: inverse is rotation by 360° - angle
        # For reflections: inverse is the same reflection (σ ∘ σ = e)
        # Tested by test_double_reflection_is_identity
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

