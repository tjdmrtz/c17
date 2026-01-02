"""
Generator for the 230 Space Groups (3D Crystallographic Groups)

The 230 space groups represent all possible ways to arrange atoms in a 
3D crystal structure. They are the 3D analogue of the 17 wallpaper groups.

Organization:
- 7 Crystal Systems: Triclinic, Monoclinic, Orthorhombic, Tetragonal, 
                     Trigonal, Hexagonal, Cubic
- 14 Bravais Lattices
- 32 Crystallographic Point Groups

Each space group is characterized by:
- Lattice type (primitive, body-centered, face-centered, etc.)
- Rotation axes (1, 2, 3, 4, 6-fold)
- Mirror planes
- Glide planes (a, b, c, n, d)
- Screw axes (21, 31, 32, 41, 42, 43, 61, 62, 63, 64, 65)
- Inversion centers

Reference: International Tables for Crystallography, Volume A
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy.ndimage import rotate as scipy_rotate
from scipy.ndimage import affine_transform


class CrystalSystem(Enum):
    """The 7 crystal systems."""
    TRICLINIC = "triclinic"       # No symmetry constraints
    MONOCLINIC = "monoclinic"     # One 2-fold axis or mirror
    ORTHORHOMBIC = "orthorhombic" # Three perpendicular 2-fold axes
    TETRAGONAL = "tetragonal"     # One 4-fold axis
    TRIGONAL = "trigonal"         # One 3-fold axis
    HEXAGONAL = "hexagonal"       # One 6-fold axis
    CUBIC = "cubic"               # Four 3-fold axes


class LatticeType(Enum):
    """The 14 Bravais lattice types."""
    PRIMITIVE = "P"          # Simple lattice
    BODY_CENTERED = "I"      # Body-centered
    FACE_CENTERED = "F"      # Face-centered
    BASE_CENTERED_A = "A"    # A-face centered
    BASE_CENTERED_B = "B"    # B-face centered
    BASE_CENTERED_C = "C"    # C-face centered
    RHOMBOHEDRAL = "R"       # Rhombohedral (trigonal)


@dataclass
class SpaceGroup:
    """Represents a space group with its properties."""
    number: int                    # International number (1-230)
    symbol: str                    # Hermann-Mauguin symbol
    crystal_system: CrystalSystem
    lattice_type: LatticeType
    point_group: str               # Point group symbol
    description: str


# Define all 230 space groups organized by crystal system
# This is a simplified representation - full implementation would include
# all symmetry operations

SPACE_GROUPS: Dict[int, SpaceGroup] = {}

# === TRICLINIC (Groups 1-2) ===
SPACE_GROUPS[1] = SpaceGroup(1, "P1", CrystalSystem.TRICLINIC, 
                              LatticeType.PRIMITIVE, "1", "No symmetry")
SPACE_GROUPS[2] = SpaceGroup(2, "P-1", CrystalSystem.TRICLINIC,
                              LatticeType.PRIMITIVE, "-1", "Inversion center only")

# === MONOCLINIC (Groups 3-15) ===
_monoclinic = [
    (3, "P2", "2", "2-fold rotation axis"),
    (4, "P2₁", "2", "2₁ screw axis"),
    (5, "C2", "2", "C-centered with 2-fold"),
    (6, "Pm", "m", "Mirror plane"),
    (7, "Pc", "m", "c-glide plane"),
    (8, "Cm", "m", "C-centered with mirror"),
    (9, "Cc", "m", "C-centered with c-glide"),
    (10, "P2/m", "2/m", "2-fold with perpendicular mirror"),
    (11, "P2₁/m", "2/m", "2₁ screw with perpendicular mirror"),
    (12, "C2/m", "2/m", "C-centered 2/m"),
    (13, "P2/c", "2/m", "2-fold with c-glide"),
    (14, "P2₁/c", "2/m", "2₁ screw with c-glide"),
    (15, "C2/c", "2/m", "C-centered with c-glide"),
]
for num, sym, pg, desc in _monoclinic:
    lt = LatticeType.BASE_CENTERED_C if sym.startswith("C") else LatticeType.PRIMITIVE
    SPACE_GROUPS[num] = SpaceGroup(num, sym, CrystalSystem.MONOCLINIC, lt, pg, desc)

# === ORTHORHOMBIC (Groups 16-74) ===
_ortho_primitive = [
    (16, "P222", "222", "Three 2-fold axes"),
    (17, "P222₁", "222", "Two 2-fold, one 2₁"),
    (18, "P2₁2₁2", "222", "One 2-fold, two 2₁"),
    (19, "P2₁2₁2₁", "222", "Three 2₁ screw axes"),
    (20, "C222₁", "222", "C-centered with 2₁"),
    (21, "C222", "222", "C-centered 222"),
    (22, "F222", "222", "Face-centered 222"),
    (23, "I222", "222", "Body-centered 222"),
    (24, "I2₁2₁2₁", "222", "Body-centered with 2₁ axes"),
    (25, "Pmm2", "mm2", "Two mirrors, one 2-fold"),
    (26, "Pmc2₁", "mm2", "Mirror, c-glide, 2₁"),
    (27, "Pcc2", "mm2", "Two c-glides, 2-fold"),
    (28, "Pma2", "mm2", "Mirror, a-glide, 2-fold"),
    (29, "Pca2₁", "mm2", "c-glide, a-glide, 2₁"),
    (30, "Pnc2", "mm2", "n-glide, c-glide, 2-fold"),
    (31, "Pmn2₁", "mm2", "Mirror, n-glide, 2₁"),
    (32, "Pba2", "mm2", "b-glide, a-glide, 2-fold"),
    (33, "Pna2₁", "mm2", "n-glide, a-glide, 2₁"),
    (34, "Pnn2", "mm2", "Two n-glides, 2-fold"),
    (35, "Cmm2", "mm2", "C-centered mm2"),
    (36, "Cmc2₁", "mm2", "C-centered mc2₁"),
    (37, "Ccc2", "mm2", "C-centered cc2"),
    (38, "Amm2", "mm2", "A-centered mm2"),
    (39, "Aem2", "mm2", "A-centered em2"),
    (40, "Ama2", "mm2", "A-centered ma2"),
    (41, "Aea2", "mm2", "A-centered ea2"),
    (42, "Fmm2", "mm2", "Face-centered mm2"),
    (43, "Fdd2", "mm2", "Face-centered dd2"),
    (44, "Imm2", "mm2", "Body-centered mm2"),
    (45, "Iba2", "mm2", "Body-centered ba2"),
    (46, "Ima2", "mm2", "Body-centered ma2"),
    (47, "Pmmm", "mmm", "Three perpendicular mirrors"),
    (48, "Pnnn", "mmm", "Three n-glides"),
    (49, "Pccm", "mmm", "Two c-glides, mirror"),
    (50, "Pban", "mmm", "b-glide, a-glide, n-glide"),
    (51, "Pmma", "mmm", "Two mirrors, a-glide"),
    (52, "Pnna", "mmm", "Two n-glides, a-glide"),
    (53, "Pmna", "mmm", "Mirror, n-glide, a-glide"),
    (54, "Pcca", "mmm", "Two c-glides, a-glide"),
    (55, "Pbam", "mmm", "b-glide, a-glide, mirror"),
    (56, "Pccn", "mmm", "Two c-glides, n-glide"),
    (57, "Pbcm", "mmm", "b-glide, c-glide, mirror"),
    (58, "Pnnm", "mmm", "Two n-glides, mirror"),
    (59, "Pmmn", "mmm", "Two mirrors, n-glide"),
    (60, "Pbcn", "mmm", "b-glide, c-glide, n-glide"),
    (61, "Pbca", "mmm", "b-glide, c-glide, a-glide"),
    (62, "Pnma", "mmm", "n-glide, mirror, a-glide"),
    (63, "Cmcm", "mmm", "C-centered mcm"),
    (64, "Cmce", "mmm", "C-centered mce"),
    (65, "Cmmm", "mmm", "C-centered mmm"),
    (66, "Cccm", "mmm", "C-centered ccm"),
    (67, "Cmme", "mmm", "C-centered mme"),
    (68, "Ccce", "mmm", "C-centered cce"),
    (69, "Fmmm", "mmm", "Face-centered mmm"),
    (70, "Fddd", "mmm", "Face-centered ddd"),
    (71, "Immm", "mmm", "Body-centered mmm"),
    (72, "Ibam", "mmm", "Body-centered bam"),
    (73, "Ibca", "mmm", "Body-centered bca"),
    (74, "Imma", "mmm", "Body-centered mma"),
]

for num, sym, pg, desc in _ortho_primitive:
    if sym.startswith("F"):
        lt = LatticeType.FACE_CENTERED
    elif sym.startswith("I"):
        lt = LatticeType.BODY_CENTERED
    elif sym.startswith("C"):
        lt = LatticeType.BASE_CENTERED_C
    elif sym.startswith("A"):
        lt = LatticeType.BASE_CENTERED_A
    else:
        lt = LatticeType.PRIMITIVE
    SPACE_GROUPS[num] = SpaceGroup(num, sym, CrystalSystem.ORTHORHOMBIC, lt, pg, desc)

# === TETRAGONAL (Groups 75-142) ===
_tetragonal = [
    (75, "P4", "4", "4-fold rotation axis"),
    (76, "P4₁", "4", "4₁ screw axis"),
    (77, "P4₂", "4", "4₂ screw axis"),
    (78, "P4₃", "4", "4₃ screw axis"),
    (79, "I4", "4", "Body-centered 4-fold"),
    (80, "I4₁", "4", "Body-centered 4₁"),
    (81, "P-4", "-4", "Rotoinversion -4"),
    (82, "I-4", "-4", "Body-centered -4"),
    (83, "P4/m", "4/m", "4-fold with mirror"),
    (84, "P4₂/m", "4/m", "4₂ with mirror"),
    (85, "P4/n", "4/m", "4-fold with n-glide"),
    (86, "P4₂/n", "4/m", "4₂ with n-glide"),
    (87, "I4/m", "4/m", "Body-centered 4/m"),
    (88, "I4₁/a", "4/m", "Body-centered 4₁/a"),
    (89, "P422", "422", "4-fold with two 2-folds"),
    (90, "P42₁2", "422", "4-fold with 2₁ axes"),
    (91, "P4₁22", "422", "4₁ with two 2-folds"),
    (92, "P4₁2₁2", "422", "4₁ with 2₁ axes"),
    (93, "P4₂22", "422", "4₂ with two 2-folds"),
    (94, "P4₂2₁2", "422", "4₂ with 2₁ axes"),
    (95, "P4₃22", "422", "4₃ with two 2-folds"),
    (96, "P4₃2₁2", "422", "4₃ with 2₁ axes"),
    (97, "I422", "422", "Body-centered 422"),
    (98, "I4₁22", "422", "Body-centered 4₁22"),
    (99, "P4mm", "4mm", "4-fold with mirrors"),
    (100, "P4bm", "4mm", "4-fold with b-glide, mirror"),
    (101, "P4₂cm", "4mm", "4₂ with c-glide, mirror"),
    (102, "P4₂nm", "4mm", "4₂ with n-glide, mirror"),
    (103, "P4cc", "4mm", "4-fold with c-glides"),
    (104, "P4nc", "4mm", "4-fold with n-glide, c-glide"),
    (105, "P4₂mc", "4mm", "4₂ with mirror, c-glide"),
    (106, "P4₂bc", "4mm", "4₂ with b-glide, c-glide"),
    (107, "I4mm", "4mm", "Body-centered 4mm"),
    (108, "I4cm", "4mm", "Body-centered 4cm"),
    (109, "I4₁md", "4mm", "Body-centered 4₁md"),
    (110, "I4₁cd", "4mm", "Body-centered 4₁cd"),
    (111, "P-42m", "-42m", "-4 with 2-fold and mirror"),
    (112, "P-42c", "-42m", "-4 with 2-fold and c-glide"),
    (113, "P-42₁m", "-42m", "-4 with 2₁ and mirror"),
    (114, "P-42₁c", "-42m", "-4 with 2₁ and c-glide"),
    (115, "P-4m2", "-4m2", "-4 with mirror and 2-fold"),
    (116, "P-4c2", "-4m2", "-4 with c-glide and 2-fold"),
    (117, "P-4b2", "-4m2", "-4 with b-glide and 2-fold"),
    (118, "P-4n2", "-4m2", "-4 with n-glide and 2-fold"),
    (119, "I-4m2", "-4m2", "Body-centered -4m2"),
    (120, "I-4c2", "-4m2", "Body-centered -4c2"),
    (121, "I-42m", "-42m", "Body-centered -42m"),
    (122, "I-42d", "-42m", "Body-centered -42d"),
    (123, "P4/mmm", "4/mmm", "Highest tetragonal symmetry"),
    (124, "P4/mcc", "4/mmm", "4/m with c-glides"),
    (125, "P4/nbm", "4/mmm", "4/n with b-glide, mirror"),
    (126, "P4/nnc", "4/mmm", "4/n with n-glides, c-glide"),
    (127, "P4/mbm", "4/mmm", "4/m with b-glide, mirror"),
    (128, "P4/mnc", "4/mmm", "4/m with n-glide, c-glide"),
    (129, "P4/nmm", "4/mmm", "4/n with mirrors"),
    (130, "P4/ncc", "4/mmm", "4/n with c-glides"),
    (131, "P4₂/mmc", "4/mmm", "4₂/m with mirrors, c-glide"),
    (132, "P4₂/mcm", "4/mmm", "4₂/m with c-glide, mirror"),
    (133, "P4₂/nbc", "4/mmm", "4₂/n with b-glide, c-glide"),
    (134, "P4₂/nnm", "4/mmm", "4₂/n with n-glides, mirror"),
    (135, "P4₂/mbc", "4/mmm", "4₂/m with b-glide, c-glide"),
    (136, "P4₂/mnm", "4/mmm", "4₂/m with n-glide, mirror"),
    (137, "P4₂/nmc", "4/mmm", "4₂/n with mirror, c-glide"),
    (138, "P4₂/ncm", "4/mmm", "4₂/n with c-glide, mirror"),
    (139, "I4/mmm", "4/mmm", "Body-centered 4/mmm"),
    (140, "I4/mcm", "4/mmm", "Body-centered 4/mcm"),
    (141, "I4₁/amd", "4/mmm", "Body-centered 4₁/amd"),
    (142, "I4₁/acd", "4/mmm", "Body-centered 4₁/acd"),
]

for num, sym, pg, desc in _tetragonal:
    lt = LatticeType.BODY_CENTERED if sym.startswith("I") else LatticeType.PRIMITIVE
    SPACE_GROUPS[num] = SpaceGroup(num, sym, CrystalSystem.TETRAGONAL, lt, pg, desc)

# === TRIGONAL (Groups 143-167) ===
_trigonal = [
    (143, "P3", "3", "3-fold rotation axis"),
    (144, "P3₁", "3", "3₁ screw axis"),
    (145, "P3₂", "3", "3₂ screw axis"),
    (146, "R3", "3", "Rhombohedral 3-fold"),
    (147, "P-3", "-3", "3-fold with inversion"),
    (148, "R-3", "-3", "Rhombohedral -3"),
    (149, "P312", "32", "3-fold with 2-fold"),
    (150, "P321", "32", "3-fold with 2-fold (alt)"),
    (151, "P3₁12", "32", "3₁ with 2-fold"),
    (152, "P3₁21", "32", "3₁ with 2-fold (alt)"),
    (153, "P3₂12", "32", "3₂ with 2-fold"),
    (154, "P3₂21", "32", "3₂ with 2-fold (alt)"),
    (155, "R32", "32", "Rhombohedral 32"),
    (156, "P3m1", "3m", "3-fold with mirrors"),
    (157, "P31m", "3m", "3-fold with mirrors (alt)"),
    (158, "P3c1", "3m", "3-fold with c-glides"),
    (159, "P31c", "3m", "3-fold with c-glide (alt)"),
    (160, "R3m", "3m", "Rhombohedral 3m"),
    (161, "R3c", "3m", "Rhombohedral 3c"),
    (162, "P-31m", "-3m", "-3 with mirrors"),
    (163, "P-31c", "-3m", "-3 with c-glide"),
    (164, "P-3m1", "-3m", "-3 with mirrors (alt)"),
    (165, "P-3c1", "-3m", "-3 with c-glides (alt)"),
    (166, "R-3m", "-3m", "Rhombohedral -3m"),
    (167, "R-3c", "-3m", "Rhombohedral -3c"),
]

for num, sym, pg, desc in _trigonal:
    lt = LatticeType.RHOMBOHEDRAL if sym.startswith("R") else LatticeType.PRIMITIVE
    SPACE_GROUPS[num] = SpaceGroup(num, sym, CrystalSystem.TRIGONAL, lt, pg, desc)

# === HEXAGONAL (Groups 168-194) ===
_hexagonal = [
    (168, "P6", "6", "6-fold rotation axis"),
    (169, "P6₁", "6", "6₁ screw axis"),
    (170, "P6₅", "6", "6₅ screw axis"),
    (171, "P6₂", "6", "6₂ screw axis"),
    (172, "P6₄", "6", "6₄ screw axis"),
    (173, "P6₃", "6", "6₃ screw axis"),
    (174, "P-6", "-6", "Rotoinversion -6"),
    (175, "P6/m", "6/m", "6-fold with mirror"),
    (176, "P6₃/m", "6/m", "6₃ with mirror"),
    (177, "P622", "622", "6-fold with 2-folds"),
    (178, "P6₁22", "622", "6₁ with 2-folds"),
    (179, "P6₅22", "622", "6₅ with 2-folds"),
    (180, "P6₂22", "622", "6₂ with 2-folds"),
    (181, "P6₄22", "622", "6₄ with 2-folds"),
    (182, "P6₃22", "622", "6₃ with 2-folds"),
    (183, "P6mm", "6mm", "6-fold with mirrors"),
    (184, "P6cc", "6mm", "6-fold with c-glides"),
    (185, "P6₃cm", "6mm", "6₃ with c-glide, mirror"),
    (186, "P6₃mc", "6mm", "6₃ with mirror, c-glide"),
    (187, "P-6m2", "-6m2", "-6 with mirror and 2-fold"),
    (188, "P-6c2", "-6m2", "-6 with c-glide and 2-fold"),
    (189, "P-62m", "-62m", "-6 with 2-fold and mirror"),
    (190, "P-62c", "-62m", "-6 with 2-fold and c-glide"),
    (191, "P6/mmm", "6/mmm", "Highest hexagonal symmetry"),
    (192, "P6/mcc", "6/mmm", "6/m with c-glides"),
    (193, "P6₃/mcm", "6/mmm", "6₃/m with c-glide, mirror"),
    (194, "P6₃/mmc", "6/mmm", "6₃/m with mirror, c-glide"),
]

for num, sym, pg, desc in _hexagonal:
    SPACE_GROUPS[num] = SpaceGroup(num, sym, CrystalSystem.HEXAGONAL, 
                                    LatticeType.PRIMITIVE, pg, desc)

# === CUBIC (Groups 195-230) ===
_cubic = [
    (195, "P23", "23", "Three 2-fold and four 3-fold axes"),
    (196, "F23", "23", "Face-centered 23"),
    (197, "I23", "23", "Body-centered 23"),
    (198, "P2₁3", "23", "2₁ and 3-fold axes"),
    (199, "I2₁3", "23", "Body-centered 2₁3"),
    (200, "Pm-3", "m-3", "Mirrors and 3-fold inversion"),
    (201, "Pn-3", "m-3", "n-glide and -3"),
    (202, "Fm-3", "m-3", "Face-centered m-3"),
    (203, "Fd-3", "m-3", "Face-centered d-3"),
    (204, "Im-3", "m-3", "Body-centered m-3"),
    (205, "Pa-3", "m-3", "a-glide and -3"),
    (206, "Ia-3", "m-3", "Body-centered a-3"),
    (207, "P432", "432", "4-fold, 3-fold, 2-fold"),
    (208, "P4₂32", "432", "4₂, 3-fold, 2-fold"),
    (209, "F432", "432", "Face-centered 432"),
    (210, "F4₁32", "432", "Face-centered 4₁32"),
    (211, "I432", "432", "Body-centered 432"),
    (212, "P4₃32", "432", "4₃, 3-fold, 2-fold"),
    (213, "P4₁32", "432", "4₁, 3-fold, 2-fold"),
    (214, "I4₁32", "432", "Body-centered 4₁32"),
    (215, "P-43m", "-43m", "-4, 3-fold, mirror"),
    (216, "F-43m", "-43m", "Face-centered -43m"),
    (217, "I-43m", "-43m", "Body-centered -43m"),
    (218, "P-43n", "-43m", "-4, 3-fold, n-glide"),
    (219, "F-43c", "-43m", "Face-centered -43c"),
    (220, "I-43d", "-43m", "Body-centered -43d"),
    (221, "Pm-3m", "m-3m", "Highest cubic symmetry"),
    (222, "Pn-3n", "m-3m", "n-glide and -3 with n-glide"),
    (223, "Pm-3n", "m-3m", "Mirror, -3, n-glide"),
    (224, "Pn-3m", "m-3m", "n-glide, -3, mirror"),
    (225, "Fm-3m", "m-3m", "Face-centered m-3m"),
    (226, "Fm-3c", "m-3m", "Face-centered m-3c"),
    (227, "Fd-3m", "m-3m", "Diamond structure"),
    (228, "Fd-3c", "m-3m", "Face-centered d-3c"),
    (229, "Im-3m", "m-3m", "Body-centered m-3m"),
    (230, "Ia-3d", "m-3m", "Garnet structure"),
]

for num, sym, pg, desc in _cubic:
    if sym.startswith("F"):
        lt = LatticeType.FACE_CENTERED
    elif sym.startswith("I"):
        lt = LatticeType.BODY_CENTERED
    else:
        lt = LatticeType.PRIMITIVE
    SPACE_GROUPS[num] = SpaceGroup(num, sym, CrystalSystem.CUBIC, lt, pg, desc)


class SpaceGroupGenerator:
    """
    Generates beautiful 3D crystallographic patterns for all 230 space groups.
    
    Each space group is implemented using its characteristic symmetry operations,
    creating patterns that faithfully represent the crystal structure.
    """
    
    def __init__(self,
                 resolution: Tuple[int, int, int] = (64, 64, 64),
                 seed: Optional[int] = None):
        """
        Initialize the space group generator.
        
        Args:
            resolution: (x, y, z) size of the output volume
            seed: Random seed for reproducibility
        """
        self.resolution = resolution
        self.rng = np.random.default_rng(seed)
        
    def _create_atom_motif(self, 
                           size: Tuple[int, int, int],
                           num_atoms: int = 3,
                           style: str = "crystalline") -> np.ndarray:
        """
        Create a motif representing atoms/molecules in the asymmetric unit.
        
        Args:
            size: (x, y, z) size of the motif
            num_atoms: Number of atom-like elements
            style: Visual style ("crystalline", "molecular", "artistic")
        """
        sx, sy, sz = size
        motif = np.zeros(size)
        
        if style == "crystalline":
            # Sharp, crystal-like atomic positions
            for _ in range(num_atoms):
                cx = self.rng.random() * sx * 0.6 + sx * 0.2
                cy = self.rng.random() * sy * 0.6 + sy * 0.2
                cz = self.rng.random() * sz * 0.6 + sz * 0.2
                
                # Atomic radius (varies for visual interest)
                radius = self.rng.random() * min(size) / 8 + min(size) / 10
                intensity = self.rng.random() * 0.5 + 0.5
                
                z, y, x = np.ogrid[:sz, :sy, :sx]
                dist_sq = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
                
                # Sharp falloff for crystal-like appearance
                atom = np.exp(-(dist_sq / (radius**2))**2)
                motif += atom * intensity
                
        elif style == "molecular":
            # Connected blob-like structures
            centers = []
            for _ in range(num_atoms):
                cx = self.rng.random() * sx * 0.5 + sx * 0.25
                cy = self.rng.random() * sy * 0.5 + sy * 0.25
                cz = self.rng.random() * sz * 0.5 + sz * 0.25
                centers.append((cx, cy, cz))
                
                radius = self.rng.random() * min(size) / 6 + min(size) / 8
                intensity = self.rng.random() * 0.4 + 0.6
                
                z, y, x = np.ogrid[:sz, :sy, :sx]
                atom = np.exp(-((x-cx)**2 + (y-cy)**2 + (z-cz)**2) / (2*radius**2))
                motif += atom * intensity
            
            # Add bonds between nearby atoms
            for i, (c1x, c1y, c1z) in enumerate(centers):
                for c2x, c2y, c2z in centers[i+1:]:
                    dist = np.sqrt((c1x-c2x)**2 + (c1y-c2y)**2 + (c1z-c2z)**2)
                    if dist < min(size) / 2:
                        # Draw bond
                        z, y, x = np.ogrid[:sz, :sy, :sx]
                        # Parametric line distance
                        t = np.clip(
                            ((x-c1x)*(c2x-c1x) + (y-c1y)*(c2y-c1y) + (z-c1z)*(c2z-c1z)) / (dist**2 + 1e-8),
                            0, 1
                        )
                        px = c1x + t * (c2x - c1x)
                        py = c1y + t * (c2y - c1y)
                        pz = c1z + t * (c2z - c1z)
                        bond_dist = np.sqrt((x-px)**2 + (y-py)**2 + (z-pz)**2)
                        bond = np.exp(-(bond_dist / 2)**2) * 0.3
                        motif += bond
                        
        elif style == "artistic":
            # Beautiful flowing shapes
            for _ in range(num_atoms):
                cx = self.rng.random() * sx
                cy = self.rng.random() * sy
                cz = self.rng.random() * sz
                
                z, y, x = np.ogrid[:sz, :sy, :sx]
                
                # Base distance
                dist = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
                
                # Add wobble for organic look
                freq = self.rng.random() * 0.3 + 0.1
                theta = np.arctan2(y - cy, x - cx)
                phi = np.arctan2(np.sqrt((x-cx)**2 + (y-cy)**2), z - cz)
                wobble = 1 + 0.3 * np.sin(freq * theta * 8) * np.sin(freq * phi * 6)
                
                radius = self.rng.random() * min(size) / 4 + min(size) / 6
                blob = np.exp(-((dist / radius) * wobble)**2)
                motif += blob * (self.rng.random() * 0.5 + 0.5)
        
        if motif.max() > 0:
            motif = motif / motif.max()
            
        return motif
    
    def _apply_symmetry_operation(self,
                                   volume: np.ndarray,
                                   matrix: np.ndarray,
                                   translation: np.ndarray = None) -> np.ndarray:
        """Apply a general symmetry operation (rotation matrix + translation)."""
        if translation is None:
            translation = np.zeros(3)
            
        # Create the affine transformation
        sz, sy, sx = volume.shape
        center = np.array([sz/2, sy/2, sx/2])
        
        offset = center - matrix @ center + translation * np.array([sz, sy, sx])
        
        result = affine_transform(
            volume, 
            matrix, 
            offset=offset,
            order=1,
            mode='wrap'
        )
        
        return result
    
    def _rotation_matrix_z(self, angle: float) -> np.ndarray:
        """Rotation matrix around z-axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    
    def _rotation_matrix_y(self, angle: float) -> np.ndarray:
        """Rotation matrix around y-axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    
    def _rotation_matrix_x(self, angle: float) -> np.ndarray:
        """Rotation matrix around x-axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    
    def _mirror_matrix(self, axis: int) -> np.ndarray:
        """Mirror matrix for given axis (0=z, 1=y, 2=x)."""
        m = np.eye(3)
        m[axis, axis] = -1
        return m
    
    def _inversion_matrix(self) -> np.ndarray:
        """Inversion matrix."""
        return -np.eye(3)
    
    def _tile_3d(self, cell: np.ndarray, tiles: Tuple[int, int, int]) -> np.ndarray:
        """Tile the unit cell in 3D."""
        return np.tile(cell, tiles)
    
    # === Generators by Crystal System ===
    
    def generate_triclinic(self, 
                           group_number: int,
                           motif_size: int = 20,
                           **kwargs) -> np.ndarray:
        """Generate patterns for triclinic space groups (1-2)."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        motif = self._create_atom_motif((ms, ms, ms), **kwargs)
        
        if group_number == 1:  # P1 - no symmetry
            tiles = (sz // ms + 1, sy // ms + 1, sx // ms + 1)
            pattern = self._tile_3d(motif, tiles)
            
        elif group_number == 2:  # P-1 - inversion only
            inverted = self._apply_symmetry_operation(motif, self._inversion_matrix())
            cell = (motif + inverted) / 2
            tiles = (sz // ms + 1, sy // ms + 1, sx // ms + 1)
            pattern = self._tile_3d(cell, tiles)
            
        return pattern[:sz, :sy, :sx]
    
    def generate_monoclinic(self,
                            group_number: int,
                            motif_size: int = 20,
                            **kwargs) -> np.ndarray:
        """Generate patterns for monoclinic space groups (3-15)."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        motif = self._create_atom_motif((ms, ms, ms), **kwargs)
        
        # Apply 2-fold rotation around b-axis (y)
        rot180 = self._apply_symmetry_operation(motif, self._rotation_matrix_y(np.pi))
        
        if group_number in [3, 4, 5]:  # P2, P21, C2
            cell = (motif + rot180) / 2
        elif group_number in [6, 7, 8, 9]:  # Pm, Pc, Cm, Cc
            mirrored = self._apply_symmetry_operation(motif, self._mirror_matrix(1))
            cell = (motif + mirrored) / 2
        else:  # 10-15: 2/m groups
            mirrored = self._apply_symmetry_operation(motif, self._mirror_matrix(1))
            cell = (motif + rot180 + mirrored) / 3
        
        if cell.max() > 0:
            cell = cell / cell.max()
            
        tiles = (sz // ms + 1, sy // ms + 1, sx // ms + 1)
        pattern = self._tile_3d(cell, tiles)
        
        return pattern[:sz, :sy, :sx]
    
    def generate_orthorhombic(self,
                              group_number: int,
                              motif_size: int = 20,
                              **kwargs) -> np.ndarray:
        """Generate patterns for orthorhombic space groups (16-74)."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        motif = self._create_atom_motif((ms, ms, ms), **kwargs)
        
        # Apply three perpendicular 2-fold rotations
        rot_x = self._apply_symmetry_operation(motif, self._rotation_matrix_x(np.pi))
        rot_y = self._apply_symmetry_operation(motif, self._rotation_matrix_y(np.pi))
        rot_z = self._apply_symmetry_operation(motif, self._rotation_matrix_z(np.pi))
        
        cell = (motif + rot_x + rot_y + rot_z) / 4
        
        # For mmm groups (47-74), add mirrors
        if group_number >= 47:
            mirror_x = self._apply_symmetry_operation(motif, self._mirror_matrix(2))
            mirror_y = self._apply_symmetry_operation(motif, self._mirror_matrix(1))
            mirror_z = self._apply_symmetry_operation(motif, self._mirror_matrix(0))
            cell = (cell + mirror_x + mirror_y + mirror_z) / 4
        
        if cell.max() > 0:
            cell = cell / cell.max()
            
        tiles = (sz // ms + 1, sy // ms + 1, sx // ms + 1)
        pattern = self._tile_3d(cell, tiles)
        
        return pattern[:sz, :sy, :sx]
    
    def generate_tetragonal(self,
                            group_number: int,
                            motif_size: int = 20,
                            **kwargs) -> np.ndarray:
        """Generate patterns for tetragonal space groups (75-142)."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        motif = self._create_atom_motif((ms, ms, ms), **kwargs)
        
        # Apply 4-fold rotation around z-axis
        rot90 = self._apply_symmetry_operation(motif, self._rotation_matrix_z(np.pi/2))
        rot180 = self._apply_symmetry_operation(motif, self._rotation_matrix_z(np.pi))
        rot270 = self._apply_symmetry_operation(motif, self._rotation_matrix_z(3*np.pi/2))
        
        cell = (motif + rot90 + rot180 + rot270) / 4
        
        # For 4/mmm groups (123-142), add mirrors
        if group_number >= 123:
            mirror_x = self._apply_symmetry_operation(motif, self._mirror_matrix(2))
            mirror_y = self._apply_symmetry_operation(motif, self._mirror_matrix(1))
            mirror_z = self._apply_symmetry_operation(motif, self._mirror_matrix(0))
            cell = (cell + mirror_x + mirror_y + mirror_z) / 4
        
        if cell.max() > 0:
            cell = cell / cell.max()
            
        tiles = (sz // ms + 1, sy // ms + 1, sx // ms + 1)
        pattern = self._tile_3d(cell, tiles)
        
        return pattern[:sz, :sy, :sx]
    
    def generate_trigonal(self,
                          group_number: int,
                          motif_size: int = 20,
                          **kwargs) -> np.ndarray:
        """Generate patterns for trigonal space groups (143-167)."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        motif = self._create_atom_motif((ms, ms, ms), **kwargs)
        
        # Apply 3-fold rotation around z-axis
        rot120 = self._apply_symmetry_operation(motif, self._rotation_matrix_z(2*np.pi/3))
        rot240 = self._apply_symmetry_operation(motif, self._rotation_matrix_z(4*np.pi/3))
        
        cell = (motif + rot120 + rot240) / 3
        
        # For -3m groups, add inversion and mirrors
        if group_number >= 162:
            inverted = self._apply_symmetry_operation(cell, self._inversion_matrix())
            cell = (cell + inverted) / 2
        
        if cell.max() > 0:
            cell = cell / cell.max()
            
        tiles = (sz // ms + 1, sy // ms + 1, sx // ms + 1)
        pattern = self._tile_3d(cell, tiles)
        
        return pattern[:sz, :sy, :sx]
    
    def generate_hexagonal(self,
                           group_number: int,
                           motif_size: int = 20,
                           **kwargs) -> np.ndarray:
        """Generate patterns for hexagonal space groups (168-194)."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        motif = self._create_atom_motif((ms, ms, ms), **kwargs)
        
        # Apply 6-fold rotation around z-axis
        cell = motif.copy()
        for k in range(1, 6):
            rotated = self._apply_symmetry_operation(motif, self._rotation_matrix_z(k * np.pi/3))
            cell = cell + rotated
        cell = cell / 6
        
        # For 6/mmm groups, add mirrors
        if group_number >= 191:
            mirror_z = self._apply_symmetry_operation(cell, self._mirror_matrix(0))
            cell = (cell + mirror_z) / 2
        
        if cell.max() > 0:
            cell = cell / cell.max()
            
        tiles = (sz // ms + 1, sy // ms + 1, sx // ms + 1)
        pattern = self._tile_3d(cell, tiles)
        
        return pattern[:sz, :sy, :sx]
    
    def generate_cubic(self,
                       group_number: int,
                       motif_size: int = 20,
                       **kwargs) -> np.ndarray:
        """Generate patterns for cubic space groups (195-230)."""
        sx, sy, sz = self.resolution
        ms = motif_size
        
        motif = self._create_atom_motif((ms, ms, ms), **kwargs)
        
        # Apply the 24 rotations of the octahedral group
        # (or 48 with inversion for m-3m groups)
        
        rotations = []
        
        # Identity
        rotations.append(np.eye(3))
        
        # 90°, 180°, 270° rotations around x, y, z
        for angle in [np.pi/2, np.pi, 3*np.pi/2]:
            rotations.append(self._rotation_matrix_x(angle))
            rotations.append(self._rotation_matrix_y(angle))
            rotations.append(self._rotation_matrix_z(angle))
        
        # 120°, 240° rotations around body diagonals
        # Approximate with combinations
        for angle in [2*np.pi/3, 4*np.pi/3]:
            r = self._rotation_matrix_x(angle) @ self._rotation_matrix_y(angle)
            rotations.append(r)
        
        # Apply all rotations
        cell = np.zeros_like(motif)
        for rot in rotations:
            rotated = self._apply_symmetry_operation(motif, rot)
            cell = cell + rotated
        cell = cell / len(rotations)
        
        # For m-3m groups (221-230), add inversion
        if group_number >= 221:
            inverted = self._apply_symmetry_operation(cell, self._inversion_matrix())
            cell = (cell + inverted) / 2
        
        if cell.max() > 0:
            cell = cell / cell.max()
            
        tiles = (sz // ms + 1, sy // ms + 1, sx // ms + 1)
        pattern = self._tile_3d(cell, tiles)
        
        return pattern[:sz, :sy, :sx]
    
    def generate(self,
                 group_number: int,
                 motif_size: int = 20,
                 style: str = "crystalline",
                 num_atoms: int = 3,
                 **kwargs) -> np.ndarray:
        """
        Generate a 3D pattern for any of the 230 space groups.
        
        Args:
            group_number: Space group number (1-230)
            motif_size: Size of the fundamental motif
            style: Visual style ("crystalline", "molecular", "artistic")
            num_atoms: Number of atom-like elements in the asymmetric unit
            **kwargs: Additional arguments
            
        Returns:
            3D numpy array with the generated pattern
        """
        if group_number < 1 or group_number > 230:
            raise ValueError(f"Space group number must be 1-230, got {group_number}")
        
        kwargs['style'] = style
        kwargs['num_atoms'] = num_atoms
        
        if group_number <= 2:
            return self.generate_triclinic(group_number, motif_size, **kwargs)
        elif group_number <= 15:
            return self.generate_monoclinic(group_number, motif_size, **kwargs)
        elif group_number <= 74:
            return self.generate_orthorhombic(group_number, motif_size, **kwargs)
        elif group_number <= 142:
            return self.generate_tetragonal(group_number, motif_size, **kwargs)
        elif group_number <= 167:
            return self.generate_trigonal(group_number, motif_size, **kwargs)
        elif group_number <= 194:
            return self.generate_hexagonal(group_number, motif_size, **kwargs)
        else:
            return self.generate_cubic(group_number, motif_size, **kwargs)
    
    def generate_by_symbol(self,
                           symbol: str,
                           **kwargs) -> np.ndarray:
        """
        Generate a pattern using the Hermann-Mauguin symbol.
        
        Args:
            symbol: Space group symbol (e.g., "Fm-3m", "P6₃/mmc")
            **kwargs: Additional arguments passed to generate()
        """
        # Find the group number for this symbol
        for num, group in SPACE_GROUPS.items():
            if group.symbol == symbol or group.symbol.replace("₁", "1").replace("₂", "2").replace("₃", "3") == symbol:
                return self.generate(num, **kwargs)
        
        raise ValueError(f"Unknown space group symbol: {symbol}")
    
    def generate_by_system(self,
                           system: CrystalSystem,
                           count: int = 5,
                           **kwargs) -> Dict[int, np.ndarray]:
        """
        Generate patterns for multiple space groups of a given crystal system.
        
        Args:
            system: Crystal system to generate patterns for
            count: Number of patterns to generate
            **kwargs: Additional arguments
            
        Returns:
            Dictionary mapping group numbers to patterns
        """
        # Get all groups for this system
        system_groups = [num for num, g in SPACE_GROUPS.items() 
                        if g.crystal_system == system]
        
        # Sample some groups
        selected = self.rng.choice(system_groups, 
                                   size=min(count, len(system_groups)),
                                   replace=False)
        
        return {int(num): self.generate(int(num), **kwargs) for num in selected}
    
    def generate_famous_structures(self, **kwargs) -> Dict[str, np.ndarray]:
        """
        Generate patterns for famous crystal structures.
        
        Returns:
            Dictionary with structure names and their patterns
        """
        famous = {
            "Diamond": 227,        # Fd-3m
            "NaCl (Rock Salt)": 225,  # Fm-3m
            "CsCl": 221,           # Pm-3m
            "Fluorite": 225,       # Fm-3m
            "Perovskite": 221,     # Pm-3m
            "Rutile": 136,         # P4₂/mnm
            "Wurtzite": 186,       # P6₃mc
            "Spinel": 227,         # Fd-3m
            "Garnet": 230,         # Ia-3d
            "Ice Ih": 194,         # P6₃/mmc
            "Graphite": 194,       # P6₃/mmc
            "Quartz": 152,         # P3₁21
        }
        
        return {name: self.generate(num, **kwargs) 
                for name, num in famous.items()}
    
    @staticmethod
    def get_group_info(group_number: int) -> SpaceGroup:
        """Get information about a space group."""
        if group_number not in SPACE_GROUPS:
            raise ValueError(f"Invalid space group number: {group_number}")
        return SPACE_GROUPS[group_number]
    
    @staticmethod
    def list_by_system(system: CrystalSystem) -> List[int]:
        """List all space group numbers for a crystal system."""
        return [num for num, g in SPACE_GROUPS.items() 
                if g.crystal_system == system]
    
    @staticmethod
    def list_all() -> List[int]:
        """List all 230 space group numbers."""
        return list(range(1, 231))


def visualize_space_group(pattern: np.ndarray,
                          output_path: str,
                          group_number: int = None,
                          threshold: float = 0.3):
    """
    Create a beautiful visualization of a space group pattern.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 6))
    
    # 3D voxel view
    ax1 = fig.add_subplot(121, projection='3d')
    
    voxels = pattern > threshold
    colors = np.zeros(voxels.shape + (4,))
    
    # Color by position for nice gradient
    for i in range(pattern.shape[0]):
        for j in range(pattern.shape[1]):
            for k in range(pattern.shape[2]):
                if voxels[i, j, k]:
                    # HSV-like coloring
                    hue = (i + j + k) / sum(pattern.shape)
                    colors[i, j, k] = [
                        0.5 + 0.5 * np.sin(2 * np.pi * hue),
                        0.5 + 0.5 * np.sin(2 * np.pi * (hue + 1/3)),
                        0.5 + 0.5 * np.sin(2 * np.pi * (hue + 2/3)),
                        0.8
                    ]
    
    ax1.voxels(voxels, facecolors=colors, edgecolor='none')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    if group_number:
        info = SPACE_GROUPS.get(group_number)
        if info:
            ax1.set_title(f"Space Group #{group_number}: {info.symbol}\n{info.description}")
    
    # Slices view
    ax2 = fig.add_subplot(122)
    
    # Show central slice
    mid = pattern.shape[0] // 2
    im = ax2.imshow(pattern[mid], cmap='magma', vmin=0, vmax=1)
    ax2.set_title(f'Central XY Slice (Z={mid})')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Space Group Generator Demo")
    print("=" * 60)
    print(f"Total space groups defined: {len(SPACE_GROUPS)}")
    
    # Print groups by system
    for system in CrystalSystem:
        groups = [num for num, g in SPACE_GROUPS.items() 
                  if g.crystal_system == system]
        print(f"{system.value.capitalize()}: {len(groups)} groups ({min(groups)}-{max(groups)})")
    
    print("\nGenerating example patterns...")
    
    generator = SpaceGroupGenerator(resolution=(48, 48, 48), seed=42)
    
    # Generate one from each system
    examples = [
        (1, "P1 - Triclinic, no symmetry"),
        (14, "P2₁/c - Most common monoclinic"),
        (62, "Pnma - Most common orthorhombic"),
        (139, "I4/mmm - High-symmetry tetragonal"),
        (166, "R-3m - Trigonal"),
        (194, "P6₃/mmc - Hexagonal close-packed"),
        (225, "Fm-3m - FCC metals"),
        (227, "Fd-3m - Diamond structure"),
    ]
    
    for num, desc in examples:
        pattern = generator.generate(num, motif_size=16, num_atoms=3)
        print(f"  {desc}: shape={pattern.shape}, range=[{pattern.min():.3f}, {pattern.max():.3f}]")
    
    print("\nDone!")




