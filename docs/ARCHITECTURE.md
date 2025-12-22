# System Architecture

Architecture of the Crystallographic Pattern Generator and Phase Transition system.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data Pipeline](#2-data-pipeline)
3. [Model Components](#3-model-components)
4. [Training Pipeline](#4-training-pipeline)
5. [Inference Pipeline](#5-inference-pipeline)
6. [File Organization](#6-file-organization)

---

## 1. Overview

The system consists of three main components:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HIGH-LEVEL ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────┐     ┌───────────────┐     ┌────────────────────┐    │
│   │   PATTERN    │     │     VAE       │     │   FLOW MATCHING    │    │
│   │   GENERATOR  │────▶│   (Encoder/   │────▶│   (Transition      │    │
│   │              │     │    Decoder)   │     │    Learning)       │    │
│   └──────────────┘     └───────────────┘     └────────────────────┘    │
│                                                                          │
│   ▸ Generates the 17   ▸ Learns latent      ▸ Learns continuous       │
│     wallpaper groups     representations      transformations          │
│   ▸ RGB 256×256        ▸ 64-dimensional     ▸ Between any two         │
│   ▸ Mathematically     ▸ Clustered by         symmetry groups         │
│     correct symmetry     symmetry group                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Pipeline

### 2.1 Pattern Generation

The `WallpaperGroupGenerator` creates mathematically correct patterns for all 17 groups:

```python
# src/dataset/pattern_generator.py

generator = WallpaperGroupGenerator(resolution=256, seed=42)

# Each group uses specific symmetry operations:
# - p1: Only translations
# - p2: 180° rotation centers
# - p4m: 90° rotation + 4 mirror axes
# - p6m: 60° rotation + 6 mirror axes
# etc.

pattern = generator.generate("p6m", motif_size=64, complexity=4)
```

### 2.2 Dataset Structure

The HDF5 dataset format:

```
crystallographic_patterns_colored.h5
├── patterns: (8500, 256, 256, 3)  # RGB images, float64 [0,1]
├── labels: (8500,)                 # Group indices 0-16
├── group_names: (17,)              # ['p1', 'p2', ..., 'p6m']
└── metadata/
    ├── complexity: (8500,)
    ├── motif_size: (8500,)
    └── motif_type: (8500,)

splits.npz
├── train: (5950,)   # 70%
├── val: (1275,)     # 15%
└── test: (1275,)    # 15%
```

### 2.3 Transition Dataset

For Flow Matching training, we create transition pairs:

```python
# src/dataset/transition_dataset.py

class TransitionDataset(Dataset):
    """
    Creates (z_source, z_target, source_idx, target_idx) pairs.
    
    Strategies:
    - Cross-pattern: Different patterns from different groups
    - Same-pattern: Same latent, different target groups
    - Mixed: 50/50 combination (best results)
    """
```

---

## 3. Model Components

### 3.1 RGB VAE

**Purpose**: Learn compact latent representations of crystallographic patterns.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SimpleVAE Architecture                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ENCODER                                                             │
│  ─────────────────────────────────────────────────────────────────  │
│  Input: (B, 3, 256, 256) RGB                                        │
│      ↓                                                               │
│  Conv2d(3→32, k=4, s=2) + BN + LeakyReLU + ResBlock  → (B, 32, 128) │
│      ↓                                                               │
│  Conv2d(32→64, k=4, s=2) + BN + LeakyReLU + ResBlock → (B, 64, 64)  │
│      ↓                                                               │
│  Conv2d(64→128, k=4, s=2) + BN + LeakyReLU + ResBlock → (B, 128, 32)│
│      ↓                                                               │
│  Conv2d(128→256, k=4, s=2) + BN + LeakyReLU + ResBlock → (B, 256, 16)│
│      ↓                                                               │
│  Conv2d(256→512, k=4, s=2) + BN + LeakyReLU + ResBlock → (B, 512, 8)│
│      ↓                                                               │
│  Flatten → (B, 32768)                                                │
│      ↓                                                               │
│  ├── Linear(32768→64) → μ (mean)                                    │
│  └── Linear(32768→64) → log σ² (log variance)                       │
│                                                                      │
│  REPARAMETERIZATION                                                  │
│  ─────────────────────────────────────────────────────────────────  │
│  z = μ + σ × ε,  where ε ~ N(0, I)                                  │
│                                                                      │
│  DECODER                                                             │
│  ─────────────────────────────────────────────────────────────────  │
│  Input: z (B, 64)                                                    │
│      ↓                                                               │
│  Linear(64→32768) → Reshape → (B, 512, 8, 8)                        │
│      ↓                                                               │
│  ConvT(512→256) + BN + LeakyReLU + ResBlock → (B, 256, 16)         │
│      ↓                                                               │
│  ConvT(256→128) + BN + LeakyReLU + ResBlock → (B, 128, 32)         │
│      ↓                                                               │
│  ConvT(128→64) + BN + LeakyReLU + ResBlock → (B, 64, 64)           │
│      ↓                                                               │
│  ConvT(64→32) + BN + LeakyReLU + ResBlock → (B, 32, 128)           │
│      ↓                                                               │
│  ConvT(32→3) + Sigmoid → (B, 3, 256, 256)                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Loss Function**:
```
L = L_reconstruction + β × L_KL

L_reconstruction = MSE(x, x̂)
L_KL = -0.5 × Σ(1 + log σ² - μ² - σ²)
```

### 3.2 Flow Matching Transition Model

**Purpose**: Learn continuous transformations between symmetry groups.

```
┌─────────────────────────────────────────────────────────────────────┐
│                   FlowMatchingTransition Architecture                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUTS                                                              │
│  ─────────────────────────────────────────────────────────────────  │
│  z:           (B, 64)   - Current latent position                   │
│  t:           (B,)      - Time in [0, 1]                            │
│  source_idx:  (B,)      - Source group index (0-16)                 │
│  target_idx:  (B,)      - Target group index (0-16)                 │
│                                                                      │
│  EMBEDDINGS                                                          │
│  ─────────────────────────────────────────────────────────────────  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ Time Embed  │  │ Source Embed│  │ Target Embed│                 │
│  │ Sinusoidal  │  │ Learnable   │  │ Learnable   │                 │
│  │ → 64 dim    │  │ → 64 dim    │  │ → 64 dim    │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
│         │                │                │                         │
│         └────────────────┴────────────────┘                         │
│                          │                                           │
│                          ▼                                           │
│  CONCATENATION: [z, t_emb, source_emb, target_emb] → (B, 256)       │
│                                                                      │
│  VELOCITY NETWORK                                                    │
│  ─────────────────────────────────────────────────────────────────  │
│  Linear(256→512) + GELU                                             │
│      ↓                                                               │
│  [ResidualBlock + Self-Attention] × 6 layers                        │
│      ↓                                                               │
│  LayerNorm + Linear(512→512) + GELU + Linear(512→64)               │
│      ↓                                                               │
│  OUTPUT: v(z, t) ∈ ℝ^64 (predicted velocity)                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Training (Flow Matching Loss)**:
```python
# Sample random time t ~ Uniform[0, 1]
t = torch.rand(batch_size)

# Linear interpolation (Optimal Transport path)
z_t = (1 - t) * z_source + t * z_target

# Target velocity is constant for linear interpolation
v_target = z_target - z_source

# Predicted velocity
v_pred = model.velocity_net(z_t, t, source_idx, target_idx)

# Loss = MSE between predicted and target velocity
loss = F.mse_loss(v_pred, v_target)
```

**Inference (Euler Integration)**:
```python
def sample_trajectory(z_start, source_idx, target_idx, n_steps=50):
    dt = 1.0 / n_steps
    z = z_start
    trajectory = [z]
    
    for i in range(n_steps - 1):
        t = i * dt
        v = velocity_net(z, t, source_idx, target_idx)
        z = z + v * dt
        trajectory.append(z)
    
    return trajectory  # [n_steps, batch, latent_dim]
```

### 3.3 Group Embeddings

Each of the 17 groups gets a learnable 64-dimensional embedding, initialized with geometric properties:

```python
# Initialization based on group properties
embedding[i, 0] = rotation_order / 6.0           # Normalized rotation
embedding[i, 1] = sin(2π × rotation_order / 6)   # Periodic encoding
embedding[i, 2] = cos(2π × rotation_order / 6)
embedding[i, 3] = lattice_type / 3.0             # 0=oblique, 1=rect, 2=square, 3=hex
embedding[i, 4] = 1.0 if has_reflection else 0.0
embedding[i, 5] = 1.0 if has_glide else 0.0
embedding[i, 6:] = random_init(0, 0.1)           # Learned during training
```

---

## 4. Training Pipeline

### 4.1 Two-Phase Training

```
┌────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: Train VAE (~100 epochs)                 │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Objective: Learn latent representations for all 17 groups          │
│                                                                     │
│  Loss = L_reconstruction + β × L_KL                                │
│                                                                     │
│  β schedule: 0.0001 → 0.001 (gradually increase KL weight)         │
│                                                                     │
│  Output: checkpoints/vae_checkpoint.pt                              │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                  PHASE 2: Train Flow Matching (~100 epochs)         │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Prerequisites: Trained VAE (frozen during training)                │
│                                                                     │
│  1. Precompute all latent representations z = VAE.encode(pattern)  │
│  2. Create transition pairs (z_source, z_target)                   │
│  3. Train velocity network with Flow Matching loss                  │
│                                                                     │
│  Loss = ||v_pred(z_t, t) - v_target||²                             │
│                                                                     │
│  Output: checkpoints/flow_matching_checkpoint.pt                    │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### 4.2 Hyperparameters

```python
# VAE Training
VAE_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'lr': 1e-4,
    'beta': 0.001,           # KL weight
    'latent_dim': 64,
    'optimizer': 'AdamW',
    'weight_decay': 1e-5,
    'scheduler': 'CosineAnnealing',
}

# Flow Matching Training
FLOW_CONFIG = {
    'epochs': 100,
    'batch_size': 128,
    'lr': 1e-4,
    'hidden_dim': 512,
    'num_layers': 6,
    'embedding_dim': 64,
    'pairs_per_epoch': 20000,
    'optimizer': 'AdamW',
    'weight_decay': 1e-4,
    'scheduler': 'CosineAnnealing',
    'use_amp': True,         # Mixed precision training
}
```

---

## 5. Inference Pipeline

### 5.1 Generate Transition

```python
# 1. Load models
vae = load_vae("checkpoints/vae.pt")
flow = load_flow_matching("checkpoints/flow.pt")

# 2. Encode source pattern
source_pattern = load_pattern("p4m_example.png")
z_source = vae.encode(source_pattern)

# 3. Generate trajectory
trajectory = flow.sample_trajectory(
    z_start=z_source,
    source_idx=GROUP_TO_IDX["p4m"],
    target_idx=GROUP_TO_IDX["p6m"],
    n_steps=60
)

# 4. Decode each frame
frames = [vae.decode(z) for z in trajectory]

# 5. Create animation
create_gif(frames, "p4m_to_p6m.gif", fps=5)
```

### 5.2 Visualization Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                      VISUALIZATION OUTPUTS                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Every N epochs during training:                                     │
│                                                                      │
│  ├── Latent Space                                                    │
│  │   └── UMAP/PCA projection with all 17 groups color-coded        │
│  │                                                                   │
│  ├── Reconstructions                                                 │
│  │   └── Grid: Original → Reconstructed for each group              │
│  │                                                                   │
│  ├── Transition Strips                                               │
│  │   └── 12-frame horizontal strips showing evolution               │
│  │                                                                   │
│  ├── Transition GIFs                                                 │
│  │   └── 60-frame animated transitions at 5 FPS                     │
│  │                                                                   │
│  └── Training Curves                                                 │
│      └── Loss, learning rate, etc. over epochs                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. File Organization

```
src/
├── dataset/
│   ├── __init__.py
│   ├── pattern_generator.py      # WallpaperGroupGenerator
│   ├── color_patterns.py         # ColorSchemeGenerator
│   ├── dataset.py                # CrystallographicDataset
│   ├── dataset_colored.py        # ColoredPatternDataset
│   ├── transition_dataset.py     # TransitionDataset, H5PatternDataset
│   └── turing_patterns.py        # Turing pattern generation
│
├── models/
│   ├── __init__.py
│   ├── vae_simple_rgb.py         # SimpleVAE, SimpleVAEConfig
│   ├── flow_matching_transition.py  # FlowMatchingTransition
│   ├── neural_ode_transition.py  # NeuralODETransition (alternative)
│   ├── vae.py                    # Legacy VAE
│   └── trainer.py                # Training utilities
│
├── visualization/
│   ├── __init__.py
│   ├── visualize.py              # Pattern plotting
│   ├── latent_explorer.py        # Latent space visualization
│   └── training_logger.py        # Training progress logging
│
└── analysis/
    ├── __init__.py
    └── symmetry_verifier.py      # Verify symmetry correctness
```

---

## Summary

The architecture provides:
- Pattern generation for all 17 wallpaper groups
- Latent representations via VAE (256×256×3 → 64 dimensions)
- Continuous phase transitions via Flow Matching

