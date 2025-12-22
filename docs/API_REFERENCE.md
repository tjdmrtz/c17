# API Reference

Reference for scripts and modules in the Crystallographic Pattern Generator.

---

## Table of Contents

1. [Training Scripts](#1-training-scripts)
2. [Data Generation Scripts](#2-data-generation-scripts)
3. [Visualization Scripts](#3-visualization-scripts)
4. [Core Modules](#4-core-modules)
5. [Model Modules](#5-model-modules)
6. [Dataset Modules](#6-dataset-modules)

---

## 1. Training Scripts

### `scripts/train_flow_matching.py`

**Main training script for Flow Matching phase transitions.**

```bash
python scripts/train_flow_matching.py \
    --vae-checkpoint <path>     # Required: Path to trained VAE
    --data-path <path>          # HDF5 dataset path
    --epochs 100                # Number of training epochs
    --batch-size 128            # Batch size
    --lr 1e-4                   # Learning rate
    --hidden-dim 512            # Hidden dimension of flow network
    --num-layers 6              # Number of layers in flow network
    --pairs-per-epoch 20000     # Transition pairs per epoch
    --viz-interval 10           # Epochs between visualizations
    --full-viz-interval 50      # Epochs between full visualizations
    --n-frames 60               # Frames in transition animations
    --no-amp                    # Disable mixed precision training
    --output-dir output         # Output directory
```

**Example:**
```bash
python scripts/train_flow_matching.py \
    --vae-checkpoint output/simple_vae_20251218/best_model.pt \
    --epochs 100 \
    --batch-size 128
```

**Output:**
```
output/flow_matching_YYYYMMDD_HHMMSS/
├── best_model.pt           # Best checkpoint
├── final_model.pt          # Final checkpoint
├── config.json             # Training configuration
├── history.json            # Loss history
├── training_curves.png     # Training visualization
├── live_state.json         # Live state for dashboard
├── visualizations/         # Per-epoch latent space plots
└── full_visualizations/    # Complete transition visualizations
    ├── transitions/*.gif   # Animated GIFs
    └── latent_space/       # Latent space plots
```

---

### `scripts/train_simple_vae.py`

**Train the RGB VAE for pattern reconstruction.**

```bash
python scripts/train_simple_vae.py \
    --data-path <path>          # HDF5 dataset path
    --epochs 100                # Number of epochs
    --batch-size 32             # Batch size
    --latent-dim 64             # Latent space dimension
    --lr 1e-4                   # Learning rate
    --beta 0.001                # KL divergence weight
    --output-dir output         # Output directory
```

**Output:**
```
output/simple_vae_YYYYMMDD_HHMMSS/
├── best_model.pt           # Best checkpoint
├── final_model.pt          # Final checkpoint
├── config.json             # Configuration
├── training_curves.png     # Loss curves
└── reconstructions/        # Sample reconstructions per epoch
```

---

### `scripts/train_neural_ode_transitions.py`

**Alternative: Train Neural ODE for transitions (slower, less stable).**

```bash
python scripts/train_neural_ode_transitions.py \
    --vae-checkpoint <path>     # Path to trained VAE
    --epochs 100                # Number of epochs
    --batch-size 64             # Batch size
    --lr 1e-3                   # Learning rate
    --solver dopri5             # ODE solver
    --use-adjoint               # Use adjoint method for memory efficiency
```

---

## 2. Data Generation Scripts

### `scripts/generate_colored_dataset.py`

**Generate the full RGB crystallographic pattern dataset.**

```bash
python scripts/generate_colored_dataset.py \
    --output-dir data/colored_crystallographic  # Output directory
    --samples-per-group 500                     # Samples per group (×17 groups)
    --resolution 256                            # Image resolution
    --seed 42                                   # Random seed
```

**Output:**
```
data/colored_crystallographic/
├── crystallographic_patterns_colored.h5    # Main dataset
├── splits.npz                              # Train/val/test splits
└── images/                                 # Sample images per group
    ├── p1/
    ├── p2/
    └── ...
```

---

### `scripts/generate_dataset.py`

**Generate grayscale crystallographic patterns (legacy).**

```bash
python scripts/generate_dataset.py \
    --output-dir data/crystallographic \
    --samples-per-group 500 \
    --resolution 128
```

---

### `scripts/generate_all_17_samples.py`

**Generate sample images for all 17 groups for documentation.**

```bash
python scripts/generate_all_17_samples.py \
    --output-dir docs/images/wallpaper_groups \
    --resolution 512
```

---

## 3. Visualization Scripts

### `scripts/dashboard_viewer.py`

**Live dashboard for monitoring training progress.**

```bash
python scripts/dashboard_viewer.py \
    --output-dir output/flow_matching_XXXX  # Training output directory
    --port 8080                              # Dashboard port
```

Opens a web browser with:
- Real-time loss curves
- Latent space visualization
- Transition animations
- Training logs

---

### `scripts/visualize_patterns.py`

**Generate visualizations of wallpaper group patterns.**

```bash
python scripts/visualize_patterns.py \
    --output-dir output/visualizations \
    --groups p4m p6m cmm           # Specific groups (or all)
    --with-symmetry                # Show symmetry annotations
    --style dark                   # dark or light theme
```

---

### `scripts/generate_visualizations.py`

**Generate comprehensive visualizations from a trained model.**

```bash
python scripts/generate_visualizations.py \
    --vae-checkpoint output/vae/best_model.pt \
    --flow-checkpoint output/flow/best_model.pt \
    --output-dir output/visualizations \
    --all-pairs                    # Generate all 17×17 transitions
```

---

## 4. Core Modules

### `src.dataset.pattern_generator`

**Core pattern generation for all 17 wallpaper groups.**

```python
from src.dataset.pattern_generator import WallpaperGroupGenerator

# Create generator
generator = WallpaperGroupGenerator(
    resolution=256,     # Output image size
    seed=42             # Random seed for reproducibility
)

# Generate single pattern
pattern = generator.generate(
    group="p6m",        # Wallpaper group name
    motif_size=64,      # Size of fundamental domain
    complexity=4,       # Number of elements in motif
    motif_type="gaussian"  # "gaussian", "geometric", or "mixed"
)

# Generate all 17 groups
all_patterns = generator.generate_all(motif_size=64)
# Returns: Dict[str, np.ndarray]
```

**Available groups:**
```python
ALL_17_GROUPS = [
    'p1', 'p2', 'pm', 'pg', 'cm', 
    'pmm', 'pmg', 'pgg', 'cmm',
    'p4', 'p4m', 'p4g', 
    'p3', 'p3m1', 'p31m', 'p6', 'p6m'
]
```

---

### `src.dataset.color_patterns`

**Generate beautiful color schemes for patterns.**

```python
from src.dataset.color_patterns import ColorSchemeGenerator

color_gen = ColorSchemeGenerator(seed=42)

# Generate random color scheme
colors = color_gen.generate_scheme(n_colors=5)

# Apply to grayscale pattern
colored_pattern = color_gen.apply_to_pattern(
    pattern,        # Grayscale pattern [H, W]
    scheme="warm"   # "warm", "cool", "vibrant", "pastel", "random"
)
# Returns: RGB image [H, W, 3]
```

---

## 5. Model Modules

### `src.models.vae_simple_rgb`

**Simple and effective RGB VAE.**

```python
from src.models.vae_simple_rgb import SimpleVAE, SimpleVAEConfig, SimpleVAELoss

# Configuration
config = SimpleVAEConfig(
    resolution=256,
    in_channels=3,
    latent_dim=64,
    hidden_dims=(32, 64, 128, 256, 512),
    beta=0.001,      # KL weight
)

# Create model
model = SimpleVAE(config)

# Forward pass
outputs = model(images, group_name="p4m")
# Returns: {'recon': Tensor, 'mu': Tensor, 'logvar': Tensor, 'z': Tensor}

# Encode
z = model.encode(images, group_name="p4m")  # Returns mu only

# Decode
reconstructed = model.decode(z, group_name="p4m")

# Sample from prior
samples = model.sample(n_samples=16, device='cuda')

# Loss computation
loss_fn = SimpleVAELoss(config)
losses = loss_fn(images, outputs, group_name="p4m")
# Returns: {'loss': Tensor, 'recon_loss': Tensor, 'kl_loss': Tensor}
```

---

### `src.models.flow_matching_transition`

**Conditional Flow Matching for phase transitions.**

```python
from src.models.flow_matching_transition import (
    FlowMatchingTransition,
    FlowMatchingConfig,
    FlowMatchingMetrics,
    ALL_17_GROUPS,
    GROUP_TO_IDX,
    IDX_TO_GROUP,
)

# Configuration
config = FlowMatchingConfig(
    latent_dim=64,
    hidden_dim=512,
    num_layers=6,
    embedding_dim=64,
    time_embedding_dim=64,
    use_attention=True,
    num_heads=8,
    dropout=0.1,
    sigma_min=1e-4,
    use_optimal_transport=True,
    lambda_velocity=0.01,
)

# Create model
model = FlowMatchingTransition(config)

# Training: compute loss
losses = model.compute_loss(
    z_source,       # [B, 64] source latents
    z_target,       # [B, 64] target latents
    source_idx,     # [B] source group indices
    target_idx,     # [B] target group indices
)
# Returns: {'loss': Tensor, 'flow_loss': Tensor, 'velocity_reg': Tensor}

# Inference: sample trajectory
trajectory = model.sample_trajectory(
    z_start,        # [B, 64] starting points
    source_idx,     # [B] source group indices
    target_idx,     # [B] target group indices
    n_steps=50,     # Number of integration steps
)
# Returns: [n_steps, B, 64] trajectory

# Inference: get endpoint only
z_end = model.sample_endpoint(z_start, source_idx, target_idx, n_steps=20)

# Compute metrics
metrics = FlowMatchingMetrics.compute_metrics(
    model, z_source, z_target, source_idx, target_idx
)
# Returns: {'endpoint_mse': float, 'smoothness': float, 
#           'path_length': float, 'straightness': float}
```

---

### `src.models.neural_ode_transition`

**Alternative: Neural ODE for transitions (legacy).**

```python
from src.models.neural_ode_transition import NeuralODETransition

model = NeuralODETransition(
    latent_dim=64,
    hidden_dim=256,
    embedding_dim=32,
    use_adjoint=True,
    solver='dopri5',
)

# Similar API to FlowMatchingTransition
trajectory = model(z_start, source_idx, target_idx, n_steps=50)
```

---

## 6. Dataset Modules

### `src.dataset.transition_dataset`

**Dataset classes for Flow Matching training.**

```python
from src.dataset.transition_dataset import H5PatternDataset, TransitionDataset

# Load patterns from HDF5
h5_dataset = H5PatternDataset(
    h5_path="data/crystallographic.h5",
    split="train",      # "train", "val", or "test"
    transform=None,     # Optional transforms
)

# Access patterns
pattern = h5_dataset[0]                          # Get by index
pattern = h5_dataset.get_pattern_by_group(5, 0)  # Get by group, index
indices = h5_dataset.indices_by_group[5]         # Get all indices for group

# Create transition pairs
transition_dataset = TransitionDataset(
    vae=vae,                    # Pretrained VAE
    h5_dataset=h5_dataset,      # Pattern dataset
    device='cuda',
    pairs_per_epoch=20000,      # Number of pairs per epoch
    seed=42,
)

# Get transition pair
z_source, z_target, source_idx, target_idx = transition_dataset[0]

# Regenerate pairs (call each epoch for variety)
transition_dataset.regenerate_pairs()
```

---

### `src.dataset.dataset`

**PyTorch Dataset classes for crystallographic patterns.**

```python
from src.dataset.dataset import CrystallographicDataset, create_dataloaders

# Create dataset
dataset = CrystallographicDataset(
    num_samples_per_group=500,
    resolution=256,
    representation="image",  # "image" or "graph"
    seed=42,
)

# Get sample
image, label = dataset[0]
# image: [C, H, W] tensor
# label: int (group index 0-16)

# Create dataloaders
train_loader, val_loader = create_dataloaders(
    batch_size=32,
    num_samples_per_group=500,
    split_ratio=0.8,
)
```

---

## Quick Reference

### Group Index Mapping

```python
GROUP_TO_IDX = {
    'p1': 0, 'p2': 1, 'pm': 2, 'pg': 3, 'cm': 4,
    'pmm': 5, 'pmg': 6, 'pgg': 7, 'cmm': 8,
    'p4': 9, 'p4m': 10, 'p4g': 11,
    'p3': 12, 'p3m1': 13, 'p31m': 14, 'p6': 15, 'p6m': 16
}

IDX_TO_GROUP = {v: k for k, v in GROUP_TO_IDX.items()}
```

### Common Imports

```python
# Pattern generation
from src.dataset.pattern_generator import WallpaperGroupGenerator

# Models
from src.models.vae_simple_rgb import SimpleVAE, SimpleVAEConfig
from src.models.flow_matching_transition import FlowMatchingTransition, FlowMatchingConfig

# Datasets
from src.dataset.transition_dataset import H5PatternDataset, TransitionDataset

# Constants
from src.models.flow_matching_transition import ALL_17_GROUPS, GROUP_TO_IDX, IDX_TO_GROUP
```

### Typical Training Pipeline

```python
import torch
from src.models.vae_simple_rgb import SimpleVAE, SimpleVAEConfig
from src.models.flow_matching_transition import FlowMatchingTransition, FlowMatchingConfig
from src.dataset.transition_dataset import H5PatternDataset, TransitionDataset

# 1. Load pretrained VAE
vae = SimpleVAE(SimpleVAEConfig(latent_dim=64))
vae.load_state_dict(torch.load("vae.pt")["model_state_dict"])
vae.eval()

# 2. Create datasets
h5_data = H5PatternDataset("patterns.h5", split="train")
transition_data = TransitionDataset(vae, h5_data, device="cuda")

# 3. Create Flow Matching model
flow = FlowMatchingTransition(FlowMatchingConfig(latent_dim=64))

# 4. Training loop
optimizer = torch.optim.AdamW(flow.parameters(), lr=1e-4)
for epoch in range(100):
    for z_src, z_tgt, src_idx, tgt_idx in dataloader:
        losses = flow.compute_loss(z_src, z_tgt, src_idx, tgt_idx)
        losses['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()

# 5. Generate transition
trajectory = flow.sample_trajectory(z_start, src_idx, tgt_idx, n_steps=60)
frames = [vae.decode(z) for z in trajectory]
```

