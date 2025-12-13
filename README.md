# Crystallographic Pattern Generator

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

A Python library for generating, visualizing, and learning from the **17 wallpaper groups** (plane crystallographic groups). Designed for training Variational Autoencoders (VAE) with CNN and GCN architectures.

## 🔷 The 17 Wallpaper Groups

The wallpaper groups represent the only 17 distinct ways to tile a 2D plane with a repeating pattern using combinations of:

| Lattice Type | Groups | Symmetries |
|-------------|--------|------------|
| **Oblique** | p1, p2 | Translation only, 2-fold rotation |
| **Rectangular** | pm, pg, cm, pmm, pmg, pgg, cmm | Reflections, glide reflections |
| **Square** | p4, p4m, p4g | 4-fold rotation with reflections |
| **Hexagonal** | p3, p3m1, p31m, p6, p6m | 3-fold and 6-fold rotations |

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd cristalography

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Generate Patterns

```python
from src.dataset import WallpaperGroupGenerator

# Create generator
generator = WallpaperGroupGenerator(resolution=256, seed=42)

# Generate a single pattern
pattern = generator.generate("p6m", motif_size=64, complexity=4)

# Generate all 17 groups
all_patterns = generator.generate_all(motif_size=64)
```

### Visualize

```python
from src.visualization import plot_all_groups, PatternVisualizer

# Plot all 17 groups
fig = plot_all_groups(save_path="all_groups.png")

# Visualize with symmetry annotations
viz = PatternVisualizer(style='dark')
viz.plot_symmetry_annotations(pattern, "p6m", save_path="symmetry.png")
```

### Create Dataset

```python
from src.dataset import CrystallographicDataset, create_dataloaders

# Create dataset
dataset = CrystallographicDataset(
    num_samples_per_group=100,
    resolution=128,
    representation="image"  # or "graph" for GCN
)

# Get dataloaders
train_loader, val_loader = create_dataloaders(
    batch_size=32,
    num_samples_per_group=500
)
```

## 📁 Project Structure

```
cristalography/
├── src/
│   ├── dataset/
│   │   ├── pattern_generator.py  # Core pattern generation
│   │   └── dataset.py            # PyTorch Dataset
│   ├── visualization/
│   │   └── visualize.py          # Visualization utilities
│   └── models/                   # VAE models (coming soon)
├── scripts/
│   ├── generate_dataset.py       # CLI for dataset generation
│   └── visualize_patterns.py     # CLI for visualization
├── data/                         # Generated datasets
├── output/                       # Visualizations
├── requirements.txt
└── README.md
```

## 🛠️ Command Line Tools

### Generate Dataset

```bash
# Generate full dataset (8500 samples: 500 per group × 17 groups)
python scripts/generate_dataset.py \
    --output-dir ./data/crystallographic \
    --samples-per-group 500 \
    --resolution 128

# Quick test with fewer samples
python scripts/generate_dataset.py -n 10 -r 64
```

### Visualize Patterns

```bash
# Generate all visualizations
python scripts/visualize_patterns.py --output-dir ./output/viz

# Specific groups with symmetry annotations
python scripts/visualize_patterns.py --groups p4m p6m cmm --with-symmetry

# Light theme
python scripts/visualize_patterns.py --style light
```

## 🎨 Pattern Examples

Each wallpaper group has unique symmetry properties:

| Group | Rotation | Reflection | Glide | Description |
|-------|----------|------------|-------|-------------|
| p1 | - | - | - | Translation only |
| p2 | 2-fold | - | - | 180° rotation |
| pm | - | ✓ | - | Mirror lines |
| pg | - | - | ✓ | Glide reflection |
| cm | - | ✓ | ✓ | Mirror + glide |
| pmm | 2-fold | ✓ | - | Perpendicular mirrors |
| pmg | 2-fold | ✓ | ✓ | Mirror + perpendicular glide |
| pgg | 2-fold | - | ✓ | Perpendicular glides |
| cmm | 2-fold | ✓ | ✓ | Centered rectangle |
| p4 | 4-fold | - | - | 90° rotation |
| p4m | 4-fold | ✓ | ✓ | Square with all mirrors |
| p4g | 4-fold | ✓ | ✓ | Square with glides |
| p3 | 3-fold | - | - | 120° rotation |
| p3m1 | 3-fold | ✓ | - | Mirrors through centers |
| p31m | 3-fold | ✓ | - | Mirrors between centers |
| p6 | 6-fold | - | - | 60° rotation |
| p6m | 6-fold | ✓ | ✓ | Full hexagonal symmetry |

## 🧠 Data Representations

The dataset supports two representations:

### Image (for CNN)
- Grayscale images: `[1, H, W]`
- Normalized to `[0, 1]`

### Graph (for GCN)
- Node features: intensity + local gradients
- Edges: spatial proximity-based
- Compatible with PyTorch Geometric

```python
# Image representation
dataset_cnn = CrystallographicDataset(representation="image")
image, label = dataset_cnn[0]  # [1, 128, 128], int

# Graph representation
dataset_gcn = CrystallographicDataset(representation="graph")
graph, label = dataset_gcn[0]  # {'x': [N, 3], 'edge_index': [2, E]}, int
```

## 📊 Dataset Statistics

With default settings (`500 samples/group × 17 groups`):
- **Total samples**: 8,500
- **Train/Val/Test split**: 70% / 15% / 15%
- **File size**: ~150 MB (HDF5, compressed)

## 🧠 Variational Autoencoder (VAE)

The project includes a complete VAE implementation for learning latent representations of crystallographic patterns.

### Architecture

```
Input [B, 1, 128, 128]
    ↓
Encoder (CNN with residual blocks)
    ↓
Latent Distribution (μ, log σ²)
    ↓
Reparametrization: z = μ + σ × ε
    ↓
Decoder (Transposed CNN)
    ↓
Output [B, 1, 128, 128]
```

### Training

```bash
# Quick training (fewer samples)
python scripts/train_vae.py --epochs 50 --samples-per-group 50

# Full training
python scripts/train_vae.py \
    --epochs 100 \
    --samples-per-group 200 \
    --latent-dim 128 \
    --batch-size 32 \
    --beta 1.0

# Resume from checkpoint
python scripts/train_vae.py --resume checkpoints/best_model.pt
```

### Evaluation

```bash
# Generate visualizations from trained model
python scripts/evaluate_vae.py --checkpoint checkpoints/best_model.pt
```

This generates:
- **Random samples** from the prior distribution
- **Reconstructions** for each wallpaper group
- **Latent interpolations** between different groups
- **t-SNE visualization** of the latent space

### Loss Function

The VAE uses a combined loss:

```
Loss = Reconstruction Loss + β × KL Divergence + γ × Classification Loss
```

- **Reconstruction**: MSE between input and output
- **KL Divergence**: Regularizes latent space toward N(0,1)
- **Classification**: Optional auxiliary task for group prediction

### Model Features

- **Interpolation**: Smooth transitions between patterns in latent space
- **Sampling**: Generate new patterns from the prior
- **Classification**: Predict wallpaper group from latent representation

```python
from src.models import CrystallographicVAE

model = CrystallographicVAE(latent_dim=128)

# Encode
mu, logvar = model.encode(images)

# Sample new patterns
samples = model.sample(num_samples=16, device='cuda')

# Interpolate between two patterns
interpolations = model.interpolate(img1, img2, steps=10)
```

## 🔬 Why CNN over GCN?

For these 2D pattern images, **CNN is the optimal choice**:

| Aspect | CNN | GCN |
|--------|-----|-----|
| Data type | Regular grid (images) ✓ | Irregular graphs |
| Spatial locality | Built-in convolution ✓ | Requires graph construction |
| Efficiency | Highly optimized ✓ | Additional overhead |
| Use case | Pattern images ✓ | Molecular structures |

**GCN would be appropriate if**: Working with actual crystal structures (atoms as nodes, bonds as edges), molecular graphs, or point cloud representations.

## 🔜 Future Improvements

- [ ] Conditional VAE (specify wallpaper group)
- [ ] β-VAE with annealing schedule
- [ ] Symmetry-aware loss functions
- [ ] Disentangled representations

## 📚 References

- [The 17 Wallpaper Groups](https://en.wikipedia.org/wiki/Wallpaper_group)
- [International Tables for Crystallography](https://it.iucr.org/)

## 📄 License

MIT License - See LICENSE file for details.

