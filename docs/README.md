# 📖 Documentation Index

Welcome to the Crystallographic Pattern Generator documentation. This folder contains detailed technical documentation for the project.

---

## Quick Navigation

| Document | Description | Audience |
|----------|-------------|----------|
| [**Architecture**](ARCHITECTURE.md) | System design, data flow, model components | Developers, Contributors |
| [**API Reference**](API_REFERENCE.md) | Scripts, modules, and function documentation | Developers |
| [**Wallpaper Groups Math**](WALLPAPER_GROUPS_MATH.md) | Mathematical foundations of the 17 groups | Researchers, Students |
| [**Neural ODE Plan**](NEURAL_ODE_IMPLEMENTATION_PLAN.md) | Detailed implementation plan (reference) | Developers |
| [**Jupyter Guide**](Wallpaper_Groups_Guide.ipynb) | Interactive tutorial notebook | Learners |

---

## Document Descriptions

### 🏗️ [Architecture](ARCHITECTURE.md)

Complete system architecture including:
- High-level system overview
- Data pipeline (pattern generation → dataset → training)
- Model components (VAE, Flow Matching)
- Training and inference pipelines
- File organization

**Best for**: Understanding how the system works end-to-end.

---

### 📚 [API Reference](API_REFERENCE.md)

Complete API documentation including:
- All training scripts with command-line arguments
- Data generation scripts
- Visualization scripts
- Core Python modules and classes
- Usage examples and code snippets

**Best for**: Using the project in your own code or running scripts.

---

### 🔢 [Wallpaper Groups Math](WALLPAPER_GROUPS_MATH.md)

Mathematical foundations including:
- Periodic patterns and symmetry operations
- The crystallographic restriction (why only orders 1,2,3,4,6)
- Group theory basics
- Detailed analysis of each of the 17 groups
- Symmetry verification methods
- Implementation formulas

**Best for**: Understanding the mathematics behind wallpaper groups.

---

### 🔬 [Neural ODE Implementation Plan](NEURAL_ODE_IMPLEMENTATION_PLAN.md)

Comprehensive implementation plan including:
- Neural ODE mathematical framework
- VAE architecture specifications
- Dashboard design
- Training pipeline details
- Implementation checklist

**Best for**: Reference for the original implementation plan.

---

### 📓 [Jupyter Guide](Wallpaper_Groups_Guide.ipynb)

Interactive Jupyter notebook including:
- Visual examples of all 17 groups
- Step-by-step pattern generation
- Interactive exploration
- Symmetry visualization

**Best for**: Learning by doing, interactive exploration.

---

## Image Assets

All documentation images are organized in the `images/` folder:

```
docs/images/
├── wallpaper_groups/          # Individual group examples
│   ├── all_17_groups_overview.png
│   ├── p1.png ... p6m.png
│   └── lattice_comparison.png
│
├── transitions/               # Phase transition animations
│   ├── p1_to_p6m.gif
│   ├── p2_to_p4.gif
│   ├── p3_to_p6.gif
│   └── *.png (transition strips)
│
├── latent_space/              # Latent space visualizations
│   └── latent_epoch_*.png
│
└── flow_matching/             # Training visualizations
    └── training_curves.png
```

---

## For LLMs (AI Assistants)

This project is designed to be easily understood by language models:

### Key Facts
- **17 Wallpaper Groups**: The only ways to tile a 2D plane with repeating patterns
- **VAE (Variational Autoencoder)**: Learns 64-dimensional latent representations of 256×256 RGB patterns
- **Flow Matching**: State-of-the-art method for learning continuous transformations between distributions
- **Transitions**: The model can smoothly transform patterns from any symmetry group to any other

### Important Files
- `src/models/flow_matching_transition.py` - Core Flow Matching model
- `src/models/vae_simple_rgb.py` - RGB VAE implementation
- `src/dataset/pattern_generator.py` - Pattern generation for 17 groups
- `scripts/train_flow_matching.py` - Main training script

### Typical Workflow
1. Generate dataset with `generate_colored_dataset.py`
2. Train VAE with `train_simple_vae.py`
3. Train Flow Matching with `train_flow_matching.py`
4. Visualizations saved to `output/flow_matching_*/`

### Group Naming Convention
```
p1, p2                    # Oblique (no special symmetry, 180° rotation)
pm, pg, cm               # Rectangular with reflections/glides
pmm, pmg, pgg, cmm       # Rectangular with 2-fold rotation
p4, p4m, p4g             # Square lattice (90° rotation)
p3, p3m1, p31m, p6, p6m  # Hexagonal lattice (60°/120° rotation)
```

---

## Contributing

When adding documentation:

1. **Follow the existing style** - Use consistent formatting and structure
2. **Include code examples** - Show practical usage
3. **Add images** - Place them in `images/` with descriptive names
4. **Update this index** - Add new documents to the navigation table
5. **Be LLM-friendly** - Clear structure, explicit explanations, avoid ambiguity

---

## Related Resources

- [Main README](../README.md) - Project overview and quick start
- [Wallpaper Groups on Wikipedia](https://en.wikipedia.org/wiki/Wallpaper_group)
- [Flow Matching Paper](https://arxiv.org/abs/2210.02747) - Lipman et al. 2023
- [International Tables for Crystallography](https://it.iucr.org/)

