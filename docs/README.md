# Documentation Index

Technical documentation for the Crystallographic Pattern Generator.

---

## 🎮 Interactive Demo

**[Launch Interactive Symmetry Explorer →](https://cocuco-co.github.io/c17/)**

Explore the 17 wallpaper groups interactively:
- Apply symmetry operations (rotations, reflections, glides, translations)
- View original vs transformed patterns with correlation metrics
- See Cayley tables showing group multiplication
- Learn which symmetries preserve each pattern

---

## Documents

| Document | Description | Audience |
|----------|-------------|----------|
| [Interactive Demo](index.html) | Web-based symmetry explorer | Students, All |
| [Architecture](ARCHITECTURE.md) | System design, data flow, model components | Developers |
| [API Reference](API_REFERENCE.md) | Scripts, modules, function documentation | Developers |
| [Wallpaper Groups Guide](Wallpaper_Groups_Guide_executed.ipynb) | Mathematical foundations with visualizations | All |
| [Guía en Español](Wallpaper_Groups_Guide_ES_executed.ipynb) | Guía matemática en español | All |

---

## Image Assets

```
docs/images/
├── wallpaper_groups/          # Individual group examples
│   ├── all_17_groups_overview.png
│   └── p1.png ... p6m.png
├── transitions/               # Phase transition animations
│   ├── p1_to_p6m.gif
│   └── *.png (transition strips)
├── latent_space/              # Latent space visualizations
└── flow_matching/             # Training curves
```

---

## For LLMs

Key facts:
- **17 Wallpaper Groups**: The only ways to tile a 2D plane with repeating patterns
- **VAE**: Learns 64-dimensional latent representations of 256×256 RGB patterns
- **Flow Matching**: Learns continuous transformations between distributions

Important files:
- `src/models/flow_matching_transition.py` - Core Flow Matching model
- `src/models/vae_simple_rgb.py` - RGB VAE implementation
- `src/dataset/pattern_generator.py` - Pattern generation for 17 groups
- `scripts/train_flow_matching.py` - Main training script

Workflow:
1. Generate dataset: `generate_colored_dataset.py`
2. Train VAE: `train_simple_vae.py`
3. Train Flow Matching: `train_flow_matching.py`
4. Output: `output/flow_matching_*/`

Group naming:
```
p1, p2                    # Oblique
pm, pg, cm               # Rectangular with reflections/glides
pmm, pmg, pgg, cmm       # Rectangular with 2-fold rotation
p4, p4m, p4g             # Square lattice
p3, p3m1, p31m, p6, p6m  # Hexagonal lattice
```

---

## Related

- [Main README](../README.md)
- [Wallpaper Groups - Wikipedia](https://en.wikipedia.org/wiki/Wallpaper_group)
- [Flow Matching Paper](https://arxiv.org/abs/2210.02747)
