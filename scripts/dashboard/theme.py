"""
Color theme for the training dashboard.
"""

THEME = {
    'background': '#0a0a0a',
    'card': '#1a1a1a',
    'card_hover': '#252525',
    'primary': '#4ecdc4',
    'secondary': '#ff6b6b',
    'accent': '#ffd93d',
    'text': '#ffffff',
    'text_muted': '#888888',
    
    # Loss component colors
    'loss_total': '#4ecdc4',
    'loss_endpoint': '#ff6b6b',
    'loss_smooth': '#ffd93d',
    'loss_velocity': '#a29bfe',
}

# Colors for each wallpaper group (by lattice type)
GROUP_COLORS = {
    # Oblique (blue tones)
    'p1': '#1a5276',
    'p2': '#2980b9',
    
    # Rectangular (green tones)
    'pm': '#27ae60',
    'pg': '#2ecc71',
    'cm': '#58d68d',
    'pmm': '#145a32',
    'pmg': '#196f3d',
    'pgg': '#1d8348',
    'cmm': '#239b56',
    
    # Square (purple tones)
    'p4': '#6c3483',
    'p4m': '#8e44ad',
    'p4g': '#a569bd',
    
    # Hexagonal (orange/red tones)
    'p3': '#b9770e',
    'p3m1': '#d68910',
    'p31m': '#f39c12',
    'p6': '#c0392b',
    'p6m': '#e74c3c',
}

LATTICE_COLORS = {
    'Oblique': '#2980b9',
    'Rectangular': '#27ae60',
    'Square': '#8e44ad',
    'Hexagonal': '#e74c3c',
}

# CSS styles
GLOBAL_CSS = """
:root {
    --bg-color: #0a0a0a;
    --card-color: #1a1a1a;
    --primary: #4ecdc4;
    --secondary: #ff6b6b;
    --accent: #ffd93d;
}

body {
    background-color: var(--bg-color) !important;
    color: #ffffff;
}

.nicegui-card {
    background-color: var(--card-color) !important;
    border: 1px solid #333 !important;
}

.q-tab--active {
    color: var(--primary) !important;
}

.q-linear-progress__track {
    background-color: #333 !important;
}

.q-linear-progress__model {
    background-color: var(--primary) !important;
}
"""



