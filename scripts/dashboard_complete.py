#!/usr/bin/env python3
"""
Flow Matching Dashboard - Clean & Functional Design.

Usage:
    python scripts/dashboard_complete.py [--output-dir OUTPUT_DIR] [--port 8080]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import base64
from pathlib import Path
from io import BytesIO

from nicegui import ui, app
import plotly.graph_objects as go


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_latest_output_dir(base_dir: Path = Path('output')) -> Path:
    """Find the latest training output directory."""
    if not base_dir.exists():
        return None
    dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('flow_matching_')],
        key=lambda d: d.name, reverse=True
    )
    return dirs[0] if dirs else None


def load_json(filepath: Path) -> dict:
    """Safely load JSON file."""
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}


def image_to_base64(path: Path) -> str:
    """Convert image to base64 string."""
    if not path.exists():
        return ''
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_all_images(output_dir: Path, pattern: str) -> list:
    """Recursively find images matching pattern."""
    images = []
    for p in output_dir.rglob(pattern):
        images.append(p)
    return sorted(images, key=lambda x: x.stem)


# ============================================================================
# DASHBOARD APP
# ============================================================================

def create_dashboard(output_dir: Path, port: int = 8080):
    """Create and run the dashboard."""
    
    # Serve static files from output directory
    app.add_static_files('/static', str(output_dir))
    
    @ui.page('/')
    async def main_page():
        state = load_json(output_dir / 'live_state.json')
        config = load_json(output_dir / 'config.json')
        
        # Inject custom CSS for full-screen layout
        ui.add_head_html('''
        <style>
            * { box-sizing: border-box; }
            body, html { 
                margin: 0; padding: 0; 
                background: #0d1117; 
                color: #e6edf3;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            .nicegui-content { padding: 0 !important; }
            .main-container { 
                min-height: 100vh; 
                padding: 16px 24px; 
            }
            .card {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 12px;
                padding: 16px;
            }
            .card-header {
                font-size: 14px;
                font-weight: 600;
                color: #8b949e;
                margin-bottom: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .stat-value { 
                font-size: 32px; 
                font-weight: 700; 
                color: #58a6ff;
            }
            .stat-label { 
                font-size: 12px; 
                color: #8b949e; 
                margin-top: 4px;
            }
            .img-container {
                background: #0d1117;
                border-radius: 8px;
                padding: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                transition: all 0.2s;
                border: 1px solid transparent;
            }
            .img-container:hover {
                border-color: #58a6ff;
                transform: scale(1.02);
            }
            .transition-card {
                background: #21262d;
                border-radius: 8px;
                padding: 12px;
                text-align: center;
            }
            .transition-label {
                font-size: 13px;
                font-weight: 600;
                color: #c9d1d9;
                margin-bottom: 8px;
            }
            .section-title {
                font-size: 18px;
                font-weight: 600;
                color: #f0f6fc;
                margin-bottom: 16px;
                padding-bottom: 8px;
                border-bottom: 1px solid #30363d;
            }
            .tab-active { 
                border-bottom: 2px solid #58a6ff !important; 
            }
            /* Scrollbar */
            ::-webkit-scrollbar { width: 8px; height: 8px; }
            ::-webkit-scrollbar-track { background: #161b22; }
            ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: #484f58; }
        </style>
        ''')
        
        # ========== HEADER ==========
        with ui.element('div').classes('w-full').style('background: #161b22; border-bottom: 1px solid #30363d; padding: 12px 24px;'):
            with ui.row().classes('w-full items-center justify-between'):
                with ui.row().classes('items-center gap-3'):
                    ui.icon('psychology', size='md').style('color: #58a6ff;')
                    ui.label('Flow Matching Dashboard').style('font-size: 20px; font-weight: 700; color: #f0f6fc;')
                
                with ui.row().classes('items-center gap-4'):
                    # Status indicator
                    epoch = state.get('epoch', 0)
                    total = state.get('total_epochs', 0)
                    is_training = epoch < total and epoch > 0
                    
                    if is_training:
                        with ui.row().classes('items-center gap-2'):
                            ui.element('div').style('width: 8px; height: 8px; background: #3fb950; border-radius: 50%; animation: pulse 1.5s infinite;')
                            ui.label(f'Training: {epoch}/{total}').style('color: #3fb950; font-weight: 600;')
                    else:
                        with ui.row().classes('items-center gap-2'):
                            ui.element('div').style('width: 8px; height: 8px; background: #f0883e; border-radius: 50%;')
                            ui.label('Complete' if epoch > 0 else 'Idle').style('color: #f0883e; font-weight: 600;')
                    
                    ui.button(icon='refresh', on_click=lambda: ui.navigate.reload()).props('flat round').style('color: #8b949e;')
        
        # ========== MAIN CONTENT ==========
        with ui.element('div').classes('main-container'):
            
            # Tabs
            with ui.tabs().classes('w-full').style('background: transparent;') as tabs:
                overview_tab = ui.tab('overview', label='Overview').style('color: #c9d1d9;')
                transitions_tab = ui.tab('transitions', label='Transitions').style('color: #c9d1d9;')
                latent_tab = ui.tab('latent', label='Latent Space').style('color: #c9d1d9;')
                gallery_tab = ui.tab('gallery', label='Gallery').style('color: #c9d1d9;')
            
            with ui.tab_panels(tabs, value=overview_tab).classes('w-full').style('background: transparent;'):
                
                # ========== OVERVIEW TAB ==========
                with ui.tab_panel(overview_tab).style('padding: 20px 0;'):
                    
                    # Stats row
                    with ui.row().classes('w-full gap-4 mb-6'):
                        for label, value, color in [
                            ('Epoch', f"{state.get('epoch', 0)}/{state.get('total_epochs', 0)}", '#58a6ff'),
                            ('Loss', f"{state.get('current_loss', 0):.4f}" if state.get('current_loss') else 'N/A', '#3fb950'),
                            ('Flow Loss', f"{state.get('flow_loss', 0):.4f}" if state.get('flow_loss') else 'N/A', '#a371f7'),
                        ]:
                            with ui.card().classes('card flex-1 text-center'):
                                ui.label(label).classes('stat-label')
                                ui.label(value).style(f'font-size: 28px; font-weight: 700; color: {color};')
                    
                    # Loss chart
                    with ui.card().classes('card w-full mb-6'):
                        ui.label('Training Progress').classes('card-header')
                        
                        history = state.get('losses', {})
                        if history and 'loss' in history:
                            losses = history['loss']
                            epochs_list = list(range(1, len(losses) + 1))
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=epochs_list, y=losses,
                                mode='lines', name='Total Loss',
                                line=dict(color='#58a6ff', width=2)
                            ))
                            if 'flow_loss' in history:
                                fig.add_trace(go.Scatter(
                                    x=epochs_list, y=history['flow_loss'],
                                    mode='lines', name='Flow Loss',
                                    line=dict(color='#a371f7', width=2)
                                ))
                            
                            fig.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#8b949e'),
                                margin=dict(l=40, r=20, t=20, b=40),
                                height=300,
                                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                                xaxis=dict(gridcolor='#30363d', title='Epoch'),
                                yaxis=dict(gridcolor='#30363d', title='Loss'),
                            )
                            ui.plotly(fig).classes('w-full')
                        else:
                            ui.label('No training data yet...').style('color: #8b949e;')
                    
                    # Config preview
                    with ui.card().classes('card w-full'):
                        ui.label('Configuration').classes('card-header')
                        if config:
                            with ui.row().classes('flex-wrap gap-4'):
                                for key, val in list(config.items())[:8]:
                                    with ui.element('div').style('background: #0d1117; padding: 8px 12px; border-radius: 6px;'):
                                        ui.label(key).style('font-size: 11px; color: #8b949e;')
                                        ui.label(str(val)[:20]).style('font-size: 14px; font-weight: 600; color: #c9d1d9;')
                        else:
                            ui.label('No config available').style('color: #8b949e;')
                
                # ========== TRANSITIONS TAB ==========
                with ui.tab_panel(transitions_tab).style('padding: 20px 0;'):
                    
                    # Helper to get static URL
                    def static_url(path: Path) -> str:
                        try:
                            rel = path.relative_to(output_dir)
                            return f'/static/{rel}'
                        except ValueError:
                            return f'data:image/png;base64,{image_to_base64(path)}'
                    
                    # Find all GIFs and strips
                    gifs = get_all_images(output_dir, '*.gif')
                    strips = [p for p in get_all_images(output_dir, '*.png') if '_to_' in p.stem and 'latent' not in p.stem]
                    
                    ui.label('Transitions').classes('section-title')
                    ui.label(f'{len(gifs)} animations • {len(strips)} strips').style('color: #8b949e; margin-bottom: 16px; margin-top: -8px;')
                    
                    # Modal for viewing details
                    dialog = ui.dialog().props('maximized')
                    
                    with dialog:
                        with ui.card().style('background: #0d1117; width: 100%; min-height: 100vh; padding: 32px;'):
                            with ui.row().classes('w-full justify-between items-center mb-8'):
                                dialog_title = ui.label('').style('font-size: 28px; font-weight: 700; color: #f0f6fc;')
                                ui.button('✕ Close', on_click=dialog.close).props('flat').style('color: #f85149; font-size: 16px;')
                            
                            with ui.row().classes('w-full gap-8'):
                                # Left: Animation (larger)
                                with ui.column().classes('flex-1'):
                                    ui.label('🎬 Animation').style('color: #58a6ff; font-size: 16px; font-weight: 600; margin-bottom: 12px;')
                                    with ui.card().style('background: #161b22; border-radius: 12px; padding: 20px;'):
                                        dialog_animation = ui.image('').style('width: 400px; height: 400px; object-fit: contain; border-radius: 8px;')
                                
                                # Right: Strip + Latent
                                with ui.column().classes('flex-1'):
                                    ui.label('📊 Frame Progression').style('color: #3fb950; font-size: 16px; font-weight: 600; margin-bottom: 12px;')
                                    with ui.card().style('background: #161b22; border-radius: 12px; padding: 16px; margin-bottom: 24px;'):
                                        dialog_strip = ui.image('').style('width: 100%; max-height: 150px; border-radius: 8px; object-fit: contain;')
                                    
                                    ui.label('🎯 Latent Space Trajectory').style('color: #f0883e; font-size: 16px; font-weight: 600; margin-bottom: 12px;')
                                    with ui.card().style('background: #161b22; border-radius: 12px; padding: 16px;'):
                                        dialog_latent = ui.image('').style('width: 100%; max-height: 350px; border-radius: 8px; object-fit: contain;')
                    
                    def show_transition(name: str, gif_path: Path):
                        dialog_title.set_text(name)
                        
                        # Set animation using static file
                        if gif_path.exists():
                            dialog_animation.set_source(static_url(gif_path))
                        
                        # Find matching strip
                        strip_candidates = list(gif_path.parent.glob(f'{gif_path.stem}*.png'))
                        strip_candidates = [s for s in strip_candidates if 'latent' not in s.stem]
                        if strip_candidates:
                            dialog_strip.set_source(static_url(strip_candidates[0]))
                        
                        # Find matching latent
                        latent_path = gif_path.parent / f'{gif_path.stem}_latent.png'
                        if latent_path.exists():
                            dialog_latent.set_source(static_url(latent_path))
                        else:
                            general_latent = list(output_dir.rglob('latent_space*.png'))
                            if general_latent:
                                dialog_latent.set_source(static_url(general_latent[0]))
                        
                        dialog.open()
                    
                    if gifs:
                        # Grid of transitions using flexbox
                        with ui.row().classes('flex-wrap gap-4'):
                            for gif in gifs[:50]:
                                name = gif.stem.replace('_to_', ' → ')
                                with ui.card().classes('transition-card').style('width: 200px; cursor: pointer;').on('click', lambda e, n=name, g=gif: show_transition(n, g)):
                                    ui.label(name).classes('transition-label')
                                    ui.image(static_url(gif)).style('width: 100%; height: 160px; object-fit: contain; border-radius: 6px;')
                    elif strips:
                        with ui.row().classes('flex-wrap gap-4'):
                            for strip in strips[:50]:
                                name = strip.stem.replace('_to_', ' → ').replace('_s1', '').replace('_s2', '')
                                with ui.card().classes('transition-card').style('width: 220px;'):
                                    ui.label(name).classes('transition-label')
                                    ui.image(static_url(strip)).style('width: 100%; border-radius: 6px;')
                    else:
                        with ui.element('div').style('text-align: center; padding: 60px;'):
                            ui.icon('hourglass_empty', size='xl').style('color: #484f58;')
                            ui.label('No transitions yet').style('color: #8b949e; font-size: 18px; margin-top: 16px;')
                            ui.label('Transitions will appear after training generates them').style('color: #484f58; font-size: 14px;')
                
                # ========== LATENT SPACE TAB ==========
                with ui.tab_panel(latent_tab).style('padding: 20px 0;'):
                    
                    # Reuse static_url helper
                    def latent_static_url(path: Path) -> str:
                        try:
                            rel = path.relative_to(output_dir)
                            return f'/static/{rel}'
                        except ValueError:
                            return f'data:image/png;base64,{image_to_base64(path)}'
                    
                    ui.label('Latent Space Visualization').classes('section-title')
                    
                    latent_images = get_all_images(output_dir, 'latent_space*.png')
                    
                    if latent_images:
                        latest = latent_images[-1]
                        with ui.card().classes('card').style('padding: 24px;'):
                            ui.image(latent_static_url(latest)).style('width: 100%; max-height: 75vh; object-fit: contain; border-radius: 8px;')
                            ui.label(f'{latest.stem}').style('color: #8b949e; font-size: 12px; margin-top: 12px;')
                    else:
                        with ui.element('div').style('text-align: center; padding: 60px;'):
                            ui.icon('scatter_plot', size='xl').style('color: #484f58;')
                            ui.label('No latent space visualization yet').style('color: #8b949e; font-size: 18px; margin-top: 16px;')
                
                # ========== GALLERY TAB ==========
                with ui.tab_panel(gallery_tab).style('padding: 20px 0;'):
                    
                    def gallery_static_url(path: Path) -> str:
                        try:
                            rel = path.relative_to(output_dir)
                            return f'/static/{rel}'
                        except ValueError:
                            return f'data:image/png;base64,{image_to_base64(path)}'
                    
                    ui.label('Image Gallery').classes('section-title')
                    
                    recon_images = get_all_images(output_dir, 'recon*.png')
                    gallery_images = get_all_images(output_dir, 'gallery*.png')
                    all_groups_images = get_all_images(output_dir, 'all_groups*.png')
                    
                    all_gallery = recon_images + gallery_images + all_groups_images
                    
                    if all_gallery:
                        with ui.element('div').style('display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px;'):
                            for img in all_gallery[:20]:
                                with ui.card().classes('card').style('padding: 16px;'):
                                    ui.image(gallery_static_url(img)).style('width: 100%; border-radius: 8px;')
                                    ui.label(img.stem).style('color: #8b949e; font-size: 12px; margin-top: 8px;')
                    else:
                        with ui.element('div').style('text-align: center; padding: 60px;'):
                            ui.icon('photo_library', size='xl').style('color: #484f58;')
                            ui.label('No gallery images yet').style('color: #8b949e; font-size: 18px; margin-top: 16px;')
        
        # Auto-refresh timer
        async def refresh_state():
            new_state = load_json(output_dir / 'live_state.json')
            if new_state.get('epoch', 0) != state.get('epoch', 0):
                ui.navigate.reload()
        
        ui.timer(10, refresh_state)
    
    # Run the app
    ui.run(
        host='0.0.0.0',
        port=port,
        title='Flow Matching Dashboard',
        favicon='🔬',
        dark=True,
        reload=False,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = get_latest_output_dir()
    
    if output_dir is None or not output_dir.exists():
        print("No training output found. Starting dashboard anyway...")
        output_dir = Path('output/empty')
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Dashboard for: {output_dir}")
    print(f"Open: http://localhost:{args.port}")
    
    create_dashboard(output_dir, args.port)


if __name__ == '__main__':
    main()
