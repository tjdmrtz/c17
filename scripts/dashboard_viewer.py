#!/usr/bin/env python3
"""
Live Dashboard Viewer for Flow Matching Training.

Reads training state from output directory and displays in real-time.
Run this in a separate terminal while training is running.

Usage:
    python scripts/dashboard_viewer.py --output-dir output/flow_matching_XXXX
    
Then open http://localhost:8080 in your browser.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import asyncio

from nicegui import ui, app
import plotly.graph_objects as go

# Group colors
GROUP_COLORS = {
    'p1': '#FF6B6B', 'p2': '#4ECDC4', 'pm': '#45B7D1', 'pg': '#96CEB4',
    'cm': '#FFEAA7', 'pmm': '#DDA0DD', 'pmg': '#98D8C8', 'pgg': '#F7DC6F',
    'cmm': '#BB8FCE', 'p4': '#85C1E9', 'p4m': '#F8B500', 'p4g': '#00CED1',
    'p3': '#FF6347', 'p3m1': '#7B68EE', 'p31m': '#3CB371', 'p6': '#FF69B4',
    'p6m': '#00FA9A',
}


class DashboardViewer:
    """Dashboard that reads from output directory."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.live_state_path = self.output_dir / 'live_state.json'
        self.history_path = self.output_dir / 'history.json'
        self.viz_dir = self.output_dir / 'visualizations'
        
        # State
        self.current_state = {}
        self.last_update = 0
        
        # UI elements
        self.epoch_label = None
        self.loss_label = None
        self.status_label = None
        self.progress = None
        self.loss_chart = None
        self.latent_image = None
        self.log_area = None
    
    def load_state(self):
        """Load current training state from files."""
        state = {}
        
        # Load live state
        if self.live_state_path.exists():
            try:
                with open(self.live_state_path, 'r') as f:
                    state = json.load(f)
                self.last_update = self.live_state_path.stat().st_mtime
            except:
                pass
        
        # Load history
        if self.history_path.exists():
            try:
                with open(self.history_path, 'r') as f:
                    state['full_history'] = json.load(f)
            except:
                pass
        
        self.current_state = state
        return state
    
    def get_latest_viz(self):
        """Get path to latest visualization image."""
        if not self.viz_dir.exists():
            return None
        
        images = sorted(self.viz_dir.glob('latent_epoch_*.png'))
        if images:
            return images[-1]
        return None
    
    def setup(self):
        """Create the dashboard UI."""
        ui.add_head_html('''
        <style>
            body { background: #0a0a0a !important; }
            .nicegui-content { background: #0a0a0a !important; }
            .q-card { background: #1a1a1a !important; border: 1px solid #333 !important; }
            .q-linear-progress { background: #333 !important; }
        </style>
        ''')
        
        # Header
        with ui.header().classes('bg-gray-900 text-white items-center'):
            ui.label('🔬 Flow Matching Training Dashboard').classes('text-xl font-bold')
            ui.space()
            self.status_label = ui.label('Connecting...').classes('text-yellow-400')
            ui.button(icon='refresh', on_click=self.refresh).props('flat color=white')
        
        with ui.column().classes('w-full p-4 gap-4'):
            # Status bar
            with ui.card().classes('w-full'):
                with ui.row().classes('w-full items-center gap-6'):
                    self.epoch_label = ui.label('Epoch: -/-').classes('text-lg text-white')
                    ui.separator().props('vertical')
                    self.loss_label = ui.label('Loss: -').classes('text-lg text-cyan-400')
                    ui.separator().props('vertical')
                    self.best_label = ui.label('Best: -').classes('text-lg text-green-400')
                
                self.progress = ui.linear_progress(value=0, show_value=False).classes('w-full mt-2')
            
            # Main content
            with ui.row().classes('w-full gap-4'):
                # Loss chart
                with ui.card().classes('flex-1'):
                    ui.label('Training Loss').classes('text-white font-bold mb-2')
                    self.loss_chart = ui.plotly({}).classes('w-full h-64')
                
                # Latent space visualization
                with ui.card().classes('flex-1'):
                    ui.label('Latent Space').classes('text-white font-bold mb-2')
                    self.latent_image = ui.image('').classes('w-full')
            
            # Output directory info
            with ui.card().classes('w-full'):
                ui.label(f'📁 Monitoring: {self.output_dir}').classes('text-gray-400 text-sm')
        
        # Start auto-refresh
        ui.timer(2.0, self.refresh)
    
    def refresh(self):
        """Refresh dashboard with latest data."""
        state = self.load_state()
        
        if not state:
            self.status_label.text = '⏳ Waiting for training...'
            self.status_label.classes(remove='text-green-400', add='text-yellow-400')
            return
        
        # Update status
        is_training = state.get('is_training', False)
        if is_training:
            self.status_label.text = '🟢 Training in progress'
            self.status_label.classes(remove='text-yellow-400', add='text-green-400')
        else:
            self.status_label.text = '✅ Training complete'
            self.status_label.classes(remove='text-yellow-400', add='text-green-400')
        
        # Update epoch and loss
        epoch = state.get('epoch', 0)
        total = state.get('total_epochs', 100)
        loss = state.get('loss', 0)
        best = state.get('best_loss', float('inf'))
        
        self.epoch_label.text = f'Epoch: {epoch}/{total}'
        self.loss_label.text = f'Loss: {loss:.4f}'
        self.best_label.text = f'Best: {best:.4f}'
        self.progress.value = epoch / max(total, 1)
        
        # Update loss chart
        history = state.get('history', state.get('full_history', {}))
        if history and 'loss' in history:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history['loss'],
                mode='lines',
                name='Total Loss',
                line=dict(color='#00CED1', width=2)
            ))
            if 'flow_loss' in history:
                fig.add_trace(go.Scatter(
                    y=history['flow_loss'],
                    mode='lines',
                    name='Flow Loss',
                    line=dict(color='#FF69B4', width=2)
                ))
            
            fig.update_layout(
                paper_bgcolor='#1a1a1a',
                plot_bgcolor='#1a1a1a',
                font=dict(color='white'),
                margin=dict(l=40, r=20, t=20, b=40),
                legend=dict(
                    bgcolor='rgba(0,0,0,0.5)',
                    font=dict(color='white')
                ),
                xaxis=dict(
                    title='Epoch',
                    gridcolor='#333',
                    zerolinecolor='#333'
                ),
                yaxis=dict(
                    title='Loss',
                    gridcolor='#333',
                    zerolinecolor='#333'
                )
            )
            self.loss_chart.update_figure(fig)
        
        # Update latent space image
        viz_path = self.get_latest_viz()
        if viz_path:
            # Add cache-busting query param
            self.latent_image.source = f'/viz/{viz_path.name}?t={int(time.time())}'


def find_latest_training_dir(base_dir: Path = Path('output')) -> Path:
    """Find the most recent flow_matching training directory."""
    if not base_dir.exists():
        return None
    
    # Look for flow_matching_* directories
    dirs = sorted(base_dir.glob('flow_matching_*'), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if dirs:
        return dirs[0]
    return None


def main():
    parser = argparse.ArgumentParser(description='Live Dashboard Viewer')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Path to training output directory (auto-detects latest if not specified)')
    parser.add_argument('--port', type=int, default=8080,
                       help='Dashboard port')
    args = parser.parse_args()
    
    # Auto-detect latest training directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = find_latest_training_dir()
        if output_dir:
            print(f"📂 Auto-detected latest training: {output_dir}")
        else:
            print("❌ No training directories found in output/")
            print("Start training first:")
            print("  python scripts/train_flow_matching.py --vae-checkpoint ...")
            sys.exit(1)
    
    if not output_dir.exists():
        print(f"Error: Directory {output_dir} does not exist")
        print("Start training first, then run this dashboard.")
        sys.exit(1)
    
    print(f"📊 Starting dashboard for: {output_dir}")
    print(f"🌐 Open http://localhost:{args.port} in your browser")
    print()
    
    dashboard = DashboardViewer(output_dir)
    
    @ui.page('/')
    def main_page():
        dashboard.setup()
    
    # Serve visualization images
    viz_dir = output_dir / 'visualizations'
    if viz_dir.exists():
        app.add_static_files('/viz', str(viz_dir))
    
    ui.run(
        port=args.port,
        title='Flow Matching Dashboard',
        dark=True,
        reload=False,
        show=False,
    )


if __name__ == '__main__':
    main()

