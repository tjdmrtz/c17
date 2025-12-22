"""
NiceGUI Training Dashboard for Neural ODE.

Real-time monitoring of:
- Loss curves (total, endpoint, smoothness, velocity)
- Latent space visualization
- Pattern reconstructions
- Phase transition animations
- Training logs
"""

import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime
import json

from nicegui import ui, app
import plotly.graph_objects as go

from .training_state import TrainingState
from .theme import THEME, GROUP_COLORS, GLOBAL_CSS


ALL_17_GROUPS = [
    'p1', 'p2', 'pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg', 'cmm',
    'p4', 'p4m', 'p4g', 'p3', 'p3m1', 'p31m', 'p6', 'p6m'
]


class TrainingDashboard:
    """
    NiceGUI dashboard for monitoring Neural ODE training.
    """
    
    def __init__(self, state: TrainingState, output_dir: Optional[Path] = None):
        self.state = state
        self.output_dir = output_dir or Path('output')
        
        # UI elements (will be set during setup)
        self.epoch_label = None
        self.loss_label = None
        self.lr_label = None
        self.eta_label = None
        self.progress = None
        self.loss_chart = None
        self.components_chart = None
        self.latent_image = None
        self.recon_image = None
        self.log_container = None
        self.transition_container = None
        
        # Update timer
        self.update_timer = None
    
    def setup(self):
        """Create the dashboard UI."""
        # Add custom CSS
        ui.add_head_html(f'<style>{GLOBAL_CSS}</style>')
        
        # Header
        with ui.header().classes('bg-gray-900 text-white'):
            ui.label('🔬 Neural ODE Training Dashboard').classes('text-xl font-bold')
            ui.space()
            with ui.row().classes('gap-2'):
                ui.button(icon='refresh', on_click=self.refresh_all).props('flat color=white')
                self.dark_mode = ui.dark_mode(True)
        
        # Status bar
        with ui.card().classes('w-full bg-gray-800'):
            with ui.row().classes('w-full items-center gap-4'):
                self.epoch_label = ui.label('Epoch: 0/0').classes('text-lg')
                ui.separator().props('vertical')
                self.loss_label = ui.label('Loss: -').classes('text-lg')
                ui.separator().props('vertical')
                self.lr_label = ui.label('LR: -').classes('text-lg')
                ui.separator().props('vertical')
                self.eta_label = ui.label('ETA: -').classes('text-lg')
            
            self.progress = ui.linear_progress(value=0, show_value=False).classes('w-full mt-2')
        
        # Tabs
        with ui.tabs().classes('w-full bg-gray-800') as tabs:
            tab_losses = ui.tab('📉 Loss Curves')
            tab_latent = ui.tab('🎯 Latent Space')
            tab_recon = ui.tab('🖼️ Reconstructions')
            tab_trans = ui.tab('🔄 Transitions')
            tab_logs = ui.tab('📋 Logs')
        
        with ui.tab_panels(tabs, value=tab_losses).classes('w-full'):
            with ui.tab_panel(tab_losses):
                self._setup_loss_panel()
            with ui.tab_panel(tab_latent):
                self._setup_latent_panel()
            with ui.tab_panel(tab_recon):
                self._setup_recon_panel()
            with ui.tab_panel(tab_trans):
                self._setup_transition_panel()
            with ui.tab_panel(tab_logs):
                self._setup_logs_panel()
        
        # Start update timer
        self.update_timer = ui.timer(1.0, self.update_ui)
    
    def _setup_loss_panel(self):
        """Loss curves panel."""
        with ui.row().classes('w-full gap-4'):
            with ui.card().classes('flex-1 bg-gray-800'):
                ui.label('Total Loss').classes('text-lg font-bold mb-2')
                self.loss_chart = ui.plotly(self._create_empty_chart()).classes('w-full h-64')
            
            with ui.card().classes('flex-1 bg-gray-800'):
                ui.label('Loss Components').classes('text-lg font-bold mb-2')
                self.components_chart = ui.plotly(self._create_empty_chart()).classes('w-full h-64')
        
        # Metrics table
        with ui.card().classes('w-full mt-4 bg-gray-800'):
            ui.label('Latest Metrics').classes('text-lg font-bold mb-2')
            self.metrics_table = ui.table(
                columns=[
                    {'name': 'metric', 'label': 'Metric', 'field': 'metric', 'align': 'left'},
                    {'name': 'value', 'label': 'Value', 'field': 'value', 'align': 'right'},
                ],
                rows=[],
            ).classes('w-full')
    
    def _setup_latent_panel(self):
        """Latent space visualization panel."""
        with ui.card().classes('w-full bg-gray-800'):
            ui.label('2D Latent Space Visualization').classes('text-lg font-bold mb-2')
            
            with ui.row().classes('w-full items-center gap-4 mb-4'):
                ui.label('Epoch:')
                self.latent_epoch_select = ui.select(
                    options=['Latest'],
                    value='Latest',
                    on_change=self._on_latent_epoch_change
                ).classes('w-32')
                
                ui.checkbox('Show Trajectories', value=True)
                ui.checkbox('Show Clusters', value=True)
            
            self.latent_image = ui.image().classes('w-full max-w-4xl mx-auto')
            self.latent_image.set_source('/static/placeholder.png')
    
    def _setup_recon_panel(self):
        """Reconstructions panel."""
        with ui.card().classes('w-full bg-gray-800'):
            ui.label('Pattern Reconstructions by Group').classes('text-lg font-bold mb-2')
            ui.label('Original → Encoded → Decoded').classes('text-gray-400 mb-4')
            
            self.recon_image = ui.image().classes('w-full max-w-5xl mx-auto')
            self.recon_image.set_source('/static/placeholder.png')
    
    def _setup_transition_panel(self):
        """Transitions panel."""
        with ui.card().classes('w-full bg-gray-800'):
            ui.label('Phase Transition Visualization').classes('text-lg font-bold mb-2')
            
            with ui.row().classes('gap-4 mb-4'):
                self.source_select = ui.select(
                    label='Source Group',
                    options=ALL_17_GROUPS,
                    value='p1'
                ).classes('w-32')
                
                ui.label('→').classes('text-2xl self-center')
                
                self.target_select = ui.select(
                    label='Target Group',
                    options=ALL_17_GROUPS,
                    value='p6m'
                ).classes('w-32')
                
                ui.button('Load Transition', on_click=self._load_transition).props('color=primary')
            
            # Transition display area
            self.transition_container = ui.column().classes('w-full')
            
            with self.transition_container:
                ui.label('Select a transition to view').classes('text-gray-400')
        
        # Canonical transitions grid
        with ui.card().classes('w-full mt-4 bg-gray-800'):
            ui.label('Canonical Transitions').classes('text-lg font-bold mb-2')
            
            canonical = [
                ('p1', 'p6m'), ('p1', 'p4m'), ('p2', 'p4'), ('p3', 'p6'),
                ('pm', 'pmm'), ('p4', 'p4m'), ('p3', 'p3m1'), ('cmm', 'p4m')
            ]
            
            with ui.row().classes('flex-wrap gap-2'):
                for source, target in canonical:
                    color = GROUP_COLORS.get(target, '#888')
                    ui.button(
                        f'{source}→{target}',
                        on_click=lambda s=source, t=target: self._quick_transition(s, t)
                    ).props(f'color="{color}"').classes('text-xs')
    
    def _setup_logs_panel(self):
        """Logs panel."""
        with ui.card().classes('w-full bg-gray-800'):
            with ui.row().classes('w-full justify-between mb-2'):
                ui.label('Training Logs').classes('text-lg font-bold')
                with ui.row().classes('gap-2'):
                    ui.button('Clear', on_click=self._clear_logs).props('flat')
                    ui.button('Export', on_click=self._export_logs).props('flat')
                    self.autoscroll = ui.checkbox('Auto-scroll', value=True)
            
            self.log_container = ui.scroll_area().classes('w-full h-96 bg-gray-900 rounded p-2')
    
    def _create_empty_chart(self) -> dict:
        """Create empty Plotly chart config."""
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(gridcolor='#333', title='Epoch'),
            yaxis=dict(gridcolor='#333', title='Loss'),
        )
        return fig.to_dict()
    
    def update_ui(self):
        """Update all UI elements from state."""
        # Status bar
        self.epoch_label.text = f'Epoch: {self.state.epoch}/{self.state.total_epochs}'
        self.loss_label.text = f'Loss: {self.state.current_loss:.4f}'
        self.lr_label.text = f'LR: {self.state.current_lr:.2e}'
        
        eta = self.state.get_eta_seconds()
        if eta is not None:
            hours = int(eta // 3600)
            minutes = int((eta % 3600) // 60)
            self.eta_label.text = f'ETA: {hours}h {minutes}m'
        else:
            self.eta_label.text = 'ETA: -'
        
        self.progress.value = self.state.get_progress()
        
        # Update charts
        self._update_loss_charts()
        
        # Update visualizations if new ones available
        self._update_visualizations()
        
        # Update logs
        self._update_logs()
    
    def _update_loss_charts(self):
        """Update loss curve charts."""
        if not self.state.losses['total']:
            return
        
        # Total loss chart
        epochs = list(range(1, len(self.state.losses['total']) + 1))
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=epochs,
            y=self.state.losses['total'],
            mode='lines',
            name='Total Loss',
            line=dict(color=THEME['loss_total'], width=2)
        ))
        fig1.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(gridcolor='#333', title='Epoch'),
            yaxis=dict(gridcolor='#333', title='Loss', type='log'),
            showlegend=False,
        )
        self.loss_chart.update_figure(fig1)
        
        # Components chart
        fig2 = go.Figure()
        
        components = [
            ('endpoint', 'Endpoint', THEME['loss_endpoint']),
            ('smoothness', 'Smoothness', THEME['loss_smooth']),
            ('velocity', 'Velocity', THEME['loss_velocity']),
        ]
        
        for key, name, color in components:
            if self.state.losses.get(key):
                fig2.add_trace(go.Scatter(
                    x=epochs[:len(self.state.losses[key])],
                    y=self.state.losses[key],
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=2)
                ))
        
        fig2.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(gridcolor='#333', title='Epoch'),
            yaxis=dict(gridcolor='#333', title='Loss', type='log'),
            legend=dict(x=0.7, y=0.95),
        )
        self.components_chart.update_figure(fig2)
        
        # Update metrics table
        if self.state.losses['total']:
            rows = [
                {'metric': 'Total Loss', 'value': f'{self.state.losses["total"][-1]:.6f}'},
            ]
            for key in ['endpoint', 'smoothness', 'velocity']:
                if self.state.losses.get(key):
                    rows.append({
                        'metric': f'{key.capitalize()} Loss',
                        'value': f'{self.state.losses[key][-1]:.6f}'
                    })
            self.metrics_table.rows = rows
    
    def _update_visualizations(self):
        """Update visualization images."""
        if self.state.latest_latent_viz:
            self.latent_image.set_source(self.state.latest_latent_viz)
        
        if self.state.latest_recon_viz:
            self.recon_image.set_source(self.state.latest_recon_viz)
    
    def _update_logs(self):
        """Update log display."""
        if not self.state.log_messages:
            return
        
        # Only update if there are new messages
        with self.log_container:
            self.log_container.clear()
            for msg in self.state.log_messages[-100:]:  # Show last 100
                ui.label(msg).classes('font-mono text-xs text-gray-300')
    
    def _on_latent_epoch_change(self, e):
        """Handle latent epoch selection change."""
        pass  # TODO: Load specific epoch visualization
    
    async def _load_transition(self):
        """Load selected transition visualization."""
        source = self.source_select.value
        target = self.target_select.value
        
        self.transition_container.clear()
        
        with self.transition_container:
            # Look for transition files
            key = f'{source}_to_{target}'
            
            if key in self.state.latest_transitions:
                path = self.state.latest_transitions[key]
                if path.endswith('.gif'):
                    ui.image(path).classes('w-full max-w-2xl mx-auto')
                else:
                    ui.image(path).classes('w-full max-w-4xl mx-auto')
            else:
                # Try to find in output directory
                viz_dir = self.output_dir / 'visualizations' / f'epoch_{self.state.epoch:03d}'
                
                gif_path = viz_dir / 'transitions' / f'{source}_to_{target}.gif'
                png_path = viz_dir / 'transitions' / f'{source}_to_{target}.png'
                
                if gif_path.exists():
                    ui.image(str(gif_path)).classes('w-full max-w-2xl mx-auto')
                elif png_path.exists():
                    ui.image(str(png_path)).classes('w-full max-w-4xl mx-auto')
                else:
                    ui.label(f'Transition {source}→{target} not yet generated').classes('text-gray-400')
    
    def _quick_transition(self, source: str, target: str):
        """Quick load a canonical transition."""
        self.source_select.value = source
        self.target_select.value = target
        asyncio.create_task(self._load_transition())
    
    def _clear_logs(self):
        """Clear log messages."""
        self.state.log_messages.clear()
        self.log_container.clear()
    
    def _export_logs(self):
        """Export logs to file."""
        if self.output_dir:
            log_path = self.output_dir / 'training_logs.txt'
            with open(log_path, 'w') as f:
                f.write('\n'.join(self.state.log_messages))
            ui.notify(f'Logs exported to {log_path}')
    
    def refresh_all(self):
        """Manually refresh all UI elements."""
        self.update_ui()
        ui.notify('Dashboard refreshed')


def run_dashboard(state: TrainingState, port: int = 8080, output_dir: Optional[Path] = None):
    """
    Run the dashboard in background (non-blocking).
    
    Args:
        state: TrainingState instance to monitor
        port: Port to run dashboard on
        output_dir: Output directory for visualizations
    """
    import threading
    
    dashboard = TrainingDashboard(state, output_dir)
    
    @ui.page('/')
    def main_page():
        dashboard.setup()
    
    # Serve static files from output directory
    if output_dir:
        app.add_static_files('/output', str(output_dir))
    
    # Run in background thread
    def run_server():
        ui.run(port=port, title='Flow Matching Training', dark=True, reload=False, show=False)
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()


if __name__ == '__main__':
    # Test dashboard standalone
    state = TrainingState()
    state.total_epochs = 100
    state.epoch = 25
    state.current_loss = 0.0234
    state.losses = {
        'total': [0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.035],
        'endpoint': [0.4, 0.25, 0.15, 0.1, 0.07, 0.05, 0.04, 0.035, 0.03, 0.025],
        'smoothness': [0.08, 0.04, 0.03, 0.035, 0.02, 0.02, 0.015, 0.01, 0.008, 0.008],
        'velocity': [0.02, 0.01, 0.02, 0.005, 0.01, 0.01, 0.005, 0.005, 0.002, 0.002],
    }
    state.log_messages = [
        '[14:30:00] Training started',
        '[14:30:15] Epoch 1 completed | Loss: 0.5000',
        '[14:30:30] Epoch 2 completed | Loss: 0.3000',
    ]
    
    run_dashboard(state, port=8080)


