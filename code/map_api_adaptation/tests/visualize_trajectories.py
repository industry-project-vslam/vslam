"""
3D Trajectory Visualization for Multi-Camera SLAM3R Tracker
===========================================================

Reads trajectory data from online_multicam_tracker.py output and creates
an interactive 3D visualization of camera positions and directions using Plotly.

Usage:
    python visualize_trajectories.py \
        --trajectory_file results_online/trajectories.json \
        --output_html trajectory_viz.html \
        --arrow_scale 0.5
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_trajectories(filepath: str) -> Dict:
    """Load trajectory data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_camera_color(cam_id: str, num_cameras: int) -> str:
    """Assign a distinct color to each camera."""
    colors = [
        '#FF6B6B',  # Red
        '#4ECDC4',  # Teal
        '#45B7D1',  # Blue
        '#FFA07A',  # Light salmon
        '#98D8C8',  # Mint
        '#F7DC6F',  # Yellow
        '#BB8FCE',  # Purple
        '#85C1E2',  # Sky blue
    ]
    cam_index = hash(cam_id) % len(colors)
    return colors[cam_index]


def create_arrow_trace(
    start: np.ndarray,
    direction: np.ndarray,
    scale: float,
    color: str,
    name: str,
) -> go.Scatter3d:
    """Create a Plotly trace for a directional arrow.
    
    Args:
        start: (3,) start position
        direction: (3,) normalized direction vector
        scale: length of arrow
        color: color for the arrow
        name: trace name for legend
    """
    end = start + direction * scale
    
    return go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode='lines',
        line=dict(color=color, width=4),
        name=name,
        showlegend=False,
        hoverinfo='text',
        text=[f"{name} direction"],
    )


def visualize_trajectories(
    trajectory_file: str,
    output_html: str,
    arrow_scale: float = 0.3,
    show_browser: bool = True,
    conf_threshold: float = 2.0,
) -> None:
    """Create 3D interactive visualization of camera trajectories.
    
    Args:
        trajectory_file: Path to trajectories.json
        output_html: Output HTML file path
        arrow_scale: Length scale for direction arrows
        show_browser: Whether to open in browser after generation
        conf_threshold: Minimum confidence to include frame (default 2.0 filters tracking losses)
    """
    
    # Load data
    print(f"Loading trajectories from {trajectory_file}...")
    data = load_trajectories(trajectory_file)
    print(f"Confidence threshold: {conf_threshold} (frames with conf < {conf_threshold} filtered out)")
    
    metadata = data.get('metadata', {})
    cameras = metadata.get('cameras', [])
    num_cameras = len(cameras)
    
    print(f"Cameras: {cameras}")
    print(f"Frames per camera: {metadata.get('num_frames', 'unknown')}")
    
    # Create figure
    fig = go.Figure()
    
    # Track bounds for axis scaling
    all_positions = []
    
    # Add trajectories and positions for each camera
    for cam_id in cameras:
        if cam_id not in data or cam_id.startswith('metadata'):
            continue
        
        cam_data = data[cam_id]
        frames = cam_data.get('frames', [])
        
        color = get_camera_color(cam_id, num_cameras)
        
        # Collect valid positions for trajectory
        trajectory_x, trajectory_y, trajectory_z = [], [], []
        frame_numbers = []
        
        for frame_info in frames:
            if not frame_info.get('valid', False):
                continue
            if frame_info.get('conf', 0) < conf_threshold:
                continue
            
            position = np.array(frame_info['position'])
            trajectory_x.append(position[0])
            trajectory_y.append(position[1])
            trajectory_z.append(position[2])
            frame_numbers.append(frame_info['frame'])
            all_positions.append(position)
        
        # Add trajectory line
        if len(trajectory_x) > 0:
            fig.add_trace(go.Scatter3d(
                x=trajectory_x,
                y=trajectory_y,
                z=trajectory_z,
                mode='lines',
                line=dict(color=color, width=2),
                name=f'{cam_id} trajectory',
                opacity=0.7,
                hoverinfo='text',
                text=[f'{cam_id} frame {fn}' for fn in frame_numbers],
            ))
        
        # Add camera position markers with arrows
        for frame_info in frames:
            if not frame_info.get('valid', False):
                continue
            if frame_info.get('conf', 0) < conf_threshold:
                continue
            
            position = np.array(frame_info['position'])
            forward = np.array(frame_info['forward'])
            frame_num = frame_info['frame']
            
            # Add position marker
            fig.add_trace(go.Scatter3d(
                x=[position[0]],
                y=[position[1]],
                z=[position[2]],
                mode='markers',
                marker=dict(
                    size=4,
                    color=color,
                    opacity=0.8,
                ),
                name=f'{cam_id} pos',
                showlegend=(frame_num == 0),  # Only show once in legend
                hoverinfo='text',
                text=[f'{cam_id} frame {frame_num}<br>'
                      f'Pos: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})'],
            ))
            
            # Add direction arrow
            arrow_trace = create_arrow_trace(
                position,
                forward,
                arrow_scale,
                color,
                f'{cam_id} frame {frame_num}',
            )
            fig.add_trace(arrow_trace)
    
    # Calculate layout bounds
    if all_positions:
        all_positions = np.array(all_positions)
        pos_min = all_positions.min(axis=0)
        pos_max = all_positions.max(axis=0)
        center = (pos_min + pos_max) / 2
        range_val = (pos_max - pos_min).max() / 2 * 1.2
    else:
        center = np.array([0, 0, 0])
        range_val = 1
    
    # Update layout
    fig.update_layout(
        title='Multi-Camera SLAM3R Trajectories',
        scene=dict(
            xaxis=dict(title='X', range=[center[0] - range_val, center[0] + range_val]),
            yaxis=dict(title='Y', range=[center[1] - range_val, center[1] + range_val]),
            zaxis=dict(title='Z', range=[center[2] - range_val, center[2] + range_val]),
            aspectmode='cube',
        ),
        width=1200,
        height=800,
        hovermode='closest',
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
        ),
    )
    
    # Save HTML
    print(f"Saving visualization to {output_html}...")
    fig.write_html(output_html)
    print(f"Saved to {output_html}")
    
    # Open in browser
    if show_browser:
        import webbrowser
        webbrowser.open(f'file://{Path(output_html).absolute()}')
        print("Opening in browser...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize multi-camera trajectories from SLAM3R tracking'
    )
    parser.add_argument(
        '--trajectory_file',
        default='results_online/trajectories.json',
        help='Path to trajectories.json from online_multicam_tracker.py'
    )
    parser.add_argument(
        '--output_html',
        default='trajectory_visualization.html',
        help='Output HTML file path'
    )
    parser.add_argument(
        '--arrow_scale', type=float, default=0.3,
        help='Length scale for direction arrows'
    )
    parser.add_argument(
        '--conf_threshold', type=float, default=2.0,
        help='Confidence threshold: frames with conf < threshold are filtered out (default 2.0)'
    )
    parser.add_argument(
        '--no-browser', action='store_true',
        help='Do not open in browser'
    )
    
    args = parser.parse_args()
    visualize_trajectories(
        args.trajectory_file,
        args.output_html,
        arrow_scale=args.arrow_scale,
        show_browser=not args.no_browser,
        conf_threshold=args.conf_threshold,
    )
