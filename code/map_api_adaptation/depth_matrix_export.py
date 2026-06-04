"""
Tier 2: Depth Matrix Export (8x8 Grid)
- Extract minimum depth per 8x8 grid cell
- Export in API response
- Visualize as Gradio heatmap
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Any
from slam3r.utils.device import to_numpy


def extract_depth_matrix(
    pts3d_world: Any,
    conf_map: Any,
    grid_size: int = 8,
    min_depth: float = 0.01,
    max_depth: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Extract 8x8 depth and confidence matrix from full resolution output.
    
    For each 8x8 grid cell, compute:
    - Min depth (closest obstacle in cell) — for obstacle avoidance
    - Mean confidence (quality of depth estimate)
    - Valid point count (debug)
    
    Args:
        pts3d_world: (1, H, W, 3) or (H, W, 3) 3D point cloud in world frame
        conf_map: (1, H, W) or (H, W) confidence map (0-255 typically)
        grid_size: Discretization level (default 8 → 8x8 grid = 64 cells)
        min_depth: Minimum valid depth (filter numerical noise)
        max_depth: Maximum valid depth (filter outliers)
    
    Returns:
        depth_matrix: (grid_size, grid_size) min depth per cell [meters]
        conf_matrix: (grid_size, grid_size) mean confidence per cell [0-1]
        metadata: dict with cell statistics for debugging
    """
    
    # Convert to numpy if needed
    if isinstance(pts3d_world, torch.Tensor):
        pts3d_world = to_numpy(pts3d_world)
    if isinstance(conf_map, torch.Tensor):
        conf_map = to_numpy(conf_map)
    
    # Remove batch dimension if present
    if pts3d_world.ndim == 4 and pts3d_world.shape[0] == 1:
        pts3d_world = pts3d_world[0]  # (H, W, 3)
    if conf_map.ndim == 3 and conf_map.shape[0] == 1:
        conf_map = conf_map[0]  # (H, W)
    elif conf_map.ndim == 2:
        pass  # Already (H, W)
    else:
        raise ValueError(f"conf_map shape unexpected: {conf_map.shape}")
    
    if pts3d_world.ndim != 3 or pts3d_world.shape[2] != 3:
        raise ValueError(f"pts3d_world must be (H, W, 3), got {pts3d_world.shape}")
    
    H, W = pts3d_world.shape[:2]
    cell_h = H // grid_size
    cell_w = W // grid_size
    
    # Compute depth: Euclidean distance from camera origin (0, 0, 0)
    # For camera-frame points: depth = ||p|| where p = (x, y, z)
    depth = np.linalg.norm(pts3d_world, axis=2)  # (H, W)
    
    # Normalize confidence to [0, 1]
    conf_min, conf_max = conf_map.min(), conf_map.max()
    if conf_max > conf_min:
        conf_normalized = (conf_map - conf_min) / (conf_max - conf_min)
    else:
        conf_normalized = np.ones_like(conf_map)
    
    # Initialize output matrices
    depth_matrix = np.full((grid_size, grid_size), fill_value=np.nan)
    conf_matrix = np.zeros((grid_size, grid_size))
    metadata = {"total_cells": grid_size * grid_size, "valid_cells": 0, "cells": {}}
    
    # Process each grid cell
    for i in range(grid_size):
        for j in range(grid_size):
            # Extract cell boundaries
            h_start = i * cell_h
            h_end = (i + 1) * cell_h
            w_start = j * cell_w
            w_end = (j + 1) * cell_w
            
            # Extract cell data
            cell_depth = depth[h_start:h_end, w_start:w_end]
            cell_conf = conf_normalized[h_start:h_end, w_start:w_end]
            
            # Validity mask: depth in valid range
            valid = (cell_depth >= min_depth) & (cell_depth <= max_depth)
            
            if valid.sum() > 0:
                # Min depth (closest obstacle for avoidance)
                depth_matrix[i, j] = cell_depth[valid].min()
                # Mean confidence (average quality)
                conf_matrix[i, j] = cell_conf[valid].mean()
                
                # Metadata for debugging
                metadata["valid_cells"] += 1
                metadata["cells"][f"{i}_{j}"] = {
                    "min_depth": float(depth_matrix[i, j]),
                    "mean_conf": float(conf_matrix[i, j]),
                    "valid_points": int(valid.sum()),
                    "total_points": int(cell_depth.size),
                }
            else:
                # No valid points in cell
                depth_matrix[i, j] = np.nan
                conf_matrix[i, j] = 0.0
    
    # Replace NaN with maximum depth (no obstacle detected)
    depth_matrix = np.nan_to_num(depth_matrix, nan=max_depth)
    
    metadata["coverage"] = metadata["valid_cells"] / metadata["total_cells"]
    
    return depth_matrix, conf_matrix, metadata


def depth_matrix_to_json(
    depth_matrix: np.ndarray,
    conf_matrix: np.ndarray,
    metadata: dict
) -> dict:
    """Convert depth/conf matrices to JSON-serializable format."""
    return {
        "depth_matrix": depth_matrix.tolist(),
        "conf_matrix": conf_matrix.tolist(),
        "grid_size": depth_matrix.shape[0],
        "metadata": {
            "coverage": metadata.get("coverage", 0),
            "valid_cells": metadata.get("valid_cells", 0),
            "total_cells": metadata.get("total_cells", 0),
        }
    }


# ============================================================================
# INTEGRATION INTO TrackerService
# ============================================================================

TRACKER_SERVICE_MODIFICATION = """
# In api/services/tracker_service.py, modify DroneMap class:

class DroneMap:
    def __init__(self, ...):
        # ... existing init code ...
        
        # NEW: Track last depth matrix for API response
        self.last_depth_matrix = None
        self.last_conf_matrix = None
        self.last_depth_metadata = {}
    
    def _process_frame_locked(self, i2p_model, l2w_model, raw_frame, frame_label):
        # ... existing frame processing ...
        
        # AFTER pointmap_global_register() call, ADD THIS:
        if 'pts3d_world' in self.input_views[current_frame_id]:
            try:
                pts3d_world = self.input_views[current_frame_id]['pts3d_world']
                conf_map = self.per_frame_res['l2w_confs'][current_frame_id]
                
                # Extract 8x8 depth matrix
                depth_mat, conf_mat, meta = extract_depth_matrix(
                    pts3d_world,
                    conf_map,
                    grid_size=8,
                    min_depth=0.01,
                    max_depth=100.0
                )
                
                self.last_depth_matrix = depth_mat
                self.last_conf_matrix = conf_mat
                self.last_depth_metadata = meta
            except Exception as e:
                logger.warning(f"Failed to extract depth matrix for frame {current_frame_id}: {e}")
    
    def get_depth_matrix(self) -> Optional[dict]:
        \"\"\"Return latest depth matrix for API response.\"\"\"
        if self.last_depth_matrix is None:
            return None
        return depth_matrix_to_json(
            self.last_depth_matrix,
            self.last_conf_matrix,
            self.last_depth_metadata
        )

# In api/routes/frames.py, update SourceResponse model:

class SourceResponse(BaseModel):
    drone_id: str
    save_dir: str
    frame_count: int
    message: str
    last_frame: int | None = None
    position: List[float] | None = None
    forward: List[float] | None = None
    valid: bool | None = None
    # NEW: Depth matrix fields
    depth_matrix: List[List[float]] | None = None
    conf_matrix: List[List[float]] | None = None
    depth_metadata: dict | None = None

# In api/routes/frames.py, update _build_response():

def _build_response(drone_id: str, tracker_service: TrackerService) -> SourceResponse:
    drone_info = tracker_service.get_status()["drones"].get(drone_id, {})
    resp = SourceResponse(
        drone_id=drone_id,
        save_dir=drone_info.get("save_dir", ""),
        frame_count=drone_info.get("frame_count", 0),
        message=f"Reconstruction complete for drone {drone_id}",
    )
    
    # Get latest pose
    latest = tracker_service.get_latest_pose(drone_id)
    if latest is not None:
        resp.last_frame = latest.get("frame")
        resp.position = latest.get("position")
        resp.forward = latest.get("forward")
        resp.valid = latest.get("valid")
    
    # NEW: Get depth matrix
    with tracker_service._lock:
        drone_map = tracker_service._drone_maps.get(drone_id)
        if drone_map:
            depth_data = drone_map.get_depth_matrix()
            if depth_data:
                resp.depth_matrix = depth_data["depth_matrix"]
                resp.conf_matrix = depth_data["conf_matrix"]
                resp.depth_metadata = depth_data["metadata"]
    
    return resp
"""


# ============================================================================
# GRADIO UI ENHANCEMENTS
# ============================================================================

GRADIO_UI_ENHANCEMENT = """
# In app_api_gradio.py, ADD THESE FUNCTIONS:

def plot_depth_heatmap(depth_matrix: List[List[float]], conf_matrix: List[List[float]]) -> Any:
    \"\"\"Create plotly heatmap figure for depth and confidence matrices.\"\"\"
    import plotly.graph_objects as go
    import plotly.subplots as sp
    import numpy as np
    
    if depth_matrix is None:
        return None
    
    depth_arr = np.array(depth_matrix)
    conf_arr = np.array(conf_matrix) if conf_matrix else np.zeros_like(depth_arr)
    
    # Create subplots: depth on left, confidence on right
    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=("Min Depth per Cell (meters)", "Mean Confidence per Cell"),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}]]
    )
    
    # Depth heatmap
    fig.add_trace(
        go.Heatmap(
            z=depth_arr,
            colorscale="Viridis",
            name="Depth",
            colorbar=dict(title="Depth (m)", x=0.46),
            hovertemplate="Cell [%{x}, %{y}]<br>Min Depth: %{z:.2f}m<extra></extra>",
            reversescale=False,
        ),
        row=1, col=1
    )
    
    # Confidence heatmap
    fig.add_trace(
        go.Heatmap(
            z=conf_arr,
            colorscale="Hot",
            name="Confidence",
            colorbar=dict(title="Conf", x=1.02),
            hovertemplate="Cell [%{x}, %{y}]<br>Confidence: %{z:.2f}<extra></extra>",
            reversescale=True,
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Column", row=1, col=1)
    fig.update_yaxes(title_text="Row", row=1, col=1)
    fig.update_xaxes(title_text="Column", row=1, col=2)
    fig.update_yaxes(title_text="Row", row=1, col=2)
    
    fig.update_layout(
        title_text="8×8 Depth Grid: Min Distance per Segment",
        height=500,
        width=1000,
        showlegend=False,
    )
    
    return fig


def extract_depth_from_response(api_response: dict) -> Any:
    \"\"\"Extract depth heatmap from API response.\"\"\"
    if isinstance(api_response, dict) and "depth_matrix" in api_response:
        return plot_depth_heatmap(
            api_response.get("depth_matrix"),
            api_response.get("conf_matrix")
        )
    return None


# In build_demo_ui(), ADD NEW TAB AFTER "Point Cloud" TAB:

with gr.Tab("Depth Heatmap"):
    gr.Markdown(
        "## Real-time Depth Matrix (8×8 Grid)\\n"
        "Displays minimum depth (distance to closest obstacle) in each grid cell. "
        "Low values = obstacles nearby."
    )
    
    with gr.Row():
        heatmap_depth = gr.Plot(label="Depth Heatmap")
        depth_info = gr.JSON(label="Depth Info")
    
    def show_latest_depth(api_url, drone_id):
        \"\"\"Fetch and display latest depth matrix for a drone.\"\"\"
        try:
            resp = _post_json(
                api_url or API_DEFAULT,
                f"/api/upload_frames/{drone_id}",
                {"get_latest_depth": True}  # Pseudo-endpoint
            )
            if "error" in resp:
                return None, resp
            
            fig = extract_depth_from_response(resp)
            info = {
                "grid_size": 8,
                "coverage": resp.get("depth_metadata", {}).get("coverage", 0),
                "valid_cells": resp.get("depth_metadata", {}).get("valid_cells", 0),
            }
            return fig, info
        except Exception as e:
            return None, {"error": str(e)}
    
    depth_btn = gr.Button("Fetch Latest Depth", variant="primary")
    depth_drone_id = gr.Textbox(label="Drone ID", value="demo_drone")
    
    depth_btn.click(
        show_latest_depth,
        inputs=[api_url, depth_drone_id],
        outputs=[heatmap_depth, depth_info]
    )
    
    # Also show depth after reconstruction
    # Modify the reconstruction button to also output depth:
    # (See full implementation below)
"""


# ============================================================================
# FULL GRADIO RECONSTRUCTION CALLBACK WITH DEPTH
# ============================================================================

GRADIO_FULL_RECONSTRUCTION = """
# REPLACE the reconstruction button callback in build_demo_ui():

def on_reconstruction_complete(api_response: dict) -> Tuple[dict, Any]:
    \"\"\"Called after /api/upload_frames completes. Extract depth if available.\"\"\"
    depth_fig = extract_depth_from_response(api_response)
    return api_response, depth_fig

recon_btn.click(
    api_upload_frames,
    inputs=[
        api_url, drone_id_recon, frame_upload,
        keyframe_stride, initial_winsize, win_r,
        conf_thres_i2p, num_scene_frame, max_num_register,
        conf_thres_l2w, num_points_save, save_each_frame,
    ],
    outputs=[recon_out, heatmap_depth],  # ADD heatmap_depth here
    _js='''
    (response) => {
        // Pass response to both JSON output and depth visualization
        return [response, response];
    }
    '''
)
"""


# ============================================================================
# TEST HARNESS
# ============================================================================

def test_depth_extraction():
    """Quick test to verify depth extraction works."""
    import matplotlib.pyplot as plt
    
    # Create synthetic test data
    H, W = 224, 224
    
    # Synthetic point cloud: hemisphere pointing forward
    y, x = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    z = np.sqrt(np.maximum(1 - x**2 - y**2, 0.01)) + 1  # Hemisphere + offset
    pts3d = np.stack([x, y, z], axis=-1)
    pts3d = pts3d[np.newaxis, :, :, :]  # Add batch dim
    
    # Synthetic confidence: lower in corners
    conf = (x**2 + y**2) < 0.8
    conf = conf[np.newaxis, :, :]
    
    # Extract depth matrix
    depth_mat, conf_mat, meta = extract_depth_matrix(
        pts3d, conf, grid_size=8, min_depth=0.01, max_depth=100.0
    )
    
    print(f"✓ Depth extraction successful")
    print(f"  Shape: {depth_mat.shape}")
    print(f"  Depth range: {depth_mat.min():.2f} - {depth_mat.max():.2f} m")
    print(f"  Coverage: {meta['coverage']*100:.1f}%")
    print(f"  Valid cells: {meta['valid_cells']}/{meta['total_cells']}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].imshow(depth_mat, cmap='viridis', origin='lower')
    axes[0].set_title('Min Depth (m)')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    plt.colorbar(axes[0].images[0], ax=axes[0])
    
    axes[1].imshow(conf_mat, cmap='hot', origin='lower')
    axes[1].set_title('Mean Confidence')
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    plt.colorbar(axes[1].images[0], ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('depth_test.png', dpi=100)
    print(f"✓ Visualization saved to depth_test.png")
    
    return depth_mat, conf_mat, meta


if __name__ == "__main__":
    print("="*70)
    print("TIER 2: DEPTH MATRIX EXPORT")
    print("="*70)
    print("\nQuick test of depth extraction...")
    test_depth_extraction()
    print("\nIntegration instructions:")
    print("1. Copy extract_depth_matrix() to slam3r/pipeline/recon_online_pipeline.py")
    print("2. Modify TrackerService and SourceResponse (see TRACKER_SERVICE_MODIFICATION)")
    print("3. Add Gradio UI elements (see GRADIO_UI_ENHANCEMENT)")
    print("4. Rebuild API and test with sample frames")
