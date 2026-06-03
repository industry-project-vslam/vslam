"""
Pydantic models for the Multi-Drone SLAM Tracker API.

Defines request/response schemas for all endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ============================================================================
# Frame Processing Models
# ============================================================================

class FrameData(BaseModel):
    """Frame data submission."""
    drone_id: str = Field(..., description="Unique drone identifier")
    frame_idx: int = Field(..., description="Frame index")
    registered_pcd: List[List[List[float]]] = Field(..., description="(H, W, 3) point cloud")
    registered_conf: List[List[float]] = Field(..., description="(H, W) confidence map")
    rgb_img: List[List[List[float]]] = Field(..., description="(H, W, 3) RGB image")


class FrameResponse(BaseModel):
    """Response from frame processing."""
    drone_id: str
    frame_idx: int
    location: List[float] = Field(..., description="[x, y, z]")
    depth_matrix: List[List[float]] = Field(..., description="8x8 matrix")
    frame_count: int
    timestamp: str


# ============================================================================
# Point Cloud Models
# ============================================================================

class PointCloudResponse(BaseModel):
    """Point cloud response."""
    drone_id: str
    point_count: int
    has_colors: bool
    timestamp: str


class PointCloudQueryParams(BaseModel):
    """Query parameters for point cloud retrieval."""
    conf_threshold: float = Field(default=12.0, description="Confidence threshold")
    format: str = Field(default="npy", description="Output format: 'npy' or 'ply'")


# ============================================================================
# Drone Status Models
# ============================================================================

class DroneStatus(BaseModel):
    """Status of a single drone."""
    drone_id: str
    created_at: str
    last_updated: str
    frame_count: int
    max_points: int
    is_stale: bool
    location: Optional[List[float]]


class DroneStatusResponse(BaseModel):
    """Response for single drone status."""
    drone_id: str
    status: Dict
    timestamp: str


class StatusResponse(BaseModel):
    """Overall system status."""
    timestamp: str
    total_drones: int
    drones: Dict[str, dict]


# ============================================================================
# Drone Management Models
# ============================================================================

class ClearResponse(BaseModel):
    """Response from clearing drone data."""
    drone_id: str
    success: bool
    message: str


class DroneListResponse(BaseModel):
    """Response listing all drones."""
    total_drones: int
    drone_ids: List[str]
    timestamp: str


# ============================================================================
# Depth Matrix Models
# ============================================================================

class DepthMatrixResponse(BaseModel):
    """Response from depth matrix computation."""
    depth_matrix: List[List[float]]
    shape: List[int]
    min: float
    max: float
    mean: float


# ============================================================================
# Navigation Models
# ============================================================================

class NavigationFeedback(BaseModel):
    """Navigation feedback from depth matrix."""
    safe_areas: List[List[bool]]
    obstacles: List[List[bool]]
    recommended_direction: List[float]
    min_depth: float
    max_depth: float
    mean_depth: float


class TrajectoryStats(BaseModel):
    """Statistics about drone trajectory."""
    drone_id: str
    num_waypoints: int
    total_distance: float
    start_position: List[float]
    end_position: List[float]
    bounding_box: Dict[str, List[float]]


# ============================================================================
# Health & Info Models
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    service: str


class APIInfoResponse(BaseModel):
    """API information response."""
    name: str
    version: str
    endpoints: Dict[str, str]
    features: Dict[str, Any]


# ============================================================================
# Error Models
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str
    timestamp: str
