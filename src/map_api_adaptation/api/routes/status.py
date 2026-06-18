"""
Routes for status and health check endpoints.
"""

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.dependencies import get_tracker_service
from api.services.tracker_service import TrackerService

router = APIRouter(prefix="/api", tags=["status"])

_SERVICE_NAME = "Multi-Drone SLAM Tracker"
_API_VERSION = "2.0.0"


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    service: str


class StatusResponse(BaseModel):
    timestamp: str
    total_drones: int
    drones: Dict[str, Any]


class APIInfoResponse(BaseModel):
    name: str
    version: str
    endpoints: Dict[str, str]
    features: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        service=_SERVICE_NAME,
    )


@router.get("/status", response_model=StatusResponse)
def get_status(
    tracker_service: TrackerService = Depends(get_tracker_service),
) -> StatusResponse:
    """Status of all active drone maps."""
    return StatusResponse(**tracker_service.get_status())


@router.get("/info", response_model=APIInfoResponse)
def get_api_info() -> APIInfoResponse:
    """API metadata and endpoint catalogue."""
    return APIInfoResponse(
        name=f"{_SERVICE_NAME} API",
        version=_API_VERSION,
        endpoints={
            "POST /api/process_source": "Run reconstruction pipeline from a server-side folder",
            "POST /api/upload_frames/{drone_id}": "Upload frames and run reconstruction",
            "GET  /api/pointcloud/{drone_id}": "Download latest point cloud (ply or npy)",
            "GET  /api/drones": "List all active drones",
            "GET  /api/drones/{drone_id}": "Get drone status",
            "DELETE /api/drones/{drone_id}": "Remove drone from registry",
            "GET  /api/status": "Status of all drones",
            "GET  /api/health": "Health check",
            "GET  /api/info": "This endpoint",
        },
        features={
            "multi_drone_support": True,
            "auto_cleanup_hours": 6,
            "point_cloud_formats": ["ply", "npy"],
            "pipeline": "scene_recon_pipeline_online",
        },
    )