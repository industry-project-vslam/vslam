"""
Routes for drone management endpoints.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import get_tracker_service
from api.services.tracker_service import TrackerService

router = APIRouter(prefix="/api/drones", tags=["drones"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class DroneStatusResponse(BaseModel):
    drone_id: str
    status: Dict[str, Any]
    timestamp: str


class ClearResponse(BaseModel):
    drone_id: str
    success: bool
    message: str


class DroneListResponse(BaseModel):
    total_drones: int
    drone_ids: List[str]
    timestamp: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=DroneListResponse)
def list_drones(
    tracker_service: TrackerService = Depends(get_tracker_service),
) -> DroneListResponse:
    """List all active drones."""
    status = tracker_service.get_status()
    return DroneListResponse(
        total_drones=status["total_drones"],
        drone_ids=list(status["drones"].keys()),
        timestamp=status["timestamp"],
    )


@router.get("/{drone_id}", response_model=DroneStatusResponse)
def get_drone_status(
    drone_id: str,
    tracker_service: TrackerService = Depends(get_tracker_service),
) -> DroneStatusResponse:
    """Get status of a specific drone."""
    status = tracker_service.get_status()

    if drone_id not in status["drones"]:
        raise HTTPException(status_code=404, detail=f"No map found for drone '{drone_id}'")

    return DroneStatusResponse(
        drone_id=drone_id,
        status=status["drones"][drone_id],
        timestamp=status["timestamp"],
    )


@router.delete("/{drone_id}", response_model=ClearResponse)
def clear_drone(
    drone_id: str,
    tracker_service: TrackerService = Depends(get_tracker_service),
) -> ClearResponse:
    """Remove a drone from the registry (output files on disk are kept)."""
    if not tracker_service.clear_drone_map(drone_id):
        raise HTTPException(status_code=404, detail=f"No map found for drone '{drone_id}'")

    return ClearResponse(
        drone_id=drone_id,
        success=True,
        message=f"Successfully cleared registry entry for drone '{drone_id}'",
    )