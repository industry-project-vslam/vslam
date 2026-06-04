"""
Routes for point cloud download endpoints.
"""

import io
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from api.dependencies import get_tracker_service
from api.services.tracker_service import TrackerService

router = APIRouter(prefix="/api/pointcloud", tags=["pointcloud"])

_PLY_MEDIA_TYPE = "model/vnd.ply"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get("/{drone_id}")
def get_pointcloud(
    drone_id: str,
    tracker_service: TrackerService = Depends(get_tracker_service),
):
    """
    Serve the latest point cloud PLY file for a drone.

    The pipeline writes PLY files to the drone's save_dir; this endpoint
    locates the most recently written one and streams it back.

    Args:
        drone_id: Unique drone identifier.
    """
    status = tracker_service.get_status()

    if drone_id not in status["drones"]:
        raise HTTPException(status_code=404, detail=f"No map found for drone '{drone_id}'")

    save_dir = Path(status["drones"][drone_id]["save_dir"])
    ply_files = sorted(save_dir.glob("*.ply"))

    if not ply_files:
        raise HTTPException(
            status_code=404,
            detail=f"No point cloud files found for drone '{drone_id}' in {save_dir}",
        )

    latest_ply = ply_files[-1]

    return FileResponse(
        path=str(latest_ply),
        media_type=_PLY_MEDIA_TYPE,
        filename=f"pointcloud_{drone_id}.ply",
    )


@router.get("/{drone_id}/trajectories")
def get_trajectories(
    drone_id: str,
    tracker_service: TrackerService = Depends(get_tracker_service),
):
    """
    Get all trajectory frames for a drone (camera positions and orientations).

    Args:
        drone_id: Unique drone identifier.

    Returns:
        Dictionary with metadata and list of frames, each containing:
        - frame: frame index
        - valid: whether pose is valid
        - position: [x, y, z] camera position
        - forward: [x, y, z] forward direction vector
        - axes: [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]] rotation axes
        - conf: confidence score
        - timestamp: unix time
    """
    trajectories = tracker_service.get_all_trajectories(drone_id)

    if trajectories is None:
        raise HTTPException(status_code=404, detail=f"No map found for drone '{drone_id}'")

    return trajectories


@router.get("/{drone_id}/trajectories")
def get_trajectories(
    drone_id: str,
    tracker_service: TrackerService = Depends(get_tracker_service),
):
    """
    Return the trajectories.json contents for a drone if present.
    """
    status = tracker_service.get_status()

    if drone_id not in status["drones"]:
        raise HTTPException(status_code=404, detail=f"No map found for drone '{drone_id}'")

    save_dir = Path(status["drones"][drone_id]["save_dir"])
    traj_file = save_dir / "trajectories.json"

    if not traj_file.exists():
        raise HTTPException(status_code=404, detail=f"No trajectories.json for drone '{drone_id}'")

    try:
        import json

        with open(traj_file, "r") as f:
            data = json.load(f)
        return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read trajectories.json: {exc}")