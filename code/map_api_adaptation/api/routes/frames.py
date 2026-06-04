"""
Routes for frame source processing endpoints.
"""

import argparse
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from api.dependencies import get_tracker_service
from api.services.tracker_service import TrackerService
from slam3r.pipeline.recon_online_pipeline import FrameReader

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["frames"])

_SLAM3R_DEVICE = os.environ.get("SLAM3R_DEVICE", "cuda")

# ---------------------------------------------------------------------------
# Shared pipeline parameter model
#
# Single source of truth for defaults — used by both the JSON body endpoint
# and the multipart upload endpoint. _PIPELINE_DEFAULTS is derived from it so
# argparse.Namespace construction stays DRY.
# ---------------------------------------------------------------------------


class PipelineParams(BaseModel):
    """Parameters forwarded to scene_recon_pipeline_online."""

    keyframe_stride: int = 1
    initial_winsize: int = 2
    win_r: int = 3
    conf_thres_i2p: float = 1.5
    num_scene_frame: int = 10
    max_num_register: int = 10
    conf_thres_l2w: float = 12.0
    num_points_save: int = 2_000_000
    norm_input: bool = False
    save_frequency: int = 3
    save_each_frame: bool = True
    retrieve_freq: int = 1
    update_buffer_intv: int = 1
    buffer_size: int = 100
    buffer_strategy: str = "reservoir"
    save_online: bool = False
    save_all_views: bool = False
    save_preds: bool = False
    save_for_eval: bool = False
    keyframe_adapt_min: int = 1
    keyframe_adapt_max: int = 20
    keyframe_adapt_stride: int = 1
    perframe: int = 1
    seed: int = 11
    device: str = _SLAM3R_DEVICE


# Pre-built defaults dict so Namespace construction is a one-liner.
_PIPELINE_DEFAULTS: Dict[str, Any] = PipelineParams().model_dump()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class SourceRequest(PipelineParams):
    """JSON body for /process_source — adds routing fields."""

    drone_id: str
    source_path: str
    map_id: str | None = None
    frame_timeout_secs: float | None = None  # Per-frame timeout in seconds; None disables


class SourceResponse(BaseModel):
    drone_id: str
    save_dir: str
    frame_count: int
    message: str
    map_id: str | None = None
    last_frame: int | None = None
    position: List[float] | None = None
    forward: List[float] | None = None
    valid: bool | None = None
    skipped_frames: List[int] | None = None  # Frames skipped due to timeout


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _build_response(
    drone_id: str, 
    tracker_service: TrackerService,
    map_id: str | None = None,
) -> SourceResponse:
    drone_info = tracker_service.get_status()["drones"].get(drone_id, {})
    resp = SourceResponse(
        drone_id=drone_id,
        save_dir=drone_info.get("save_dir", ""),
        frame_count=drone_info.get("frame_count", 0),
        message=f"Reconstruction complete for drone {drone_id}",
        map_id=map_id,
    )
    # Attach latest pose if available
    latest = tracker_service.get_latest_pose(drone_id)
    if latest is not None:
        resp.last_frame = latest.get("frame")
        resp.position = latest.get("position")
        resp.forward = latest.get("forward")
        resp.valid = latest.get("valid")
    # Attach skipped frames if any
    skipped = tracker_service.get_skipped_frames(drone_id)
    if skipped:
        resp.skipped_frames = skipped
    return resp


def _run_pipeline(
    drone_id: str,
    source_path: str,
    params: Dict[str, Any],
    tracker_service: TrackerService,
    map_id: str | None = None,
) -> SourceResponse:
    """Open a FrameReader, build args, run the pipeline, return a response."""
    try:
        source = FrameReader(source_path)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot open source path '{source_path}': {exc}",
        ) from exc

    # Extract timeout parameter if provided
    frame_timeout = params.pop("frame_timeout_secs", None)

    args = argparse.Namespace(**{**_PIPELINE_DEFAULTS, **params})

    try:
        # Set frame timeout if specified
        if frame_timeout is not None:
            tracker_service.set_frame_timeout(drone_id, frame_timeout)
        
        tracker_service.process_source(drone_id, source, args, map_id=map_id)
    except Exception:
        logger.exception("Reconstruction failed for drone '%s'", drone_id)
        raise HTTPException(
            status_code=500,
            detail="Reconstruction failed. Check server logs for details.",
        )

    return _build_response(drone_id, tracker_service, map_id=map_id)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/process_source", response_model=SourceResponse)
def process_source(
    request: SourceRequest,
    tracker_service: TrackerService = Depends(get_tracker_service),
) -> SourceResponse:
    """Run reconstruction from a server-side folder path."""
    params = {
        k: v
        for k, v in request.model_dump().items()
        if k not in ("drone_id", "source_path", "map_id")
    }
    return _run_pipeline(
        request.drone_id, 
        request.source_path, 
        params, 
        tracker_service,
        map_id=request.map_id,
    )


@router.post("/upload_frames/{drone_id}", response_model=SourceResponse)
async def upload_frames(
    drone_id: str,
    files: List[UploadFile] = File(..., description="Frame image files (jpg/png)"),
    map_id: str | None = Form(None, description="Optional shared map ID for multi-drone stitching"),
    frame_timeout_secs: float | None = Form(None, description="Per-frame timeout in seconds; frames exceeding this are skipped"),
    # Expose the most commonly tuned params; others use pipeline defaults.
    keyframe_stride: int = Form(_PIPELINE_DEFAULTS["keyframe_stride"]),
    initial_winsize: int = Form(_PIPELINE_DEFAULTS["initial_winsize"]),
    win_r: int = Form(_PIPELINE_DEFAULTS["win_r"]),
    conf_thres_i2p: float = Form(_PIPELINE_DEFAULTS["conf_thres_i2p"]),
    num_scene_frame: int = Form(_PIPELINE_DEFAULTS["num_scene_frame"]),
    max_num_register: int = Form(_PIPELINE_DEFAULTS["max_num_register"]),
    conf_thres_l2w: float = Form(_PIPELINE_DEFAULTS["conf_thres_l2w"]),
    num_points_save: int = Form(_PIPELINE_DEFAULTS["num_points_save"]),
    save_each_frame: bool = Form(_PIPELINE_DEFAULTS["save_each_frame"]),
    device: str = Form(_SLAM3R_DEVICE),
    tracker_service: TrackerService = Depends(get_tracker_service),
) -> SourceResponse:
    """
    Upload frame images and run reconstruction.

    Files are written to a temporary server-side folder (sorted by filename to
    preserve frame order) then passed to the pipeline via FrameReader — the
    same path as /process_source but without requiring filesystem access.
    
    If map_id is specified, the drone's reconstruction will be stitched with
    other drones in the same map for multi-drone scenarios.
    
    If frame_timeout_secs is specified, any frame exceeding this processing time
    will be skipped and not included in the reconstruction (no marks).
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    tmp_dir = Path("results_online") / "uploads" / f"{drone_id}_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for i, upload in enumerate(sorted(files, key=lambda f: f.filename or "")):
            suffix = Path(upload.filename).suffix.lower() if upload.filename else ".jpg"
            dest = tmp_dir / f"{i:06d}{suffix}"
            dest.write_bytes(await upload.read())

        params = dict(
            keyframe_stride=keyframe_stride,
            initial_winsize=initial_winsize,
            win_r=win_r,
            conf_thres_i2p=conf_thres_i2p,
            num_scene_frame=num_scene_frame,
            max_num_register=max_num_register,
            conf_thres_l2w=conf_thres_l2w,
            num_points_save=num_points_save,
            save_each_frame=save_each_frame,
            device=device,
            frame_timeout_secs=frame_timeout_secs,
        )
        return _run_pipeline(drone_id, str(tmp_dir), params, tracker_service, map_id=map_id)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Cancellation and timeout control endpoints
# ---------------------------------------------------------------------------


class CancelResponse(BaseModel):
    drone_id: str
    message: str
    cancelled: bool


class TimeoutResponse(BaseModel):
    drone_id: str
    timeout_secs: float | None
    message: str


class SkippedFramesResponse(BaseModel):
    drone_id: str
    skipped_frames: List[int]


@router.post("/cancel/{drone_id}", response_model=CancelResponse)
def cancel_processing(
    drone_id: str,
    tracker_service: TrackerService = Depends(get_tracker_service),
) -> CancelResponse:
    """Request cancellation of ongoing frame processing for a drone."""
    cancelled = tracker_service.cancel_processing(drone_id)
    if not cancelled:
        raise HTTPException(
            status_code=404,
            detail=f"Drone '{drone_id}' not found or not currently processing.",
        )
    return CancelResponse(
        drone_id=drone_id,
        message=f"Cancellation requested for drone '{drone_id}'",
        cancelled=True,
    )


@router.post("/set_frame_timeout/{drone_id}", response_model=TimeoutResponse)
def set_frame_timeout(
    drone_id: str,
    timeout_secs: float | None = None,
    tracker_service: TrackerService = Depends(get_tracker_service),
) -> TimeoutResponse:
    """Set or disable per-frame timeout for a drone."""
    success = tracker_service.set_frame_timeout(drone_id, timeout_secs)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Drone '{drone_id}' not found.",
        )
    return TimeoutResponse(
        drone_id=drone_id,
        timeout_secs=timeout_secs,
        message=(
            f"Frame timeout for drone '{drone_id}' set to {timeout_secs}s"
            if timeout_secs is not None
            else f"Frame timeout for drone '{drone_id}' disabled"
        ),
    )


@router.get("/skipped_frames/{drone_id}", response_model=SkippedFramesResponse)
def get_skipped_frames(
    drone_id: str,
    tracker_service: TrackerService = Depends(get_tracker_service),
) -> SkippedFramesResponse:
    """Get list of frames skipped due to timeout for a drone."""
    skipped = tracker_service.get_skipped_frames(drone_id)
    if skipped is None:
        raise HTTPException(
            status_code=404,
            detail=f"Drone '{drone_id}' not found.",
        )
    return SkippedFramesResponse(drone_id=drone_id, skipped_frames=skipped)