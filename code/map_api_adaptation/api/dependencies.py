"""Shared dependency providers for the Multi-Drone SLAM Tracker API."""

import os
from typing import Optional

from api.services.tracker_service import TrackerService

_tracker_service: Optional[TrackerService] = None


def init_tracker_service(
    output_dir: str = "results_online/",
    stale_threshold_hours: int = 6,
    cleanup_interval: int = 300,
) -> None:
    """
    Load SLAM3R models and initialise the singleton TrackerService.
    Call this once at application startup (lifespan).
    Mirrors the model-loading block in online_tracker.py main().
    """
    global _tracker_service

    device = os.environ.get("SLAM3R_DEVICE", "cuda")
    i2p_weights = os.environ.get("SLAM3R_I2P_WEIGHTS")
    l2w_weights = os.environ.get("SLAM3R_L2W_WEIGHTS")

    from slam3r.models import Image2PointsModel, Local2WorldModel

    if i2p_weights:
        import torch
        from slam3r.models import Image2PointsModel as _I2P
        ckpt = torch.load(i2p_weights, map_location=device)
        i2p_model = _I2P()
        i2p_model.load_state_dict(ckpt["model"], strict=False)
        del ckpt
    else:
        i2p_model = Image2PointsModel.from_pretrained("siyan824/slam3r_i2p")

    if l2w_weights:
        import torch
        from slam3r.models import Local2WorldModel as _L2W
        ckpt = torch.load(l2w_weights, map_location=device)
        l2w_model = _L2W()
        l2w_model.load_state_dict(ckpt["model"], strict=False)
        del ckpt
    else:
        l2w_model = Local2WorldModel.from_pretrained("siyan824/slam3r_l2w")

    i2p_model.to(device).eval()
    l2w_model.to(device).eval()

    _tracker_service = TrackerService(
        i2p_model=i2p_model,
        l2w_model=l2w_model,
        output_dir=output_dir,
        stale_threshold_hours=stale_threshold_hours,
        cleanup_interval=cleanup_interval,
    )


def get_tracker_service() -> TrackerService:
    """FastAPI dependency — returns the shared TrackerService instance."""
    if _tracker_service is None:
        raise RuntimeError("TrackerService has not been initialised. Call init_tracker_service() at startup.")
    return _tracker_service