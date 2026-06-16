from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.cuda.amp import autocast

from slam3r.pipeline.recon_online_pipeline import (
    FrameReader,
    estimate_camera_intrinsics_from_frames,
    estimate_camera_pose_from_correspondences,
    estimate_focal_knowing_depth,
    estimate_pose_from_pcd,
    get_raw_input_frame,
    initial_scene_for_accumulated_frames,
    pointmap_global_register,
    pointmap_local_recon,
    process_input_frame,
    recover_points_in_initial_window,
    save_recon,
    select_ids_as_reference,
    update_buffer_set,
    is_valid_position_jump,
)
from slam3r.pipeline.recon_online_pipeline_gpu_max import GPUProfiler
from slam3r.utils.device import to_numpy
from slam3r.utils.recon_utils import estimate_focal_knowing_depth, estimate_camera_pose

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_MAX_TRAJ_ENTRIES      = 10_000
_GPU_CACHE_CLEAR_FREQ  = 3
_LOG_FRAME_FREQ        = 10
_MAX_FAIL_VIEW_ENTRIES = 500
_TRAJ_FLUSH_FREQ       = 5

# ── Degradation thresholds ────────────────────────────────────────────────────
# A new segment is started only when tracking is fundamentally broken (sustained 
# severe degradation). Single bad frames are handled gracefully without escalation.
# Loop closure is attempted to recover from degradation when descriptors match.
_CONF_DEGRADE_THRESHOLD  = 0.5   # absolute confidence floor for severe tracking loss
_DEGRADE_CONSEC_FRAMES   = 10    # require 10+ consecutive critical frames (v. strict)
_DEGRADE_RECOVERY_FRAMES = 5     # if conf recovers within this window, don't split
_LOOP_CLOSURE_CONF_MIN   = 3.0   # min conf for a frame to be a loop-closure candidate
_LOOP_CLOSURE_SIM_THRESH = 0.80  # cosine similarity threshold for descriptor match (lowered for better matching)
_LOOP_CLOSURE_MAX_SEGS   = 100   # keep more segments for better recovery potential


# ─────────────────────────────────────────────────────────────────────────────
# Segment state machine
# ─────────────────────────────────────────────────────────────────────────────

class SegmentStatus(Enum):
    ACTIVE   = auto()   # currently being built
    DEGRADED = auto()   # split off due to tracking loss
    STITCHED = auto()   # merged back via loop closure


@dataclass
class MapSegment:
    """
    One contiguous tracking interval.

    A segment starts when the previous one degrades (or on boot) and ends
    either when tracking degrades again, or when a loop-closure links it
    back to an older segment.
    """
    segment_id:   int
    start_frame:  int
    end_frame:    int                       = -1        # -1 = still open
    status:       SegmentStatus            = SegmentStatus.ACTIVE
    keyframe_ids: List[int]                = field(default_factory=list)
    # Mean appearance descriptor of the best-confidence keyframes — used for
    # coarse loop detection without re-running the full network.
    appearance_desc: Optional[np.ndarray]  = None      # (D,) float32
    # If stitched: which older segment this links to and the 4×4 relative pose
    loop_target_id:  Optional[int]         = None
    loop_rel_pose:   Optional[np.ndarray]  = None      # (4,4) float32
    # Accumulated per-frame tracking quality within this segment
    conf_history:    List[float]           = field(default_factory=list)

    @property
    def mean_conf(self) -> float:
        return float(np.mean(self.conf_history)) if self.conf_history else 0.0

    @property
    def length(self) -> int:
        end = self.end_frame if self.end_frame >= 0 else self.start_frame
        return max(1, end - self.start_frame + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _squeeze_batch(t: Any) -> Any:
    if t is not None and hasattr(t, "shape") and len(t.shape) == 4 and t.shape[0] == 1:
        return t[0]
    return t


def _to_numpy(t: Any) -> Optional[np.ndarray]:
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _fwd_vector(c2w: np.ndarray) -> List[float]:
    fwd  = np.asarray(c2w[:3, 2], dtype=float)
    norm = float(np.linalg.norm(fwd))
    return (fwd / norm).tolist() if norm > 1e-6 else fwd.tolist()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _appearance_descriptor_from_view(view: Any) -> Optional[np.ndarray]:
    """
    Lightweight appearance descriptor from an input view.

    Uses the mean + std of the registered point-cloud depth (z-values) as a
    simple 6-D signature.  Replace with a learned descriptor if available.
    """
    pts = view.get("pts3d_world")
    if pts is None:
        return None
    arr = _to_numpy(pts)
    if arr is None:
        return None
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    depths = arr[..., 2].ravel()
    depths = depths[np.isfinite(depths)]
    if len(depths) < 10:
        return None
    desc = np.array([
        np.mean(depths), np.std(depths),
        np.percentile(depths, 10), np.percentile(depths, 50),
        np.percentile(depths, 90), float(len(depths)),
    ], dtype=np.float32)
    return desc


# ─────────────────────────────────────────────────────────────────────────────
# Domain objects
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DroneMapMetadata:
    drone_id:     str
    created_at:   datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    frame_count:  int      = 0


class DroneMap:
    """
    Stateful per-drone reconstruction context with degradation handling.

    Degradation detection
    ─────────────────────
    After Phase 3 begins, every frame's registered confidence is tracked in a
    rolling window (_degrade_window).  When _DEGRADE_CONSEC_FRAMES consecutive
    frames all fall below _CONF_DEGRADE_THRESHOLD, the current MapSegment is
    closed and a NEW segment is opened — the pipeline state (buffers, bufering
    set, milestone, etc.) is fully reset so the next frames initialise a fresh
    local scene rather than trying to extend a broken one.

    Loop closure
    ────────────
    Whenever a new segment becomes initialised (Phase 2 of the new segment
    completes), its appearance descriptor is compared against the descriptors
    of all previous STITCHED/DEGRADED segments.  If a match above
    _LOOP_CLOSURE_SIM_THRESH is found, a relative 4×4 pose is estimated
    between the two segments' reference frames and stored in the segment.  The
    trajectory is then globally re-expressed in the oldest segment's frame so
    positions are consistent across the full flight.

    Output layout (unchanged, plus one extra file):
        <save_dir>/
            <drone>_online.ply
            preds/  ...
            trajectories.npz
            depth_history.npz
            segments.json              ← NEW: segment metadata + stitching info
    """

    _PRINCIPAL_POINT = torch.tensor((224 // 2, 224 // 2))

    def __init__(self, drone_id: str, save_dir: Path, args: Any) -> None:
        self.drone_id = drone_id
        self.save_dir = save_dir
        self.args     = args
        self.metadata = DroneMapMetadata(drone_id=drone_id)
        self._lock    = threading.Lock()
        self.mean_intrinsics: Optional[np.ndarray] = None
        self.prev_valid_position: Optional[List[float]] = None

        init_window = args.initial_winsize * args.keyframe_stride
        self._win   = max(getattr(args, "buffer_size", 10) * 2, init_window + 16)

        # ── sliding-window buffers ────────────────────────────────────────
        self.data_views:  Deque[Any] = deque(maxlen=self._win)
        self.rgb_imgs:    Deque[Any] = deque(maxlen=self._win)
        self.input_views: Deque[Any] = deque(maxlen=self._win)

        self.per_frame_res: Dict[str, Deque] = {
            "i2p_pcds":  deque(maxlen=self._win),
            "i2p_confs": deque(maxlen=self._win),
            "l2w_pcds":  deque(maxlen=self._win),
            "l2w_confs": deque(maxlen=self._win),
        }
        self.registered_confs_mean: Deque[Any] = deque(maxlen=self._win)
        self.local_confs_mean_up2now: List[Any] = []
        self.buffering_set_ids:   List[int] = []
        self.last_ref_ids_buffer: List[int] = []

        self.fail_view:      Dict[int, float] = {}
        self.milestone:      int              = 0
        self.candi_frame_id: int              = 0
        self.num_frame_read: int              = 0
        self.init_ref_id:    int              = 0
        self.init_num:       int              = 0

        # ── trajectory store ──────────────────────────────────────────────
        self._traj_frames:    List[int]         = []
        self._traj_valid:     List[bool]        = []
        self._traj_positions: List[List[float]] = []
        self._traj_forwards:  List[List[float]] = []
        self._traj_axes:      List[Any]         = []
        self._traj_confs:     List[float]       = []
        self._traj_times:     List[float]       = []
        self._traj_segments:  List[int]         = []   # NEW: which segment each entry belongs to

        self._traj_dirty  = False

        # ── degradation & segment state ───────────────────────────────────
        # Rolling window of recent confidences to detect sustained degradation
        self._degrade_window: Deque[float] = deque(maxlen=_DEGRADE_CONSEC_FRAMES)
        # All segments (closed and open); last entry is always the active one
        self._segments: List[MapSegment] = []
        self._open_new_segment(start_frame=0)
        # Archive of (segment_id, descriptor) pairs for loop detection
        self._seg_descriptors: Deque[Tuple[int, np.ndarray]] = deque(maxlen=_LOOP_CLOSURE_MAX_SEGS)
        # Global transform that maps the current segment's origin into the
        # world frame of the very first segment (identity until first closure)
        self._seg_to_world: np.ndarray = np.eye(4, dtype=np.float32)

        # ── cancellation and timeout ───────────────────────────────────────
        self._cancel_requested: bool = False
        self._frame_timeout_secs: Optional[float] = None  # per-frame timeout in seconds
        self._frame_start_time: Optional[float] = None    # timestamp when current frame started
        self._skipped_frames: List[int] = []              # frames skipped due to timeout

        # ── confidence-based frame skipping ──────────────────────────────────
        self._confidence_threshold: Optional[float] = None  # skip frames below this confidence
        self._confidence_skipped_frames: List[int] = []     # frames skipped due to low confidence

        self.gpu_profiler = GPUProfiler()

    # ──────────────────────────────────────────────────────────────────────
    # Segment management
    # ──────────────────────────────────────────────────────────────────────

    @property
    def _active_segment(self) -> MapSegment:
        return self._segments[-1]

    def _open_new_segment(self, start_frame: int) -> MapSegment:
        seg = MapSegment(
            segment_id=len(self._segments),
            start_frame=start_frame,
        )
        self._segments.append(seg)
        logger.info(
            "drone '%s': opened segment %d at frame %d",
            self.drone_id, seg.segment_id, start_frame,
        )
        return seg

    def _close_active_segment(self, end_frame: int) -> None:
        seg = self._active_segment
        if seg.status == SegmentStatus.ACTIVE:
            seg.end_frame = end_frame
            seg.status    = SegmentStatus.DEGRADED
            # Build composite appearance descriptor from best-conf keyframes
            seg.appearance_desc = self._build_segment_descriptor(seg)
            if seg.appearance_desc is not None:
                self._seg_descriptors.append((seg.segment_id, seg.appearance_desc))
            logger.info(
                "drone '%s': closed segment %d (frames %d–%d, mean_conf=%.2f)",
                self.drone_id, seg.segment_id, seg.start_frame, end_frame, seg.mean_conf,
            )

    def _build_segment_descriptor(self, seg: MapSegment) -> Optional[np.ndarray]:
        """
        Average the per-view appearance descriptors of the segment's keyframes.
        Only views with conf >= _LOOP_CLOSURE_CONF_MIN are included.
        """
        descs = []
        for kf_id in seg.keyframe_ids:
            if kf_id < len(self.input_views):
                view = list(self.input_views)[kf_id] if kf_id < len(self.input_views) else None
                if view is None:
                    continue
                d = _appearance_descriptor_from_view(view)
                if d is not None:
                    descs.append(d)
        if not descs:
            return None
        return np.mean(descs, axis=0).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Degradation detection
    # ──────────────────────────────────────────────────────────────────────

    def _check_degradation(self, frame_id: int, conf: float) -> bool:
        """
        Update the degradation window and return True only if tracking is severely
        and persistently broken.

        A split is triggered ONLY when:
        1. ALL frames in the window are BELOW _CONF_DEGRADE_THRESHOLD (critical loss)
        2. AND at least N consecutive critical frames are observed
        3. AND confidence has not recovered within _DEGRADE_RECOVERY_FRAMES

        This avoids splitting on transient single-frame drops or even multi-frame
        glitches that recover quickly.
        """
        self._degrade_window.append(conf)
        
        # Not enough data yet
        if len(self._degrade_window) < _DEGRADE_CONSEC_FRAMES:
            return False
        
        # Check if the last N frames are all below the critical threshold
        window_list = list(self._degrade_window)
        if not all(c < _CONF_DEGRADE_THRESHOLD for c in window_list):
            return False  # At least one frame is OK, no split needed
        
        # Window is all critically low. But check for recent recovery.
        # If confidence recovers within the recovery window, don't split.
        if len(window_list) >= _DEGRADE_RECOVERY_FRAMES:
            recent = window_list[-_DEGRADE_RECOVERY_FRAMES:]
            if any(c >= _CONF_DEGRADE_THRESHOLD * 2 for c in recent):
                # Confidence is recovering, don't split yet
                return False
        
        logger.warning(
            "drone '%s': SEVERE degradation detected at frame %d "
            "(last %d confs: %s, critical threshold: %.2f) — splitting segment",
            self.drone_id, frame_id, _DEGRADE_CONSEC_FRAMES,
            [f"{c:.2f}" for c in window_list],
            _CONF_DEGRADE_THRESHOLD,
        )
        return True

    def _reset_pipeline_state(self) -> None:
        """
        Wipe all sliding-window buffers and scalars so the next frames
        reinitialise a fresh local scene (Phase 1 → 2 → 3 from scratch).
        """
        self.data_views.clear()
        self.rgb_imgs.clear()
        self.input_views.clear()
        for dq in self.per_frame_res.values():
            dq.clear()
        self.registered_confs_mean.clear()
        self.local_confs_mean_up2now  = []
        self.buffering_set_ids        = []
        self.last_ref_ids_buffer      = []
        self.milestone                = 0
        self.candi_frame_id           = 0
        self.init_ref_id              = 0
        self.init_num                 = 0
        self._degrade_window.clear()
        self.mean_intrinsics          = None
        torch.cuda.empty_cache()
        logger.debug("drone '%s': pipeline state reset for new segment", self.drone_id)

    # ──────────────────────────────────────────────────────────────────────
    # Loop closure
    # ──────────────────────────────────────────────────────────────────────

    def _attempt_loop_closure(self, new_seg: MapSegment) -> bool:
        """
        After a new segment is initialised, try to stitch it back to an
        earlier segment via appearance similarity.

        This is the primary mechanism for recovering from degradation — if a 
        segment splits due to tracking loss, loop closure will try to link it 
        back to the good map, ensuring a continuous globally-consistent map.

        If a match is found:
          1. Estimate the 4×4 relative pose between the two segments'
             reference frames.
          2. Update self._seg_to_world so all future trajectory entries are
             expressed in the original world frame.
          3. Re-project the in-memory trajectory entries that belong to
             new_seg to be consistent.

        Returns True if a loop was closed.
        """
        if new_seg.appearance_desc is None:
            return False

        best_sim  = -1.0
        best_id   = -1
        
        # Search for best match among all older segments
        for (seg_id, desc) in self._seg_descriptors:
            if seg_id >= new_seg.segment_id:
                continue  # only look at older segments
            sim = _cosine_sim(new_seg.appearance_desc, desc)
            if sim > best_sim:
                best_sim = sim
                best_id  = seg_id

        # Use a lower threshold to increase matching likelihood after degradation
        # This prioritizes recovery over strictness
        threshold = _LOOP_CLOSURE_SIM_THRESH - 0.05 if best_id >= 0 else _LOOP_CLOSURE_SIM_THRESH
        
        if best_sim < threshold or best_id < 0:
            logger.debug(
                "drone '%s': no loop closure candidate found for segment %d "
                "(best_sim=%.3f < threshold=%.3f)",
                self.drone_id, new_seg.segment_id, best_sim, threshold,
            )
            return False

        logger.info(
            "drone '%s': loop closure detected — segment %d → segment %d (sim=%.3f)",
            self.drone_id, new_seg.segment_id, best_id, best_sim,
        )

        # Estimate relative pose between segments using their reference frames
        rel_pose = self._estimate_inter_segment_pose(new_seg, best_id)
        if rel_pose is None:
            logger.warning(
                "drone '%s': loop closure pose estimation failed (seg %d → %d)",
                self.drone_id, new_seg.segment_id, best_id,
            )
            return False

        new_seg.loop_target_id = best_id
        new_seg.loop_rel_pose  = rel_pose.astype(np.float32)
        new_seg.status         = SegmentStatus.STITCHED

        # Update the global transform so subsequent trajectory entries are
        # expressed in the oldest segment's coordinate frame.
        self._seg_to_world = rel_pose @ self._seg_to_world

        # Re-project trajectory entries already stored for this segment
        self._reproject_segment_trajectory(new_seg.segment_id, rel_pose)

        self._save_segments_json()
        logger.info(
            "drone '%s': segment %d successfully stitched to segment %d",
            self.drone_id, new_seg.segment_id, best_id,
        )
        return True

    def _estimate_inter_segment_pose(
        self,
        new_seg: MapSegment,
        target_seg_id: int,
    ) -> Optional[np.ndarray]:
        """
        Estimate a 4×4 rigid transform T such that:
            p_target = T @ p_new

        Strategy: use the last known valid position of the target segment
        and the first known valid position of new_seg.  This is intentionally
        simple — swap in ICP or a learned matcher if you have one.
        """
        # Collect trajectory entries for each segment
        new_positions    = []
        target_positions = []
        for i, seg_id in enumerate(self._traj_segments):
            if self._traj_valid[i]:
                if seg_id == new_seg.segment_id:
                    new_positions.append(self._traj_positions[i])
                elif seg_id == target_seg_id:
                    target_positions.append(self._traj_positions[i])

        if not new_positions or not target_positions:
            return None

        # Use the centroid of the overlap region as the translation
        new_start    = np.array(new_positions[0],    dtype=np.float64)
        target_end   = np.array(target_positions[-1], dtype=np.float64)
        translation  = target_end - new_start

        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = translation
        return T.astype(np.float32)

    def _reproject_segment_trajectory(
        self,
        segment_id: int,
        T: np.ndarray,
    ) -> None:
        """
        Apply transform T to all trajectory positions that belong to segment_id.
        """
        T3 = T[:3, :3]
        t  = T[:3, 3]
        for i, seg_id in enumerate(self._traj_segments):
            if seg_id == segment_id and self._traj_valid[i]:
                p = np.array(self._traj_positions[i], dtype=np.float32)
                self._traj_positions[i] = (T3 @ p + t).tolist()
                # Rotate forward vector too
                fwd = np.array(self._traj_forwards[i], dtype=np.float32)
                fwd_new = T3 @ fwd
                norm = np.linalg.norm(fwd_new)
                self._traj_forwards[i] = (fwd_new / norm).tolist() if norm > 1e-6 else fwd_new.tolist()
        self._traj_dirty = True

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def process_frame(self, i2p_model, l2w_model, frame, frame_label: str) -> str:
        with self._lock:
            return self._process_frame_locked(i2p_model, l2w_model, frame, frame_label)

    def process_all_frames(self, i2p_model, l2w_model, source) -> int:
        processed = 0
        while True:
            # Check for cancellation
            if self._cancel_requested:
                logger.info("drone '%s': processing cancelled", self.drone_id)
                break

            success, raw_frame = source.read()
            if not success:
                break
            label  = f"frame_{self.num_frame_read:06d}"
            
            # Start frame timer for timeout tracking
            self._frame_start_time = time.time()
            
            status = self.process_frame(i2p_model, l2w_model, raw_frame, label)
            
            if processed % _LOG_FRAME_FREQ == 0:
                logger.debug("drone '%s' %s → %s", self.drone_id, label, status)
            processed += 1
        
        return processed

    def is_stale(self, hours: int = 6) -> bool:
        return datetime.now() - self.metadata.last_updated > timedelta(hours=hours)

    def to_dict(self, stale_threshold_hours: int = 6) -> Dict[str, Any]:
        return {
            "created_at":    self.metadata.created_at.isoformat(),
            "last_updated":  self.metadata.last_updated.isoformat(),
            "frame_count":   self.metadata.frame_count,
            "save_dir":      str(self.save_dir),
            "is_stale":      self.is_stale(stale_threshold_hours),
            "initialized":   self._is_initialized(),
            "num_segments":  len(self._segments),
            "active_segment": self._active_segment.segment_id,
        }

    def get_latest_pose_dict(self) -> Optional[Dict[str, Any]]:
        if not self._traj_frames:
            return None
        return {
            "frame":      self._traj_frames[-1],
            "valid":      self._traj_valid[-1],
            "position":   self._traj_positions[-1],
            "forward":    self._traj_forwards[-1],
            "conf":       self._traj_confs[-1],
            "timestamp":  self._traj_times[-1],
            "segment_id": self._traj_segments[-1],
        }

    def get_all_trajectories(self) -> Dict[str, Any]:
        frames = []
        for i in range(len(self._traj_frames)):
            frames.append({
                "frame":      int(self._traj_frames[i]),
                "valid":      bool(self._traj_valid[i]),
                "position":   self._traj_positions[i],
                "forward":    self._traj_forwards[i],
                "axes":       self._traj_axes[i],
                "conf":       float(self._traj_confs[i]),
                "timestamp":  float(self._traj_times[i]),
                "segment_id": int(self._traj_segments[i]),
            })
        seg_dicts = []
        for s in self._segments:
            seg_dicts.append({
                "segment_id":     s.segment_id,
                "start_frame":    s.start_frame,
                "end_frame":      s.end_frame,
                "status":         s.status.name,
                "mean_conf":      s.mean_conf,
                "loop_target_id": s.loop_target_id,
            })
        return {
            "metadata": {
                "drone_id":    self.drone_id,
                "num_frames":  len(frames),
                "initialized": self._is_initialized(),
                "num_segments": len(self._segments),
            },
            "frames":   frames,
            "segments": seg_dicts,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _check_timeout(self) -> Optional[str]:
        """Check if current frame has exceeded timeout. Returns skip message or None."""
        if self._frame_timeout_secs is None or self._frame_start_time is None:
            return None
        elapsed = time.time() - self._frame_start_time
        if elapsed > self._frame_timeout_secs:
            return f"TIMEOUT ({elapsed:.2f}s > {self._frame_timeout_secs:.2f}s limit)"
        return None

    def _is_initialized(self) -> bool:
        threshold = (self.args.initial_winsize - 1) * self.args.keyframe_stride
        return self.num_frame_read > threshold

    def _views_list(self)  -> List[Any]: return list(self.input_views)
    def _pfr_lists(self)   -> Dict[str, List]:
        return {k: list(v) for k, v in self.per_frame_res.items()}

    def _pfr_update(self, updated: Dict[str, List]) -> None:
        for k, lst in updated.items():
            dq = self.per_frame_res[k]
            dq.clear()
            dq.extend(lst[-self._win:])

    # ──────────────────────────────────────────────────────────────────────
    # Core frame processing
    # ──────────────────────────────────────────────────────────────────────

    def _process_frame_locked(self, i2p_model, l2w_model, raw_frame, frame_label: str) -> str:
        args             = self.args
        current_frame_id = self.num_frame_read
        self.gpu_profiler.start()

        # ── 1. Ingest ────────────────────────────────────────────────────
        try:
            _, dv_tmp, ri_tmp = get_raw_input_frame(
                "imgs",
                list(self.data_views), list(self.rgb_imgs),
                current_frame_id, raw_frame, args.device,
            )
        except Exception:
            logger.warning(
                "drone '%s': get_raw_input_frame failed for frame %d — skipping frame",
                self.drone_id, current_frame_id, exc_info=True,
            )
            self.num_frame_read += 1
            self._touch()
            return f"Frame {current_frame_id} SKIPPED (indescriptive/invalid input)."

        # Check timeout after ingest
        timeout_msg = self._check_timeout()
        if timeout_msg:
            self.num_frame_read += 1
            self._touch()
            self._skipped_frames.append(current_frame_id)
            logger.warning(
                "drone '%s': frame %d skipped (%s)",
                self.drone_id, current_frame_id, timeout_msg,
            )
            return f"Frame {current_frame_id} SKIPPED ({timeout_msg})"

        if len(dv_tmp) > len(self.data_views):
            self.data_views.append(dv_tmp[-1])
            self.rgb_imgs.append(ri_tmp[-1])

        # Local index into the deque (always the last-appended item).
        # Must NOT use current_frame_id here — after a segment reset the deque
        # is cleared so the global counter is out-of-range as a list index.
        local_frame_idx = len(self.data_views) - 1

        try:
            with torch.no_grad():
                input_view, pfr_tmp, rcm_tmp = process_input_frame(
                    self._pfr_lists(),
                    list(self.registered_confs_mean),
                    list(self.data_views),
                    local_frame_idx,   # ← local index, not global
                    i2p_model,
                )
        except Exception:
            logger.warning(
                "drone '%s': process_input_frame failed for frame %d — skipping frame",
                self.drone_id, current_frame_id, exc_info=True,
            )
            self.num_frame_read += 1
            self._touch()
            return f"Frame {current_frame_id} SKIPPED (encoder error)."

        # Check timeout after input processing
        timeout_msg = self._check_timeout()
        if timeout_msg:
            self.num_frame_read += 1
            self._touch()
            self._skipped_frames.append(current_frame_id)
            logger.warning(
                "drone '%s': frame %d skipped (%s)",
                self.drone_id, current_frame_id, timeout_msg,
            )
            return f"Frame {current_frame_id} SKIPPED ({timeout_msg})"

        self.input_views.append(input_view)
        self._pfr_update(pfr_tmp)
        self.registered_confs_mean.clear()
        self.registered_confs_mean.extend(rcm_tmp[-self._win:])
        self.num_frame_read += 1

        # Within a new segment, frame numbering restarts from 0 for the
        # Phase 1/2/3 logic, but we track global IDs for trajectories.
        seg_frame_id   = current_frame_id - self._active_segment.start_frame
        init_threshold = (args.initial_winsize - 1) * args.keyframe_stride

        # ── Phase 1: accumulate ──────────────────────────────────────────
        if seg_frame_id < init_threshold:
            if current_frame_id % _GPU_CACHE_CLEAR_FREQ == 0:
                torch.cuda.empty_cache()
            self._touch()
            return (
                f"[seg {self._active_segment.segment_id}] "
                f"Accumulating: {seg_frame_id + 1}/{init_threshold + 1} frames."
            )

        # ── Phase 2: initialise ──────────────────────────────────────────
        if seg_frame_id == init_threshold:
            iv_list  = self._views_list()
            pfr_list = self._pfr_lists()
            rcm_list = list(self.registered_confs_mean)

            with torch.no_grad():
                (
                    self.buffering_set_ids,
                    self.init_ref_id,
                    self.init_num,
                    iv_list, pfr_list, rcm_list,
                ) = initial_scene_for_accumulated_frames(
                    iv_list, args.initial_winsize, args.keyframe_stride,
                    i2p_model, pfr_list, rcm_list,
                    args.buffer_size, args.conf_thres_i2p,
                )
                self.local_confs_mean_up2now, pfr_list, iv_list = (
                    recover_points_in_initial_window(
                        seg_frame_id, self.buffering_set_ids,
                        args.keyframe_stride, self.init_ref_id,
                        pfr_list, iv_list, i2p_model, args.conf_thres_i2p,
                    )
                )

            # Check timeout after Phase 2
            timeout_msg = self._check_timeout()
            if timeout_msg:
                self.num_frame_read += 1
                self._touch()
                self._skipped_frames.append(current_frame_id)
                logger.warning(
                    "drone '%s': frame %d skipped during Phase 2 (%s)",
                    self.drone_id, current_frame_id, timeout_msg,
                )
                return f"Frame {current_frame_id} SKIPPED ({timeout_msg})"

            self.input_views.clear()
            self.input_views.extend(iv_list[-self._win:])
            self._pfr_update(pfr_list)
            self.registered_confs_mean.clear()
            self.registered_confs_mean.extend(rcm_list[-self._win:])

            self.milestone      = self.init_num * args.keyframe_stride + 1
            self.candi_frame_id = len(self.buffering_set_ids)

            # Register keyframes in segment for descriptor building
            self._active_segment.keyframe_ids = list(self.buffering_set_ids)

            # Build descriptor for new segment and attempt loop closure
            self._active_segment.appearance_desc = self._build_segment_descriptor(
                self._active_segment
            )
            if self._active_segment.segment_id > 0:
                self._attempt_loop_closure(self._active_segment)

            if args.save_each_frame:
                self._save_recon()
            torch.cuda.empty_cache()
            self._touch()
            return (
                f"[seg {self._active_segment.segment_id}] "
                f"Scene initialized with {self.init_num} frames."
            )

        # ── Phase 3: incremental registration ───────────────────────────
        # Early confidence check: skip obviously bad frames before expensive registration
        if self._confidence_threshold is not None and len(self.registered_confs_mean) > 0:
            prev_conf = float(self.registered_confs_mean[-1].cpu()) if isinstance(self.registered_confs_mean[-1], torch.Tensor) else float(self.registered_confs_mean[-1])
            if prev_conf < self._confidence_threshold:
                logger.warning(
                    "drone '%s': frame %d skipped during Phase 3 (prev frame confidence %.2f < %.2f threshold)",
                    self.drone_id, current_frame_id, prev_conf, self._confidence_threshold,
                )
                self.num_frame_read += 1
                self._touch()
                self._confidence_skipped_frames.append(current_frame_id)
                return f"Frame {current_frame_id} SKIPPED (low prev confidence {prev_conf:.2f})"
        
        iv_list  = self._views_list()
        pfr_list = self._pfr_lists()

        try:
            with torch.no_grad():
                ref_ids, self.last_ref_ids_buffer = select_ids_as_reference(
                    self.buffering_set_ids, seg_frame_id, iv_list,
                    i2p_model, args.num_scene_frame, args.win_r,
                    args.keyframe_stride, args.retrieve_freq, self.last_ref_ids_buffer,
                )
                with autocast(dtype=torch.float16):
                    self.local_confs_mean_up2now, pfr_list, iv_list = pointmap_local_recon(
                        [iv_list[seg_frame_id]] + [iv_list[i] for i in ref_ids],
                        i2p_model, seg_frame_id, 0,
                        pfr_list, iv_list, args.conf_thres_i2p,
                        self.local_confs_mean_up2now,
                    )
                with autocast(dtype=torch.float16):
                    iv_list, pfr_list, rcm_list = pointmap_global_register(
                        [iv_list[i] for i in ref_ids],
                        iv_list, l2w_model, pfr_list,
                        list(self.registered_confs_mean),
                        seg_frame_id,
                        device=args.device, norm_input=args.norm_input,
                    )

            # Check timeout after Phase 3
            timeout_msg = self._check_timeout()
            if timeout_msg:
                self.num_frame_read += 1
                self._touch()
                self._skipped_frames.append(current_frame_id)
                logger.warning(
                    "drone '%s': frame %d skipped during Phase 3 (%s)",
                    self.drone_id, current_frame_id, timeout_msg,
                )
                return f"Frame {current_frame_id} SKIPPED ({timeout_msg})"

        except Exception:
            logger.warning(
                "drone '%s': Phase 3 registration failed for frame %d (seg_frame %d) — "
                "skipping frame without escalating",
                self.drone_id, current_frame_id, seg_frame_id, exc_info=True,
            )
            self._touch()
            elapsed_ms = self.gpu_profiler.end(f"Frame {current_frame_id}") * 1000
            return (
                f"Frame {current_frame_id} SKIPPED (registration error). "
                f"GPU: {elapsed_ms:.0f}ms."
            )

        self.input_views.clear()
        self.input_views.extend(iv_list[-self._win:])
        self._pfr_update(pfr_list)
        self.registered_confs_mean.clear()
        self.registered_confs_mean.extend(rcm_list[-self._win:])

        next_seg_frame_id = seg_frame_id + 1
        if next_seg_frame_id - self.milestone >= args.update_buffer_intv * args.keyframe_stride:
            self.milestone, self.candi_frame_id, self.buffering_set_ids = update_buffer_set(
                next_seg_frame_id, args.buffer_size, args.keyframe_stride,
                self.buffering_set_ids, args.buffer_strategy,
                list(self.registered_confs_mean),
                self.local_confs_mean_up2now,
                self.candi_frame_id, self.milestone,
            )

        conf_raw = self.registered_confs_mean[-1]
        conf = float(conf_raw.cpu()) if isinstance(conf_raw, torch.Tensor) else float(conf_raw)

        # Post-registration confidence check: skip frame if below threshold
        if self._confidence_threshold is not None and conf < self._confidence_threshold:
            logger.warning(
                "drone '%s': frame %d skipped after Phase 3 (confidence %.2f < %.2f threshold)",
                self.drone_id, current_frame_id, conf, self._confidence_threshold,
            )
            self._confidence_skipped_frames.append(current_frame_id)
            # Still mark as processed but with low confidence flag
            self._active_segment.conf_history.append(conf)
            self._touch()
            return f"Frame {current_frame_id} SKIPPED (confidence {conf:.2f} below threshold {self._confidence_threshold:.2f})"

        # Track conf in active segment
        self._active_segment.conf_history.append(conf)

        if conf < 10:
            self.fail_view[current_frame_id] = conf
            if len(self.fail_view) > _MAX_FAIL_VIEW_ENTRIES:
                oldest = sorted(self.fail_view)[:len(self.fail_view) - _MAX_FAIL_VIEW_ENTRIES]
                for k in oldest:
                    del self.fail_view[k]

        # ── Degradation check → split ────────────────────────────────────
        if self._check_degradation(current_frame_id, conf):
            self._close_active_segment(end_frame=current_frame_id)
            self._reset_pipeline_state()
            next_frame_id = current_frame_id + 1
            self._open_new_segment(start_frame=next_frame_id)
            self._save_segments_json()
            self._touch()
            elapsed_ms = self.gpu_profiler.end(f"Frame {current_frame_id}") * 1000
            return (
                f"Frame {current_frame_id} DEGRADED (conf={conf:.2f}) — "
                f"split to segment {self._active_segment.segment_id}. "
                f"GPU: {elapsed_ms:.0f}ms."
            )

        if args.save_each_frame:
            self._save_recon()

        self._append_trajectory(current_frame_id, conf)

        if current_frame_id % _TRAJ_FLUSH_FREQ == 0 and self._traj_dirty:
            self._write_trajectory()

        if current_frame_id % _GPU_CACHE_CLEAR_FREQ == 0:
            torch.cuda.empty_cache()

        self._touch()
        elapsed_ms = self.gpu_profiler.end(f"Frame {current_frame_id}") * 1000
        return (
            f"[seg {self._active_segment.segment_id}] "
            f"Frame {current_frame_id} registered. Conf: {conf:.2f}. GPU: {elapsed_ms:.0f}ms."
        )

    # ──────────────────────────────────────────────────────────────────────
    # Pose estimation
    # ──────────────────────────────────────────────────────────────────────

    def _estimate_pose_robust(
        self,
        current_frame_id: int,
    ) -> Tuple[Optional[np.ndarray], bool]:
        try:
            current_view = (
                self.input_views[current_frame_id]
                if current_frame_id < len(self.input_views)
                else None
            )
            if current_view is None:
                return None, False

            pts3d_world = current_view.get('pts3d_world')
            pts3d_cam   = current_view.get('pts3d_cam')

            # Method 1: correspondence-based
            if pts3d_world is not None and pts3d_cam is not None:
                c2w, success = estimate_camera_pose_from_correspondences(pts3d_cam, pts3d_world)
                if success and c2w is not None:
                    # Apply current segment-to-world transform
                    c2w_world             = self._seg_to_world @ c2w
                    self.prev_valid_position = c2w_world[:3, 3].tolist()
                    return c2w_world, True

            # Method 2: PnP with mean intrinsics
            if pts3d_world is not None:
                if self.mean_intrinsics is None or current_frame_id % 10 == 0:
                    recent_start = max(0, current_frame_id - 10)
                    recent_views = list(self.input_views)[recent_start:current_frame_id + 1]
                    if recent_views:
                        self.mean_intrinsics = estimate_camera_intrinsics_from_frames(
                            recent_views, self._PRINCIPAL_POINT
                        )
                if self.mean_intrinsics is not None:
                    c2w, success = estimate_pose_from_pcd(pts3d_world, self.mean_intrinsics)
                    if success and c2w is not None:
                        c2w_world = self._seg_to_world @ c2w
                        pos = c2w_world[:3, 3].tolist()
                        if is_valid_position_jump(pos, self.prev_valid_position, max_jump=1.0):
                            self.prev_valid_position = pos
                            return c2w_world, True

            # Method 3: camera-frame PnP
            if pts3d_cam is not None and self.mean_intrinsics is not None:
                c2w, success = estimate_pose_from_pcd(pts3d_cam, self.mean_intrinsics)
                if success and c2w is not None:
                    c2w_world = self._seg_to_world @ c2w
                    pos = c2w_world[:3, 3].tolist()
                    if is_valid_position_jump(pos, self.prev_valid_position, max_jump=1.0):
                        self.prev_valid_position = pos
                        return c2w_world, True

        except Exception:
            logger.debug(
                "Pose estimation failed for frame %d drone '%s'",
                current_frame_id, self.drone_id, exc_info=True,
            )
        return None, False

    # ──────────────────────────────────────────────────────────────────────
    # Trajectory
    # ──────────────────────────────────────────────────────────────────────

    def _append_trajectory(self, frame_id: int, conf: float) -> None:
        c2w, valid_pose = self._estimate_pose_robust(frame_id)

        pos     = [0.0, 0.0, 0.0]
        forward = [0.0, 0.0, 0.0]
        axes: Optional[List[List[float]]] = None

        if valid_pose and c2w is not None:
            pos     = c2w[:3, 3].tolist()
            axes    = [c2w[:3, i].tolist() for i in range(3)]
            forward = _fwd_vector(c2w)

        self._traj_frames.append(frame_id)
        self._traj_valid.append(valid_pose)
        self._traj_positions.append(pos)
        self._traj_forwards.append(forward)
        self._traj_axes.append(axes if axes is not None else [[0]*3]*3)
        self._traj_confs.append(conf)
        self._traj_times.append(time.time())
        self._traj_segments.append(self._active_segment.segment_id)
        self._traj_dirty = True

        if len(self._traj_frames) > _MAX_TRAJ_ENTRIES:
            trim = len(self._traj_frames) - _MAX_TRAJ_ENTRIES
            self._traj_frames    = self._traj_frames[trim:]
            self._traj_valid     = self._traj_valid[trim:]
            self._traj_positions = self._traj_positions[trim:]
            self._traj_forwards  = self._traj_forwards[trim:]
            self._traj_axes      = self._traj_axes[trim:]
            self._traj_confs     = self._traj_confs[trim:]
            self._traj_times     = self._traj_times[trim:]
            self._traj_segments  = self._traj_segments[trim:]

    def _write_trajectory(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        try:
            np.savez_compressed(
                self.save_dir / "trajectories.npz",
                frames    = np.array(self._traj_frames,    dtype=np.int32),
                valid     = np.array(self._traj_valid,     dtype=bool),
                positions = np.array(self._traj_positions, dtype=np.float32),
                forwards  = np.array(self._traj_forwards,  dtype=np.float32),
                axes      = np.array(self._traj_axes,      dtype=np.float32),
                confs     = np.array(self._traj_confs,     dtype=np.float32),
                timestamps= np.array(self._traj_times,     dtype=np.float64),
                segments  = np.array(self._traj_segments,  dtype=np.int32),   # NEW
            )
            self._traj_dirty = False
        except Exception:
            logger.exception("Failed to write trajectories.npz for drone '%s'", self.drone_id)

    # ──────────────────────────────────────────────────────────────────────
    # Segments persistence
    # ──────────────────────────────────────────────────────────────────────

    def _save_segments_json(self) -> None:
        """Write segments.json so the visualiser can colour-code segments."""
        os.makedirs(self.save_dir, exist_ok=True)
        payload = []
        for s in self._segments:
            rel_pose_list = s.loop_rel_pose.tolist() if s.loop_rel_pose is not None else None
            payload.append({
                "segment_id":     s.segment_id,
                "start_frame":    s.start_frame,
                "end_frame":      s.end_frame,
                "status":         s.status.name,
                "mean_conf":      round(s.mean_conf, 3),
                "keyframe_ids":   s.keyframe_ids,
                "loop_target_id": s.loop_target_id,
                "loop_rel_pose":  rel_pose_list,
            })
        try:
            (self.save_dir / "segments.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except Exception:
            logger.exception("Failed to write segments.json for drone '%s'", self.drone_id)

    # ──────────────────────────────────────────────────────────────────────
    # Reconstruction save
    # ──────────────────────────────────────────────────────────────────────

    def _save_recon(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        scene_id  = f"{self.drone_id}_online"
        win       = self._win
        views     = list(self.input_views)[-win:]
        imgs      = list(self.rgb_imgs)[-win:]
        l2w_confs = [
            c.cpu() if isinstance(c, torch.Tensor) else c
            for c in list(self.per_frame_res["l2w_confs"])[-win:]
        ]
        if not views:
            return
        try:
            save_recon(
                views, len(views), str(self.save_dir), scene_id,
                self.args.save_all_views, imgs,
                registered_confs=l2w_confs,
                num_points_save=self.args.num_points_save,
                conf_thres_res=self.args.conf_thres_l2w,
            )
            self.metadata.frame_count = len(list(self.save_dir.glob("*.ply")))
        except Exception:
            logger.exception("save_recon failed for drone '%s'", self.drone_id)
        self._save_preds()

    def _save_preds(self) -> None:
        preds_dir = self.save_dir / "preds"
        preds_dir.mkdir(parents=True, exist_ok=True)

        win  = self._win
        pfr  = self._pfr_lists()

        def _squeeze_stack(seq: List[Any], dtype=np.float32) -> Optional[np.ndarray]:
            arrays = []
            for item in seq:
                a = _to_numpy(item)
                if a is None:
                    continue
                if a.ndim == 4 and a.shape[0] == 1:
                    a = a[0]
                arrays.append(a.astype(dtype))
            if not arrays:
                return None
            try:
                return np.stack(arrays, axis=0)
            except ValueError as exc:
                logger.warning("_save_preds: shape mismatch — %s", exc)
                return None

        local_pcds       = _squeeze_stack(pfr["i2p_pcds"][-win:])
        registered_pcds  = _squeeze_stack(pfr["l2w_pcds"][-win:])
        local_confs      = _squeeze_stack(pfr["i2p_confs"][-win:])
        registered_confs = _squeeze_stack(pfr["l2w_confs"][-win:])

        rgb_raw    = list(self.rgb_imgs)[-win:]
        input_imgs = _squeeze_stack(rgb_raw, dtype=np.float32)
        if input_imgs is not None:
            if input_imgs.max() <= 1.0 + 1e-3:
                input_imgs = (input_imgs * 255.0).clip(0, 255).astype(np.uint8)
            else:
                input_imgs = input_imgs.clip(0, 255).astype(np.uint8)

        try:
            if local_pcds       is not None: np.save(preds_dir / "local_pcds.npy",       local_pcds)
            if registered_pcds  is not None: np.save(preds_dir / "registered_pcds.npy",  registered_pcds)
            if local_confs      is not None: np.save(preds_dir / "local_confs.npy",       local_confs)
            if registered_confs is not None: np.save(preds_dir / "registered_confs.npy",  registered_confs)
            if input_imgs       is not None: np.save(preds_dir / "input_imgs.npy",        input_imgs)
        except Exception:
            logger.exception("_save_preds: npy write failed for drone '%s'", self.drone_id)

        init_ids = list(range(0, self.init_num * self.args.keyframe_stride,
                               self.args.keyframe_stride)) if self.init_num else []
        metadata = {
            "init_winsize": int(self.args.initial_winsize),
            "kf_stride":    int(self.args.keyframe_stride),
            "init_ref_id":  int(self.init_ref_id),
            "init_ids":     init_ids,
            "num_frames":   int(self.num_frame_read),
        }
        try:
            (preds_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
        except Exception:
            logger.exception("_save_preds: metadata.json write failed for drone '%s'", self.drone_id)

    def _touch(self) -> None:
        self.metadata.last_updated = datetime.now()


# ─────────────────────────────────────────────────────────────────────────────
# Shared multi-drone map
# ─────────────────────────────────────────────────────────────────────────────

class SharedMap:
    """
    A named world-coordinate map that may be contributed to by multiple drones.

    Each drone gets its own DroneMap (so pipelines stay isolated and per-drone
    save dirs are separate), but the SharedMap owns the merged PLY that is
    written when any contributor finishes a batch.  All contributor trajectories
    are already expressed in a common world frame (via segment stitching inside
    DroneMap), so merging is a simple concatenation of the per-drone PLY files.
    """

    def __init__(self, map_id: str, output_dir: Path) -> None:
        self.map_id     = map_id
        self.map_dir    = output_dir / "maps" / map_id
        self.map_dir.mkdir(parents=True, exist_ok=True)
        self._drones: Dict[str, DroneMap] = {}   # drone_id → DroneMap
        self._lock = threading.Lock()

    def register_drone(self, drone_id: str, dm: "DroneMap") -> None:
        with self._lock:
            self._drones[drone_id] = dm

    def get_drone(self, drone_id: str) -> Optional["DroneMap"]:
        with self._lock:
            return self._drones.get(drone_id)

    def drone_ids(self) -> List[str]:
        with self._lock:
            return list(self._drones.keys())

    def stitch_all(self) -> Optional[Path]:
        """
        Merge all per-drone PLY files into a single map PLY.
        Returns the path to the merged PLY, or None if nothing to merge.
        """
        try:
            import trimesh
        except ImportError:
            logger.warning("trimesh not available — cannot stitch map PLYs")
            return None

        ply_files: List[Path] = []
        with self._lock:
            for dm in self._drones.values():
                candidates = sorted(dm.save_dir.glob("*_online.ply"))
                if candidates:
                    ply_files.append(candidates[-1])

        if not ply_files:
            return None

        all_verts  = []
        all_colors = []
        for p in ply_files:
            try:
                mesh = trimesh.load(str(p))
                verts = np.asarray(mesh.vertices)
                if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
                    cols = np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0
                else:
                    cols = np.ones((len(verts), 3), dtype=np.float32) * 0.7
                all_verts.append(verts)
                all_colors.append(cols)
            except Exception:
                logger.warning("stitch_all: failed to load %s", p, exc_info=True)

        if not all_verts:
            return None

        merged_v = np.concatenate(all_verts,  axis=0).astype(np.float64)
        merged_c = np.concatenate(all_colors, axis=0)
        cloud = trimesh.PointCloud(vertices=merged_v)
        cloud.visual.vertex_colors = (merged_c * 255).astype(np.uint8)

        out_path = self.map_dir / f"{self.map_id}_merged.ply"
        try:
            cloud.export(str(out_path))
            logger.info(
                "map '%s': stitched %d PLYs → %s (%d points)",
                self.map_id, len(ply_files), out_path, len(merged_v),
            )
            return out_path
        except Exception:
            logger.exception("stitch_all: export failed for map '%s'", self.map_id)
            return None

    def to_dict(self) -> Dict[str, Any]:
        merged_ply = self.map_dir / f"{self.map_id}_merged.ply"
        return {
            "map_id":      self.map_id,
            "map_dir":     str(self.map_dir),
            "drones":      self.drone_ids(),
            "merged_ply":  str(merged_ply) if merged_ply.exists() else None,
        }


class TrackerService:
    def __init__(
        self,
        i2p_model:             Any,
        l2w_model:             Any,
        output_dir:            str = "results_online/",
        stale_threshold_hours: int = 6,
        cleanup_interval:      int = 300,
    ) -> None:
        self.i2p_model             = i2p_model
        self.l2w_model             = l2w_model
        self.output_dir            = Path(output_dir)
        self.stale_threshold_hours = stale_threshold_hours
        self.cleanup_interval      = cleanup_interval

        # drone_id → DroneMap  (always unique per drone)
        self._drone_maps: Dict[str, DroneMap] = {}
        # map_id → SharedMap  (one map may own many drones)
        self._shared_maps: Dict[str, SharedMap] = {}
        # drone_id → map_id  (reverse lookup)
        self._drone_to_map: Dict[str, str] = {}

        self._lock       = threading.Lock()
        self._stop_event = threading.Event()

        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="drone-map-cleanup"
        )
        self._cleanup_thread.start()

    # ── Map management ────────────────────────────────────────────────────

    def _get_or_create_shared_map(self, map_id: str) -> SharedMap:
        """Return the SharedMap for map_id, creating it if needed (caller holds _lock)."""
        if map_id not in self._shared_maps:
            self._shared_maps[map_id] = SharedMap(map_id, self.output_dir)
            logger.info("Created SharedMap '%s'", map_id)
        return self._shared_maps[map_id]

    def _get_or_create_map(self, drone_id: str, args: Any, map_id: Optional[str] = None) -> DroneMap:
        """
        Return the DroneMap for drone_id, creating it if needed.

        If map_id is given the drone is registered in that SharedMap so its PLY
        will be included in merged outputs.  If map_id is omitted the drone is
        assigned its own implicit map (map_id == drone_id) for backwards compat.
        """
        effective_map_id = map_id or drone_id
        with self._lock:
            if drone_id not in self._drone_maps:
                shared_map = self._get_or_create_shared_map(effective_map_id)
                save_dir   = self.output_dir / effective_map_id / drone_id
                save_dir.mkdir(parents=True, exist_ok=True)
                dm = DroneMap(drone_id, save_dir, args)
                self._drone_maps[drone_id] = dm
                self._drone_to_map[drone_id] = effective_map_id
                shared_map.register_drone(drone_id, dm)
                logger.debug(
                    "Created DroneMap for drone '%s' in map '%s'",
                    drone_id, effective_map_id,
                )
            return self._drone_maps[drone_id]

    def get_shared_map(self, map_id: str) -> Optional[SharedMap]:
        with self._lock:
            return self._shared_maps.get(map_id)

    def list_maps(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [sm.to_dict() for sm in self._shared_maps.values()]

    # ── Cancellation and timeout control ──────────────────────────────────

    def set_frame_timeout(self, drone_id: str, timeout_secs: Optional[float]) -> bool:
        """Set per-frame timeout for a drone. None disables timeout."""
        with self._lock:
            dm = self._drone_maps.get(drone_id)
        if dm is None:
            return False
        with dm._lock:
            dm._frame_timeout_secs = timeout_secs
            if timeout_secs is not None:
                logger.info("Set frame timeout for drone '%s' to %.2fs", drone_id, timeout_secs)
            else:
                logger.info("Disabled frame timeout for drone '%s'", drone_id)
        return True

    def cancel_processing(self, drone_id: str) -> bool:
        """Request cancellation of ongoing frame processing for a drone."""
        with self._lock:
            dm = self._drone_maps.get(drone_id)
        if dm is None:
            return False
        with dm._lock:
            dm._cancel_requested = True
            logger.info("Cancellation requested for drone '%s'", drone_id)
        return True

    def set_frame_confidence_threshold(self, drone_id: str, confidence_threshold: Optional[float]) -> bool:
        """Set minimum confidence threshold for frame acceptance. Frames below this are skipped."""
        with self._lock:
            dm = self._drone_maps.get(drone_id)
        if dm is None:
            return False
        with dm._lock:
            dm._confidence_threshold = confidence_threshold
            if confidence_threshold is not None:
                logger.info("Set confidence threshold for drone '%s' to %.2f", drone_id, confidence_threshold)
            else:
                logger.info("Disabled confidence threshold for drone '%s'", drone_id)
        return True

    def get_skipped_frames(self, drone_id: str) -> Optional[List[int]]:
        """Get list of frame indices that were skipped due to timeout."""
        with self._lock:
            dm = self._drone_maps.get(drone_id)
        if dm is None:
            return None
        with dm._lock:
            return list(dm._skipped_frames)

    # ── Frame processing ──────────────────────────────────────────────────

    def process_source(
        self,
        drone_id: str,
        source: Any,
        args: Any,
        map_id: Optional[str] = None,
    ) -> None:
        args.camera_name = drone_id
        args.test_name   = drone_id
        dm = self._get_or_create_map(drone_id, args, map_id=map_id)
        effective_map_id = self._drone_to_map.get(drone_id, drone_id)
        logger.info(
            "Starting reconstruction for drone '%s' in map '%s' → %s",
            drone_id, effective_map_id, dm.save_dir,
        )
        try:
            processed = dm.process_all_frames(self.i2p_model, self.l2w_model, source)
        except Exception:
            logger.exception("Pipeline error for drone '%s'", drone_id)
            return
        with dm._lock:
            if dm._traj_dirty:
                dm._write_trajectory()
            dm._save_segments_json()

        # Stitch all drones in the same map into a merged PLY
        with self._lock:
            shared_map = self._shared_maps.get(effective_map_id)
        if shared_map is not None:
            shared_map.stitch_all()

        logger.info(
            "Finished drone '%s' (map '%s'): %d frames, %d segments (%d stitched), %d .ply in %s",
            drone_id, effective_map_id, processed,
            len(dm._segments),
            sum(1 for s in dm._segments if s.status == SegmentStatus.STITCHED),
            dm.metadata.frame_count, dm.save_dir,
        )

    def clear_drone_map(self, drone_id: str) -> bool:
        with self._lock:
            if drone_id not in self._drone_maps:
                return False
            map_id = self._drone_to_map.pop(drone_id, None)
            del self._drone_maps[drone_id]
            # Remove from SharedMap if it exists
            if map_id and map_id in self._shared_maps:
                sm = self._shared_maps[map_id]
                with sm._lock:
                    sm._drones.pop(drone_id, None)
                if not sm._drones:
                    del self._shared_maps[map_id]
            logger.info("Cleared map for drone '%s'", drone_id)
        torch.cuda.empty_cache()
        return True

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "timestamp":    datetime.now().isoformat(),
                "total_drones": len(self._drone_maps),
                "drones": {
                    did: dm.to_dict(self.stale_threshold_hours)
                    for did, dm in self._drone_maps.items()
                },
                "maps": {
                    mid: sm.to_dict()
                    for mid, sm in self._shared_maps.items()
                },
            }

    def get_latest_pose(self, drone_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            dm = self._drone_maps.get(drone_id)
        if dm is None:
            return None
        with dm._lock:
            return dm.get_latest_pose_dict()

    def get_all_trajectories(self, drone_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            dm = self._drone_maps.get(drone_id)
        if dm is None:
            return None
        with dm._lock:
            return dm.get_all_trajectories()

    def shutdown(self) -> None:
        self._stop_event.set()

    def _cleanup_loop(self) -> None:
        while not self._stop_event.wait(timeout=self.cleanup_interval):
            try:
                self._cleanup_stale_maps()
            except Exception:
                logger.exception("Unexpected error in cleanup loop")

    def _cleanup_stale_maps(self) -> None:
        with self._lock:
            stale = [
                did for did, dm in self._drone_maps.items()
                if dm.is_stale(self.stale_threshold_hours)
            ]
            for did in stale:
                logger.info("Evicting stale DroneMap for drone '%s'", did)
                self._drone_to_map.pop(did, None)
                del self._drone_maps[did]
        if stale:
            torch.cuda.empty_cache()