from __future__ import annotations

import math
from dataclasses import dataclass

from .config import SwarmConfig
from .formation import FormationModel
from .ranger import RangerReading


@dataclass(frozen=True)
class SurfaceClassification:
    classification: str
    confidence: str
    next_action: str


def detect_surface_candidate(config: SwarmConfig, formation: FormationModel, front_ranger: RangerReading) -> bool:
    if not front_ranger.valid.get("front", False) or not math.isfinite(front_ranger.front):
        return False
    corrected = formation.corrected_front_wall_distance(front_ranger.front)
    target_hit = abs(corrected - config.target_wall_offset) <= config.offset_tolerance
    fallback_hit = abs(corrected - config.fallback_wall_offset) <= config.offset_tolerance
    return target_hit or fallback_hit


def classify_surface_probe(
    config: SwarmConfig,
    front_initial: float,
    front_after: float,
    probe_shift: float,
    noisy: bool = False,
) -> SurfaceClassification:
    if noisy or not math.isfinite(front_initial):
        return SurfaceClassification("AMBIGUOUS", "LOW", "FRONTIER_SAVE")
    if not math.isfinite(front_after) or front_after > front_initial + config.clear_increase:
        return SurfaceClassification("OBSTACLE_OR_OPENING", "HIGH", "BASE_FORMATION_BYPASS")
    if probe_shift >= config.max_probe_shift - 1e-6 and abs(front_after - front_initial) <= config.offset_tolerance:
        return SurfaceClassification("WALL_OR_BOUNDARY", "MEDIUM", "RANGER_RESLOT_TURN")
    return SurfaceClassification("AMBIGUOUS", "LOW", "FRONTIER_SAVE")
