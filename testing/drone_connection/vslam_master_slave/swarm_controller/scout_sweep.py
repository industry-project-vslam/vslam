from __future__ import annotations

import math
from dataclasses import dataclass

from .config import SwarmConfig
from .ranger import RangerReading
from .safety import EnvelopeState


@dataclass(frozen=True)
class ScoutLaneState:
    position: str
    state: EnvelopeState
    reading: RangerReading
    reason: str


@dataclass(frozen=True)
class ScoutSweepResult:
    center_lane_state: ScoutLaneState
    left_lane_state: ScoutLaneState
    right_lane_state: ScoutLaneState
    rear_lane_state: ScoutLaneState
    up_clearance_safe: bool

    @property
    def movement_allowed(self) -> bool:
        required = (
            self.center_lane_state,
            self.left_lane_state,
            self.right_lane_state,
            self.rear_lane_state,
        )
        return self.up_clearance_safe and all(lane.state == EnvelopeState.FREE for lane in required)


def evaluate_lane(config: SwarmConfig, position: str, reading: RangerReading, rear: bool = False) -> ScoutLaneState:
    critical = []
    checks = (
        ("front", reading.front, config.critical_front),
        ("left", reading.left, config.critical_side),
        ("right", reading.right, config.critical_side),
        ("up", reading.up, config.critical_up),
    )
    if rear:
        checks = (
            ("back", reading.back, config.critical_back),
            ("left", reading.left, config.critical_side),
            ("right", reading.right, config.critical_side),
            ("up", reading.up, config.critical_up),
        )
    for key, value, threshold in checks:
        if reading.valid.get(key, False) and math.isfinite(value) and value < threshold:
            critical.append(f"{key}<{threshold:.2f}")
    if critical:
        return ScoutLaneState(position, EnvelopeState.CRITICAL, reading, ",".join(critical))

    unknown = [key for key, *_ in checks if not reading.valid.get(key, False)]
    if unknown:
        return ScoutLaneState(position, EnvelopeState.UNKNOWN, reading, "unknown:" + ",".join(unknown))

    occupied = []
    for key, value, threshold in checks:
        required = max(threshold, config.formation_margin)
        if math.isfinite(value) and value < required:
            occupied.append(f"{key}<{required:.2f}")
    if occupied:
        return ScoutLaneState(position, EnvelopeState.OCCUPIED, reading, ",".join(occupied))

    return ScoutLaneState(position, EnvelopeState.FREE, reading, "free")


def build_sweep_result(
    config: SwarmConfig,
    center: RangerReading,
    left: RangerReading,
    right: RangerReading,
    rear: RangerReading,
) -> ScoutSweepResult:
    up_values = [center.up, left.up, right.up, rear.up]
    up_valid = all(reading.valid.get("up", False) for reading in (center, left, right, rear))
    up_safe = up_valid and all((not math.isfinite(value)) or value >= config.critical_up for value in up_values)
    return ScoutSweepResult(
        center_lane_state=evaluate_lane(config, "center", center),
        left_lane_state=evaluate_lane(config, "left", left),
        right_lane_state=evaluate_lane(config, "right", right),
        rear_lane_state=evaluate_lane(config, "rear", rear, rear=True),
        up_clearance_safe=up_safe,
    )
