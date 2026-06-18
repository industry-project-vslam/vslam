from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .config import SwarmConfig
from .formation import FormationModel
from .ranger import RangerMonitor, RangerReading


class EnvelopeState(str, Enum):
    FREE = "FREE"
    UNKNOWN = "UNKNOWN"
    OCCUPIED = "OCCUPIED"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class PlannedPrimitive:
    command: str
    distance: float = 0.0
    direction: str = "FORWARD"


@dataclass(frozen=True)
class EnvelopeResult:
    state: EnvelopeState
    reason: str
    required: dict[str, float]
    measured: dict[str, float]

    @property
    def allowed(self) -> bool:
        return self.state == EnvelopeState.FREE


@dataclass(frozen=True)
class SurfaceClassification:
    classification: str
    confidence: str
    next_action: str


def evaluate_critical_safety(
    config: SwarmConfig,
    ranger_monitor: RangerMonitor,
    front_ranger: RangerReading,
    back_ranger: RangerReading | None = None,
) -> str | None:
    reason = ranger_monitor.critical_reason(front_ranger, prefix="X_FRONT")
    if reason is not None:
        return reason
    if config.requires_back_ranger and back_ranger is not None:
        return ranger_monitor.critical_reason(back_ranger, prefix="X_BACK")
    return None


def evaluate_formation_envelope(
    config: SwarmConfig,
    formation: FormationModel,
    front_ranger: RangerReading,
    back_ranger: RangerReading | None,
    planned: PlannedPrimitive,
) -> EnvelopeResult:
    step = max(0.0, planned.distance)
    required_front = formation.forward_extent() + step + config.formation_margin
    required_back = formation.rear_extent() + config.formation_margin
    required_side = formation.side_extent() + config.formation_margin
    required_up = config.critical_up

    measured = {
        "front": front_ranger.front,
        "left": front_ranger.left,
        "right": front_ranger.right,
        "up": min(front_ranger.up, back_ranger.up) if back_ranger is not None else front_ranger.up,
    }
    if config.requires_back_ranger:
        measured["back"] = back_ranger.back if back_ranger is not None else math.nan
    required = {
        "front": required_front,
        "left": required_side,
        "right": required_side,
        "up": required_up,
    }
    if config.requires_back_ranger:
        required["back"] = required_back

    critical = _critical_reason(config, front_ranger, back_ranger)
    if critical is not None:
        return EnvelopeResult(EnvelopeState.CRITICAL, critical, required, measured)

    unknown = []
    for key, value in measured.items():
        if key == "up" and back_ranger is not None and config.requires_back_ranger:
            if _unknown(front_ranger.up, front_ranger, "up") or _unknown(back_ranger.up, back_ranger, "up"):
                unknown.append(key)
            continue
        reading = back_ranger if key == "back" else front_ranger
        if _unknown(value, reading, key):
            unknown.append(key)
    if unknown:
        return EnvelopeResult(EnvelopeState.UNKNOWN, "UNKNOWN_" + "_".join(key.upper() for key in unknown), required, measured)

    occupied = [key for key, value in measured.items() if value < required[key]]
    if occupied:
        return EnvelopeResult(
            EnvelopeState.OCCUPIED,
            "OCCUPIED_" + "_".join(f"{key.upper()}_{measured[key]:.2f}_LT_{required[key]:.2f}" for key in occupied),
            required,
            measured,
        )

    return EnvelopeResult(EnvelopeState.FREE, "FORMATION_ENVELOPE_FREE", required, measured)


def corrected_wall_candidate(config: SwarmConfig, formation: FormationModel, front_ranger: RangerReading) -> bool:
    if not front_ranger.valid.get("front", False) or not math.isfinite(front_ranger.front):
        return False
    corrected = formation.corrected_front_wall_distance(front_ranger.front)
    return abs(corrected - config.target_wall_offset) <= config.offset_tolerance


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


def bypass_envelope_free(envelope: EnvelopeResult) -> bool:
    return envelope.state == EnvelopeState.FREE


def evaluate_reslot_path_safety(
    config: SwarmConfig,
    formation: FormationModel,
    front_ranger: RangerReading,
    back_ranger: RangerReading | None,
    turn_direction: str,
    allow_unknown_front_back_leg: bool = False,
) -> EnvelopeResult:
    turn = turn_direction.upper()
    if turn not in {"LEFT", "RIGHT"}:
        raise ValueError(f"Unknown turn direction: {turn_direction}")

    measured: dict[str, float] = {}
    required: dict[str, float] = {}
    unknown: list[str] = []
    used_sensors: set[str] = set()

    waypoints = formation.reslot_waypoints(turn).get("X_FRONT", [])
    for index, (start, end) in enumerate(zip(waypoints, waypoints[1:]), start=1):
        delta = end - start
        forward_m, left_m = formation.drawing_delta_to_motion_meters(delta)
        for sensor_key, distance in _reslot_leg_sensor_requirements(forward_m, left_m):
            used_sensors.add(sensor_key)
            label = f"x_front_leg{index}_{sensor_key}"
            measured[label] = front_ranger.value(sensor_key)
            required[label] = abs(distance) + config.formation_margin
            if (
                sensor_key == "back"
                and allow_unknown_front_back_leg
                and not math.isfinite(measured[label])
            ):
                continue
            if _unknown(measured[label], front_ranger, sensor_key):
                unknown.append(label)

    if not measured:
        required["x_front_reslot"] = formation.reslot_required_clearance(turn) + config.formation_margin
        measured["x_front_reslot"] = math.inf

    if config.requires_back_ranger and back_ranger is not None:
        back_side_key = "left" if turn == "RIGHT" else "right"
        label = f"x_back_{back_side_key}"
        measured[label] = back_ranger.value(back_side_key)
        required[label] = formation.reslot_required_clearance(turn) + config.formation_margin
        if _unknown(measured[label], back_ranger, back_side_key):
            unknown.append(label)

    critical = _reslot_critical_reason(config, front_ranger, used_sensors)
    if critical is not None:
        return EnvelopeResult(EnvelopeState.CRITICAL, critical, required, measured)
    if unknown:
        return EnvelopeResult(EnvelopeState.UNKNOWN, "UNKNOWN_RESLOT_" + "_".join(unknown).upper(), required, measured)
    occupied = [key for key, value in measured.items() if value < required[key]]
    if occupied:
        return EnvelopeResult(EnvelopeState.OCCUPIED, "RESLOT_PATH_OCCUPIED_" + "_".join(occupied).upper(), required, measured)
    return EnvelopeResult(EnvelopeState.FREE, "RESLOT_PATH_FREE", required, measured)


def _reslot_leg_sensor_requirements(forward_m: float, left_m: float) -> list[tuple[str, float]]:
    requirements: list[tuple[str, float]] = []
    if abs(forward_m) > 1e-6:
        requirements.append(("front" if forward_m > 0.0 else "back", forward_m))
    if abs(left_m) > 1e-6:
        requirements.append(("left" if left_m > 0.0 else "right", left_m))
    return requirements


def _reslot_critical_reason(config: SwarmConfig, front: RangerReading, used_sensors: set[str]) -> str | None:
    """Critical checks for X_FRONT re-slot use only sensors required by the path.

    A wall directly in front should stop forward scout motion, but it should not
    also block a Ranger-only escape path that moves sideways/backward around the
    AI row. The up sensor remains global because ceiling clearance matters for
    every motion primitive.
    """

    thresholds = {
        "front": config.critical_front,
        "back": config.critical_back,
        "left": config.critical_side,
        "right": config.critical_side,
        "up": config.critical_up,
    }
    for key in sorted(used_sensors | {"up"}):
        value = front.value(key)
        threshold = thresholds[key]
        if math.isfinite(value) and value < threshold:
            return f"X_FRONT_{key.upper()}_CRITICAL_{value:.2f}m_LT_{threshold:.2f}m"
    return None


def evaluate_bypass_envelope(
    config: SwarmConfig,
    formation: FormationModel,
    front_ranger: RangerReading,
    back_ranger: RangerReading | None,
    direction: str,
    lateral_shift: float,
    forward_distance: float,
) -> EnvelopeResult:
    lateral = evaluate_formation_envelope(
        config,
        formation,
        front_ranger,
        back_ranger,
        PlannedPrimitive("FORMATION_SIDESTEP", lateral_shift, direction),
    )
    if lateral.state != EnvelopeState.FREE:
        return lateral
    return evaluate_formation_envelope(
        config,
        formation,
        front_ranger,
        back_ranger,
        PlannedPrimitive("FORMATION_FORWARD", forward_distance),
    )


def _unknown(value: float, reading: RangerReading | None, key: str) -> bool:
    if reading is None:
        return True
    if not reading.valid.get(key, False):
        return True
    return math.isnan(value) or value <= 0.0


def _critical_reason(config: SwarmConfig, front: RangerReading, back: RangerReading | None) -> str | None:
    front_checks = (
        ("FRONT", front.front, config.critical_front),
        ("LEFT", front.left, config.critical_side),
        ("RIGHT", front.right, config.critical_side),
        ("UP", front.up, config.critical_up),
    )
    for name, value, threshold in front_checks:
        if math.isfinite(value) and value < threshold:
            return f"X_FRONT_{name}_CRITICAL_{value:.2f}m_LT_{threshold:.2f}m"
    if back is not None:
        back_checks = (
            ("BACK", back.back, config.critical_back),
            ("LEFT", back.left, config.critical_side),
            ("RIGHT", back.right, config.critical_side),
            ("UP", back.up, config.critical_up),
        )
        for name, value, threshold in back_checks:
            if math.isfinite(value) and value < threshold:
                return f"X_BACK_{name}_CRITICAL_{value:.2f}m_LT_{threshold:.2f}m"
    return None
