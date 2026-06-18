from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from .config import DroneRole, FormationConfig, SwarmConfig
from .geometry import Heading, Point, crazyflie_yaw_from_heading, heading_from_degrees, heading_vector, normalize_yaw
from .ranger import RANGE_KEYS, RangerReading


class ObservationState(str, Enum):
    IDLE = "IDLE"
    CONNECTING = "CONNECTING"
    CONFIG_LOADED = "CONFIG_LOADED"
    CONNECTED = "CONNECTED"
    SENSOR_CHECK = "SENSOR_CHECK"
    STREAMING = "STREAMING"
    TAKEOFF = "TAKEOFF"
    YAW_ALIGN = "YAW_ALIGN"
    READY = "READY"
    OBSERVE_STEP_CHECK = "OBSERVE_STEP_CHECK"
    GROUP_FORWARD_STEP = "GROUP_FORWARD_STEP"
    BLOCKED_HOVER = "BLOCKED_HOVER"
    CHOOSE_TURN_SIDE = "CHOOSE_TURN_SIDE"
    X_FRONT_RESLOT = "X_FRONT_RESLOT"
    UPDATE_HEADING = "UPDATE_HEADING"
    REALIGN_YAWS = "REALIGN_YAWS"
    SAFE_HOVER = "SAFE_HOVER"
    LANDING = "LANDING"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    HARD_MOTOR_KILL = "HARD_MOTOR_KILL"
    MISSION_COMPLETE = "MISSION_COMPLETE"
    ERROR = "ERROR"


class EmergencyState(str, Enum):
    NONE = "NONE"
    SAFE_HOVER = "SAFE_HOVER"
    LANDING = "LANDING"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    HARD_MOTOR_KILL = "HARD_MOTOR_KILL"


@dataclass(frozen=True)
class RangerSnapshot:
    front: float = math.inf
    back: float = math.inf
    left: float = math.inf
    right: float = math.inf
    up: float = math.inf
    zrange: float = math.inf
    valid: dict[str, bool] = field(default_factory=lambda: {key: False for key in RANGE_KEYS})

    @classmethod
    def from_reading(cls, reading: RangerReading) -> RangerSnapshot:
        return cls(
            front=reading.front,
            back=reading.back,
            left=reading.left,
            right=reading.right,
            up=reading.up,
            zrange=reading.zrange,
            valid=dict(reading.valid),
        )

    def value(self, key: str) -> float:
        return float(getattr(self, key))


@dataclass(frozen=True)
class FormationSlot:
    drone_id: str
    role: DroneRole
    x: float
    y: float
    yaw_deg: float

    @property
    def point(self) -> Point:
        return Point(self.x, self.y)


def heading_to_vector(heading: Heading | float) -> Point:
    return heading_vector(heading)


def heading_to_yaw(heading: Heading | float) -> float:
    return crazyflie_yaw_from_heading(heading)


def yaw_for_drone_role(role: DroneRole, heading: Heading | float) -> float:
    base = heading_to_yaw(heading)
    offsets = {
        DroneRole.FRONT_RANGER: 0.0,
        DroneRole.BACK_RANGER: 0.0,
        DroneRole.AI_LEFT_STREAM: 270.0,
        DroneRole.AI_FORWARD_STREAM: 0.0,
        DroneRole.AI_BACKWARD_STREAM: 180.0,
        DroneRole.AI_RIGHT_STREAM: 0.0,
    }
    return normalize_yaw(base + offsets.get(role, 0.0))


def get_slot_offset(
    drone_id: str,
    role: DroneRole,
    heading: Heading | float,
    formation_config: FormationConfig | None = None,
) -> Point:
    config = formation_config or FormationConfig()
    heading = heading_from_degrees(heading.value if isinstance(heading, Heading) else heading)

    if role == DroneRole.FRONT_RANGER:
        half_width = _ai_half_width(config)
        if heading == Heading.NORTH:
            return Point(0.0, -config.front_clearance_units_north_south)
        if heading == Heading.EAST:
            return Point(config.front_clearance_units_east_west + half_width, 0.0)
        if heading == Heading.SOUTH:
            return Point(0.0, config.front_clearance_units_north_south)
        if heading == Heading.WEST:
            return Point(-(config.front_clearance_units_east_west + half_width), 0.0)

    if role == DroneRole.BACK_RANGER:
        half_width = _ai_half_width(config)
        if heading == Heading.NORTH:
            return Point(0.0, config.front_clearance_units_north_south)
        if heading == Heading.EAST:
            return Point(-(config.front_clearance_units_east_west + half_width), 0.0)
        if heading == Heading.SOUTH:
            return Point(0.0, -config.front_clearance_units_north_south)
        if heading == Heading.WEST:
            return Point(config.front_clearance_units_east_west + half_width, 0.0)

    if drone_id in config.ai_slots:
        x, y = config.ai_slots[drone_id]
        return Point(x - config.center_x, y - config.center_y)

    return Point(0.0, 0.0)


def get_commanded_slot(
    formation_anchor: Point,
    drone_id: str,
    role: DroneRole,
    heading: Heading | float,
    formation_config: FormationConfig | None = None,
) -> FormationSlot:
    offset = get_slot_offset(drone_id, role, heading, formation_config)
    point = formation_anchor + offset
    return FormationSlot(drone_id, role, point.x, point.y, yaw_for_drone_role(role, heading))


def choose_turn_side(snapshot: RangerSnapshot, critical_side: float = 0.50) -> str | None:
    left = snapshot.left if _valid_distance(snapshot, "left") else -1.0
    right = snapshot.right if _valid_distance(snapshot, "right") else -1.0
    left_safe = left >= critical_side
    right_safe = right >= critical_side
    if not left_safe and not right_safe:
        return None
    if right_safe and (not left_safe or right >= left):
        return "RIGHT"
    return "LEFT"


def is_sensor_snapshot_valid(snapshot: RangerSnapshot, required: tuple[str, ...] = ("front", "left", "right", "up")) -> bool:
    return all(_valid_distance(snapshot, key) for key in required)


def is_forward_safe(snapshot: RangerSnapshot, config: SwarmConfig) -> bool:
    if not is_sensor_snapshot_valid(snapshot):
        return False
    return (
        snapshot.front >= config.critical_front
        and snapshot.left >= config.critical_side
        and snapshot.right >= config.critical_side
        and snapshot.up >= config.critical_up
    )


def compensate_body_velocity(
    vx_form: float,
    vy_form: float,
    formation_heading_deg: float,
    drone_yaw_deg: float,
) -> tuple[float, float]:
    # cflib's hover setpoint uses +Y as body-left and -Y as body-right. The
    # formation heading is map-style, where +90 means right/east; convert it to
    # Crazyflie yaw before comparing it with the drone's actual yaw.
    formation_yaw_deg = crazyflie_yaw_from_heading(formation_heading_deg)
    relative_rad = math.radians(drone_yaw_deg - formation_yaw_deg)
    vx_body = math.cos(relative_rad) * vx_form + math.sin(relative_rad) * vy_form
    vy_body = -math.sin(relative_rad) * vx_form + math.cos(relative_rad) * vy_form
    return vx_body, vy_body


def _valid_distance(snapshot: RangerSnapshot, key: str) -> bool:
    value = snapshot.value(key)
    return bool(snapshot.valid.get(key, False)) and (math.isfinite(value) and value > 0.0 or math.isinf(value))


def _ai_half_width(config: FormationConfig) -> float:
    if not config.ai_slots:
        return 0.0
    xs = [point[0] for point in config.ai_slots.values()]
    return max(abs(x - config.center_x) for x in xs)


def headingToVector(heading: Heading | float) -> Point:
    return heading_to_vector(heading)


def headingToYaw(heading: Heading | float) -> float:
    return heading_to_yaw(heading)


def yawForDroneRole(role: DroneRole, heading: Heading | float) -> float:
    return yaw_for_drone_role(role, heading)


def getSlotOffset(
    drone_id: str,
    role: DroneRole,
    heading: Heading | float,
    formation_config: FormationConfig | None = None,
) -> Point:
    return get_slot_offset(drone_id, role, heading, formation_config)


def getCommandedSlot(
    formationAnchor: Point,
    drone_id: str,
    role: DroneRole,
    heading: Heading | float,
    formation_config: FormationConfig | None = None,
) -> FormationSlot:
    return get_commanded_slot(formationAnchor, drone_id, role, heading, formation_config)


def chooseTurnSide(snapshot: RangerSnapshot, criticalSide: float = 0.50) -> str | None:
    return choose_turn_side(snapshot, criticalSide)


def isForwardSafe(snapshot: RangerSnapshot, config: SwarmConfig) -> bool:
    return is_forward_safe(snapshot, config)


def isSensorSnapshotValid(snapshot: RangerSnapshot, required: tuple[str, ...] = ("front", "left", "right", "up")) -> bool:
    return is_sensor_snapshot_valid(snapshot, required)


def compensateBodyVelocity(
    vxForm: float,
    vyForm: float,
    formationHeadingDeg: float,
    droneYawDeg: float,
) -> tuple[float, float]:
    return compensate_body_velocity(vxForm, vyForm, formationHeadingDeg, droneYawDeg)
