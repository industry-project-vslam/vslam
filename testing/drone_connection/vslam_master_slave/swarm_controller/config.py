from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DroneRole(str, Enum):
    FRONT_RANGER = "FRONT_RANGER"
    BACK_RANGER = "BACK_RANGER"
    AI_LEFT_STREAM = "AI_LEFT_STREAM"
    AI_FORWARD_STREAM = "AI_FORWARD_STREAM"
    AI_BACKWARD_STREAM = "AI_BACKWARD_STREAM"
    AI_RIGHT_STREAM = "AI_RIGHT_STREAM"


@dataclass(frozen=True)
class DroneConfig:
    uri: str
    drone_id: str
    role: DroneRole
    # Drawing-unit offset from the fixed AI row center.
    # Default scale: 1 drawing unit = 0.01 m.
    offset_x: float
    offset_y: float
    offset_z: float = 0.0
    stream_direction: str | None = None
    # Optional physical yaw offset from the current formation heading.
    # Use this when a specific drone's camera/mount direction must override
    # the generic role yaw. In the half-swarm test, O1 is side-looking while
    # O2 keeps the default forward-looking yaw from its role.
    yaw_offset_deg: float | None = None
    enabled: bool = True


@dataclass(frozen=True)
class FormationConfig:
    unit_to_meters: float = 0.01
    center_x: float = 32.0
    center_y: float = 45.0
    # Top-3 MVP geometry from the 120 x 118 start mat:
    # X_FRONT=(32,20), O1=(12,45), O2=(52,45).
    front_radius: float = 25.0
    back_radius: float = 25.0
    front_clearance_units_north_south: float = 45.0
    front_clearance_units_east_west: float = 45.0
    ai_slots: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "O1": (12.0, 45.0),
            "O2": (52.0, 45.0),
        }
    )


@dataclass(frozen=True)
class ThresholdConfig:
    target_wall_offset: float = 3.5
    fallback_wall_offset: float = 3.0
    offset_tolerance: float = 0.30
    critical_front: float = 0.70
    critical_side: float = 0.50
    critical_back: float = 0.50
    critical_up: float = 0.40
    probe_step: float = 0.25
    max_probe_shift: float = 1.25
    clear_increase: float = 0.70
    formation_margin: float = 0.50


@dataclass
class SwarmConfig:
    formation_mode: str = "TOP_3_X_FRONT_O1_O2"
    requires_back_ranger: bool = False
    formation_unit_to_meters: float = 0.01
    initial_step: float = 0.20
    step_size: float = 0.30
    max_step: float = 0.35
    side_step: float = 0.30
    target_wall_offset: float = 3.5
    fallback_wall_offset: float = 3.0
    offset_tolerance: float = 0.30
    critical_front: float = 0.70
    critical_side: float = 0.50
    critical_back: float = 0.50
    critical_up: float = 0.40
    warning_front: float = 1.00
    warning_side: float = 0.75
    probe_step: float = 0.25
    max_probe_shift: float = 1.25
    clear_increase: float = 0.70
    formation_margin: float = 0.50
    hover_time: float = 0.20
    hover_after_turn: float = 0.40
    demo_settle_time: float = 0.70
    command_timeout: float = 3.5
    test_takeoff_height: float = 0.30
    flight_height: float = 0.40
    takeoff_velocity: float = 0.30
    landing_velocity: float = 0.20
    initial_speed: float = 0.18
    speed: float = 0.25
    max_speed: float = 0.28
    absolute_max_speed: float = 0.28
    turn_reposition_total: float = 0.80
    turn_reposition_chunk: float = 0.30
    turn_reposition_chunk_max: float = 0.35
    front_recovery_target: float = 0.85
    front_recovery_step: float = 0.15
    front_recovery_max: float = 0.30
    front_recovery_speed: float = 0.15
    copy_front_recovery_to_followers: bool = True
    takeoff_stagger_delay: float = 2.0
    min_battery_v: float = 3.05
    battery_warn_v: float = 3.10
    battery_land_v: float = 3.00
    max_mission_time_s: float = 270.0
    auto_return_or_finish_time_s: float = 300.0
    forced_land_time_s: float = 330.0
    yaw_rate_deg_s: float = 72.0
    setpoint_hz: float = 20.0
    # Duration of one continuous motion command before the controller checks
    # safety/emergency and refreshes logs. Too small creates radio jitter because
    # each segment stops and restarts the hover-hold loop on every drone.
    motion_segment_s: float = 0.45
    mission_max_steps: int = 20
    video_demo_steps: int = 12
    formation: FormationConfig = field(default_factory=FormationConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    drones: list[DroneConfig] = field(default_factory=list)


def default_drone_configs() -> list[DroneConfig]:
    return [
        DroneConfig(
            uri="radio://0/82/2M/E7E7E7E701",
            drone_id="X_FRONT",
            role=DroneRole.FRONT_RANGER,
            offset_x=0.0,
            offset_y=-25.0,
        ),
        DroneConfig(
            uri="",
            drone_id="O1",
            role=DroneRole.AI_LEFT_STREAM,
            offset_x=-20.0,
            offset_y=0.0,
            stream_direction="left",
            yaw_offset_deg=270.0,
        ),
        DroneConfig(
            uri="",
            drone_id="O2",
            role=DroneRole.AI_FORWARD_STREAM,
            offset_x=20.0,
            offset_y=0.0,
            stream_direction="forward",
        ),
    ]


def default_swarm_config() -> SwarmConfig:
    return default_half_group_config()


def default_half_group_config() -> SwarmConfig:
    uri_by_id = {
        "X_FRONT": "radio://0/82/2M/E7E7E7E701",
        "O1": "radio://0/82/2M/E7E7E7E702",
        "O2": "radio://0/82/2M/E7E7E7E703",
    }
    drones = []
    for drone in default_drone_configs():
        if drone.drone_id not in uri_by_id:
            continue
        drones.append(
            DroneConfig(
                uri=uri_by_id[drone.drone_id],
                drone_id=drone.drone_id,
                role=drone.role,
                offset_x=drone.offset_x,
                offset_y=drone.offset_y,
                offset_z=drone.offset_z,
                stream_direction=drone.stream_direction,
                yaw_offset_deg=drone.yaw_offset_deg,
                enabled=drone.enabled,
            )
        )
    return SwarmConfig(
        formation_mode="TOP_3_X_FRONT_O1_O2",
        requires_back_ranger=False,
        formation_unit_to_meters=0.01,
        initial_step=0.20,
        step_size=0.30,
        max_step=0.35,
        side_step=0.30,
        test_takeoff_height=0.30,
        flight_height=0.40,
        takeoff_velocity=0.30,
        landing_velocity=0.20,
        initial_speed=0.18,
        speed=0.25,
        max_speed=0.28,
        absolute_max_speed=0.28,
        turn_reposition_total=0.80,
        turn_reposition_chunk=0.30,
        turn_reposition_chunk_max=0.35,
        front_recovery_target=0.85,
        front_recovery_step=0.15,
        front_recovery_max=0.30,
        front_recovery_speed=0.15,
        copy_front_recovery_to_followers=True,
        hover_time=0.35,
        hover_after_turn=0.80,
        demo_settle_time=0.70,
        takeoff_stagger_delay=2.0,
        min_battery_v=3.05,
        battery_warn_v=3.10,
        battery_land_v=3.00,
        motion_segment_s=0.45,
        mission_max_steps=20,
        video_demo_steps=12,
        drones=drones,
    )
