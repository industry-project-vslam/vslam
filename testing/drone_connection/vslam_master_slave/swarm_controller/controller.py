from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from .ai_streams import AIStreamManager
from .classifier import classify_surface_probe, detect_surface_candidate
from .config import DroneConfig, DroneRole, SwarmConfig, default_half_group_config, default_swarm_config
from .drones import CrazyflieDrone, DroneLike, SimulationDrone
from .emergency import EmergencyManager
from .formation import FormationModel, HalfSwarmFormationModel
from .logs import SwarmLogger
from .motion import MotionController
from .preflight import PreflightManager
from .preflight import PREFLIGHT_LOG_VARIABLES, _ranger_from_values
from .ranger import RangerMonitor, RangerReading
from .safety import (
    EnvelopeState,
    PlannedPrimitive,
    evaluate_bypass_envelope,
    evaluate_critical_safety,
    evaluate_formation_envelope,
    evaluate_reslot_path_safety,
)
from .simulation_stub import scenarios
from .top3_logic import ObservationState, RangerSnapshot, choose_turn_side


class MissionMode(str, Enum):
    IDLE = "IDLE"
    CONNECTING = "CONNECTING"
    STREAM_CHECK = "STREAM_CHECK"
    RANGER_CHECK = "RANGER_CHECK"
    FORMATION_SETUP = "FORMATION_SETUP"
    SCOUT_SWEEP = "SCOUT_SWEEP"
    MANUAL_TEST = "MANUAL_TEST"
    FULL_OBSERVATION = "FULL_OBSERVATION"
    SENSOR_CHECK = "MODE_SENSOR_CHECK"
    SINGLE_DRONE_HOVER = "MODE_SINGLE_DRONE_HOVER"
    SINGLE_RANGER_SWEEP = "MODE_SINGLE_RANGER_SWEEP"
    WALL_OBSTACLE_PROBE_TEST = "MODE_WALL_OBSTACLE_PROBE_TEST"
    FORMATION_HOVER_ONLY = "MODE_FORMATION_HOVER_ONLY"
    FORMATION_MICRO_STEP = "MODE_FORMATION_MICRO_STEP"
    FULL_OBSERVATION_SAFE = "MODE_FULL_OBSERVATION_SAFE"
    PAUSE_HOVER = "PAUSE_HOVER"
    EMERGENCY = "EMERGENCY"
    LANDING = "LANDING"
    MISSION_COMPLETE = "MISSION_COMPLETE"
    REPLAY_LOGS = "REPLAY_LOGS"


@dataclass(frozen=True)
class DroneStatus:
    drone_id: str
    role: str
    uri: str
    connected: bool = False
    airborne: bool = False
    battery_v: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw_deg: float = 0.0


@dataclass(frozen=True)
class RangerStatus:
    drone_id: str
    front: float = math.inf
    back: float = math.inf
    left: float = math.inf
    right: float = math.inf
    up: float = math.inf
    zrange: float = math.inf
    min_clearance: float = math.inf
    corrected_front_wall_distance: float = math.inf
    health: str = "UNKNOWN"


@dataclass(frozen=True)
class AIStreamStatus:
    drone_id: str
    stream_direction: str
    active: bool = False
    fps: float = 0.0
    last_frame_timestamp: float = 0.0


@dataclass(frozen=True)
class SafetyEnvelopeStatus:
    state: str = EnvelopeState.UNKNOWN.value
    reason: str = "NOT_EVALUATED"
    front_required: float = 0.0
    side_required: float = 0.0
    back_required: float = 0.0


@dataclass(frozen=True)
class MissionStatus:
    mode: str = MissionMode.IDLE.value
    state: str = "INIT"
    emergency: bool = False
    battery_summary: str = "n/a"
    radio_status: str = "disconnected"
    formation_config_loaded: bool = False
    ranger_valid: bool = False
    ai_stream_manager_initialized: bool = False
    start_observation_enabled: bool = False
    disabled_reason: str = "Load config and connect X_FRONT/X_BACK"
    emergency_manager_active: bool = False
    hard_kill_armed: bool = False
    watchdog_status: str = "not started"
    last_setpoint_age_s: float = math.inf
    preflight_passed: bool = False
    heading_deg: float = 0.0
    log_dir: str = ""
    mission_elapsed_s: float = 0.0
    mission_remaining_s: float = 0.0
    auto_land_reason: str = ""
    chunk_progress: str = ""


@dataclass(frozen=True)
class SwarmStatusSnapshot:
    mission: MissionStatus
    drones: list[DroneStatus] = field(default_factory=list)
    rangers: list[RangerStatus] = field(default_factory=list)
    ai_streams: list[AIStreamStatus] = field(default_factory=list)
    envelope: SafetyEnvelopeStatus = field(default_factory=SafetyEnvelopeStatus)
    events: list[str] = field(default_factory=list)
    formation_slots: dict[str, tuple[float, float]] = field(default_factory=dict)
    ai_yaws: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservationLoopBudget:
    forward_steps: int
    continuous: bool
    max_decision_cycles: int | None
    label: str


class SwarmController:
    """Backend facade for GUI commands.

    The GUI calls only this class. All cflib/hardware access stays in backend
    drone abstractions and is never used directly by widgets.
    """

    def __init__(self, simulation: bool = True, scenario: str = "open_space") -> None:
        self.simulation = simulation
        self.scenario = scenario
        self.config: SwarmConfig | None = None
        self.formation: FormationModel | None = None
        self.ranger_monitor: RangerMonitor | None = None
        self.streams: AIStreamManager | None = None
        self.logger: SwarmLogger | None = None
        self.emergency_manager = EmergencyManager()
        self.preflight_manager: PreflightManager | None = None
        self.drones: dict[str, DroneLike] = {}
        self.mode = MissionMode.IDLE
        self.state = ObservationState.IDLE.value
        self.emergency = False
        self.events: list[str] = []
        self.last_envelope = SafetyEnvelopeStatus()
        self._connected = False
        self._airborne = False
        self._paused = False
        self._seq = 0
        self._mission_started_at: float | None = None
        self._auto_land_reason = ""
        self._chunk_progress = ""
        self._last_normal_setpoint_ts = 0.0
        self._allow_unknown_back_reslot_escape = False
        self._post_reslot_sync_next = False
        self._last_forward_progress = False
        self.real_flight_confirm = False
        self.mode_pass: dict[str, bool] = {
            "MODE_SENSOR_CHECK": False,
            "MODE_SINGLE_DRONE_HOVER": False,
            "MODE_SINGLE_RANGER_SWEEP": False,
            "MODE_WALL_OBSTACLE_PROBE_TEST": False,
            "MODE_FORMATION_HOVER_ONLY": False,
            "MODE_FORMATION_MICRO_STEP": False,
            "MODE_HALF_SWARM_HOVER": False,
            "MODE_HALF_SWARM_MICRO_STEP": False,
            "MODE_HALF_SWARM_OBSERVATION_SAFE": False,
        }

    def load_config(self, config: SwarmConfig | None = None) -> None:
        self.config = config or default_swarm_config()
        self.formation = HalfSwarmFormationModel(self.config.drones, formation_config=self.config.formation)
        self.ranger_monitor = RangerMonitor(self.config)
        self.streams = AIStreamManager()
        self.logger = SwarmLogger(mission_id="gui_fixed_swarm")
        self.preflight_manager = PreflightManager(self.config, self.emergency_manager)
        self.emergency_manager.reset_for_new_mission()
        self.logger.write_config(self.config, self.formation)
        self._seed_simulation_readings()
        self.mode = MissionMode.IDLE
        self.state = ObservationState.CONFIG_LOADED.value
        self._event("formation config loaded")

    def load_half_group_config(self) -> None:
        self.load_config(default_half_group_config())
        self._event("half-group config loaded: X_FRONT=E701, O1=E702, O2=E703")

    def set_simulation_mode(self, enabled: bool, scenario: str = "open_space") -> None:
        self.simulation = enabled
        self.scenario = scenario
        self.real_flight_confirm = not enabled
        self._event(f"simulation mode {'enabled' if enabled else 'disabled'} scenario={scenario}; REAL_FLIGHT_CONFIRM={self.real_flight_confirm}")

    def connect_all(self) -> None:
        self._require_config()
        assert self.config is not None
        self.mode = MissionMode.CONNECTING
        self.drones.clear()
        for drone_config in self.config.drones:
            if not drone_config.enabled:
                continue
            if not self.simulation and not drone_config.uri:
                continue
            drone = SimulationDrone(drone_config) if self.simulation else CrazyflieDrone(drone_config)
            drone.connect()
            self.drones[drone_config.drone_id] = drone
        if not self.simulation:
            self.emergency_manager.start_supervisor_watchdog(self.drones)
            self._event("supervisor watchdog started at 5 Hz")
        self._connected = True
        self.state = ObservationState.CONNECTED.value
        self._event("connect all finished")

    def run_sensor_check(self) -> None:
        self._require_connected()
        assert self.config is not None and self.preflight_manager is not None and self.logger is not None
        self.mode = MissionMode.SENSOR_CHECK
        self.state = ObservationState.SENSOR_CHECK.value
        result = self.preflight_manager.run(self.drones, self.real_flight_confirm, self.simulation)
        if result.front_ranger is not None and self.ranger_monitor is not None and not self.simulation:
            self.ranger_monitor.update_front(result.front_ranger)
            self.logger.ranger(self._seq, "X_FRONT", result.front_ranger, self.state)
        if result.back_ranger is not None and self.ranger_monitor is not None and not self.simulation:
            self.ranger_monitor.update_back(result.back_ranger)
            self.logger.ranger(self._seq, "X_BACK", result.back_ranger, self.state)
        if self.ranger_monitor is not None and self.simulation:
            self.logger.ranger(self._seq, "X_FRONT", self.ranger_monitor.get_front_ranger(), self.state)
            if self.config.requires_back_ranger:
                self.logger.ranger(self._seq, "X_BACK", self.ranger_monitor.get_back_ranger(), self.state)
        for drone_result in result.drones:
            zrange_raw = drone_result.log_values.get("range.zrange", 0.0)
            state_z = drone_result.log_values.get("stateEstimate.z", 0.0)
            self.logger.zrange(self._seq, drone_result.drone_id, zrange_raw / 1000.0 if zrange_raw else math.inf, state_z, zrange_raw > 0.0)
            for check in drone_result.checks:
                self.logger.preflight(drone_result.drone_id, check.name, check.passed, check.value, check.reason)
            if drone_result.drone_id != "GLOBAL":
                self._event(_preflight_value_summary(drone_result.drone_id, drone_result.log_values))
        self.mode_pass["MODE_SENSOR_CHECK"] = result.passed
        self.logger.mode_result("MODE_SENSOR_CHECK", result.passed, result.reason or "passed")
        self.state = ObservationState.READY.value if result.passed else ObservationState.ERROR.value
        self._event("sensor check passed" if result.passed else f"sensor check FAILED: {result.reason}")

    def start_ai_streams(self) -> None:
        self._require_config()
        assert self.config is not None and self.streams is not None and self.logger is not None
        self.mode = MissionMode.STREAM_CHECK
        for event in self.streams.start_streams(self.config.drones):
            self.logger.ai_stream(event)
        self.state = ObservationState.STREAMING.value
        self._event("AI streams marked active")

    def stop_ai_streams(self) -> None:
        if self.streams is None or self.logger is None:
            return
        for event in self.streams.stop_streams():
            self.logger.ai_stream(event)
        self._event("AI streams stopped")

    def takeoff_all(self) -> None:
        self._require_connected()
        assert self.config is not None
        self._clear_soft_stop_for_new_motion()
        if not self.mode_pass.get("MODE_SENSOR_CHECK", False):
            self._event("takeoff blocked: run Sensor Check first")
            return
        if self.emergency_manager.motion_interrupted():
            self._event("takeoff blocked: emergency/soft stop active")
            return
        self.mode = MissionMode.FORMATION_SETUP
        self.state = ObservationState.TAKEOFF.value
        if self._staggered_takeoff_all(self.config.flight_height):
            self.state = ObservationState.READY.value
            self._event("takeoff all finished")
        else:
            self._event("takeoff all interrupted")

    def takeoff_drone(self, drone_id: str) -> None:
        self._require_connected()
        assert self.config is not None
        drone = self.drones.get(drone_id)
        if drone is None:
            self._event(f"takeoff {drone_id} blocked: drone is not connected")
            return
        if not self.mode_pass.get("MODE_SENSOR_CHECK", False):
            self._event(f"takeoff {drone_id} blocked: run Sensor Check first")
            return
        if self.emergency_manager.motion_interrupted():
            self._event(f"takeoff {drone_id} blocked: emergency/soft stop active")
            return
        self.mode = MissionMode.MANUAL_TEST
        self.state = ObservationState.TAKEOFF.value
        self._event(f"single-drone takeoff started: {drone_id}")
        drone.takeoff(self.config.test_takeoff_height, self.config.takeoff_velocity)
        drone.hover(5.0)
        drone.land(self.config.landing_velocity)
        self.mode = MissionMode.SINGLE_DRONE_HOVER
        self.mode_pass["MODE_SINGLE_DRONE_HOVER"] = True
        self._airborne = False
        if self.logger is not None:
            self.logger.mode_result("MODE_SINGLE_DRONE_HOVER", True, f"{drone_id} hover-land passed")
        self._event(f"single-drone hover-land finished: {drone_id}")

    def run_scout_sweep(self) -> None:
        self._require_connected()
        assert self.config is not None and self.ranger_monitor is not None and self.logger is not None
        if not self.mode_pass.get("MODE_SENSOR_CHECK", False):
            self._event("scout sweep blocked: run Sensor Check first")
            return
        x_front = self.drones.get("X_FRONT")
        if x_front is None:
            self._event("scout sweep blocked: X_FRONT not connected")
            return
        self.mode = MissionMode.SCOUT_SWEEP
        self.state = "SCOUT_SWEEP"
        self._event("scout sweep started")
        x_front.takeoff(self.config.test_takeoff_height, self.config.takeoff_velocity)
        x_front.hover(self.config.hover_time)
        ok = self._scout_sweep_in_place(self.config.initial_step, self.config.initial_speed)
        x_front.land(self.config.landing_velocity)
        self._airborne = False
        self._event("scout sweep finished" if ok else "scout sweep interrupted")
        self.mode_pass["MODE_SINGLE_RANGER_SWEEP"] = ok
        self.logger.mode_result("MODE_SINGLE_RANGER_SWEEP", ok, "front Ranger sweep with 0.05m left/right motion")

    def run_wall_obstacle_probe_test(self) -> None:
        self._require_connected()
        assert self.config is not None and self.logger is not None
        if not self.mode_pass.get("MODE_SENSOR_CHECK", False):
            self._event("wall/obstacle probe blocked: run Sensor Check first")
            return
        x_front = self.drones.get("X_FRONT")
        if x_front is None:
            self._event("wall/obstacle probe blocked: X_FRONT not connected")
            return
        self.mode = MissionMode.WALL_OBSTACLE_PROBE_TEST
        self.state = "MODE_WALL_OBSTACLE_PROBE_TEST"
        self._event("wall/obstacle probe test started")
        x_front.takeoff(self.config.test_takeoff_height, self.config.takeoff_velocity)
        x_front.hover(self.config.hover_time)
        classification, _direction, _shift = self._lateral_probe(copy_followers=False)
        x_front.land(self.config.landing_velocity)
        self._airborne = False
        ok = classification.classification in {"OBSTACLE_OR_OPENING", "WALL_OR_BOUNDARY", "AMBIGUOUS"}
        self.mode_pass["MODE_WALL_OBSTACLE_PROBE_TEST"] = ok
        self.logger.mode_result("MODE_WALL_OBSTACLE_PROBE_TEST", ok, classification.classification)
        self._event(f"wall/obstacle probe test finished: {classification.classification}")

    def run_formation_hover_only(self) -> None:
        self._require_connected()
        if not self.mode_pass.get("MODE_SENSOR_CHECK", False):
            self._event("formation hover blocked: run Sensor Check first")
            return
        self.mode = MissionMode.FORMATION_HOVER_ONLY
        self.state = "MODE_FORMATION_HOVER_ONLY"
        self._event("half-swarm hover-only started")
        if not self._staggered_takeoff_all(self.config.test_takeoff_height):
            self.mode_pass["MODE_FORMATION_HOVER_ONLY"] = False
            self.mode_pass["MODE_HALF_SWARM_HOVER"] = False
            self._event("half-swarm hover-only interrupted during takeoff")
            return
        self._motion_controller().hover_all(2.0)
        self.land_all()
        self._airborne = False
        self.mode_pass["MODE_FORMATION_HOVER_ONLY"] = True
        self.mode_pass["MODE_HALF_SWARM_HOVER"] = True
        if self.logger is not None:
            self.logger.mode_result("MODE_FORMATION_HOVER_ONLY", True, "staged hover-only passed")
            self.logger.mode_result("MODE_HALF_SWARM_HOVER", True, "staged half-swarm hover-only passed")
        self._event("formation hover-only passed")

    def run_formation_micro_step(self) -> None:
        self._require_connected()
        if not self.mode_pass.get("MODE_FORMATION_HOVER_ONLY", False):
            self._event("micro-step blocked: pass Formation Hover Only first")
            return
        self.mode = MissionMode.FORMATION_MICRO_STEP
        self.state = "MODE_FORMATION_MICRO_STEP"
        self._event("formation micro-step started")
        if not self._staggered_takeoff_all(self.config.test_takeoff_height):
            self.mode_pass["MODE_FORMATION_MICRO_STEP"] = False
            self._event("formation micro-step interrupted during takeoff")
            return
        self._seq += 1
        ok = self._motion_controller().formation_forward(self._seq, self.config.initial_step, speed=self.config.initial_speed)
        self.land_all()
        self.mode_pass["MODE_FORMATION_MICRO_STEP"] = ok
        self.mode_pass["MODE_HALF_SWARM_MICRO_STEP"] = ok
        if self.logger is not None:
            self.logger.mode_result("MODE_FORMATION_MICRO_STEP", ok, "0.05m fixed formation move at 0.05m/s")
            self.logger.mode_result("MODE_HALF_SWARM_MICRO_STEP", ok, "0.05m half-swarm move at 0.05m/s")
        self._event("formation micro-step finished" if ok else "formation micro-step failed")

    def start_full_observation_mode(self, max_steps: int | None = None) -> None:
        self._require_connected()
        assert self.config is not None
        self._clear_soft_stop_for_new_motion()
        max_steps = self.config.mission_max_steps if max_steps is None else max_steps
        if not self.mode_pass.get("MODE_SENSOR_CHECK", False):
            self._event("auto sensor check before full observation")
            self.run_sensor_check()
        if not self.mode_pass.get("MODE_SENSOR_CHECK", False):
            reason = "Sensor Check failed; fix front Ranger/connection before motion"
            self._log_decision("START_FULL_OBSERVATION_BLOCKED", reason, "NO_MOTION")
            self._event(f"start full observation blocked: {reason}")
            return
        if not self._ranger_valid():
            reason = "front Ranger readings are not valid after Sensor Check"
            self._log_decision("START_FULL_OBSERVATION_BLOCKED", reason, "NO_MOTION")
            self._event(f"start full observation blocked: {reason}")
            return
        if not self.can_start_observation()[0]:
            reason = self.can_start_observation()[1]
            self._log_decision("START_FULL_OBSERVATION_BLOCKED", reason, "NO_MOTION")
            self._event(f"start full observation blocked: {reason}")
            return
        if not self._airborne:
            self.takeoff_all()
        if not self._airborne:
            self._log_decision("START_FULL_OBSERVATION_STOPPED", "takeoff did not finish", "NO_MOTION")
            self._event("start full observation stopped: takeoff did not finish")
            return
        if self.streams is not None and not self.streams.active:
            self.start_ai_streams()
        self.mode = MissionMode.FULL_OBSERVATION
        self.state = ObservationState.OBSERVE_STEP_CHECK.value
        self._mission_started_at = time.monotonic()
        self._auto_land_reason = ""
        self._post_reslot_sync_next = False
        self._log_decision("FULL_OBSERVATION_START", "prechecks passed", "BEGIN_LOOP")
        self._event("full observation mode started")
        self._event(
            "VIDEO DEMO PROFILE: staged takeoff, yaw-settle, "
            f"Ranger-first step={self.config.step_size:.2f}m, speed={self.config.speed:.2f}m/s, "
            f"hover settle={self.config.hover_time:.2f}s, turn settle={self.config.hover_after_turn:.2f}s"
        )
        if self._airborne:
            settle = max(0.30, min(self.config.demo_settle_time, 1.20))
            self._event(f"pre-motion settle: hover all {settle:.2f}s before first observation step")
            self._motion_controller().hover_all(settle)
        if self.formation is not None:
            yaw_targets = self.formation.intended_ai_yaws()
            yaw_text = ", ".join(f"{drone_id}={yaw:.0f}deg" for drone_id, yaw in sorted(yaw_targets.items()))
            right_path = " -> ".join(
                f"({point.x:.0f},{point.y:.0f})"
                for point in self.formation.reslot_waypoints("RIGHT").get("X_FRONT", [])
            )
            self._event(
                "MISSION PLAN: staged observation = X_FRONT ranger scouts "
                f"{self.config.step_size:.2f}m first at {self.config.speed:.2f}m/s, "
                f"then O1/O2 copy only after the Ranger check passes; "
                f"setpoint segment={self.config.motion_segment_s:.2f}s"
            )
            self._event(
                "MISSION PLAN: wall/obstacle ahead = O1/O2 hover; X_FRONT re-slots to the new front "
                f"(right-turn example path {right_path}); after the re-slot all drones yaw and continue "
                "moving in the new heading"
            )
            if yaw_text:
                self._event(f"MISSION PLAN: AI yaw targets {yaw_text}; body-frame movement is compensated")
        budget = self._observation_budget(max_steps)
        forward_steps_done = 0
        decision_cycles = 0
        self._event(
            "MISSION LOOP: continue in the current heading until Ranger detects the next obstacle; "
            f"turns/re-slots do not consume the forward-step budget ({budget.label})"
        )
        while budget.continuous or forward_steps_done < budget.forward_steps:
            if self.emergency or self._paused:
                self._log_decision("LOOP_STOP", "emergency or paused", "BREAK")
                break
            decision_cycles += 1
            if budget.max_decision_cycles is not None and decision_cycles > budget.max_decision_cycles:
                self.state = "FRONTIER_SAVE"
                self._log_decision(
                    "MISSION_DECISION_GUARD",
                    "too many turn/check cycles without enough forward progress",
                    "SAFE_HOVER",
                )
                self._event(
                    "mission paused: too many safety/turn decisions without forward progress; "
                    "check obstacles and Ranger readings"
                )
                break
            budget_reason = self._mission_stop_reason()
            if budget_reason is not None:
                self._auto_land_reason = budget_reason
                self._log_decision("AUTO_SAFE_LAND", budget_reason, "SAFE_HOVER_LAND")
                self._event(f"auto safe land: {budget_reason}")
                self.safe_hover_land()
                break
            self._seq += 1
            self._last_forward_progress = False
            self._refresh_ranger_readings()
            self.state = ObservationState.OBSERVE_STEP_CHECK.value
            envelope = self._evaluate_envelope()
            self._log_decision(
                "ENVELOPE_CHECK",
                envelope.reason,
                "EVALUATE_SURFACE_OR_MOVE" if envelope.state == EnvelopeState.FREE.value else envelope.state,
                envelope_state=envelope.state,
                envelope_reason=envelope.reason,
            )
            self._event(f"envelope {envelope.state}: {envelope.reason}")
            if envelope.state == EnvelopeState.FREE.value:
                if self._surface_candidate():
                    self.state = "SURFACE_CANDIDATE"
                    self._log_decision("SURFACE_CANDIDATE", "front distance near target/wall threshold", "HANDLE_SURFACE")
                    self._event("surface candidate detected at target distance")
                    self._handle_surface_candidate()
                    if self._surface_response_should_stop():
                        break
                else:
                    self._log_decision("FORWARD_CLEAR", envelope.reason, "MOVE_BASE_FORMATION_FORWARD")
                    if not self._move_base_formation_forward():
                        break
                    if self._last_forward_progress:
                        forward_steps_done += 1
                        self._event(
                            "observation progress: "
                            f"forward step {forward_steps_done}/{budget.label} completed; "
                            f"continuing heading {self.formation.heading_deg:.0f}deg until next Ranger obstacle"
                        )
                time.sleep(0.05)
                continue
            if envelope.state == EnvelopeState.UNKNOWN.value:
                self.state = "FRONTIER_SAVE"
                self._log_decision("UNKNOWN_SPACE", envelope.reason, "FRONTIER_SAVE")
                self._event("frontier saved: unknown space blocks fixed formation")
                break
            if envelope.state == EnvelopeState.OCCUPIED.value:
                self.state = "SURFACE_CANDIDATE"
                self._log_decision("OCCUPIED_ENVELOPE", envelope.reason, "HANDLE_SURFACE")
                self._event("surface candidate detected from occupied envelope")
                self._handle_surface_candidate()
                if self._surface_response_should_stop():
                    break
                continue
            if envelope.state == EnvelopeState.CRITICAL.value:
                if self._front_blocked_reason(envelope.reason):
                    self._log_decision("CRITICAL_FRONT_ENVELOPE", envelope.reason, "FRONT_RECOVERY_THEN_TURN")
                    self._event(f"front critical: {envelope.reason}; AI drones hold, X_FRONT creates safe distance before deciding")
                    self._recover_front_personal_space(envelope.reason)
                    self._handle_surface_candidate()
                    if self._surface_response_should_stop():
                        break
                    continue
                self._log_decision("CRITICAL_ENVELOPE", envelope.reason, "EMERGENCY_HOVER")
                self.emergency_hover()
                break
        if not self.emergency and self.state != ObservationState.SAFE_HOVER.value:
            self.mode_pass["MODE_HALF_SWARM_OBSERVATION_SAFE"] = True
            self.mode = MissionMode.MISSION_COMPLETE
            self.state = ObservationState.MISSION_COMPLETE.value
            self._log_decision("MISSION_COMPLETE", "loop ended without emergency", "STOP")
            self._event(f"mission complete: forward_steps={forward_steps_done}/{budget.label}")

    def start_observation_demo(self) -> None:
        """Run the visible top-3 observation path from the GUI.

        Simulation mode is intentionally forgiving so the GUI can demonstrate
        the behavior without real hardware: it auto-loads config, connects,
        runs the sensor check, starts passive AI streams, takes off, and runs a
        short observation. Real mode keeps the staged safety gates from
        ``start_full_observation_mode``.
        """

        if self.simulation:
            if self.config is None:
                self.load_half_group_config()
            if not self._connected:
                self.connect_all()
            if not self.mode_pass.get("MODE_SENSOR_CHECK", False):
                self.run_sensor_check()
            if self.streams is not None and not self.streams.active:
                self.start_ai_streams()
            self._event(
                f"video observation demo started scenario={self.scenario}: "
                f"{self.config.video_demo_steps} successful forward group moves; "
                "turns/re-slots continue into the new heading"
            )
            self.start_full_observation_mode(max_steps=self.config.video_demo_steps)
            return
        self._event(
            "video observation demo requested in real mode: "
            f"{self.config.video_demo_steps} successful forward group moves; "
            "turns/re-slots continue into the new heading"
        )
        self.start_full_observation_mode(max_steps=self.config.video_demo_steps)

    def pause_hover(self) -> None:
        self._paused = True
        self.mode = MissionMode.PAUSE_HOVER
        self.state = ObservationState.SAFE_HOVER.value
        for drone in self.drones.values():
            drone.hover(0.2)
        self._event("pause hover")

    def resume(self) -> None:
        self._paused = False
        self.mode = MissionMode.FULL_OBSERVATION
        self._event("resume requested")

    def manual_swarm_forward(self) -> None:
        self._manual_swarm_move("FORWARD")

    def manual_swarm_back(self) -> None:
        self._manual_swarm_move("BACK")

    def manual_swarm_left(self) -> None:
        self._manual_swarm_move("LEFT")

    def manual_swarm_right(self) -> None:
        self._manual_swarm_move("RIGHT")

    def manual_swarm_yaw_left(self) -> None:
        self._manual_swarm_yaw("LEFT")

    def manual_swarm_yaw_right(self) -> None:
        self._manual_swarm_yaw("RIGHT")

    def land_all(self) -> None:
        if not self.drones:
            return
        self.mode = MissionMode.LANDING
        self.state = ObservationState.LANDING.value
        for drone in self.drones.values():
            drone.land(self.config.landing_velocity)
        self._airborne = False
        self._event("land all finished")

    def emergency_hover(self) -> None:
        self.safe_hover_land(mark_emergency=True)

    def safe_hover_land(self, mark_emergency: bool = False) -> None:
        self.emergency = mark_emergency
        self.mode = MissionMode.EMERGENCY if mark_emergency else MissionMode.LANDING
        self.state = ObservationState.SAFE_HOVER.value
        reason = "emergency_hover" if mark_emergency else "safe hover/land request"
        self._log_decision("SAFE_HOVER_LAND", reason, "SAFE_HOVER_LAND_ALL")
        result = self.emergency_manager.safe_hover_land_all(self.drones)
        if self.logger is not None:
            self.logger.emergency(
                result.action,
                result.target_count,
                result.elapsed_to_first_stop_s,
                result.elapsed_total_s,
                "GUI emergency hover land" if mark_emergency else "GUI safe hover land",
                button_pressed_ts=result.button_pressed_ts,
                event_set_ts=result.event_set_ts,
                first_stop_ts=result.first_stop_ts,
                last_normal_setpoint_ts=self._last_normal_setpoint_ts or None,
            )
        self._airborne = False
        self._event(f"SAFE HOVER/LAND sent to {result.target_count} drones in {result.elapsed_total_s*1000:.1f} ms")

    def hard_motor_kill(self) -> None:
        self.emergency = True
        self.mode = MissionMode.EMERGENCY
        self.state = ObservationState.HARD_MOTOR_KILL.value
        self._log_decision("HARD_MOTOR_KILL", "user emergency stop", "KILL_ALL")
        result = self.emergency_manager.hard_kill_all(self.drones)
        if self.logger is not None:
            self.logger.emergency(
                result.action,
                result.target_count,
                result.elapsed_to_first_stop_s,
                result.elapsed_total_s,
                "GUI hard motor kill",
                button_pressed_ts=result.button_pressed_ts,
                event_set_ts=result.event_set_ts,
                first_stop_ts=result.first_stop_ts,
                last_normal_setpoint_ts=self._last_normal_setpoint_ts or None,
            )
        self._event(f"HARD MOTOR KILL sent to {result.target_count} drones in {result.elapsed_total_s*1000:.1f} ms")

    def save_logs(self) -> str:
        if self.logger is None:
            return ""
        self._event(f"logs saved: {self.logger.run_dir}")
        return str(self.logger.run_dir)

    def close(self) -> None:
        if self.logger is not None:
            self._write_run_summary()
            self.logger.close()
        self.emergency_manager.stop_supervisor_watchdog()
        for drone in self.drones.values():
            close = getattr(drone, "close", None)
            if close is not None:
                close()

    def get_status_snapshot(self) -> SwarmStatusSnapshot:
        config_loaded = self.config is not None and self.formation is not None
        can_start, disabled_reason = self.can_start_observation()
        mission = MissionStatus(
            mode=self.mode.value,
            state=self.state,
            emergency=self.emergency,
            battery_summary=self._battery_summary(),
            radio_status=self._radio_status(),
            formation_config_loaded=config_loaded,
            ranger_valid=self._ranger_valid(),
            ai_stream_manager_initialized=self.streams is not None,
            start_observation_enabled=can_start,
            disabled_reason=disabled_reason,
            emergency_manager_active=self.emergency_manager.active,
            hard_kill_armed=self.emergency_manager.hard_kill_armed,
            watchdog_status=self._watchdog_status(),
            last_setpoint_age_s=self._last_setpoint_age_s(),
            preflight_passed=self.mode_pass.get("MODE_SENSOR_CHECK", False),
            heading_deg=self.formation.heading_deg if self.formation is not None else 0.0,
            log_dir=str(self.logger.run_dir) if self.logger is not None else "",
            mission_elapsed_s=self._mission_elapsed_s(),
            mission_remaining_s=self._mission_remaining_s(),
            auto_land_reason=self._auto_land_reason,
            chunk_progress=self._chunk_progress,
        )
        slots = {}
        ai_yaws = {}
        if self.formation is not None:
            slots = {key: (slot.x, slot.y) for key, slot in self.formation.rotated_offsets().items()}
            ai_yaws = self.formation.intended_ai_yaws()
        return SwarmStatusSnapshot(
            mission=mission,
            drones=self._drone_statuses(),
            rangers=self._ranger_statuses(),
            ai_streams=self._ai_stream_statuses(),
            envelope=self.last_envelope,
            events=list(self.events[-200:]),
            formation_slots=slots,
            ai_yaws=ai_yaws,
        )

    def can_start_observation(self) -> tuple[bool, str]:
        if self.config is None or self.formation is None:
            return False, "formation config not loaded"
        if "X_FRONT" not in self.drones:
            return False, "X_FRONT not connected"
        if self.config.requires_back_ranger and "X_BACK" not in self.drones:
            return False, "X_BACK not connected"
        if self.streams is None:
            return False, "AI stream manager not initialized"
        if self.mode_pass.get("MODE_SENSOR_CHECK", False) and not self._ranger_valid():
            return False, "Ranger readings are not valid"
        if not self.simulation:
            if not self.real_flight_confirm:
                return False, "real flight confirmation is disabled"
            if not self.emergency_manager.active:
                return False, "emergency manager inactive"
        return True, ""

    def _evaluate_envelope(self) -> SafetyEnvelopeStatus:
        assert self.config is not None and self.formation is not None and self.ranger_monitor is not None
        result = evaluate_formation_envelope(
            self.config,
            self.formation,
            self.ranger_monitor.get_front_ranger(),
            self.ranger_monitor.get_back_ranger(),
            PlannedPrimitive("FORMATION_FORWARD", self.config.step_size),
        )
        self.last_envelope = SafetyEnvelopeStatus(
            state=result.state.value,
            reason=result.reason,
            front_required=result.required.get("front", 0.0),
            side_required=result.required.get("left", 0.0),
            back_required=result.required.get("back", 0.0),
        )
        return self.last_envelope

    def _surface_candidate(self) -> bool:
        assert self.config is not None and self.formation is not None and self.ranger_monitor is not None
        return detect_surface_candidate(self.config, self.formation, self.ranger_monitor.get_front_ranger())

    @staticmethod
    def _observation_budget(max_steps: int) -> ObservationLoopBudget:
        continuous = max_steps <= 0
        return ObservationLoopBudget(
            forward_steps=max_steps,
            continuous=continuous,
            max_decision_cycles=None if continuous else max(20, max_steps * 8 + 8),
            label="continuous" if continuous else str(max_steps),
        )

    def _surface_response_should_stop(self) -> bool:
        return self.state in {"FRONTIER_SAVE", ObservationState.SAFE_HOVER.value}

    def _can_continue_after_surface_response(self) -> bool:
        return not self.emergency and self.state not in {
            "FRONTIER_SAVE",
            ObservationState.SAFE_HOVER.value,
            ObservationState.EMERGENCY_STOP.value,
            ObservationState.HARD_MOTOR_KILL.value,
        }

    def _move_base_formation_forward(self) -> bool:
        assert self.config is not None
        if self._post_reslot_sync_next:
            return self._move_post_reslot_formation_forward()
        self.state = ObservationState.GROUP_FORWARD_STEP.value
        self._log_decision(
            "GROUP_FORWARD_STEP",
            f"forward envelope clear; X_FRONT scouts first, then AI drones copy; step={self.config.step_size:.2f}m speed={self.config.speed:.2f}m/s",
            "COMMAND_X_FRONT_THEN_AI_COPY",
        )
        motion = self._motion_controller()
        ai_ids = [drone_id for drone_id in ("O1", "O2", "O3", "O4") if drone_id in self.drones]
        self._event(f"step {self._seq}: Ranger big step X_FRONT forward {self.config.step_size:.2f}m")
        moved = motion.selected_forward(
            self._seq,
            "X_FRONT_SCOUT_FORWARD",
            ["X_FRONT"],
            self.config.step_size,
            speed=self.config.speed,
        )
        self._refresh_ranger_readings()
        if not moved:
            reason = self._emergency_reason() or "motion command failed"
            if self._front_blocked_reason(reason):
                return self._recover_and_turn_after_front_block(
                    motion,
                    "X_FRONT_SCOUT_FORWARD_BLOCKED_BY_RANGER",
                    reason,
                    f"step {self._seq}: X_FRONT scout blocked by Ranger front ({reason}); "
                    "AI drones hold, creating safe distance then choosing turn from Ranger left/right data",
                )
            self._mark_motion_interrupted("X_FRONT_SCOUT_FORWARD")
            return False
        self._event(f"step {self._seq}: X_FRONT scout DONE; checking Ranger before AI copy")
        reason = self._emergency_reason()
        if reason is not None:
            self._log_decision("AI_COPY_BLOCKED_AFTER_SCOUT", reason, "SAFE_HOVER")
            self._event(f"AI copy blocked after X_FRONT scout: {reason}")
            self.state = ObservationState.SAFE_HOVER.value
            return False
        if ai_ids:
            self._event(f"step {self._seq}: AI big step copy forward {self.config.step_size:.2f}m ({','.join(ai_ids)})")
            moved = motion.selected_forward(
                self._seq,
                "AI_COPY_FORWARD",
                ai_ids,
                self.config.step_size,
                speed=self.config.speed,
            )
            self._refresh_ranger_readings()
            if not moved:
                self._mark_motion_interrupted("AI_COPY_FORWARD")
                return False
        self.state = ObservationState.OBSERVE_STEP_CHECK.value
        self._log_decision("GROUP_FORWARD_DONE", "X_FRONT scout and AI copy completed", "NEXT_RANGER_CHECK")
        self._event(f"base formation staged forward DONE {self.config.step_size:.2f}m")
        self._last_forward_progress = True
        return True

    def _move_post_reslot_formation_forward(self) -> bool:
        """Move all active drones together after a Ranger re-slot turn.

        Straight exploration uses a staged pattern where X_FRONT proves one
        step first and the AI drones copy it. Immediately after a turn, however,
        X_FRONT has already moved alone into the new front slot. The next
        visible motion must be synchronized so the AI drones do not look stuck
        while the Ranger starts scouting again.
        """

        assert self.config is not None
        self._post_reslot_sync_next = False
        self.state = ObservationState.GROUP_FORWARD_STEP.value
        drone_ids = ",".join(self.drones)
        self._log_decision(
            "POST_RESLOT_GROUP_FORWARD_STEP",
            (
                "Ranger re-slot already proved the new direction; "
                f"all drones move together one step={self.config.step_size:.2f}m"
            ),
            "COMMAND_SYNCHRONIZED_FORMATION_FORWARD",
        )
        self._event(
            f"step {self._seq}: post-turn synchronized formation forward "
            f"{self.config.step_size:.2f}m ({drone_ids})"
        )
        moved = self._motion_controller().formation_forward(
            self._seq,
            self.config.step_size,
            speed=self.config.speed,
        )
        self._refresh_ranger_readings()
        if not moved:
            reason = self._emergency_reason() or "motion command failed"
            if self._front_blocked_reason(reason):
                return self._recover_and_turn_after_front_block(
                    self._motion_controller(),
                    "POST_RESLOT_FORWARD_BLOCKED_BY_RANGER",
                    reason,
                    f"step {self._seq}: post-turn synchronized move blocked by Ranger front ({reason}); "
                    "AI drones hold, X_FRONT creates safe distance and chooses the next turn",
                )
            self._mark_motion_interrupted("POST_RESLOT_FORMATION_FORWARD")
            return False
        self.state = ObservationState.OBSERVE_STEP_CHECK.value
        self._log_decision(
            "POST_RESLOT_GROUP_FORWARD_DONE",
            "all drones moved together after turn and yaw realignment",
            "NEXT_RANGER_CHECK",
        )
        self._event(f"post-turn synchronized formation forward DONE {self.config.step_size:.2f}m")
        self._last_forward_progress = True
        return True

    def _recover_and_turn_after_front_block(
        self,
        motion: MotionController,
        decision: str,
        reason: str,
        event_message: str,
    ) -> bool:
        self._log_decision(decision, reason, "AI_HOVER_FRONT_RECOVERY_THEN_TURN")
        self._event(event_message)
        motion.hover_all(0.20)
        self._recover_front_personal_space(reason)
        self._handle_surface_candidate()
        return self._can_continue_after_surface_response()

    def _mark_motion_interrupted(self, command: str) -> None:
        reason = self._emergency_reason() or "motion command failed"
        next_action = "EMERGENCY_STOP" if self.emergency_manager.emergency_event.is_set() or self.emergency_manager.killed else "SAFE_HOVER"
        self._log_decision(f"{command}_INTERRUPTED", reason, next_action)
        self._event(f"{command} INTERRUPTED: {reason}")
        if next_action == "EMERGENCY_STOP":
            self.emergency = True
            self.mode = MissionMode.EMERGENCY
            self.state = ObservationState.EMERGENCY_STOP.value
        else:
            self.state = ObservationState.SAFE_HOVER.value

    @staticmethod
    def _front_blocked_reason(reason: str) -> bool:
        return reason.startswith("X_FRONT_FRONT_CRITICAL_")

    def _recover_front_personal_space(self, reason: str) -> bool:
        """Move X_FRONT backward a little when a person/object is too close ahead.

        This is a formation recovery, not a turn. When the front Ranger only
        needs to create personal space, followers copy the backward primitive so
        the group stays readable on video. The later re-slot turn is still
        Ranger-only: AI drones hover while X_FRONT moves around to the new front
        slot.
        """

        assert self.config is not None and self.ranger_monitor is not None and self.logger is not None
        if not self._front_blocked_reason(reason):
            return False

        motion = self._front_recovery_motion_controller()
        moved_total = 0.0
        self.state = ObservationState.BLOCKED_HOVER.value
        ai_ids = [drone_id for drone_id in ("O1", "O2", "O3", "O4") if drone_id in self.drones]
        copy_targets = ["X_FRONT"]
        if self.config.copy_front_recovery_to_followers:
            copy_targets.extend(ai_ids)
        target_text = ",".join(copy_targets)
        self._event(
            f"front recovery started: target={self.config.front_recovery_target:.2f}m "
            f"max_backoff={self.config.front_recovery_max:.2f}m; "
            f"backoff targets={target_text}"
        )

        while moved_total < self.config.front_recovery_max - 1e-6:
            self._refresh_ranger_readings()
            reading = self.ranger_monitor.get_front_ranger()
            if reading.front >= self.config.front_recovery_target:
                self._event(f"front recovery already safe: front={_fmt(reading.front)}m")
                break

            remaining = self.config.front_recovery_max - moved_total
            needed = self.config.front_recovery_target - reading.front if math.isfinite(reading.front) else self.config.front_recovery_step
            step = min(self.config.front_recovery_step, remaining, max(0.0, needed))
            if step < 0.02:
                break

            required_back = step + self.config.critical_back
            back_valid = reading.valid.get("back", False)
            if not back_valid:
                self._log_decision("FRONT_RECOVERY_BLOCKED", "X_FRONT back Ranger unknown", "TRY_TURN_WITHOUT_BACKOFF")
                self._event("front recovery skipped: X_FRONT back Ranger unknown; trying side re-slot instead")
                self._allow_unknown_back_reslot_escape = True
                break
            if math.isfinite(reading.back) and reading.back < required_back:
                self._log_decision(
                    "FRONT_RECOVERY_BLOCKED",
                    f"back={reading.back:.2f}m < required {required_back:.2f}m",
                    "TRY_TURN_WITHOUT_BACKOFF",
                )
                self._event(
                    f"front recovery skipped: back clearance {_fmt(reading.back)}m "
                    f"< required {required_back:.2f}m; trying side re-slot instead"
                )
                break

            if len(copy_targets) > 1:
                motion.hover_all(0.05)
            self._event(
                f"front recovery backoff: {target_text} backward {step:.2f}m "
                f"(front={_fmt(reading.front)}m back={_fmt(reading.back)}m)"
            )
            moved = motion.selected_backward(
                self._seq,
                "FORMATION_FRONT_RECOVERY_BACKOFF" if len(copy_targets) > 1 else "X_FRONT_FRONT_RECOVERY_BACKOFF",
                copy_targets,
                step,
                speed=self.config.front_recovery_speed,
            )
            if not moved:
                self._log_decision("FRONT_RECOVERY_INTERRUPTED", self._front_recovery_emergency_reason() or "motion failed", "TRY_TURN")
                self._event("front recovery interrupted; trying Ranger side decision")
                break
            moved_total += step
            self._apply_simulated_front_recovery_after_if_needed(step)

        self._refresh_ranger_readings()
        reading = self.ranger_monitor.get_front_ranger()
        self._log_decision(
            "FRONT_RECOVERY_DONE" if moved_total > 0.0 else "FRONT_RECOVERY_NOT_MOVED",
            f"front={_fmt(reading.front)}m back={_fmt(reading.back)}m moved={moved_total:.2f}m",
            "CHOOSE_TURN_FROM_RANGER",
            ranger=reading,
            extra={"moved_m": moved_total},
        )
        self._event(
            f"front recovery finished: moved={moved_total:.2f}m "
            f"front={_fmt(reading.front)}m; now choosing side from Ranger data"
        )
        return moved_total > 0.0

    def _manual_swarm_move(self, direction: str) -> None:
        self._require_connected()
        assert self.config is not None
        if not self._airborne:
            self._event(f"manual {direction} blocked: Takeoff All first")
            return
        self._seq += 1
        self.state = f"MANUAL_{direction}"
        distance = min(self.config.step_size, self.config.initial_step)
        speed = min(self.config.speed, self.config.initial_speed)
        motion = self._motion_controller()
        if direction == "FORWARD":
            moved = motion.formation_forward(self._seq, distance, speed=speed)
        elif direction == "BACK":
            moved = motion.formation_backward(self._seq, distance, speed=speed)
        else:
            moved = motion.formation_sidestep(self._seq, direction, distance, speed=speed)
        self._refresh_ranger_readings()
        self._event(f"manual swarm {direction} {'DONE' if moved else 'BLOCKED'} {distance:.2f}m")

    def _manual_swarm_yaw(self, direction: str) -> None:
        self._require_connected()
        assert self.config is not None
        if not self._airborne:
            self._event(f"manual yaw {direction} blocked: Takeoff All first")
            return
        self._seq += 1
        self.state = f"MANUAL_YAW_{direction}"
        moved = self._motion_controller().formation_yaw(self._seq, direction, 15.0)
        self._refresh_ranger_readings()
        self._event(f"manual swarm yaw {direction} {'DONE' if moved else 'BLOCKED'} 15deg")

    def _handle_surface_candidate(self) -> None:
        """Deterministic half-swarm wall/obstacle response.

        This is intentionally simple and visible in the lab:

        1. The full group moves forward only while X_FRONT says the forward
           envelope is clear.
        2. If X_FRONT sees a wall/obstacle/blocked forward envelope, O1 and O2
           do not copy any probe motion. They keep hovering in their current
           places.
        3. The controller chooses LEFT or RIGHT from the X_FRONT side ranger
           readings. The side with more measured clearance wins.
        4. Only X_FRONT moves into the new front slot. For a RIGHT turn from the
           starting heading this is exactly:
              X_FRONT (32,0) -> (97,0) -> (97,45)
           At 0.01 m/unit this is a 0.65 m side leg and a 0.45 m back leg.
        5. After X_FRONT reaches the new front slot, the formation heading is
           updated. O1/O2 yaw targets are updated for streaming direction.
           The next loop performs one synchronized full-group step before
           returning to the normal X_FRONT-scout-then-AI-copy rhythm.

        This replaces the previous lateral-probe/bypass behavior for normal
        observation because that behavior looked chaotic with the 3-drone half
        group and no shared x/y localization.
        """

        assert self.config is not None and self.formation is not None and self.ranger_monitor is not None
        reading = self.ranger_monitor.get_front_ranger()
        direction = self._choose_turn_direction(reading)
        if direction is None:
            self.state = ObservationState.SAFE_HOVER.value
            self.mode = MissionMode.PAUSE_HOVER
            self._log_decision(
                "TURN_SIDE_REJECTED",
                "neither side has safe clearance",
                "SAFE_HOVER",
                ranger=reading,
            )
            self._motion_controller().hover_all(0.25)
            self._event(
                "front blocked -> SAFE_HOVER: neither side has safe clearance; "
                f"ranger f/l/r/u={_fmt(reading.front)}/{_fmt(reading.left)}/"
                f"{_fmt(reading.right)}/{_fmt(reading.up)}"
            )
            return
        self.state = ObservationState.BLOCKED_HOVER.value
        self._log_decision(
            "TURN_SIDE_SELECTED",
            f"{direction} has better/safe side clearance",
            "AI_HOVER_X_FRONT_RESLOT",
            ranger=reading,
            extra={"direction": direction},
        )
        self._event(
            "front blocked -> AI hover, Ranger re-slot turn "
            f"{direction}; ranger f/l/r/u={_fmt(reading.front)}/{_fmt(reading.left)}/"
            f"{_fmt(reading.right)}/{_fmt(reading.up)}"
        )
        self.state = ObservationState.BLOCKED_HOVER.value
        self._motion_controller().hover_all(0.10)
        self.state = ObservationState.CHOOSE_TURN_SIDE.value
        self._turn_reslot(direction)
        if self.state not in {"FRONTIER_SAVE", ObservationState.SAFE_HOVER.value}:
            self.state = ObservationState.OBSERVE_STEP_CHECK.value

    def _lateral_probe(self, copy_followers: bool = True):
        assert self.config is not None and self.formation is not None and self.ranger_monitor is not None and self.logger is not None
        self.state = "LATERAL_PROBE"
        initial_reading = self.ranger_monitor.get_front_ranger()
        initial = self.formation.corrected_front_wall_distance(initial_reading.front)
        direction = "LEFT" if initial_reading.left >= initial_reading.right else "RIGHT"
        shift = 0.0
        after = initial
        mode = "group-copy" if copy_followers else "ranger-only"
        motion = self._motion_controller()
        self._event(f"lateral probe started mode={mode} direction={direction} corrected_front={_fmt(initial)}")
        while shift < self.config.max_probe_shift:
            if copy_followers:
                moved = motion.formation_sidestep(self._seq, direction, self.config.probe_step, speed=self.config.speed)
            else:
                moved = motion.ranger_front_probe(self._seq, direction, self.config.probe_step, speed=self.config.speed)
            if not moved:
                break
            shift += self.config.probe_step
            self._apply_simulated_probe_after_if_needed()
            self._refresh_ranger_readings()
            after_reading = self.ranger_monitor.get_front_ranger()
            after = self.formation.corrected_front_wall_distance(after_reading.front)
            self._event(f"lateral probe sample shift={shift:.2f}m corrected_front={_fmt(after)}")
            if math.isfinite(after) and math.isfinite(initial) and after > initial + self.config.clear_increase:
                break
        classification = classify_surface_probe(self.config, initial, after, shift)
        if shift > 0.0:
            return_direction = "RIGHT" if direction == "LEFT" else "LEFT"
            if copy_followers:
                motion.formation_sidestep(self._seq, return_direction, shift, speed=self.config.speed)
            else:
                motion.ranger_front_probe(self._seq, return_direction, shift, speed=self.config.speed)
            self._refresh_ranger_readings()
        self.logger.classification(
            self._seq,
            initial,
            direction,
            shift,
            after,
            classification.classification,
            classification.confidence,
            classification.next_action,
        )
        self._event(
            f"lateral probe result: {classification.classification} confidence={classification.confidence} "
            f"side={direction} shift={shift:.2f}m"
        )
        return classification, direction, shift

    def _try_bypass(self, direction: str, shift: float) -> None:
        assert self.config is not None and self.formation is not None and self.ranger_monitor is not None
        self.state = "BASE_FORMATION_BYPASS"
        self._refresh_ranger_readings()
        envelope = evaluate_bypass_envelope(
            self.config,
            self.formation,
            self.ranger_monitor.get_front_ranger(),
            self.ranger_monitor.get_back_ranger(),
            direction,
            shift,
            self.config.step_size * 2.0,
        )
        if envelope.state != EnvelopeState.FREE:
            self.state = "FRONTIER_SAVE"
            self._event(f"bypass blocked: {envelope.reason}")
            return
        moved = self._motion_controller().full_formation_bypass(self._seq, direction, shift, self.config.step_size * 2.0)
        self._refresh_ranger_readings()
        self._event("bypass finished" if moved else "bypass interrupted")

    def _turn_reslot(self, direction: str) -> None:
        assert self.config is not None and self.formation is not None and self.ranger_monitor is not None and self.logger is not None
        turn = "LEFT" if direction.upper() == "LEFT" else "RIGHT"
        self.state = ObservationState.X_FRONT_RESLOT.value
        points = self.formation.reslot_waypoints(turn).get("X_FRONT", [])
        segment_lengths = [
            (abs((end - start).x) + abs((end - start).y)) * self.formation.unit_to_meters
            for start, end in zip(points, points[1:])
        ]
        segment_text = " then ".join(f"{distance:.2f}m" for distance in segment_lengths) or "0.00m"
        waypoint_text = " -> ".join(f"({point.x:.0f},{point.y:.0f})" for point in points) or "(none)"
        self._log_decision(
            "RANGER_RESLOT_PLAN",
            f"planned X_FRONT path {segment_text}; AI drones hold",
            f"TRY_TURN_{turn}",
            extra={
                "turn": turn,
                "segment_lengths_m": segment_lengths,
                "waypoints": [(point.x, point.y) for point in points],
            },
        )
        self._event(
            f"Ranger re-slot plan {turn}: AI drones hold; X_FRONT moves formation-{turn} "
            f"path {waypoint_text} = {segment_text} to become front of new heading"
        )
        if not self._try_reslot_turn(turn):
            opposite = "RIGHT" if turn == "LEFT" else "LEFT"
            self._log_decision("RANGER_RESLOT_RETRY", f"{turn} blocked or failed", f"TRY_TURN_{opposite}")
            self._event(f"Ranger re-slot {turn} blocked; trying {opposite}")
            if not self._try_reslot_turn(opposite):
                self.state = ObservationState.SAFE_HOVER.value
                self.mode = MissionMode.PAUSE_HOVER
                self._log_decision("RANGER_RESLOT_FAILED", "both turn paths blocked or failed", "SAFE_HOVER")
                self._motion_controller().hover_all(0.25)
                self._event("SAFE_HOVER: both Ranger re-slot turn paths blocked")
                self._allow_unknown_back_reslot_escape = False
                return
        self._allow_unknown_back_reslot_escape = False
        self.state = ObservationState.UPDATE_HEADING.value
        self._log_decision("RANGER_RESLOT_DONE", "X_FRONT is at new front slot", "UPDATE_HEADING")
        self._event(f"Ranger moved to front of new heading {self.formation.heading_deg:.0f}deg")
        self.state = ObservationState.REALIGN_YAWS.value
        self._post_reslot_sync_next = True
        if self.formation is not None:
            yaws = self.formation.intended_ai_yaws()
            yaw_text = ", ".join(f"{drone_id}={yaw:.0f}deg" for drone_id, yaw in sorted(yaws.items()))
            if yaw_text:
                self._event(f"post-turn AI yaw targets applied: {yaw_text}")
        self._event("post-turn next move will be synchronized: X_FRONT and AI drones move together")
        self._apply_simulated_reslot_after_if_needed()
        self._refresh_ranger_readings()

    def _choose_turn_direction(self, reading: RangerReading) -> str | None:
        """Pick the safer turn direction from X_FRONT side ranger readings."""

        assert self.config is not None
        return choose_turn_side(RangerSnapshot.from_reading(reading), self.config.critical_side)

    def _try_reslot_turn(self, turn: str) -> bool:
        assert self.config is not None and self.formation is not None and self.ranger_monitor is not None and self.logger is not None
        self._refresh_ranger_readings()
        safety = evaluate_reslot_path_safety(
            self.config,
            self.formation,
            self.ranger_monitor.get_front_ranger(),
            self.ranger_monitor.get_back_ranger() if self.config.requires_back_ranger else None,
            turn,
            allow_unknown_front_back_leg=self._allow_unknown_back_reslot_escape,
        )
        self.logger.reslot_path(self._seq, turn, safety.state.value, safety.reason, safety.measured, safety.required)
        targets = "X_FRONT,X_BACK" if self.config.requires_back_ranger else "X_FRONT"
        self.logger.command(
            self._seq,
            self.state,
            f"TURN_{turn}_90_CHECK",
            targets,
            0.0,
            self.formation.heading_deg,
            safety.state.value,
            safety.reason,
        )
        self._log_decision(
            f"TURN_{turn}_90_CHECK",
            safety.reason,
            "EXECUTE_RANGER_RESLOT" if safety.state == EnvelopeState.FREE else "BLOCK_TURN",
            envelope_state=safety.state.value,
            envelope_reason=safety.reason,
            extra={"measured": safety.measured, "required": safety.required},
        )
        self._event(f"turn {turn} safety: {safety.state.value} {safety.reason}")
        if safety.state != EnvelopeState.FREE:
            return False
        # During re-slot the Ranger is escaping a known front blockage by
        # moving sideways/backward. The path safety check above already
        # validates the required side/back/up clearances, so the motion loop
        # must not be interrupted by the same old front obstacle that triggered
        # the turn. It still stops on user emergency, battery, side, back, and
        # ceiling critical readings.
        motion = self._front_recovery_motion_controller()
        moved = motion.ranger_reslot_turn(self._seq, turn)
        self._log_decision(
            f"TURN_{turn}_90_EXECUTE",
            "motion command completed" if moved else "motion command failed/interrupted",
            "DONE" if moved else "FAILED",
        )
        self._event(f"turn {turn} Ranger re-slot {'DONE' if moved else 'FAILED'}")
        return moved

    def _motion_controller(self) -> MotionController:
        assert self.config is not None and self.formation is not None and self.logger is not None
        return MotionController(
            self.config,
            self.formation,
            self.drones,
            self.logger,
            self._emergency_reason,
            refresh_callback=self._refresh_ranger_readings,
            progress_callback=self._set_chunk_progress,
            setpoint_callback=self._mark_normal_setpoint,
        )

    def _front_recovery_motion_controller(self) -> MotionController:
        assert self.config is not None and self.formation is not None and self.logger is not None
        return MotionController(
            self.config,
            self.formation,
            self.drones,
            self.logger,
            self._front_recovery_emergency_reason,
            refresh_callback=self._refresh_ranger_readings,
            progress_callback=self._set_chunk_progress,
            setpoint_callback=self._mark_normal_setpoint,
        )

    def _front_recovery_emergency_reason(self) -> str | None:
        if self.emergency or self.emergency_manager.motion_interrupted():
            return "USER_EMERGENCY_OR_PAUSE"
        if self.config is None or self.ranger_monitor is None:
            return None
        battery_reason = self._battery_stop_reason()
        if battery_reason is not None:
            return battery_reason
        reading = self.ranger_monitor.get_front_ranger()
        checks = (
            ("BACK", reading.back, self.config.critical_back),
            ("LEFT", reading.left, self.config.critical_side),
            ("RIGHT", reading.right, self.config.critical_side),
            ("UP", reading.up, self.config.critical_up),
        )
        for name, value, threshold in checks:
            if math.isfinite(value) and value < threshold:
                return f"X_FRONT_{name}_CRITICAL_{value:.2f}m_LT_{threshold:.2f}m"
        return None

    def _set_chunk_progress(self, progress: str) -> None:
        self._chunk_progress = progress

    def _mark_normal_setpoint(self) -> None:
        self._last_normal_setpoint_ts = time.time()

    def _clear_soft_stop_for_new_motion(self) -> None:
        if self.emergency_manager.killed or self.emergency_manager.emergency_event.is_set():
            return
        self.emergency_manager.soft_stop_event.clear()
        self.emergency = False
        self._paused = False

    def _required_staged_modes(self) -> list[str]:
        return [
            "MODE_SENSOR_CHECK",
            "MODE_SINGLE_DRONE_HOVER",
            "MODE_SINGLE_RANGER_SWEEP",
            "MODE_WALL_OBSTACLE_PROBE_TEST",
            "MODE_FORMATION_HOVER_ONLY",
            "MODE_FORMATION_MICRO_STEP",
        ]

    def _staggered_takeoff_all(self, height: float) -> bool:
        assert self.config is not None
        items = list(self.drones.items())
        for index, (drone_id, drone) in enumerate(items):
            if self.emergency_manager.motion_interrupted():
                return False
            self._event(f"staggered takeoff: {drone_id} height={height:.2f}m")
            drone.takeoff(height, self.config.takeoff_velocity)
            drone.hover(self.config.hover_time)
            if not self.simulation and index < len(items) - 1:
                time.sleep(self.config.takeoff_stagger_delay)
        self._airborne = True
        self.state = ObservationState.YAW_ALIGN.value
        if not self._motion_controller().align_yaws_to_formation(self._seq):
            return False
        self._event("formation yaw aligned: Rangers face heading, AI decks face stream directions")
        if self.formation is not None:
            yaws = self.formation.intended_ai_yaws()
            yaw_text = ", ".join(f"{drone_id}={yaw:.0f}deg" for drone_id, yaw in sorted(yaws.items()))
            if yaw_text:
                self._event(f"AI yaw targets: {yaw_text}; movement uses compensated body-frame hover setpoints")
        settle = max(self.config.hover_after_turn, self.config.demo_settle_time)
        self._event(f"takeoff/yaw settle: hover all {settle:.2f}s before motion")
        self._motion_controller().hover_all(settle)
        return True

    def _scout_sweep_in_place(self, distance: float, speed: float) -> bool:
        motion = self._motion_controller()
        if not motion.ranger_front_probe(self._seq, "LEFT", distance, speed=speed):
            return False
        self._refresh_ranger_readings()
        if not motion.ranger_front_probe(self._seq, "RIGHT", distance * 2.0, speed=speed):
            return False
        self._refresh_ranger_readings()
        if not motion.ranger_front_probe(self._seq, "LEFT", distance, speed=speed):
            return False
        self._refresh_ranger_readings()
        return True

    def _emergency_reason(self) -> str | None:
        if self.emergency or self.emergency_manager.motion_interrupted():
            return "USER_EMERGENCY_OR_PAUSE"
        if self.config is None or self.ranger_monitor is None:
            return None
        battery_reason = self._battery_stop_reason()
        if battery_reason is not None:
            return battery_reason
        return evaluate_critical_safety(
            self.config,
            self.ranger_monitor,
            self.ranger_monitor.get_front_ranger(),
            self.ranger_monitor.get_back_ranger() if self.config.requires_back_ranger else None,
        )

    def _mission_elapsed_s(self) -> float:
        if self._mission_started_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self._mission_started_at)

    def _mission_remaining_s(self) -> float:
        if self.config is None:
            return 0.0
        return max(0.0, self.config.max_mission_time_s - self._mission_elapsed_s())

    def _mission_stop_reason(self) -> str | None:
        if self.config is None:
            return None
        battery_reason = self._battery_stop_reason()
        if battery_reason is not None:
            return battery_reason
        elapsed = self._mission_elapsed_s()
        if elapsed >= self.config.forced_land_time_s:
            return f"FORCED_LAND_TIME_{elapsed:.1f}s"
        if elapsed >= self.config.auto_return_or_finish_time_s:
            return f"AUTO_FINISH_TIME_{elapsed:.1f}s"
        if elapsed >= self.config.max_mission_time_s:
            return f"MAX_MISSION_TIME_{elapsed:.1f}s"
        return None

    def _battery_stop_reason(self) -> str | None:
        if self.config is None:
            return None
        values = []
        for drone_id, drone in self.drones.items():
            voltage = drone.get_battery()
            if voltage > 0.0:
                values.append((drone_id, voltage))
                if self.logger is not None:
                    status = "LAND" if voltage < self.config.battery_land_v else "WARN" if voltage < self.config.battery_warn_v else "OK"
                    self.logger.battery(self._seq, drone_id, voltage, status)
        if not values:
            return None
        low_land = [(drone_id, value) for drone_id, value in values if value < self.config.battery_land_v]
        if low_land:
            drone_id, value = min(low_land, key=lambda item: item[1])
            return f"BATTERY_LAND_{drone_id}_{value:.2f}V_LT_{self.config.battery_land_v:.2f}V"
        return None

    def _refresh_ranger_readings(self) -> None:
        if self.ranger_monitor is None or self.config is None:
            return
        if self.simulation:
            return
        front_drone = self.drones.get("X_FRONT")
        if front_drone is not None:
            values = front_drone.read_log_snapshot(PREFLIGHT_LOG_VARIABLES)
            if values:
                reading = _ranger_from_values(values)
                self.ranger_monitor.update_front(reading)
                if self.logger is not None:
                    self.logger.ranger(self._seq, "X_FRONT", reading, self.state)
        if self.config.requires_back_ranger:
            back_drone = self.drones.get("X_BACK")
            if back_drone is not None:
                values = back_drone.read_log_snapshot(PREFLIGHT_LOG_VARIABLES)
                if values:
                    reading = _ranger_from_values(values)
                    self.ranger_monitor.update_back(reading)
                    if self.logger is not None:
                        self.logger.ranger(self._seq, "X_BACK", reading, self.state)

    def _apply_simulated_probe_after_if_needed(self) -> None:
        if not self.simulation or self.ranger_monitor is None:
            return
        scenario = scenarios().get(self.scenario)
        if scenario is None or scenario.probe_after is None:
            return
        current = self.ranger_monitor.get_front_ranger()
        updated = RangerReading(
            front=scenario.probe_after,
            back=current.back,
            left=current.left,
            right=current.right,
            up=current.up,
            valid=dict(current.valid),
        )
        for _ in range(3):
            self.ranger_monitor.update_front(updated)

    def _apply_simulated_reslot_after_if_needed(self) -> None:
        if not self.simulation or self.ranger_monitor is None:
            return
        if self.scenario not in {
            "wall_ahead",
            "local_obstacle",
            "local_obstacle_left_clear",
            "ambiguous_wide_obstacle",
            "front_critical",
            "front_critical_back_unknown",
        }:
            return
        current = self.ranger_monitor.get_front_ranger()
        updated = RangerReading(
            front=4.0,
            back=current.back,
            left=3.0,
            right=3.0,
            up=current.up,
            valid={"front": True, "back": True, "left": True, "right": True, "up": True},
        )
        for _ in range(3):
            self.ranger_monitor.update_front(updated)
        self._event("simulation: new heading is clear after Ranger re-slot")

    def _apply_simulated_front_recovery_after_if_needed(self, distance: float) -> None:
        if not self.simulation or self.ranger_monitor is None:
            return
        current = self.ranger_monitor.get_front_ranger()
        front = current.front + distance if math.isfinite(current.front) else current.front
        back = max(0.01, current.back - distance) if math.isfinite(current.back) else current.back
        updated = RangerReading(
            front=front,
            back=back,
            left=current.left,
            right=current.right,
            up=current.up,
            zrange=current.zrange,
            valid=dict(current.valid),
        )
        for _ in range(3):
            self.ranger_monitor.update_front(updated)

    def _seed_simulation_readings(self) -> None:
        if not self.simulation or self.ranger_monitor is None:
            return
        scenario = scenarios().get(self.scenario, scenarios()["open_space"])
        self.ranger_monitor.update_front(scenario.front)
        self.ranger_monitor.update_back(scenario.back)

    def _reading_safe(self, reading: RangerReading) -> bool:
        assert self.config is not None
        return (
            reading.front >= self.config.critical_front
            and reading.left >= self.config.critical_side
            and reading.right >= self.config.critical_side
            and reading.up >= self.config.critical_up
        )

    def _ranger_valid(self) -> bool:
        if self.ranger_monitor is None:
            return False
        front = self.ranger_monitor.get_front_ranger()
        back = self.ranger_monitor.get_back_ranger()
        if self.config is not None and not self.config.requires_back_ranger:
            return any(front.valid.values())
        return any(front.valid.values()) and any(back.valid.values())

    def _drone_statuses(self) -> list[DroneStatus]:
        if self.config is None:
            return []
        statuses: list[DroneStatus] = []
        for drone_config in self.config.drones:
            drone = self.drones.get(drone_config.drone_id)
            state = drone.get_state() if drone is not None else None
            statuses.append(
                DroneStatus(
                    drone_id=drone_config.drone_id,
                    role=drone_config.role.value,
                    uri=drone_config.uri,
                    connected=drone is not None,
                    airborne=bool(state and state.z > 0.05),
                    battery_v=drone.get_battery() if drone is not None else 0.0,
                    x=state.x if state is not None else 0.0,
                    y=state.y if state is not None else 0.0,
                    z=state.z if state is not None else 0.0,
                    yaw_deg=state.yaw_deg if state is not None else 0.0,
                )
            )
        return statuses

    def _ranger_statuses(self) -> list[RangerStatus]:
        if self.ranger_monitor is None or self.formation is None or self.config is None:
            return []
        front = self.ranger_monitor.get_front_ranger()
        back = self.ranger_monitor.get_back_ranger()
        statuses = [self._ranger_status("X_FRONT", front)]
        if self.config.requires_back_ranger:
            statuses.append(self._ranger_status("X_BACK", back))
        return statuses

    def _ranger_status(self, drone_id: str, reading: RangerReading) -> RangerStatus:
        assert self.formation is not None
        values = [reading.front, reading.back, reading.left, reading.right, reading.up]
        finite = [value for value in values if math.isfinite(value)]
        min_clearance = min(finite) if finite else math.inf
        health = "SAFE"
        if min_clearance < 0.40:
            health = "CRITICAL"
        elif min_clearance < 0.70:
            health = "WARNING"
        return RangerStatus(
            drone_id=drone_id,
            front=reading.front,
            back=reading.back,
            left=reading.left,
            right=reading.right,
            up=reading.up,
            zrange=reading.zrange,
            min_clearance=min_clearance,
            corrected_front_wall_distance=self.formation.corrected_front_wall_distance(reading.front)
            if drone_id == "X_FRONT"
            else math.inf,
            health=health,
        )

    def _ai_stream_statuses(self) -> list[AIStreamStatus]:
        if self.config is None:
            return []
        active = self.streams.active if self.streams is not None else {}
        statuses = []
        for drone in self.config.drones:
            if drone.stream_direction is None:
                continue
            event = active.get(drone.drone_id)
            statuses.append(
                AIStreamStatus(
                    drone_id=drone.drone_id,
                    stream_direction=drone.stream_direction,
                    active=event is not None,
                    fps=0.0,
                    last_frame_timestamp=event.timestamp if event is not None else 0.0,
                )
            )
        return statuses

    def _battery_summary(self) -> str:
        statuses = self._drone_statuses()
        values = [status.battery_v for status in statuses if status.connected and status.battery_v > 0.0]
        if not values:
            return "sim/unknown"
        return f"min {min(values):.2f} V"

    def _last_setpoint_age_s(self) -> float:
        stamps = [getattr(drone, "last_setpoint_ts", 0.0) for drone in self.drones.values()]
        stamps = [stamp for stamp in stamps if stamp]
        if not stamps:
            return math.inf
        return time.time() - max(stamps)

    def _radio_status(self) -> str:
        mode = "simulation" if self.simulation else "real Crazyradio"
        state = "connected" if self._connected else "disconnected"
        return f"{mode} {state}"

    def _watchdog_status(self) -> str:
        if self.simulation:
            return "simulation"
        if not self.emergency_manager.watchdog_running():
            return "stopped"
        age = time.time() - self.emergency_manager.last_watchdog_keepalive_ts if self.emergency_manager.last_watchdog_keepalive_ts else math.inf
        return "running" if age < 1.0 else f"stale {age:.1f}s"

    def _event(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.events.append(f"[{stamp}] {text}")
        if self.logger is not None:
            self.logger.event(self._seq, self._mode_value(), self.state, text)
            self._log_snapshot(text[:80])

    def _mode_value(self) -> str:
        return self.mode.value if isinstance(self.mode, MissionMode) else str(self.mode)

    def _log_decision(
        self,
        decision: str,
        reason: str,
        action: str,
        ranger: RangerReading | None = None,
        envelope_state: str = "",
        envelope_reason: str = "",
        extra: dict[str, object] | None = None,
    ) -> None:
        if self.logger is None:
            return
        if ranger is None and self.ranger_monitor is not None:
            ranger = self.ranger_monitor.get_front_ranger()
        heading = self.formation.heading_deg if self.formation is not None else 0.0
        self.logger.decision(
            self._seq,
            self._mode_value(),
            self.state,
            decision,
            reason,
            action,
            ranger=ranger,
            envelope_state=envelope_state or self.last_envelope.state,
            envelope_reason=envelope_reason or self.last_envelope.reason,
            heading=heading,
            extra=extra,
        )

    def _log_snapshot(self, label: str) -> None:
        if self.logger is None:
            return
        front_ranger = self.ranger_monitor.get_front_ranger() if self.ranger_monitor is not None else None
        drone_states = {status.drone_id: asdict(status) for status in self._drone_statuses()}
        formation_slots: dict[str, tuple[float, float]] = {}
        ai_yaws: dict[str, float] = {}
        heading = 0.0
        if self.formation is not None:
            heading = self.formation.heading_deg
            formation_slots = {key: (slot.x, slot.y) for key, slot in self.formation.rotated_offsets().items()}
            ai_yaws = self.formation.intended_ai_yaws()
        self.logger.state_snapshot(
            self._seq,
            label,
            self._mode_value(),
            self.state,
            self.emergency,
            self._connected,
            self._airborne,
            self._paused,
            heading,
            self._battery_summary(),
            self._radio_status(),
            self.last_envelope.state,
            self.last_envelope.reason,
            front_ranger,
            drone_states,
            formation_slots,
            ai_yaws,
            chunk_progress=self._chunk_progress,
            auto_land_reason=self._auto_land_reason,
        )

    def _write_run_summary(self) -> None:
        if self.logger is None:
            return
        self.logger.run_summary(
            {
                "mode": self._mode_value(),
                "state": self.state,
                "emergency": self.emergency,
                "connected": self._connected,
                "airborne": self._airborne,
                "simulation": self.simulation,
                "scenario": self.scenario,
                "seq": self._seq,
                "battery_summary": self._battery_summary(),
                "radio_status": self._radio_status(),
                "heading_deg": self.formation.heading_deg if self.formation is not None else 0.0,
                "last_envelope": asdict(self.last_envelope),
                "auto_land_reason": self._auto_land_reason,
                "chunk_progress": self._chunk_progress,
                "recent_events": self.events[-50:],
            }
        )

    def _require_config(self) -> None:
        if self.config is None:
            self.load_config()

    def _require_connected(self) -> None:
        self._require_config()
        if not self._connected:
            self.connect_all()


def _fmt(value: float) -> str:
    return "inf" if not math.isfinite(value) else f"{value:.2f}"


def _preflight_value_summary(drone_id: str, values: dict[str, float]) -> str:
    def raw(name: str) -> str:
        if name not in values:
            return "missing"
        return f"{values[name]:.2f}"

    def meters(name: str) -> str:
        if name not in values:
            return "missing"
        return f"{values[name] / 1000.0:.2f}m"

    return (
        f"sensor values {drone_id}: "
        f"vbat={raw('pm.vbat')}V "
        f"zrange={meters('range.zrange')} "
        f"state_z={raw('stateEstimate.z')}m "
        f"ranger f/l/r/b/u={meters('range.front')}/{meters('range.left')}/"
        f"{meters('range.right')}/{meters('range.back')}/{meters('range.up')}"
    )
