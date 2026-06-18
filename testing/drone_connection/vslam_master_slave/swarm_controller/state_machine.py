from __future__ import annotations

import math
from enum import Enum

from .ai_streams import AIStreamManager
from .classifier import classify_surface_probe, detect_surface_candidate
from .config import DroneRole, SwarmConfig
from .drones import DroneLike
from .formation import FormationModel
from .frontiers import FrontierManager
from .logs import SwarmLogger
from .motion import MotionController
from .ranger import RangerMonitor, RangerReading
from .safety import (
    EnvelopeState,
    PlannedPrimitive,
    evaluate_bypass_envelope,
    evaluate_critical_safety,
    evaluate_formation_envelope,
    evaluate_reslot_path_safety,
)
from .scout_sweep import build_sweep_result


class SwarmState(str, Enum):
    INIT = "INIT"
    TAKEOFF = "TAKEOFF"
    AI_STREAM_ON = "AI_STREAM_ON"
    HOLD_BASE_FORMATION = "HOLD_BASE_FORMATION"
    SCOUT_SWEEP = "SCOUT_SWEEP"
    FORMATION_ENVELOPE_CHECK = "FORMATION_ENVELOPE_CHECK"
    BASE_FORMATION_MOVE = "BASE_FORMATION_MOVE"
    SURFACE_CANDIDATE = "SURFACE_CANDIDATE"
    LATERAL_PROBE = "LATERAL_PROBE"
    OBSTACLE_CLASSIFIED = "OBSTACLE_CLASSIFIED"
    BASE_FORMATION_BYPASS = "BASE_FORMATION_BYPASS"
    WALL_CONFIRMED = "WALL_CONFIRMED"
    RANGER_RESLOT_TURN = "RANGER_RESLOT_TURN"
    FRONTIER_SAVE = "FRONTIER_SAVE"
    MISSION_COMPLETE = "MISSION_COMPLETE"
    EMERGENCY_HOVER = "EMERGENCY_HOVER"
    LAND = "LAND"


class RangerLedStreamingSwarm:
    def __init__(
        self,
        config: SwarmConfig,
        formation: FormationModel,
        drones: dict[str, DroneLike],
        ranger_monitor: RangerMonitor,
        streams: AIStreamManager,
        logger: SwarmLogger,
    ) -> None:
        self.config = config
        self.formation = formation
        self.drones = drones
        self.ranger_monitor = ranger_monitor
        self.streams = streams
        self.logger = logger
        self.state = SwarmState.INIT
        self.seq = 0
        self.stop_reason: str | None = None
        self.frontiers = FrontierManager()
        self.motion = MotionController(config, formation, drones, logger, self._emergency_reason)

    def run(self, max_steps: int | None = None) -> None:
        max_steps = self.config.mission_max_steps if max_steps is None else max_steps
        self.logger.write_config(self.config, self.formation)
        try:
            self._connect_all()
            self._takeoff_all()
            self._start_ai_streams()
            self._hold_base_formation()
            self._scout_sweep()

            while self.seq < max_steps:
                self.seq += 1
                reason = self._emergency_reason()
                if reason is not None:
                    self.stop_reason = reason
                    self.state = SwarmState.EMERGENCY_HOVER
                    break

                envelope = self._check_forward_envelope()
                if envelope.state == EnvelopeState.FREE:
                    if detect_surface_candidate(self.config, self.formation, self.ranger_monitor.get_front_ranger()):
                        self._surface_candidate()
                    else:
                        self._base_formation_move()
                    continue

                if envelope.state == EnvelopeState.CRITICAL:
                    self.stop_reason = envelope.reason
                    self.state = SwarmState.EMERGENCY_HOVER
                    break

                if envelope.state == EnvelopeState.UNKNOWN:
                    self._frontier_save("UNKNOWN_SPACE_BLOCKS_FIXED_FORMATION", "UNKNOWN", "LOW")
                    self._choose_turn_or_land()
                    continue

                if envelope.state == EnvelopeState.OCCUPIED:
                    self._surface_candidate()
                    continue

            if self.state != SwarmState.EMERGENCY_HOVER:
                self.state = SwarmState.MISSION_COMPLETE
        finally:
            if self.state == SwarmState.EMERGENCY_HOVER:
                self.motion.hover_all(self.config.hover_time)
            self._land_all()
            for event in self.streams.stop_streams():
                self.logger.ai_stream(event)

    def _connect_all(self) -> None:
        self.state = SwarmState.INIT
        for drone in self.drones.values():
            drone.connect()

    def _takeoff_all(self) -> None:
        self.state = SwarmState.TAKEOFF
        for drone in self.drones.values():
            drone.takeoff(self.config.flight_height, self.config.takeoff_velocity)
            self.logger.command(self.seq, self.state.value, "TAKEOFF", drone.config.drone_id, self.config.flight_height, self.formation.heading_deg, "DONE", "ACK_DONE")

    def _start_ai_streams(self) -> None:
        self.state = SwarmState.AI_STREAM_ON
        for event in self.streams.start_streams(self.config.drones):
            self.logger.ai_stream(event)

    def _hold_base_formation(self) -> None:
        self.state = SwarmState.HOLD_BASE_FORMATION
        self.motion.hover_all(self.config.hover_time)

    def _scout_sweep(self) -> None:
        self.state = SwarmState.SCOUT_SWEEP
        front = self.ranger_monitor.get_front_ranger()
        self.logger.scout_sweep(self.seq, "center", front, self._wide_allowed(front))
        self.motion.ranger_front_probe(self.seq, "LEFT", self.config.side_step)
        reading = self.ranger_monitor.get_front_ranger()
        self.logger.scout_sweep(self.seq, "left", reading, self._wide_allowed(reading))
        self.motion.ranger_front_probe(self.seq, "RIGHT", self.config.side_step * 2.0)
        reading = self.ranger_monitor.get_front_ranger()
        self.logger.scout_sweep(self.seq, "right", reading, self._wide_allowed(reading))
        self.motion.ranger_front_probe(self.seq, "LEFT", self.config.side_step)
        reading = self.ranger_monitor.get_front_ranger()
        self.logger.scout_sweep(self.seq, "center_return", reading, self._wide_allowed(reading))
        back = self.ranger_monitor.get_back_ranger()
        if self.config.requires_back_ranger:
            self.logger.ranger(self.seq, "X_BACK", back, self.state.value)
            sweep = build_sweep_result(self.config, front, reading, reading, back)
            sweep_result = "ALLOW" if sweep.movement_allowed else "BLOCK"
            sweep_detail = f"center={sweep.center_lane_state.state.value};rear={sweep.rear_lane_state.state.value}"
        else:
            sweep_result = "ALLOW" if self._wide_allowed(reading) else "BLOCK"
            sweep_detail = "half_group_front_ranger_only"
        self.logger.command(
            self.seq,
            self.state.value,
            "SCOUT_SWEEP",
            "X_FRONT" if not self.config.requires_back_ranger else "X_FRONT,X_BACK",
            0.0,
            self.formation.heading_deg,
            sweep_result,
            sweep_detail,
        )

    def _check_forward_envelope(self):
        self.state = SwarmState.FORMATION_ENVELOPE_CHECK
        front = self.ranger_monitor.get_front_ranger()
        back = self.ranger_monitor.get_back_ranger()
        self.logger.ranger(self.seq, "X_FRONT", front, self.state.value)
        self.logger.ranger(self.seq, "X_BACK", back, self.state.value)
        envelope = evaluate_formation_envelope(
            self.config,
            self.formation,
            front,
            back,
            PlannedPrimitive("FORMATION_FORWARD", self.config.step_size),
        )
        self.logger.command(self.seq, self.state.value, "ENVELOPE_CHECK", "ALL", self.config.step_size, self.formation.heading_deg, envelope.state.value, envelope.reason)
        return envelope

    def _base_formation_move(self) -> None:
        self.state = SwarmState.BASE_FORMATION_MOVE
        self.motion.formation_forward(self.seq, self.config.step_size)

    def _surface_candidate(self) -> None:
        self.state = SwarmState.SURFACE_CANDIDATE
        classification, direction, shift = self._lateral_probe()
        if classification.classification == "OBSTACLE_OR_OPENING":
            self.state = SwarmState.OBSTACLE_CLASSIFIED
            self._try_bypass(direction, shift)
        elif classification.classification == "WALL_OR_BOUNDARY":
            self.state = SwarmState.WALL_CONFIRMED
            self._turn_reslot(direction)
        else:
            self._frontier_save("AMBIGUOUS_SURFACE", classification.classification, classification.confidence)
            self._choose_turn_or_land()

    def _lateral_probe(self):
        self.state = SwarmState.LATERAL_PROBE
        initial_reading = self.ranger_monitor.get_front_ranger()
        initial = self.formation.corrected_front_wall_distance(initial_reading.front)
        direction = self._clearer_side()
        shift = 0.0
        after = initial
        while shift < self.config.max_probe_shift:
            self.motion.ranger_front_probe(self.seq, direction, self.config.probe_step)
            shift += self.config.probe_step
            after_reading = self.ranger_monitor.get_front_ranger()
            after = self.formation.corrected_front_wall_distance(after_reading.front)
            if math.isfinite(after) and math.isfinite(initial) and after > initial + self.config.clear_increase:
                break
        classification = classify_surface_probe(self.config, initial, after, shift)
        if shift > 0.0:
            self.motion.ranger_front_probe(self.seq, self._opposite_side(direction), shift)
        self.logger.classification(
            self.seq,
            initial,
            direction,
            shift,
            after,
            classification.classification,
            classification.confidence,
            classification.next_action,
        )
        return classification, direction, shift

    def _try_bypass(self, direction: str, shift: float) -> None:
        self.state = SwarmState.BASE_FORMATION_BYPASS
        front = self.ranger_monitor.get_front_ranger()
        back = self.ranger_monitor.get_back_ranger()
        envelope = evaluate_bypass_envelope(
            self.config,
            self.formation,
            front,
            back,
            direction,
            shift,
            self.config.step_size * 2.0,
        )
        if envelope.state != EnvelopeState.FREE:
            self._frontier_save("BYPASS_BLOCKED_FOR_FIXED_FORMATION", "BLOCKED_FOR_FIXED_FORMATION", "HIGH")
            return
        self.motion.full_formation_bypass(self.seq, direction, shift, self.config.step_size * 2.0)

    def _turn_reslot(self, direction: str) -> None:
        self.state = SwarmState.RANGER_RESLOT_TURN
        turn = "LEFT" if direction == "LEFT" else "RIGHT"
        if not self._try_reslot_turn(turn):
            opposite = "LEFT" if turn == "RIGHT" else "RIGHT"
            if not self._try_reslot_turn(opposite):
                self._frontier_save("TURN_BLOCKED", "WALL_OR_BOUNDARY", "LOW")
                self.stop_reason = "BOTH_RESLOT_TURNS_BLOCKED"
                self.state = SwarmState.EMERGENCY_HOVER
                return
        self._scout_sweep()

    def _frontier_save(self, reason: str, classification: str, confidence: str) -> None:
        self.state = SwarmState.FRONTIER_SAVE
        front = self.ranger_monitor.get_front_ranger()
        frontier = self.frontiers.save_frontier(
            heading=self.formation.heading_deg,
            state=self.state.value,
            front_initial=front.front,
            probe_direction="",
            front_after=front.front,
            classification=classification,
            confidence=confidence,
            status=reason,
        )
        self.logger.frontier(self.seq, reason, classification, confidence, "RETURN_LATER")
        self.logger.command(self.seq, self.state.value, "FRONTIER_SAVE", "ALL", 0.0, self.formation.heading_deg, frontier.frontier_id, reason)

    def _choose_turn_or_land(self) -> None:
        front = self.ranger_monitor.get_front_ranger()
        if math.isfinite(front.left) or math.isfinite(front.right):
            direction = "LEFT" if front.left > front.right else "RIGHT"
            self._turn_reslot(direction)
        else:
            self.stop_reason = "NO_VERIFIED_FIXED_FORMATION_ROUTE"
            self.state = SwarmState.EMERGENCY_HOVER

    def _land_all(self) -> None:
        self.state = SwarmState.LAND
        self.motion.land_all()

    def _emergency_reason(self) -> str | None:
        return evaluate_critical_safety(
            self.config,
            self.ranger_monitor,
            self.ranger_monitor.get_front_ranger(),
            self.ranger_monitor.get_back_ranger(),
        )

    def _clearer_side(self) -> str:
        front = self.ranger_monitor.get_front_ranger()
        return "LEFT" if front.left >= front.right else "RIGHT"

    def _wide_allowed(self, reading: RangerReading) -> bool:
        return (
            reading.valid.get("front", False)
            and reading.valid.get("left", False)
            and reading.valid.get("right", False)
            and reading.valid.get("up", False)
            and reading.front >= self.config.critical_front
            and reading.left >= self.config.critical_side
            and reading.right >= self.config.critical_side
            and reading.up >= self.config.critical_up
        )

    def _try_reslot_turn(self, turn: str) -> bool:
        front = self.ranger_monitor.get_front_ranger()
        back = self.ranger_monitor.get_back_ranger()
        safety = evaluate_reslot_path_safety(self.config, self.formation, front, back, turn)
        self.logger.command(
            self.seq,
            self.state.value,
            f"TURN_{turn}_90_CHECK",
            "X_FRONT,X_BACK" if self.config.requires_back_ranger else "X_FRONT",
            0.0,
            self.formation.heading_deg,
            safety.state.value,
            safety.reason,
        )
        if safety.state != EnvelopeState.FREE:
            return False
        return self.motion.ranger_reslot_turn(self.seq, turn)

    @staticmethod
    def _opposite_side(direction: str) -> str:
        return "RIGHT" if direction.upper() == "LEFT" else "LEFT"
