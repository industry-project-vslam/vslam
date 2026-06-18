from __future__ import annotations

import time
from collections.abc import Callable
from threading import Thread

from .config import DroneRole, SwarmConfig
from .drones import DroneLike
from .formation import FormationModel
from .geometry import Point, crazyflie_yaw_from_heading
from .logs import SwarmLogger


EmergencyCheck = Callable[[], str | None]
RefreshCallback = Callable[[], None]
ProgressCallback = Callable[[str], None]
SetpointCallback = Callable[[], None]


class MotionController:
    def __init__(
        self,
        config: SwarmConfig,
        formation: FormationModel,
        drones: dict[str, DroneLike],
        logger: SwarmLogger,
        emergency_check: EmergencyCheck,
        refresh_callback: RefreshCallback | None = None,
        progress_callback: ProgressCallback | None = None,
        setpoint_callback: SetpointCallback | None = None,
    ) -> None:
        self.config = config
        self.formation = formation
        self.drones = drones
        self.logger = logger
        self.emergency_check = emergency_check
        self.refresh_callback = refresh_callback
        self.progress_callback = progress_callback
        self.setpoint_callback = setpoint_callback

    def formation_forward(self, seq: int, distance: float, speed: float | None = None) -> bool:
        speed = self.config.speed if speed is None else speed
        return self._move_all(seq, "FORMATION_FORWARD", speed, 0.0, distance, speed=speed, chunk_distance=self.config.step_size)

    def formation_backward(self, seq: int, distance: float, speed: float | None = None) -> bool:
        speed = self.config.speed if speed is None else speed
        return self._move_all(seq, "FORMATION_BACKWARD", -speed, 0.0, distance, speed=speed, chunk_distance=self.config.step_size)

    def selected_forward(self, seq: int, label: str, drone_ids: list[str], distance: float, speed: float | None = None) -> bool:
        speed = self.config.speed if speed is None else speed
        return self._move_selected(seq, label, drone_ids, speed, 0.0, distance, speed=speed, chunk_distance=self.config.step_size)

    def selected_backward(self, seq: int, label: str, drone_ids: list[str], distance: float, speed: float | None = None) -> bool:
        speed = self.config.speed if speed is None else speed
        return self._move_selected(seq, label, drone_ids, -speed, 0.0, distance, speed=speed, chunk_distance=self.config.step_size)

    def formation_sidestep(self, seq: int, direction: str, distance: float, speed: float | None = None) -> bool:
        speed = self.config.speed if speed is None else speed
        vy = speed if direction.upper() == "LEFT" else -speed
        return self._move_all(seq, f"FORMATION_{direction.upper()}", 0.0, vy, distance, speed=speed, chunk_distance=self.config.step_size)

    def formation_yaw(self, seq: int, direction: str, degrees: float) -> bool:
        turn = direction.upper()
        old_heading = self.formation.heading_deg
        delta = degrees if turn == "RIGHT" else -degrees
        new_heading = (old_heading + delta) % 360.0
        self.formation.update_heading(new_heading)
        for drone_id, yaw in self._target_yaws_by_drone().items():
            if self.emergency_check() is not None:
                return False
            self.drones[drone_id].set_yaw(yaw, self.config.yaw_rate_deg_s)
        self.logger.command(seq, "MOTION", f"FORMATION_YAW_{turn}", ",".join(self.drones), degrees, new_heading, "DONE", "ACK_DONE")
        return True

    def ranger_front_probe(self, seq: int, direction: str, distance: float, speed: float | None = None) -> bool:
        x_front = self._drone_by_role(DroneRole.FRONT_RANGER)
        if x_front is None:
            return False
        speed = self.config.speed if speed is None else speed
        vy = speed if direction.upper() == "LEFT" else -speed
        return self._move_selected(
            seq,
            "RANGER_FRONT_PROBE",
            [x_front.config.drone_id],
            0.0,
            vy,
            distance,
            speed=speed,
            chunk_distance=self.config.turn_reposition_chunk,
        )

    def full_formation_bypass(self, seq: int, direction: str, lateral_shift: float, forward_distance: float) -> bool:
        if not self.formation_sidestep(seq, direction, lateral_shift):
            self.logger.bypass(seq, direction, lateral_shift, forward_distance, "SIDESTEP_FAILED")
            return False
        if not self.formation_forward(seq, forward_distance):
            self.logger.bypass(seq, direction, lateral_shift, forward_distance, "FORWARD_FAILED")
            return False
        return_direction = "RIGHT" if direction.upper() == "LEFT" else "LEFT"
        if not self.formation_sidestep(seq, return_direction, lateral_shift):
            self.logger.bypass(seq, direction, lateral_shift, forward_distance, "RETURN_SHIFT_FAILED")
            return False
        self.logger.bypass(seq, direction, lateral_shift, forward_distance, "DONE")
        return True

    def ranger_reslot_turn(self, seq: int, turn_direction: str) -> bool:
        old_heading = self.formation.heading_deg
        new_heading = self.formation.heading_after_turn(turn_direction).value
        waypoints = self.formation.reslot_waypoints(turn_direction)
        self._hover_non_rangers()
        for drone_id, points in waypoints.items():
            if drone_id not in self.drones:
                continue
            if not self.move_ranger_through_waypoints(seq, drone_id, points):
                self.logger.turn_reslot(seq, turn_direction, old_heading, old_heading, f"{drone_id}_FAILED")
                return False
        self.formation.update_heading(new_heading)
        self._hover_non_rangers()
        target_yaws = self._target_yaws_by_drone()
        for drone_id, yaw in target_yaws.items():
            if self.emergency_check() is not None:
                return False
            self.drones[drone_id].set_yaw(yaw, self.config.yaw_rate_deg_s)
        yaw_text = ",".join(f"{drone_id}:{yaw:.0f}" for drone_id, yaw in sorted(target_yaws.items()))
        self.logger.command(
            seq,
            "MOTION",
            "FORMATION_YAW_AFTER_RESLOT",
            yaw_text,
            0.0,
            self.formation.heading_deg,
            "DONE",
            "ALL_YAWS_SENT",
        )
        self._hover_selected(list(self.drones), self.config.hover_after_turn)
        self.logger.turn_reslot(seq, turn_direction, old_heading, self.formation.heading_deg, "DONE")
        return True

    def align_yaws_to_formation(self, seq: int) -> bool:
        target_yaws = self._target_yaws_by_drone()
        for drone_id, yaw in target_yaws.items():
            if self.emergency_check() is not None:
                return False
            self.drones[drone_id].set_yaw(yaw, self.config.yaw_rate_deg_s)
        yaw_text = ",".join(f"{drone_id}:{yaw:.0f}" for drone_id, yaw in sorted(target_yaws.items()))
        self.logger.command(seq, "MOTION", "FORMATION_YAW_ALIGN", yaw_text, 0.0, self.formation.heading_deg, "DONE", "ALL_YAWS_SENT")
        return True

    def move_ranger_through_waypoints(self, seq: int, drone_id: str, waypoints: list[Point]) -> bool:
        completed: list[tuple[Point, Point]] = []
        for start, end in zip(waypoints, waypoints[1:]):
            delta = end - start
            if not self._move_selected_delta_units(seq, "RANGER_RESLOT_WAYPOINT", [drone_id], delta):
                for done_start, done_end in reversed(completed):
                    reverse = done_start - done_end
                    self._move_selected_delta_units(seq, "RANGER_RESLOT_REVERSE", [drone_id], reverse)
                return False
            completed.append((start, end))
        return True

    def hover_all(self, duration: float | None = None) -> None:
        duration = self.config.hover_time if duration is None else duration
        self._hover_selected(list(self.drones), duration)

    def land_all(self) -> None:
        for drone in self.drones.values():
            drone.land(self.config.landing_velocity)

    def _move_all(
        self,
        seq: int,
        command: str,
        vx: float,
        vy: float,
        distance: float,
        speed: float | None = None,
        chunk_distance: float | None = None,
    ) -> bool:
        return self._move_selected(seq, command, list(self.drones), vx, vy, distance, speed=speed, chunk_distance=chunk_distance)

    def _move_selected(
        self,
        seq: int,
        command: str,
        drone_ids: list[str],
        vx: float,
        vy: float,
        distance: float,
        speed: float | None = None,
        chunk_distance: float | None = None,
    ) -> bool:
        requested_speed = self.config.speed if speed is None else speed
        speed = min(abs(requested_speed), self.config.absolute_max_speed)
        if abs(vx) > speed:
            vx = speed if vx > 0.0 else -speed
        if abs(vy) > speed:
            vy = speed if vy > 0.0 else -speed
        chunk_distance = self.config.step_size if chunk_distance is None else chunk_distance
        chunk_distance = max(0.01, min(chunk_distance, self.config.max_step))
        remaining_distance = max(0.0, distance)
        setpoint_period = 1.0 / max(self.config.setpoint_hz, 10.0)
        segment = max(setpoint_period, min(self.config.motion_segment_s, 0.50))
        chunk_index = 0
        while remaining_distance > 1e-6:
            reason = self.emergency_check()
            if reason is not None:
                self.logger.command(seq, "EMERGENCY_HOVER", command, ",".join(drone_ids), distance, self.formation.heading_deg, reason, "INTERRUPTED")
                return False
            chunk_index += 1
            chunk = min(chunk_distance, remaining_distance)
            duration_remaining = chunk / max(speed, 0.01)
            if self.progress_callback is not None:
                self.progress_callback(f"{command} chunk {chunk_index}: {chunk:.2f}m/{distance:.2f}m")
            while duration_remaining > 1e-6:
                reason = self.emergency_check()
                if reason is not None:
                    self.logger.command(seq, "EMERGENCY_HOVER", command, ",".join(drone_ids), distance, self.formation.heading_deg, reason, "INTERRUPTED")
                    return False
                duration = min(segment, duration_remaining)
                threads: list[Thread] = []
                for drone_id in drone_ids:
                    reason = self.emergency_check()
                    if reason is not None:
                        self.logger.command(seq, "EMERGENCY_HOVER", command, ",".join(drone_ids), distance, self.formation.heading_deg, reason, "INTERRUPTED")
                        return False
                    if self.setpoint_callback is not None:
                        self.setpoint_callback()
                    reason = self.emergency_check()
                    if reason is not None:
                        self.logger.command(seq, "EMERGENCY_HOVER", command, ",".join(drone_ids), distance, self.formation.heading_deg, reason, "INTERRUPTED")
                        return False
                    self.logger.setpoint_loop(
                        drone_id,
                        command,
                        0.0,
                        "SEND",
                        seq=seq,
                        vx=vx,
                        vy=vy,
                        vz=0.0,
                        yaw_rate=0.0,
                        emergency_flag=False,
                    )
                    drone = self.drones[drone_id]
                    thread = Thread(
                        target=drone.send_formation_velocity,
                        args=(vx, vy, 0.0, 0.0, self.formation.heading_deg, duration),
                        daemon=True,
                    )
                    thread.start()
                    threads.append(thread)
                for thread in threads:
                    thread.join(timeout=duration + 0.5)
                duration_remaining -= duration
                time.sleep(0.01)
            remaining_distance -= chunk
            if self.refresh_callback is not None:
                self.refresh_callback()
            reason = self.emergency_check()
            if reason is not None:
                self.logger.command(seq, "EMERGENCY_HOVER", command, ",".join(drone_ids), distance, self.formation.heading_deg, reason, "INTERRUPTED")
                return False
        self._hover_selected(drone_ids, self.config.hover_time)
        self.logger.command(seq, "MOTION", command, ",".join(drone_ids), distance, self.formation.heading_deg, "DONE", "ACK_DONE")
        self.logger.breadcrumb(
            seq,
            command,
            distance,
            self.formation.heading_deg,
            self.formation.center.x,
            self.formation.center.y,
            "FIXED_FORMATION",
        )
        return True

    def _move_selected_delta_units(self, seq: int, command: str, drone_ids: list[str], delta: Point) -> bool:
        forward_m, left_m = self.formation.drawing_delta_to_motion_meters(delta)
        if abs(forward_m) > 1e-6:
            vx = self.config.speed if forward_m > 0.0 else -self.config.speed
            if not self._move_selected(
                seq,
                command,
                drone_ids,
                vx,
                0.0,
                abs(forward_m),
                speed=self.config.speed,
                chunk_distance=self.config.turn_reposition_chunk,
            ):
                return False
        if abs(left_m) > 1e-6:
            vy = self.config.speed if left_m > 0.0 else -self.config.speed
            if not self._move_selected(
                seq,
                command,
                drone_ids,
                0.0,
                vy,
                abs(left_m),
                speed=self.config.speed,
                chunk_distance=self.config.turn_reposition_chunk,
            ):
                return False
        return True

    def _drone_by_role(self, role: DroneRole) -> DroneLike | None:
        for drone in self.drones.values():
            if drone.config.role == role:
                return drone
        return None

    def _hover_non_rangers(self) -> None:
        self._hover_selected(
            [
                drone_id
                for drone_id, drone in self.drones.items()
                if drone.config.role not in {DroneRole.FRONT_RANGER, DroneRole.BACK_RANGER}
            ],
            0.05,
        )

    def _hover_selected(self, drone_ids: list[str], duration: float) -> None:
        threads: list[Thread] = []
        for drone_id in drone_ids:
            drone = self.drones.get(drone_id)
            if drone is None:
                continue
            thread = Thread(target=drone.hover, args=(duration,), daemon=True)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join(timeout=duration + 0.5)

    def _target_yaws_by_drone(self) -> dict[str, float]:
        yaws = self.formation.intended_ai_yaws()
        for drone_id, drone in self.drones.items():
            if drone.config.role in {DroneRole.FRONT_RANGER, DroneRole.BACK_RANGER}:
                yaws[drone_id] = crazyflie_yaw_from_heading(self.formation.heading_deg)
        return {drone_id: yaw for drone_id, yaw in yaws.items() if drone_id in self.drones}
