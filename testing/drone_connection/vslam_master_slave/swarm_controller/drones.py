from __future__ import annotations

import math
import time
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Protocol

from .config import DroneConfig
from .top3_logic import compensate_body_velocity

try:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crazyflie.syncLogger import SyncLogger
except ImportError:  # pragma: no cover - exercised only without cflib installed
    cflib = None
    Crazyflie = None
    LogConfig = None
    SyncCrazyflie = None
    SyncLogger = None


@dataclass
class DroneState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw_deg: float = 0.0
    battery_v: float = 0.0


class DroneLike(Protocol):
    config: DroneConfig

    def connect(self) -> None: ...
    def takeoff(self, height: float, velocity: float = 0.20) -> None: ...
    def land(self, velocity: float = 0.20) -> None: ...
    def hover(self, duration: float) -> None: ...
    def send_formation_velocity(
        self,
        vx_form: float,
        vy_form: float,
        vz: float,
        yaw_rate: float,
        heading: float,
        duration: float,
    ) -> None: ...
    def set_yaw(self, yaw_deg: float, rate_deg_s: float = 72.0) -> None: ...
    def hard_kill(self) -> None: ...
    def safe_hover_land(self) -> None: ...
    def supervisor_watchdog_ping(self) -> None: ...
    def read_param(self, name: str) -> str | None: ...
    def read_log_snapshot(self, variables: list[str]) -> dict[str, float]: ...
    def stop(self) -> None: ...
    def get_battery(self) -> float: ...
    def get_state(self) -> DroneState: ...


class CrazyflieDrone:
    def __init__(self, config: DroneConfig, cache: str = "./cache", rate_hz: float = 20.0) -> None:
        self.config = config
        self.cache = cache
        self.rate_hz = rate_hz
        self.state = DroneState()
        self._scf: SyncCrazyflie | None = None
        self._command_lock = Lock()
        self._hover_stop = Event()
        self._hover_thread: Thread | None = None
        self._motion_stop = Event()
        self.last_setpoint_ts: float = 0.0

    def connect(self) -> None:
        if cflib is None or Crazyflie is None or SyncCrazyflie is None:
            raise RuntimeError("cflib is not installed; use SimulationDrone or install cflib")
        cflib.crtp.init_drivers(enable_debug_driver=False)
        self._scf = SyncCrazyflie(self.config.uri, cf=Crazyflie(rw_cache=self.cache))
        self._scf.open_link()

    def takeoff(self, height: float, velocity: float = 0.20) -> None:
        self._motion_stop.clear()
        self._arm(True)
        self._ramp_height(height, seconds=self._vertical_move_duration(height, velocity, min_seconds=1.0))
        self._start_hover_hold(height)

    def land(self, velocity: float = 0.20) -> None:
        self._stop_hover_hold()
        self._ramp_height(0.02, seconds=self._vertical_move_duration(0.02, velocity, min_seconds=1.0), ignore_motion_stop=True)
        self.stop()
        self._arm(False)

    def hover(self, duration: float) -> None:
        target_z = self.state.z or 0.40
        if self._hover_thread is not None and self._hover_thread.is_alive():
            time.sleep(duration)
            return
        self._send_hover(0.0, 0.0, 0.0, target_z, duration)
        self._start_hover_hold(target_z)

    def send_formation_velocity(
        self,
        vx_form: float,
        vy_form: float,
        vz: float,
        yaw_rate: float,
        heading: float,
        duration: float,
    ) -> None:
        self._stop_hover_hold()
        self._motion_stop.clear()
        vx_body, vy_body = formation_velocity_to_body(vx_form, vy_form, heading, self.state.yaw_deg)
        z = max(0.02, self.state.z + vz * duration)
        self._send_hover(vx_body, vy_body, yaw_rate, z, duration)
        self._start_hover_hold(z)

    def set_yaw(self, yaw_deg: float, rate_deg_s: float = 72.0) -> None:
        self._stop_hover_hold()
        self._motion_stop.clear()
        delta = normalize_signed_degrees(yaw_deg - self.state.yaw_deg)
        yaw_rate = abs(rate_deg_s) if delta >= 0 else -abs(rate_deg_s)
        duration = abs(delta) / max(abs(rate_deg_s), 1.0)
        self._send_hover(0.0, 0.0, yaw_rate, self.state.z or 0.40, duration)
        self.state.yaw_deg = yaw_deg
        self._start_hover_hold(self.state.z or 0.40)

    def stop(self) -> None:
        self._motion_stop.set()
        self._stop_hover_hold()
        if self._scf is not None:
            with self._command_lock:
                self._scf.cf.commander.send_stop_setpoint()

    def hard_kill(self) -> None:
        self._motion_stop.set()
        self._stop_hover_hold()
        if self._scf is None:
            return
        for _ in range(3):
            with self._command_lock:
                supervisor = getattr(self._scf.cf, "supervisor", None)
                emergency_stop = getattr(supervisor, "send_emergency_stop", None) if supervisor is not None else None
                if emergency_stop is not None:
                    emergency_stop()
                else:
                    self._scf.cf.commander.send_stop_setpoint()
            time.sleep(0.03)

    def safe_hover_land(self) -> None:
        self._motion_stop.set()
        self._stop_hover_hold()
        if self._scf is None:
            return
        z = self.state.z if self.state.z > 0.02 else 0.20
        self._send_hover(0.0, 0.0, 0.0, z, 0.25, ignore_motion_stop=True)
        self._ramp_height(0.02, seconds=self._vertical_move_duration(0.02, 0.20, min_seconds=1.5), ignore_motion_stop=True)
        self.stop()
        self._arm(False)

    def supervisor_watchdog_ping(self) -> None:
        if self._scf is None:
            return
        supervisor = getattr(self._scf.cf, "supervisor", None)
        for method_name in ("send_watchdog_reset", "send_supervisor_keepalive", "send_keepalive"):
            method = getattr(supervisor, method_name, None) if supervisor is not None else None
            if method is not None:
                method()
                return

    def read_param(self, name: str) -> str | None:
        if self._scf is None:
            return None
        try:
            return str(self._scf.cf.param.get_value(name))
        except Exception:
            return None

    def read_log_snapshot(self, variables: list[str]) -> dict[str, float]:
        if self._scf is None or LogConfig is None or SyncLogger is None:
            return {}
        values: dict[str, float] = {}
        for variable in variables:
            for fetch_as in (None, "float"):
                log_config = LogConfig(name=f"preflight_{variable.replace('.', '_')}", period_in_ms=100)
                try:
                    if fetch_as is None:
                        log_config.add_variable(variable)
                    else:
                        log_config.add_variable(variable, fetch_as)
                    with SyncLogger(self._scf, log_config) as logger:
                        _timestamp, data, _logconf = next(logger)
                    if variable in data:
                        values[variable] = float(data[variable])
                        break
                except Exception:
                    continue
        if "pm.vbat" in values:
            self.state.battery_v = values["pm.vbat"]
        if "stateEstimate.z" in values:
            self.state.z = values["stateEstimate.z"]
        return values

    def get_battery(self) -> float:
        return self.state.battery_v

    def get_state(self) -> DroneState:
        return self.state

    def close(self) -> None:
        self._stop_hover_hold()
        if self._scf is not None:
            self._scf.close_link()
            self._scf = None

    def _send_hover(
        self,
        vx: float,
        vy: float,
        yaw_rate: float,
        z: float,
        duration: float,
        ignore_motion_stop: bool = False,
    ) -> None:
        if self._scf is None:
            raise RuntimeError(f"{self.config.drone_id} is not connected")
        period = 1.0 / self.rate_hz
        end = time.monotonic() + duration
        while time.monotonic() < end:
            if self._motion_stop.is_set() and not ignore_motion_stop:
                return
            with self._command_lock:
                self._scf.cf.commander.send_hover_setpoint(vx, vy, yaw_rate, z)
                self.last_setpoint_ts = time.time()
            time.sleep(period)
        self.state.z = z

    def _ramp_height(self, target_height: float, seconds: float, ignore_motion_stop: bool = False) -> None:
        start = self.state.z
        steps = max(1, int(seconds * self.rate_hz))
        for index in range(steps):
            if self._motion_stop.is_set() and not ignore_motion_stop:
                return
            z = start + (target_height - start) * float(index + 1) / float(steps)
            self._send_hover(0.0, 0.0, 0.0, z, 1.0 / self.rate_hz, ignore_motion_stop=ignore_motion_stop)

    def _vertical_move_duration(self, target_height: float, velocity: float, min_seconds: float) -> float:
        distance = abs(target_height - self.state.z)
        return max(min_seconds, distance / max(velocity, 0.05))

    def _arm(self, armed: bool) -> None:
        if self._scf is None:
            return
        supervisor = getattr(self._scf.cf, "supervisor", None)
        if supervisor is not None:
            supervisor.send_arming_request(armed)

    def _start_hover_hold(self, z: float) -> None:
        if self._scf is None:
            return
        if self._hover_thread is not None and self._hover_thread.is_alive():
            return
        self._hover_stop.clear()
        self._hover_thread = Thread(target=self._hover_hold_loop, args=(z,), daemon=True)
        self._hover_thread.start()

    def _stop_hover_hold(self) -> None:
        self._hover_stop.set()
        if self._hover_thread is not None and self._hover_thread.is_alive():
            self._hover_thread.join(timeout=0.5)
        self._hover_thread = None

    def _hover_hold_loop(self, z: float) -> None:
        if self._scf is None:
            return
        period = 1.0 / self.rate_hz
        while not self._hover_stop.is_set():
            if self._motion_stop.is_set():
                return
            with self._command_lock:
                self._scf.cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, z)
                self.last_setpoint_ts = time.time()
            time.sleep(period)


class SimulationDrone:
    def __init__(self, config: DroneConfig) -> None:
        self.config = config
        self.state = DroneState()
        self.connected = False
        self.commands: list[str] = []
        self.last_setpoint_ts: float = 0.0

    def connect(self) -> None:
        self.connected = True
        self.commands.append("connect")
        print(f"[sim:{self.config.drone_id}] connect {self.config.uri or '(sim)'}")

    def takeoff(self, height: float, velocity: float = 0.20) -> None:
        self.state.z = height
        self.commands.append(f"takeoff:{height:.2f},velocity={velocity:.2f}")
        print(f"[sim:{self.config.drone_id}] takeoff {height:.2f}m velocity={velocity:.2f}m/s")

    def land(self, velocity: float = 0.20) -> None:
        self.state.z = 0.0
        self.commands.append(f"land:velocity={velocity:.2f}")
        print(f"[sim:{self.config.drone_id}] land velocity={velocity:.2f}m/s")

    def hover(self, duration: float) -> None:
        self.commands.append(f"hover:{duration:.2f}")
        print(f"[sim:{self.config.drone_id}] hover {duration:.2f}s")

    def send_formation_velocity(
        self,
        vx_form: float,
        vy_form: float,
        vz: float,
        yaw_rate: float,
        heading: float,
        duration: float,
    ) -> None:
        self.state.x += vx_form * duration
        self.state.y += vy_form * duration
        self.state.z = max(0.0, self.state.z + vz * duration)
        self.state.yaw_deg = normalize_degrees(self.state.yaw_deg + yaw_rate * duration)
        vx_body, vy_body = formation_velocity_to_body(vx_form, vy_form, heading, self.state.yaw_deg)
        self.commands.append(
            f"vel:{vx_form:.2f},{vy_form:.2f},{duration:.2f},heading={heading:.1f},"
            f"body={vx_body:.2f},{vy_body:.2f},yaw={self.state.yaw_deg:.1f}"
        )
        self.last_setpoint_ts = time.time()
        print(
            f"[sim:{self.config.drone_id}] v_form=({vx_form:.2f},{vy_form:.2f}) "
            f"v_body=({vx_body:.2f},{vy_body:.2f}) yaw_rate={yaw_rate:.1f} "
            f"heading={heading:.1f} yaw={self.state.yaw_deg:.1f} duration={duration:.2f}s"
        )

    def set_yaw(self, yaw_deg: float, rate_deg_s: float = 72.0) -> None:
        self.state.yaw_deg = normalize_degrees(yaw_deg)
        self.commands.append(f"yaw:{yaw_deg:.1f},rate={rate_deg_s:.1f}")
        print(f"[sim:{self.config.drone_id}] yaw {yaw_deg:.1f}deg rate={rate_deg_s:.1f}deg/s")

    def hard_kill(self) -> None:
        self.commands.append("hard_kill")
        self.state.z = 0.0
        print(f"[sim:{self.config.drone_id}] hard motor kill")

    def safe_hover_land(self) -> None:
        self.commands.append("safe_hover_land")
        self.state.z = 0.0
        print(f"[sim:{self.config.drone_id}] safe hover land")

    def supervisor_watchdog_ping(self) -> None:
        self.commands.append("watchdog_ping")

    def read_param(self, name: str) -> str | None:
        if name == "deck.bcFlow2":
            return "1"
        return None

    def read_log_snapshot(self, variables: list[str]) -> dict[str, float]:
        values = {variable: 1.0 for variable in variables}
        values["pm.vbat"] = 4.0
        values["stateEstimate.z"] = self.state.z
        values["range.zrange"] = max(0.05, self.state.z or 0.40) * 1000.0
        for key in ("range.front", "range.back", "range.left", "range.right", "range.up"):
            values[key] = 2500.0
        return values

    def stop(self) -> None:
        self.commands.append("stop")
        print(f"[sim:{self.config.drone_id}] stop")

    def get_battery(self) -> float:
        return 4.0

    def get_state(self) -> DroneState:
        return self.state


def normalize_degrees(value: float) -> float:
    return value % 360.0


def normalize_signed_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def formation_velocity_to_body(
    vx_form: float,
    vy_form: float,
    formation_heading_deg: float,
    drone_yaw_deg: float,
) -> tuple[float, float]:
    """Convert a formation-relative velocity into the drone's body frame.

    The formation can move north/east/south/west while individual drones look
    sideways or backward for streaming. Hover setpoints are body-frame
    velocities, so compensate by the difference between formation heading and
    the drone's current yaw.
    """

    return compensate_body_velocity(vx_form, vy_form, formation_heading_deg, drone_yaw_deg)
