from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Event, Lock, Thread

from .drones import DroneLike


@dataclass(frozen=True)
class EmergencyResult:
    action: str
    target_count: int
    elapsed_to_first_stop_s: float
    elapsed_total_s: float
    button_pressed_ts: float
    event_set_ts: float
    first_stop_ts: float


class EmergencyManager:
    """Level-0 emergency layer.

    This object is deliberately small and thread-safe. The GUI may call
    hard_kill_all() directly instead of routing it through the worker command
    queue.
    """

    def __init__(self) -> None:
        self.emergency_event = Event()
        self.soft_stop_event = Event()
        self.active = True
        self.hard_kill_armed = True
        self.killed = False
        self._lock = Lock()
        self._watchdog_stop = Event()
        self._watchdog_thread: Thread | None = None
        self.last_watchdog_keepalive_ts: float = 0.0
        self._last_result: EmergencyResult | None = None

    @property
    def last_result(self) -> EmergencyResult | None:
        return self._last_result

    def reset_for_new_mission(self) -> None:
        with self._lock:
            self.emergency_event.clear()
            self.soft_stop_event.clear()
            self.killed = False
            self.hard_kill_armed = True
            self.active = True
            self._last_result = None

    def motion_interrupted(self) -> bool:
        return self.emergency_event.is_set() or self.soft_stop_event.is_set() or self.killed

    def assert_no_emergency_before_motion(self) -> None:
        if self.motion_interrupted():
            raise RuntimeError("Emergency/soft stop active; refusing to send normal motion setpoints")

    def hard_kill_all(self, drones: dict[str, DroneLike]) -> EmergencyResult:
        start = time.perf_counter()
        button_pressed_ts = time.time()
        first_stop = start
        with self._lock:
            self.emergency_event.set()
            self.soft_stop_event.set()
            self.killed = True
            self.hard_kill_armed = False
            event_set_ts = time.time()

        for index, drone in enumerate(list(drones.values())):
            hard_kill = getattr(drone, "hard_kill", None)
            if hard_kill is not None:
                hard_kill()
            else:
                drone.stop()
            if index == 0:
                first_stop = time.perf_counter()
                first_stop_ts = time.time()
        if not drones:
            first_stop_ts = event_set_ts

        result = EmergencyResult(
            action="HARD_MOTOR_KILL",
            target_count=len(drones),
            elapsed_to_first_stop_s=first_stop - start,
            elapsed_total_s=time.perf_counter() - start,
            button_pressed_ts=button_pressed_ts,
            event_set_ts=event_set_ts,
            first_stop_ts=first_stop_ts,
        )
        self._last_result = result
        return result

    def safe_hover_land_all(self, drones: dict[str, DroneLike]) -> EmergencyResult:
        start = time.perf_counter()
        button_pressed_ts = time.time()
        first_stop = start
        with self._lock:
            self.soft_stop_event.set()
            event_set_ts = time.time()

        for index, drone in enumerate(list(drones.values())):
            safe_hover_land = getattr(drone, "safe_hover_land", None)
            if safe_hover_land is not None:
                safe_hover_land()
            else:
                drone.hover(0.2)
                drone.land()
            if index == 0:
                first_stop = time.perf_counter()
                first_stop_ts = time.time()
        if not drones:
            first_stop_ts = event_set_ts

        result = EmergencyResult(
            action="SAFE_HOVER_LAND",
            target_count=len(drones),
            elapsed_to_first_stop_s=first_stop - start,
            elapsed_total_s=time.perf_counter() - start,
            button_pressed_ts=button_pressed_ts,
            event_set_ts=event_set_ts,
            first_stop_ts=first_stop_ts,
        )
        self._last_result = result
        return result

    def start_supervisor_watchdog(self, drones: dict[str, DroneLike], rate_hz: float = 5.0) -> None:
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return
        self._watchdog_stop.clear()
        self._watchdog_thread = Thread(target=self._watchdog_loop, args=(drones, rate_hz), daemon=True)
        self._watchdog_thread.start()

    def stop_supervisor_watchdog(self) -> None:
        self._watchdog_stop.set()
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=0.5)
        self._watchdog_thread = None

    def watchdog_running(self) -> bool:
        return self._watchdog_thread is not None and self._watchdog_thread.is_alive()

    def _watchdog_loop(self, drones: dict[str, DroneLike], rate_hz: float) -> None:
        period = 1.0 / max(rate_hz, 1.0)
        while not self._watchdog_stop.is_set() and not self.emergency_event.is_set():
            for drone in list(drones.values()):
                ping = getattr(drone, "supervisor_watchdog_ping", None)
                if ping is not None:
                    ping()
                    self.last_watchdog_keepalive_ts = time.time()
            time.sleep(period)
