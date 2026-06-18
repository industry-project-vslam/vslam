from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .config import DroneRole, SwarmConfig
from .drones import DroneLike
from .emergency import EmergencyManager
from .ranger import RangerReading, mm_to_m


PREFLIGHT_LOG_VARIABLES = [
    "pm.vbat",
    "stateEstimate.z",
    "range.zrange",
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.up",
]


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    passed: bool
    value: str = ""
    reason: str = ""


@dataclass
class DronePreflight:
    drone_id: str
    checks: list[PreflightCheck] = field(default_factory=list)
    log_values: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


@dataclass
class PreflightResult:
    passed: bool
    reason: str
    drones: list[DronePreflight]
    front_ranger: RangerReading | None = None
    back_ranger: RangerReading | None = None


class PreflightManager:
    def __init__(self, config: SwarmConfig, emergency_manager: EmergencyManager) -> None:
        self.config = config
        self.emergency_manager = emergency_manager
        self.last_result: PreflightResult | None = None

    def run(self, drones: dict[str, DroneLike], real_flight_confirm: bool, simulation: bool) -> PreflightResult:
        results: list[DronePreflight] = []
        front_ranger: RangerReading | None = None
        back_ranger: RangerReading | None = None

        configured_ids = [drone.drone_id for drone in self.config.drones]
        missing = [drone_id for drone_id in configured_ids if drone_id not in drones]
        global_checks = [
            PreflightCheck("all_configured_uris_connected", not missing, ",".join(configured_ids), f"missing={missing}" if missing else ""),
            PreflightCheck("emergency_manager_active", self.emergency_manager.active, str(self.emergency_manager.active), ""),
            PreflightCheck("hard_kill_armed", self.emergency_manager.hard_kill_armed, str(self.emergency_manager.hard_kill_armed), ""),
            PreflightCheck("real_flight_confirm", simulation or real_flight_confirm, str(real_flight_confirm), "REAL_FLIGHT_CONFIRM is false"),
        ]
        results.append(DronePreflight("GLOBAL", global_checks, {}))

        for config in self.config.drones:
            drone = drones.get(config.drone_id)
            checks: list[PreflightCheck] = []
            values: dict[str, float] = {}
            checks.append(PreflightCheck("connected", drone is not None, config.uri, "not connected" if drone is None else ""))
            checks.append(PreflightCheck("role_loaded", True, config.role.value, ""))
            if drone is not None:
                values = drone.read_log_snapshot(PREFLIGHT_LOG_VARIABLES)
                battery = values.get("pm.vbat", drone.get_battery())
                battery_valid = simulation or battery >= self.config.battery_warn_v
                checks.append(PreflightCheck("battery", battery_valid, f"{battery:.2f} V", f"below warning {self.config.battery_warn_v:.2f} V"))

                flow_param = _read_param(drone, "deck.bcFlow2")
                flow_detected = simulation or _truthy_param(flow_param)
                checks.append(
                    PreflightCheck(
                        "deck_bcFlow2",
                        flow_detected,
                        "sim" if simulation else str(flow_param),
                        "Flow deck missing or not reported by deck.bcFlow2",
                    )
                )

                zrange_m = mm_to_m(values.get("range.zrange", 0.0))
                # On the launch pad the Flow deck range can be around 1 cm.
                # Treat Flow/zrange as diagnostic only. Some valid Ranger/AI
                # setups report 0 here before takeoff, while the Ranger sensors
                # themselves are still usable for safety decisions.
                zrange_valid = simulation or (math.isfinite(zrange_m) and 0.005 <= zrange_m <= 2.0)
                checks.append(
                    PreflightCheck(
                        "flow_zrange",
                        True,
                        f"{zrange_m:.2f} m" if math.isfinite(zrange_m) else "invalid",
                        "" if zrange_valid else "Flow/zrange diagnostic warning only",
                    )
                )

                if config.role == DroneRole.FRONT_RANGER:
                    front_ranger = _ranger_from_values(values)
                    checks.append(PreflightCheck("x_front_ranger_valid", _ranger_required_valid(front_ranger), _ranger_summary(front_ranger), "front Ranger invalid"))
                if config.role == DroneRole.BACK_RANGER and self.config.requires_back_ranger:
                    back_ranger = _ranger_from_values(values)
                    checks.append(PreflightCheck("x_back_ranger_valid", _ranger_required_valid(back_ranger), _ranger_summary(back_ranger), "back Ranger invalid"))

            results.append(DronePreflight(config.drone_id, checks, values))

        failed = [f"{drone.drone_id}:{check.name}:{check.reason}" for drone in results for check in drone.checks if not check.passed]
        result = PreflightResult(not failed, "; ".join(failed), results, front_ranger, back_ranger)
        self.last_result = result
        return result


def _ranger_from_values(values: dict[str, float]) -> RangerReading:
    converted = {
        "front": mm_to_m(values.get("range.front", 0.0)),
        "back": mm_to_m(values.get("range.back", 0.0)),
        "left": mm_to_m(values.get("range.left", 0.0)),
        "right": mm_to_m(values.get("range.right", 0.0)),
        "up": mm_to_m(values.get("range.up", 0.0)),
    }
    valid = {}
    for key, value in converted.items():
        raw_mm = values.get(f"range.{key}", 0.0)
        sensor_reported_clear = raw_mm >= 8000.0
        valid[key] = (math.isfinite(value) and value > 0.0) or sensor_reported_clear
    return RangerReading(timestamp=time.time(), valid=valid, zrange=mm_to_m(values.get("range.zrange", 0.0)), **converted)


def _ranger_required_valid(reading: RangerReading | None) -> bool:
    if reading is None:
        return False
    return reading.valid.get("front", False) and reading.valid.get("left", False) and reading.valid.get("right", False) and reading.valid.get("up", False)


def _ranger_summary(reading: RangerReading | None) -> str:
    if reading is None:
        return "missing"
    return f"f/l/r/b/u={reading.front:.2f}/{reading.left:.2f}/{reading.right:.2f}/{reading.back:.2f}/{reading.up:.2f}"


def _read_param(drone: DroneLike, name: str) -> str | None:
    read_param = getattr(drone, "read_param", None)
    if read_param is None:
        return None
    try:
        value = read_param(name)
    except Exception:
        return None
    return None if value is None else str(value)


def _truthy_param(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}
