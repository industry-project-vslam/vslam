from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .ai_streams import AIStreamEvent
from .config import SwarmConfig
from .formation import FormationModel
from .ranger import RangerReading


class SwarmLogger:
    def __init__(self, root: str | Path = "stream_out/fixed_formation_missions", mission_id: str = "fixed_swarm") -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(root) / f"{mission_id}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = time.monotonic()
        self._files: list[Any] = []
        self._handles: dict[str, Any] = {}
        self._writers: dict[str, csv.DictWriter] = {}
        self._open_all()
        self._write_log_readme()

    def write_config(self, config: SwarmConfig, formation: FormationModel) -> None:
        payload = {
            "config": {key: value for key, value in asdict(config).items() if key != "drones"},
            "drones": [asdict(drone) for drone in config.drones],
            "heading_deg": formation.heading_deg,
            "rotated_offsets": {key: asdict(slot) for key, slot in formation.rotated_offsets().items()},
        }
        (self.run_dir / "formation_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def close(self) -> None:
        for handle in self._files:
            handle.close()
        self._files.clear()

    def event(self, seq: int, mode: str, state: str, text: str) -> None:
        self._write(
            "event_log",
            {
                "seq": seq,
                "mode": mode,
                "state": state,
                "event": text,
            },
        )

    def decision(
        self,
        seq: int,
        mode: str,
        state: str,
        decision: str,
        reason: str,
        action: str,
        ranger: RangerReading | None = None,
        envelope_state: str = "",
        envelope_reason: str = "",
        heading: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._write(
            "decision_log",
            {
                "seq": seq,
                "mode": mode,
                "state": state,
                "decision": decision,
                "reason": reason,
                "action": action,
                "heading": heading,
                "envelope_state": envelope_state,
                "envelope_reason": envelope_reason,
                "front": ranger.front if ranger is not None else "",
                "back": ranger.back if ranger is not None else "",
                "left": ranger.left if ranger is not None else "",
                "right": ranger.right if ranger is not None else "",
                "up": ranger.up if ranger is not None else "",
                "ranger_valid": ranger.valid if ranger is not None else {},
                "extra_json": extra or {},
            },
        )

    def state_snapshot(
        self,
        seq: int,
        label: str,
        mode: str,
        state: str,
        emergency: bool,
        connected: bool,
        airborne: bool,
        paused: bool,
        heading: float,
        battery_summary: str,
        radio_status: str,
        envelope_state: str,
        envelope_reason: str,
        front_ranger: RangerReading | None,
        drone_states: dict[str, Any],
        formation_slots: dict[str, Any],
        ai_yaws: dict[str, Any],
        chunk_progress: str = "",
        auto_land_reason: str = "",
    ) -> None:
        self._write(
            "state_snapshot_log",
            {
                "seq": seq,
                "label": label,
                "mode": mode,
                "state": state,
                "emergency": emergency,
                "connected": connected,
                "airborne": airborne,
                "paused": paused,
                "heading": heading,
                "battery_summary": battery_summary,
                "radio_status": radio_status,
                "envelope_state": envelope_state,
                "envelope_reason": envelope_reason,
                "front": front_ranger.front if front_ranger is not None else "",
                "back": front_ranger.back if front_ranger is not None else "",
                "left": front_ranger.left if front_ranger is not None else "",
                "right": front_ranger.right if front_ranger is not None else "",
                "up": front_ranger.up if front_ranger is not None else "",
                "ranger_valid": front_ranger.valid if front_ranger is not None else {},
                "drone_states_json": drone_states,
                "formation_slots_json": formation_slots,
                "ai_yaws_json": ai_yaws,
                "chunk_progress": chunk_progress,
                "auto_land_reason": auto_land_reason,
            },
        )

    def reslot_path(
        self,
        seq: int,
        turn: str,
        result: str,
        reason: str,
        measured: dict[str, float],
        required: dict[str, float],
    ) -> None:
        labels = sorted(set(measured) | set(required))
        if not labels:
            self._write(
                "reslot_path_log",
                {
                    "seq": seq,
                    "turn": turn,
                    "check": "none",
                    "measured": "",
                    "required": "",
                    "result": result,
                    "reason": reason,
                },
            )
            return
        for label in labels:
            self._write(
                "reslot_path_log",
                {
                    "seq": seq,
                    "turn": turn,
                    "check": label,
                    "measured": measured.get(label, ""),
                    "required": required.get(label, ""),
                    "result": result,
                    "reason": reason,
                },
            )

    def error(self, seq: int, mode: str, state: str, error_type: str, message: str, context: dict[str, Any] | None = None) -> None:
        self._write(
            "error_log",
            {
                "seq": seq,
                "mode": mode,
                "state": state,
                "error_type": error_type,
                "message": message,
                "context_json": context or {},
            },
        )

    def run_summary(self, summary: dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_s": f"{time.monotonic() - self.started_at:.3f}",
            **summary,
        }
        (self.run_dir / "run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def ranger(self, seq: int, source: str, reading: RangerReading, state: str) -> None:
        self._write(
            "ranger_log",
            {
                "seq": seq,
                "source": source,
                "state": state,
                "front": reading.front,
                "back": reading.back,
                "left": reading.left,
                "right": reading.right,
                "up": reading.up,
                "valid": json.dumps(reading.valid),
            },
        )

    def scout_sweep(self, seq: int, position: str, reading: RangerReading, wide_allowed: bool) -> None:
        self._write(
            "scout_sweep_log",
            {
                "seq": seq,
                "sweep_position": position,
                "front": reading.front,
                "back": reading.back,
                "left": reading.left,
                "right": reading.right,
                "up": reading.up,
                "wide_allowed": int(wide_allowed),
            },
        )

    def classification(
        self,
        seq: int,
        front_initial: float,
        probe_direction: str,
        probe_shift: float,
        front_after: float,
        classification: str,
        confidence: str,
        next_action: str,
    ) -> None:
        self._write(
            "classification_log",
            {
                "seq": seq,
                "front_initial": front_initial,
                "probe_direction": probe_direction,
                "probe_shift": probe_shift,
                "front_after_probe": front_after,
                "classification": classification,
                "confidence": confidence,
                "next_action": next_action,
            },
        )

    def bypass(self, seq: int, direction: str, lateral_shift: float, forward_distance: float, result: str) -> None:
        self._write(
            "bypass_log",
            {
                "seq": seq,
                "direction": direction,
                "lateral_shift": lateral_shift,
                "forward_distance": forward_distance,
                "result": result,
            },
        )

    def turn_reslot(self, seq: int, direction: str, old_heading: float, new_heading: float, result: str) -> None:
        self._write(
            "turn_reslot_log",
            {
                "seq": seq,
                "direction": direction,
                "old_heading": old_heading,
                "new_heading": new_heading,
                "result": result,
            },
        )

    def frontier(self, seq: int, reason: str, classification: str, confidence: str, status: str) -> None:
        self._write(
            "frontier_log",
            {
                "seq": seq,
                "frontier_id": f"frontier_{seq:04d}",
                "reason": reason,
                "classification": classification,
                "confidence": confidence,
                "status": status,
            },
        )

    def breadcrumb(
        self,
        seq: int,
        command: str,
        distance: float,
        heading: float,
        center_x: float,
        center_y: float,
        formation_mode: str,
        ranger_snapshot_id: int | None = None,
    ) -> None:
        self._write(
            "breadcrumb_log",
            {
                "seq": seq,
                "command": command,
                "distance": distance,
                "heading": heading,
                "center_x": center_x,
                "center_y": center_y,
                "formation_mode": formation_mode,
                "ranger_snapshot_id": ranger_snapshot_id if ranger_snapshot_id is not None else "",
            },
        )

    def ai_stream(self, event: AIStreamEvent) -> None:
        self._write("ai_stream_log", asdict(event))

    def command(
        self,
        seq: int,
        state: str,
        command: str,
        target_drones: str,
        distance: float,
        heading: float,
        result: str,
        ack_done: str,
        ranger_snapshot_id: int | None = None,
    ) -> None:
        self._write(
            "command_log",
            {
                "seq": seq,
                "state": state,
                "command": command,
                "target_drones": target_drones,
                "distance": distance,
                "heading": heading,
                "result": result,
                "ack_done": ack_done,
                "ranger_snapshot_id": ranger_snapshot_id if ranger_snapshot_id is not None else "",
            },
        )

    def safety(self, seq: int, level: str, event: str, detail: str) -> None:
        self._write("safety_log", {"seq": seq, "level": level, "event": event, "detail": detail})

    def emergency(
        self,
        action: str,
        target_count: int,
        first_stop_s: float,
        total_s: float,
        detail: str,
        button_pressed_ts: float | None = None,
        event_set_ts: float | None = None,
        first_stop_ts: float | None = None,
        last_normal_setpoint_ts: float | None = None,
    ) -> None:
        self._write(
            "emergency_log",
            {
                "action": action,
                "target_count": target_count,
                "elapsed_to_first_stop_s": first_stop_s,
                "elapsed_total_s": total_s,
                "detail": detail,
                "button_pressed_ts": button_pressed_ts if button_pressed_ts is not None else "",
                "emergency_event_set_ts": event_set_ts if event_set_ts is not None else "",
                "first_stop_packet_ts": first_stop_ts if first_stop_ts is not None else "",
                "last_normal_setpoint_ts": last_normal_setpoint_ts if last_normal_setpoint_ts is not None else "",
                "time_to_first_stop_packet_ms": first_stop_s * 1000.0,
            },
        )

    def preflight(self, drone_id: str, check: str, passed: bool, value: str, reason: str) -> None:
        self._write(
            "preflight_log",
            {
                "drone_id": drone_id,
                "check": check,
                "passed": int(passed),
                "value": value,
                "reason": reason,
            },
        )

    def setpoint_loop(
        self,
        drone_id: str,
        command: str,
        age_s: float,
        status: str,
        seq: int | None = None,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        yaw_rate: float = 0.0,
        emergency_flag: bool = False,
        ranger_snapshot_id: int | None = None,
    ) -> None:
        self._write(
            "setpoint_loop_log",
            {
                "seq": seq if seq is not None else "",
                "drone_id": drone_id,
                "command": command,
                "last_setpoint_age_s": age_s,
                "vx": vx,
                "vy": vy,
                "vz": vz,
                "yaw_rate": yaw_rate,
                "emergency_flag": int(emergency_flag),
                "ranger_snapshot_id": ranger_snapshot_id if ranger_snapshot_id is not None else "",
                "status": status,
            },
        )

    def zrange(self, seq: int, drone_id: str, zrange_m: float, state_z: float, valid: bool) -> None:
        self._write(
            "zrange_log",
            {
                "seq": seq,
                "drone_id": drone_id,
                "zrange_m": zrange_m,
                "state_estimate_z": state_z,
                "valid": int(valid),
            },
        )

    def mode_result(self, mode: str, passed: bool, reason: str) -> None:
        path = self.run_dir / "mode_test_results.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = {}
        payload[mode] = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "passed": bool(passed),
            "reason": reason,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def battery(self, seq: int, drone_id: str, voltage: float, status: str) -> None:
        self._write("battery_log", {"seq": seq, "drone_id": drone_id, "voltage": voltage, "status": status})

    def mission_timing(self, seq: int, phase: str, duration_s: float, total_s: float, reason: str = "") -> None:
        self._write(
            "mission_timing_log",
            {"seq": seq, "phase": phase, "duration_s": duration_s, "total_mission_time_s": total_s, "reason": reason},
        )

    def _open_all(self) -> None:
        self._open("event_log", ["seq", "mode", "state", "event"])
        self._open(
            "decision_log",
            [
                "seq",
                "mode",
                "state",
                "decision",
                "reason",
                "action",
                "heading",
                "envelope_state",
                "envelope_reason",
                "front",
                "back",
                "left",
                "right",
                "up",
                "ranger_valid",
                "extra_json",
            ],
        )
        self._open(
            "state_snapshot_log",
            [
                "seq",
                "label",
                "mode",
                "state",
                "emergency",
                "connected",
                "airborne",
                "paused",
                "heading",
                "battery_summary",
                "radio_status",
                "envelope_state",
                "envelope_reason",
                "front",
                "back",
                "left",
                "right",
                "up",
                "ranger_valid",
                "drone_states_json",
                "formation_slots_json",
                "ai_yaws_json",
                "chunk_progress",
                "auto_land_reason",
            ],
        )
        self._open("reslot_path_log", ["seq", "turn", "check", "measured", "required", "result", "reason"])
        self._open("error_log", ["seq", "mode", "state", "error_type", "message", "context_json"])
        self._open("safety_log", ["seq", "level", "event", "detail"])
        self._open(
            "emergency_log",
            [
                "action",
                "target_count",
                "elapsed_to_first_stop_s",
                "elapsed_total_s",
                "detail",
                "button_pressed_ts",
                "emergency_event_set_ts",
                "first_stop_packet_ts",
                "last_normal_setpoint_ts",
                "time_to_first_stop_packet_ms",
            ],
        )
        self._open("preflight_log", ["drone_id", "check", "passed", "value", "reason"])
        self._open(
            "setpoint_loop_log",
            [
                "seq",
                "drone_id",
                "command",
                "last_setpoint_age_s",
                "vx",
                "vy",
                "vz",
                "yaw_rate",
                "emergency_flag",
                "ranger_snapshot_id",
                "status",
            ],
        )
        self._open("zrange_log", ["seq", "drone_id", "zrange_m", "state_estimate_z", "valid"])
        self._open("battery_log", ["seq", "drone_id", "voltage", "status"])
        self._open("mission_timing_log", ["seq", "phase", "duration_s", "total_mission_time_s", "reason"])
        self._open("ranger_log", ["seq", "source", "state", "front", "back", "left", "right", "up", "valid"])
        self._open("scout_sweep_log", ["seq", "sweep_position", "front", "back", "left", "right", "up", "wide_allowed"])
        self._open(
            "classification_log",
            ["seq", "front_initial", "probe_direction", "probe_shift", "front_after_probe", "classification", "confidence", "next_action"],
        )
        self._open("bypass_log", ["seq", "direction", "lateral_shift", "forward_distance", "result"])
        self._open("turn_reslot_log", ["seq", "direction", "old_heading", "new_heading", "result"])
        self._open("frontier_log", ["seq", "frontier_id", "reason", "classification", "confidence", "status"])
        self._open(
            "breadcrumb_log",
            ["seq", "command", "distance", "heading", "center_x", "center_y", "formation_mode", "ranger_snapshot_id"],
        )
        self._open("ai_stream_log", ["drone_id", "stream_direction", "frame_id", "file_path", "status"])
        self._open(
            "command_log",
            ["seq", "state", "command", "target_drones", "distance", "heading", "result", "ack_done", "ranger_snapshot_id"],
        )

    def _open(self, key: str, fields: list[str]) -> None:
        filename = key if key.endswith(".csv") else f"{key}.csv"
        handle = (self.run_dir / filename).open("w", newline="", encoding="utf-8")
        fieldnames = ["timestamp", "elapsed_s"] + fields
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        handle.flush()
        self._files.append(handle)
        self._handles[key] = handle
        self._writers[key] = writer

    def _write(self, key: str, row: dict[str, Any]) -> None:
        writer = self._writers[key]
        payload = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_s": f"{time.monotonic() - self.started_at:.3f}",
        }
        payload.update({name: _csv_value(value) for name, value in row.items()})
        writer.writerow(payload)
        self._handles[key].flush()

    def _write_log_readme(self) -> None:
        text = """# Swarm Mission Logs

Start here when debugging a real run:

1. `event_log.csv` - readable timeline of what the controller did.
2. `decision_log.csv` - why each movement/turn/stop decision was made.
3. `state_snapshot_log.csv` - drone/ranger/formation state at important points.
4. `command_log.csv` - every motion command and whether it completed.
5. `reslot_path_log.csv` - exact clearance checks before X_FRONT re-slot turns.
6. `ranger_log.csv` - raw Ranger readings from X_FRONT/X_BACK.
7. `safety_log.csv` and `emergency_log.csv` - stop/land/emergency causes.
8. `run_summary.json` - final compact summary if the controller closed cleanly.

If the swarm behaves badly, send this whole folder. The most useful files are
`event_log.csv`, `decision_log.csv`, `state_snapshot_log.csv`,
`command_log.csv`, `reslot_path_log.csv`, and `ranger_log.csv`.
"""
        (self.run_dir / "README_LOGS.md").write_text(text, encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    return value
