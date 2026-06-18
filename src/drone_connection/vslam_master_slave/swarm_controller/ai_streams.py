from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .config import DroneConfig


@dataclass(frozen=True)
class AIStreamEvent:
    drone_id: str
    stream_direction: str
    timestamp: float
    frame_id: int | None = None
    file_path: str | None = None
    status: str = "ACTIVE"


class AIStreamManager:
    def __init__(self, output_root: str | Path = "stream_out/fixed_formation_streams") -> None:
        self.output_root = Path(output_root)
        self.active: dict[str, AIStreamEvent] = {}
        self.events: list[AIStreamEvent] = []

    def start_streams(self, drones: list[DroneConfig]) -> list[AIStreamEvent]:
        events: list[AIStreamEvent] = []
        for drone in drones:
            if drone.stream_direction is None:
                continue
            event = AIStreamEvent(
                drone_id=drone.drone_id,
                stream_direction=drone.stream_direction,
                timestamp=time.time(),
                status="STREAM_MARKED_ACTIVE",
            )
            self.active[drone.drone_id] = event
            self.events.append(event)
            events.append(event)
        return events

    def record_frame(self, drone_id: str, frame_id: int, file_path: str | None = None) -> AIStreamEvent:
        direction = self.active.get(drone_id, AIStreamEvent(drone_id, "unknown", time.time())).stream_direction
        event = AIStreamEvent(
            drone_id=drone_id,
            stream_direction=direction,
            timestamp=time.time(),
            frame_id=frame_id,
            file_path=file_path,
            status="FRAME",
        )
        self.events.append(event)
        return event

    def stop_streams(self) -> list[AIStreamEvent]:
        stopped: list[AIStreamEvent] = []
        for drone_id, event in list(self.active.items()):
            stop_event = AIStreamEvent(
                drone_id=drone_id,
                stream_direction=event.stream_direction,
                timestamp=time.time(),
                status="STREAM_STOPPED",
            )
            self.events.append(stop_event)
            stopped.append(stop_event)
        self.active.clear()
        return stopped

