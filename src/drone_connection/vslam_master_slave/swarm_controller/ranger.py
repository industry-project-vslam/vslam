from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from statistics import median

from .config import SwarmConfig


RANGE_KEYS = ("front", "back", "left", "right", "up")


@dataclass
class RangerReading:
    front: float = math.inf
    back: float = math.inf
    left: float = math.inf
    right: float = math.inf
    up: float = math.inf
    zrange: float = math.inf
    timestamp: float = field(default_factory=time.time)
    valid: dict[str, bool] = field(default_factory=lambda: {key: False for key in RANGE_KEYS})

    def as_dict(self) -> dict[str, float]:
        values = {key: getattr(self, key) for key in RANGE_KEYS}
        values["zrange"] = self.zrange
        return values

    def value(self, key: str) -> float:
        return float(getattr(self, key))


class MedianSmoother:
    def __init__(self, window: int = 3) -> None:
        self.samples: dict[str, deque[float]] = {key: deque(maxlen=window) for key in RANGE_KEYS}

    def update(self, raw: RangerReading) -> RangerReading:
        values: dict[str, float] = {}
        valid: dict[str, bool] = {}
        for key in RANGE_KEYS:
            value = raw.value(key)
            if raw.valid.get(key, False) and not math.isfinite(value):
                # The Multi-ranger reports "no object in range" as a large raw
                # value that is converted to infinity. That is valid clear
                # space, not an unknown sensor failure. Preserve it so the
                # navigation can turn toward open space instead of blocking.
                self.samples[key].clear()
                values[key] = math.inf
                valid[key] = True
                continue
            if math.isfinite(value) and value > 0.0:
                self.samples[key].append(value)
            if self.samples[key]:
                values[key] = float(median(self.samples[key]))
                valid[key] = True
            else:
                values[key] = math.inf
                valid[key] = False
        return RangerReading(timestamp=raw.timestamp, valid=valid, zrange=raw.zrange, **values)


class RangerMonitor:
    def __init__(self, config: SwarmConfig) -> None:
        self.config = config
        self._front = RangerReading()
        self._back = RangerReading()
        self._front_smoother = MedianSmoother()
        self._back_smoother = MedianSmoother()

    def update_front(self, reading: RangerReading) -> None:
        self._front = self._front_smoother.update(reading)

    def update_back(self, reading: RangerReading) -> None:
        self._back = self._back_smoother.update(reading)

    def get_front_ranger(self) -> RangerReading:
        return self._front

    def get_back_ranger(self) -> RangerReading:
        return self._back

    def is_critical(self, reading: RangerReading) -> bool:
        return self.critical_reason(reading) is not None

    def critical_reason(self, reading: RangerReading, prefix: str = "RANGER") -> str | None:
        checks = (
            ("front", self.config.critical_front),
            ("back", self.config.critical_back),
            ("left", self.config.critical_side),
            ("right", self.config.critical_side),
            ("up", self.config.critical_up),
        )
        for key, threshold in checks:
            value = reading.value(key)
            if math.isfinite(value) and value < threshold:
                return f"{prefix}_{key.upper()}_CRITICAL_{value:.2f}m_LT_{threshold:.2f}m"
        return None

    def min_clearance(self, reading: RangerReading) -> float:
        values = [reading.value(key) for key in RANGE_KEYS if math.isfinite(reading.value(key))]
        return min(values) if values else math.inf


def mm_to_m(value_mm: float) -> float:
    if value_mm <= 0:
        return math.inf
    if value_mm >= 8000:
        return math.inf
    return value_mm / 1000.0
