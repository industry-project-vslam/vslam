from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)

    def scale(self, value: float) -> Point:
        return Point(self.x * value, self.y * value)


class Heading(Enum):
    NORTH = 0
    EAST = 90
    SOUTH = 180
    WEST = 270

    def right(self) -> Heading:
        return heading_from_degrees(self.value + 90)

    def left(self) -> Heading:
        return heading_from_degrees(self.value - 90)


def heading_from_degrees(value: float) -> Heading:
    normalized = int(round(normalize_yaw(value) / 90.0) * 90) % 360
    return {
        0: Heading.NORTH,
        90: Heading.EAST,
        180: Heading.SOUTH,
        270: Heading.WEST,
    }[normalized]


def heading_vector(heading: Heading | float) -> Point:
    if not isinstance(heading, Heading):
        heading = heading_from_degrees(heading)
    return {
        Heading.NORTH: Point(0.0, -1.0),
        Heading.EAST: Point(1.0, 0.0),
        Heading.SOUTH: Point(0.0, 1.0),
        Heading.WEST: Point(-1.0, 0.0),
    }[heading]


def left_vector(heading: Heading | float) -> Point:
    vector = heading_vector(heading)
    return Point(vector.y, -vector.x)


def normalize_yaw(value: float) -> float:
    return value % 360.0


def crazyflie_yaw_from_heading(heading: Heading | float) -> float:
    """Convert drawing/formation heading to Crazyflie yaw.

    The formation model uses map-style headings where 90 deg means turn right
    toward +x. Crazyflie yaw is positive for a physical left turn, so the yaw
    sent to the drone is the negative of the formation heading.
    """

    value = heading.value if isinstance(heading, Heading) else heading
    return normalize_yaw(-float(value))


def local_to_world(center: Point, local: Point) -> Point:
    return center + local
