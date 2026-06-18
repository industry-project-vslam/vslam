from __future__ import annotations

from dataclasses import dataclass

from .config import DroneConfig, DroneRole, FormationConfig
from .geometry import Heading, Point, crazyflie_yaw_from_heading, heading_from_degrees, heading_vector, left_vector, normalize_yaw
from .top3_logic import yaw_for_drone_role


AI_CAMERA_YAW_OFFSETS = {
    # In the lab half-swarm setup, O1's physical camera/yaw direction is correct
    # at 270 deg from the formation heading. Movement commands are compensated
    # later so it still translates with the group.
    DroneRole.AI_LEFT_STREAM: 270.0,
    DroneRole.AI_FORWARD_STREAM: 0.0,
    DroneRole.AI_BACKWARD_STREAM: 180.0,
    # In the current half-swarm MVP the right AI drone keeps looking along the
    # formation heading. Only O1 is side-looking; this avoids symmetric sideways
    # yaw during the three-drone test.
    DroneRole.AI_RIGHT_STREAM: 0.0,
}


@dataclass(frozen=True)
class Slot:
    drone_id: str
    role: DroneRole
    x: float
    y: float
    z: float

    @property
    def point(self) -> Point:
        return Point(self.x, self.y)


class FormationModel:
    """Fixed top-3 AI-core formation with Ranger re-slot targets.

    Coordinates are drawing units on the 120 x 118 launch model:
    x grows right, y grows down, and NORTH is toward smaller y.

    MVP geometry:
      X_FRONT=(32,20), O1=(12,45), O2=(52,45) for NORTH.
      When turning east/west, X_FRONT uses extra side clearance so it does not
      cut close to the AI drones while it moves to the new front slot.

    O1/O2 keep their physical row. When the formation turns, the AI drones
    hold position and only update yaw. X_FRONT moves around the outside to the
    new front slot for the next heading.
    """

    def __init__(
        self,
        drones: list[DroneConfig],
        heading_deg: float = 0.0,
        formation_config: FormationConfig | None = None,
    ) -> None:
        self.drones = list(drones)
        self.heading_deg = normalize_yaw(heading_deg)
        self.config = formation_config or FormationConfig()

    @property
    def center(self) -> Point:
        return Point(self.config.center_x, self.config.center_y)

    @property
    def unit_to_meters(self) -> float:
        return self.config.unit_to_meters

    @property
    def heading(self) -> Heading:
        return heading_from_degrees(self.heading_deg)

    def update_heading(self, new_heading: float) -> None:
        self.heading_deg = normalize_yaw(new_heading)

    def slots_by_heading(self, heading: Heading | float | None = None) -> dict[str, Slot]:
        heading = self.heading if heading is None else heading
        slots: dict[str, Slot] = {}
        for drone in self.drones:
            point = self._target_for_role(drone.drone_id, drone.role, heading)
            slots[drone.drone_id] = Slot(drone.drone_id, drone.role, point.x, point.y, drone.offset_z)
        return slots

    def rotated_offsets(self) -> dict[str, Slot]:
        """Compatibility name used by the GUI/logger.

        The AI core is no longer rotated. Only X_FRONT/X_BACK positions change
        with heading.
        """
        return self.slots_by_heading()

    def ranger_slot(self, front: bool, heading: Heading | float | None = None) -> Point:
        heading = self.heading if heading is None else heading
        if front:
            return self._front_slot(heading)
        return self._back_slot(heading)

    def target_slots_after_turn(self, turn_direction: str) -> dict[str, Slot]:
        return self.slots_by_heading(self.heading_after_turn(turn_direction))

    def heading_after_turn(self, turn_direction: str) -> Heading:
        turn = turn_direction.upper()
        if turn == "LEFT":
            return self.heading.left()
        if turn == "RIGHT":
            return self.heading.right()
        raise ValueError(f"Unknown turn direction: {turn_direction}")

    def reslot_waypoints(self, turn_direction: str) -> dict[str, list[Point]]:
        old_heading = self.heading
        new_heading = self.heading_after_turn(turn_direction)
        old_front = self._front_slot(old_heading)
        new_front = self._front_slot(new_heading)
        front_corner = self._outside_corner(old_front, new_front, old_heading)

        old_back = self._back_slot(old_heading)
        new_back = self._back_slot(new_heading)
        back_corner = self._outside_corner(old_back, new_back, old_heading)

        return {
            "X_FRONT": [old_front, front_corner, new_front],
            "X_BACK": [old_back, back_corner, new_back],
        }

    def intended_ai_yaws(self) -> dict[str, float]:
        yaws: dict[str, float] = {}
        for drone in self.drones:
            if drone.yaw_offset_deg is not None:
                yaws[drone.drone_id] = normalize_yaw(crazyflie_yaw_from_heading(self.heading_deg) + drone.yaw_offset_deg)
            elif drone.role in AI_CAMERA_YAW_OFFSETS:
                yaws[drone.drone_id] = yaw_for_drone_role(drone.role, self.heading)
        return yaws

    def corrected_front_wall_distance(self, x_front_reading: float) -> float:
        return x_front_reading + self.front_offset_meters(self.heading)

    def front_offset_meters(self, heading: Heading | float | None = None) -> float:
        heading = self.heading if heading is None else heading
        vector = heading_vector(heading)
        front = self._front_slot(heading)
        delta = front - self.center
        return max(0.0, delta.x * vector.x + delta.y * vector.y) * self.unit_to_meters

    def reslot_required_clearance(self, turn_direction: str) -> float:
        points = self.reslot_waypoints(turn_direction).get("X_FRONT", [])
        if len(points) < 2:
            return self.config.front_radius * self.unit_to_meters
        longest = max(
            abs((end - start).x) + abs((end - start).y)
            for start, end in zip(points, points[1:])
        )
        return longest * self.unit_to_meters

    def formation_width(self) -> float:
        xs = [slot.x for slot in self.slots_by_heading().values()]
        return (max(xs) - min(xs)) * self.unit_to_meters if xs else 0.0

    def formation_length(self) -> float:
        ys = [slot.y for slot in self.slots_by_heading().values()]
        return (max(ys) - min(ys)) * self.unit_to_meters if ys else 0.0

    def forward_extent(self) -> float:
        projections = self._heading_projections()
        return max(projections, default=0.0) * self.unit_to_meters

    def rear_extent(self) -> float:
        projections = self._heading_projections()
        return abs(min(projections, default=0.0)) * self.unit_to_meters

    def side_extent(self) -> float:
        lateral = self._left_projections()
        return max((abs(value) for value in lateral), default=0.0) * self.unit_to_meters

    def drawing_delta_to_motion_meters(self, delta: Point) -> tuple[float, float]:
        forward = heading_vector(self.heading)
        left = left_vector(self.heading)
        forward_units = delta.x * forward.x + delta.y * forward.y
        left_units = delta.x * left.x + delta.y * left.y
        return forward_units * self.unit_to_meters, left_units * self.unit_to_meters

    def _target_for_role(self, drone_id: str, role: DroneRole, heading: Heading | float) -> Point:
        if role == DroneRole.FRONT_RANGER:
            return self.ranger_slot(front=True, heading=heading)
        if role == DroneRole.BACK_RANGER:
            return self.ranger_slot(front=False, heading=heading)
        if drone_id in self.config.ai_slots:
            x, y = self.config.ai_slots[drone_id]
            return Point(x, y)
        return self.center

    def _ai_bounds(self) -> tuple[float, float, float, float]:
        ids = {
            drone.drone_id
            for drone in self.drones
            if drone.role
            in {
                DroneRole.AI_LEFT_STREAM,
                DroneRole.AI_FORWARD_STREAM,
                DroneRole.AI_BACKWARD_STREAM,
                DroneRole.AI_RIGHT_STREAM,
            }
        }
        points = [self.config.ai_slots[drone_id] for drone_id in ids if drone_id in self.config.ai_slots]
        if not points:
            points = list(self.config.ai_slots.values()) or [(self.config.center_x, self.config.center_y)]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return min(xs), max(xs), min(ys), max(ys)

    def _front_slot(self, heading: Heading | float) -> Point:
        heading = heading_from_degrees(heading.value if isinstance(heading, Heading) else heading)
        min_x, max_x, min_y, max_y = self._ai_bounds()
        if heading == Heading.NORTH:
            return Point(self.config.center_x, min_y - self.config.front_clearance_units_north_south)
        if heading == Heading.SOUTH:
            return Point(self.config.center_x, max_y + self.config.front_clearance_units_north_south)
        if heading == Heading.EAST:
            return Point(max_x + self.config.front_clearance_units_east_west, self.config.center_y)
        if heading == Heading.WEST:
            return Point(min_x - self.config.front_clearance_units_east_west, self.config.center_y)
        return self.center + heading_vector(heading).scale(self.config.front_radius)

    def _back_slot(self, heading: Heading | float) -> Point:
        heading = heading_from_degrees(heading.value if isinstance(heading, Heading) else heading)
        min_x, max_x, min_y, max_y = self._ai_bounds()
        if heading == Heading.NORTH:
            return Point(self.config.center_x, max_y + self.config.front_clearance_units_north_south)
        if heading == Heading.SOUTH:
            return Point(self.config.center_x, min_y - self.config.front_clearance_units_north_south)
        if heading == Heading.EAST:
            return Point(min_x - self.config.front_clearance_units_east_west, self.config.center_y)
        if heading == Heading.WEST:
            return Point(max_x + self.config.front_clearance_units_east_west, self.config.center_y)
        return self.center - heading_vector(heading).scale(self.config.back_radius)

    @staticmethod
    def _outside_corner(old: Point, new: Point, old_heading: Heading | float) -> Point:
        old_heading = heading_from_degrees(old_heading.value if isinstance(old_heading, Heading) else old_heading)
        if old_heading in {Heading.NORTH, Heading.SOUTH}:
            return Point(new.x, old.y)
        return Point(old.x, new.y)

    def _heading_projections(self) -> list[float]:
        vector = heading_vector(self.heading)
        center = self.center
        return [
            (slot.x - center.x) * vector.x + (slot.y - center.y) * vector.y
            for slot in self.slots_by_heading().values()
        ]

    def _left_projections(self) -> list[float]:
        vector = left_vector(self.heading)
        center = self.center
        return [
            (slot.x - center.x) * vector.x + (slot.y - center.y) * vector.y
            for slot in self.slots_by_heading().values()
        ]


class HalfSwarmFormationModel(FormationModel):
    """Explicit name for the X_FRONT/O1/O2 MVP formation."""
