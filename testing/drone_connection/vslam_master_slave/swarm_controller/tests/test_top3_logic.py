from __future__ import annotations

import math
import unittest

from swarm_controller.config import DroneRole, default_half_group_config
from swarm_controller.geometry import Heading, Point
from swarm_controller.top3_logic import (
    EmergencyState,
    ObservationState,
    RangerSnapshot,
    chooseTurnSide,
    choose_turn_side,
    compensateBodyVelocity,
    compensate_body_velocity,
    getCommandedSlot,
    getSlotOffset,
    get_commanded_slot,
    get_slot_offset,
    headingToVector,
    headingToYaw,
    heading_to_vector,
    heading_to_yaw,
    isForwardSafe,
    isSensorSnapshotValid,
    is_forward_safe,
    is_sensor_snapshot_valid,
    yawForDroneRole,
    yaw_for_drone_role,
)


def snapshot(**values: float) -> RangerSnapshot:
    payload = {
        "front": values.get("front", 4.0),
        "back": values.get("back", 4.0),
        "left": values.get("left", 3.0),
        "right": values.get("right", 3.0),
        "up": values.get("up", 2.5),
    }
    return RangerSnapshot(**payload, valid={key: True for key in payload})


class Top3LogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = default_half_group_config()
        self.formation_config = self.config.formation
        self.anchor = Point(32.0, 45.0)

    def test_required_states_are_explicit(self) -> None:
        expected = {
            "IDLE",
            "CONNECTING",
            "SENSOR_CHECK",
            "STREAMING",
            "TAKEOFF",
            "YAW_ALIGN",
            "OBSERVE_STEP_CHECK",
            "GROUP_FORWARD_STEP",
            "BLOCKED_HOVER",
            "CHOOSE_TURN_SIDE",
            "X_FRONT_RESLOT",
            "UPDATE_HEADING",
            "REALIGN_YAWS",
            "SAFE_HOVER",
            "LANDING",
            "EMERGENCY_STOP",
            "HARD_MOTOR_KILL",
        }
        self.assertTrue(expected.issubset({state.value for state in ObservationState}))
        self.assertEqual(EmergencyState.HARD_MOTOR_KILL.value, "HARD_MOTOR_KILL")

    def test_heading_vectors_and_yaws(self) -> None:
        self.assertEqual(heading_to_vector(Heading.NORTH), Point(0.0, -1.0))
        self.assertEqual(heading_to_vector(90.0), Point(1.0, 0.0))
        self.assertEqual(heading_to_yaw(Heading.WEST), 90.0)
        self.assertEqual(headingToVector(180.0), Point(0.0, 1.0))
        self.assertEqual(headingToYaw(270.0), 90.0)

    def test_yaw_for_drone_roles(self) -> None:
        self.assertEqual(yaw_for_drone_role(DroneRole.FRONT_RANGER, 0.0), 0.0)
        self.assertEqual(yaw_for_drone_role(DroneRole.AI_LEFT_STREAM, 0.0), 270.0)
        self.assertEqual(yaw_for_drone_role(DroneRole.AI_FORWARD_STREAM, 90.0), 270.0)
        self.assertEqual(yawForDroneRole(DroneRole.AI_LEFT_STREAM, 180.0), 90.0)

    def test_slot_offsets_and_commanded_slots(self) -> None:
        self.assertEqual(get_slot_offset("X_FRONT", DroneRole.FRONT_RANGER, 0.0, self.formation_config), Point(0.0, -45.0))
        self.assertEqual(get_slot_offset("X_FRONT", DroneRole.FRONT_RANGER, 90.0, self.formation_config), Point(65.0, 0.0))
        self.assertEqual(getSlotOffset("O1", DroneRole.AI_LEFT_STREAM, 90.0, self.formation_config), Point(-20.0, 0.0))

        east_front = get_commanded_slot(self.anchor, "X_FRONT", DroneRole.FRONT_RANGER, 90.0, self.formation_config)
        self.assertEqual((east_front.x, east_front.y, east_front.yaw_deg), (97.0, 45.0, 270.0))

        west_o1 = getCommandedSlot(self.anchor, "O1", DroneRole.AI_LEFT_STREAM, 270.0, self.formation_config)
        self.assertEqual((west_o1.x, west_o1.y, west_o1.yaw_deg), (12.0, 45.0, 0.0))

    def test_turn_side_selection(self) -> None:
        self.assertEqual(choose_turn_side(snapshot(left=2.0, right=3.0), 0.50), "RIGHT")
        self.assertEqual(chooseTurnSide(snapshot(left=3.0, right=2.0), 0.50), "LEFT")
        self.assertIsNone(choose_turn_side(snapshot(left=0.4, right=0.4), 0.50))

    def test_sensor_snapshot_and_forward_safety(self) -> None:
        self.assertTrue(is_sensor_snapshot_valid(snapshot()))
        self.assertTrue(isSensorSnapshotValid(snapshot(front=math.inf)))
        invalid = RangerSnapshot(front=0.0, left=3.0, right=3.0, up=2.5, valid={"front": False, "left": True, "right": True, "up": True})
        self.assertFalse(is_sensor_snapshot_valid(invalid))

        self.assertTrue(is_forward_safe(snapshot(front=1.0), self.config))
        self.assertFalse(isForwardSafe(snapshot(front=0.69), self.config))
        self.assertFalse(is_forward_safe(snapshot(left=0.49), self.config))

    def test_body_velocity_compensation(self) -> None:
        self.assertAlmostEqual(compensate_body_velocity(0.08, 0.0, 0.0, 270.0)[1], 0.08)
        self.assertAlmostEqual(compensateBodyVelocity(0.08, 0.0, 90.0, 0.0)[1], -0.08)


if __name__ == "__main__":
    unittest.main()
