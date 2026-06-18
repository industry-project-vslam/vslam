from __future__ import annotations

import math
import unittest

from swarm_controller.ai_streams import AIStreamManager
from swarm_controller.classifier import classify_surface_probe
from swarm_controller.config import DroneRole, default_half_group_config, default_swarm_config
from swarm_controller.controller import SwarmController
from swarm_controller.drones import formation_velocity_to_body
from swarm_controller.formation import FormationModel
from swarm_controller.ranger import RangerReading
from swarm_controller.safety import (
    EnvelopeState,
    PlannedPrimitive,
    bypass_envelope_free,
    evaluate_bypass_envelope,
    evaluate_formation_envelope,
    evaluate_reslot_path_safety,
)


def valid_reading(**values: float) -> RangerReading:
    payload = {
        "front": values.get("front", 4.0),
        "back": values.get("back", 4.0),
        "left": values.get("left", 3.0),
        "right": values.get("right", 3.0),
        "up": values.get("up", 2.0),
    }
    return RangerReading(**payload, valid={key: True for key in payload})


class FixedFormationSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = default_swarm_config()
        self.formation = FormationModel(self.config.drones, formation_config=self.config.formation)

    def test_exact_ranger_slots_by_heading(self) -> None:
        expected = {
            0.0: {"X_FRONT": (32.0, 0.0), "O1": (12.0, 45.0), "O2": (52.0, 45.0)},
            90.0: {"X_FRONT": (97.0, 45.0), "O1": (12.0, 45.0), "O2": (52.0, 45.0)},
            180.0: {"X_FRONT": (32.0, 90.0), "O1": (12.0, 45.0), "O2": (52.0, 45.0)},
            270.0: {"X_FRONT": (-33.0, 45.0), "O1": (12.0, 45.0), "O2": (52.0, 45.0)},
        }
        for heading, slots in expected.items():
            self.formation.update_heading(heading)
            actual = self.formation.rotated_offsets()
            for drone_id, point in slots.items():
                self.assertEqual((actual[drone_id].x, actual[drone_id].y), point)

    def test_ranger_reslot_waypoints_avoid_ai_core(self) -> None:
        self.formation.update_heading(0.0)
        waypoints = self.formation.reslot_waypoints("RIGHT")
        self.assertEqual([(p.x, p.y) for p in waypoints["X_FRONT"]], [(32.0, 0.0), (97.0, 0.0), (97.0, 45.0)])
        ai_box = (12.0, 45.0, 52.0, 45.0)
        for point in waypoints["X_FRONT"]:
            inside_ai_row = ai_box[0] <= point.x <= ai_box[2] and point.y == ai_box[1]
            self.assertFalse(inside_ai_row)

    def test_wall_vs_obstacle_classification(self) -> None:
        obstacle = classify_surface_probe(self.config, 3.5, 4.3, 0.25)
        self.assertEqual(obstacle.classification, "OBSTACLE_OR_OPENING")

        wall = classify_surface_probe(self.config, 3.5, 3.55, self.config.max_probe_shift)
        self.assertEqual(wall.classification, "WALL_OR_BOUNDARY")

        ambiguous = classify_surface_probe(self.config, 3.5, 3.7, 0.25)
        self.assertEqual(ambiguous.classification, "AMBIGUOUS")

    def test_unknown_space_is_unsafe(self) -> None:
        front = RangerReading(
            front=4.0,
            back=4.0,
            left=math.nan,
            right=3.0,
            up=2.0,
            valid={"front": True, "back": True, "left": False, "right": True, "up": True},
        )
        back = valid_reading()
        envelope = evaluate_formation_envelope(
            self.config,
            self.formation,
            front,
            back,
            PlannedPrimitive("FORMATION_FORWARD", self.config.step_size),
        )
        self.assertEqual(envelope.state, EnvelopeState.UNKNOWN)
        self.assertFalse(envelope.allowed)

    def test_bypass_allowed_only_when_envelope_free(self) -> None:
        free = evaluate_bypass_envelope(
            self.config,
            self.formation,
            valid_reading(),
            valid_reading(),
            "LEFT",
            self.config.side_step,
            self.config.step_size * 2.0,
        )
        blocked = evaluate_bypass_envelope(
            self.config,
            self.formation,
            valid_reading(left=0.4),
            valid_reading(),
            "LEFT",
            self.config.side_step,
            self.config.step_size * 2.0,
        )
        self.assertTrue(bypass_envelope_free(free))
        self.assertFalse(bypass_envelope_free(blocked))

    def test_reslot_path_safety_can_block_one_turn_side(self) -> None:
        right_blocked = evaluate_reslot_path_safety(
            self.config,
            self.formation,
            valid_reading(right=0.6, left=3.0),
            valid_reading(left=0.6, right=3.0),
            "RIGHT",
        )
        left_free = evaluate_reslot_path_safety(
            self.config,
            self.formation,
            valid_reading(right=0.6, left=3.0),
            valid_reading(left=0.6, right=3.0),
            "LEFT",
        )
        self.assertNotEqual(right_blocked.state, EnvelopeState.FREE)
        self.assertEqual(left_free.state, EnvelopeState.FREE)

    def test_half_group_reslot_safety_uses_front_ranger_geometry(self) -> None:
        config = default_half_group_config()
        formation = FormationModel(config.drones, formation_config=config.formation)

        free = evaluate_reslot_path_safety(
            config,
            formation,
            valid_reading(right=2.0, left=2.0),
            None,
            "RIGHT",
        )
        blocked = evaluate_reslot_path_safety(
            config,
            formation,
            valid_reading(right=0.8, left=2.0),
            None,
            "RIGHT",
        )

        self.assertEqual(free.state, EnvelopeState.FREE)
        self.assertAlmostEqual(free.required["x_front_leg1_right"], 1.15)
        self.assertAlmostEqual(free.required["x_front_leg2_back"], 0.95)
        self.assertNotIn("x_back_left", free.required)
        self.assertEqual(blocked.state, EnvelopeState.OCCUPIED)

    def test_front_wall_does_not_block_sideways_reslot_escape(self) -> None:
        config = default_half_group_config()
        formation = FormationModel(config.drones, formation_config=config.formation)

        result = evaluate_reslot_path_safety(
            config,
            formation,
            valid_reading(front=0.40, right=2.0, back=2.0, up=2.0),
            None,
            "RIGHT",
        )

        self.assertEqual(result.state, EnvelopeState.FREE)
        self.assertNotIn("front", result.measured)

    def test_half_group_reslot_safety_checks_each_leg_direction(self) -> None:
        config = default_half_group_config()
        formation = FormationModel(config.drones, formation_config=config.formation)

        back_blocked = evaluate_reslot_path_safety(
            config,
            formation,
            valid_reading(right=3.0, back=0.60),
            None,
            "RIGHT",
        )

        self.assertEqual(back_blocked.state, EnvelopeState.OCCUPIED)
        self.assertIn("X_FRONT_LEG2_BACK", back_blocked.reason)

    def test_turn_updates_heading_and_ai_camera_yaws(self) -> None:
        self.formation.update_heading(90.0)
        yaws = self.formation.intended_ai_yaws()
        self.assertEqual(yaws["O1"], 180.0)
        self.assertEqual(yaws["O2"], 270.0)

    def test_formation_velocity_compensates_individual_camera_yaw(self) -> None:
        vx_body, vy_body = formation_velocity_to_body(0.15, 0.0, formation_heading_deg=0.0, drone_yaw_deg=270.0)
        self.assertAlmostEqual(vx_body, 0.0, places=6)
        self.assertAlmostEqual(vy_body, 0.15, places=6)

        vx_body, vy_body = formation_velocity_to_body(0.15, 0.0, formation_heading_deg=90.0, drone_yaw_deg=0.0)
        self.assertAlmostEqual(vx_body, 0.0, places=6)
        self.assertAlmostEqual(vy_body, -0.15, places=6)

        vx_body, vy_body = formation_velocity_to_body(0.0, 0.15, formation_heading_deg=0.0, drone_yaw_deg=90.0)
        self.assertAlmostEqual(vx_body, 0.15, places=6)
        self.assertAlmostEqual(vy_body, 0.0, places=6)

    def test_ai_streams_do_not_affect_navigation(self) -> None:
        streams = AIStreamManager()
        ai_drones = [drone for drone in self.config.drones if drone.role.name.startswith("AI_")]
        events = streams.start_streams(ai_drones)
        envelope = evaluate_formation_envelope(
            self.config,
            self.formation,
            valid_reading(),
            valid_reading(),
            PlannedPrimitive("FORMATION_FORWARD", self.config.step_size),
        )
        self.assertEqual(len(events), 2)
        self.assertEqual(envelope.state, EnvelopeState.FREE)

    def test_half_group_uses_three_previous_addresses(self) -> None:
        config = default_half_group_config()
        self.assertFalse(config.requires_back_ranger)
        self.assertEqual([drone.drone_id for drone in config.drones], ["X_FRONT", "O1", "O2"])
        self.assertEqual(config.drones[0].uri, "radio://0/82/2M/E7E7E7E701")
        self.assertEqual(config.drones[1].uri, "radio://0/82/2M/E7E7E7E702")
        self.assertEqual(config.drones[2].uri, "radio://0/82/2M/E7E7E7E703")

    def test_half_group_gui_controller_auto_runs_sensor_check_before_observation(self) -> None:
        controller = SwarmController(simulation=True)
        try:
            controller.load_half_group_config()
            controller.connect_all()
            can_start, reason = controller.can_start_observation()
            self.assertTrue(can_start, reason)

            controller.start_full_observation_mode(max_steps=1)

            self.assertTrue(controller.mode_pass["MODE_SENSOR_CHECK"])
            self.assertIn("auto sensor check before full observation", "\n".join(controller.events))
        finally:
            controller.close()


if __name__ == "__main__":
    unittest.main()
