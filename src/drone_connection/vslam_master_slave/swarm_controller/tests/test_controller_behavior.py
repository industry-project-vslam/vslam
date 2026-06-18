from __future__ import annotations

import unittest

from swarm_controller.config import default_half_group_config
from swarm_controller.controller import MissionMode, SwarmController
from swarm_controller.emergency import EmergencyManager
from swarm_controller.motion import MotionController
from swarm_controller.preflight import PreflightManager
from swarm_controller.ranger import RangerReading


class ControllerBehaviorSimulationTests(unittest.TestCase):
    def test_official_style_default_flight_parameters(self) -> None:
        config = default_half_group_config()

        self.assertEqual(config.test_takeoff_height, 0.30)
        self.assertEqual(config.flight_height, 0.40)
        self.assertEqual(config.takeoff_velocity, 0.30)
        self.assertEqual(config.landing_velocity, 0.20)
        self.assertEqual(config.initial_speed, 0.18)
        self.assertEqual(config.speed, 0.25)
        self.assertEqual(config.max_speed, 0.28)
        self.assertEqual(config.absolute_max_speed, 0.28)
        self.assertEqual(config.initial_step, 0.20)
        self.assertEqual(config.step_size, 0.30)
        self.assertEqual(config.max_step, 0.35)
        self.assertEqual(config.side_step, 0.30)
        self.assertEqual(config.turn_reposition_chunk, 0.30)
        self.assertEqual(config.motion_segment_s, 0.45)
        self.assertEqual(config.hover_time, 0.35)
        self.assertEqual(config.hover_after_turn, 0.80)
        self.assertEqual(config.demo_settle_time, 0.70)
        self.assertEqual(config.video_demo_steps, 12)
        self.assertEqual(config.front_recovery_target, 0.85)
        self.assertEqual(config.front_recovery_step, 0.15)
        self.assertEqual(config.front_recovery_max, 0.30)
        self.assertEqual(config.front_recovery_speed, 0.15)
        self.assertTrue(config.copy_front_recovery_to_followers)
        self.assertEqual(config.battery_warn_v, 3.10)
        self.assertEqual(config.battery_land_v, 3.00)
        self.assertEqual(config.yaw_rate_deg_s, 72.0)
        self.assertEqual(config.setpoint_hz, 20.0)

    def test_full_observation_auto_stages_sensor_streams_and_takeoff(self) -> None:
        controller = SwarmController(simulation=True, scenario="open_space")
        try:
            controller.load_half_group_config()
            controller.connect_all()

            controller.start_full_observation_mode(max_steps=1)

            events = "\n".join(controller.events)
            self.assertIn("auto sensor check before full observation", events)
            self.assertIn("AI streams marked active", events)
            self.assertIn("full observation mode started", events)
            self.assertTrue(any("takeoff" in command for drone in controller.drones.values() for command in drone.commands))
        finally:
            controller.close()

    def test_sensor_check_preserves_fake_room_ranger_readings(self) -> None:
        controller = SwarmController(simulation=True, scenario="side_unknown")
        try:
            controller.load_half_group_config()
            controller.connect_all()

            before = controller.ranger_monitor.get_front_ranger()
            controller.run_sensor_check()
            after = controller.ranger_monitor.get_front_ranger()

            self.assertFalse(before.valid["left"])
            self.assertFalse(after.valid["left"])
            self.assertTrue(controller.mode_pass["MODE_SENSOR_CHECK"])
        finally:
            controller.close()

    def test_open_space_full_observation_runs_without_emergency_or_landing(self) -> None:
        controller = self._ready_controller("open_space")
        try:
            controller.start_full_observation_mode(max_steps=2)

            events = "\n".join(controller.events)
            self.assertIn("full observation mode started", events)
            self.assertIn("Ranger big step X_FRONT forward", events)
            self.assertIn("AI big step copy forward", events)
            self.assertIn("base formation staged forward", events)
            self.assertIn("mission complete", events)
            self.assertFalse(controller.emergency)
            self.assertEqual(controller.mode, MissionMode.MISSION_COMPLETE)
            for drone in controller.drones.values():
                self.assertGreater(drone.state.z, 0.05)
                self.assertNotEqual(drone.commands[-1], "land")
        finally:
            controller.close()

    def test_single_ranger_sweep_only_flies_x_front(self) -> None:
        controller = SwarmController(simulation=True, scenario="open_space")
        try:
            controller.load_half_group_config()
            controller.connect_all()
            controller.run_sensor_check()
            controller.takeoff_drone("X_FRONT")

            controller.run_scout_sweep()

            x_front_commands = " ".join(controller.drones["X_FRONT"].commands)
            self.assertIn("takeoff:0.30,velocity=0.30", x_front_commands)
            self.assertIn("vel:0.00,0.18", x_front_commands)
            self.assertIn("land:velocity=0.20", x_front_commands)
            self.assertNotIn("takeoff", " ".join(controller.drones["O1"].commands))
            self.assertNotIn("takeoff", " ".join(controller.drones["O2"].commands))
            self.assertTrue(controller.mode_pass["MODE_SINGLE_RANGER_SWEEP"])
        finally:
            controller.close()

    def test_full_observation_needs_sensor_check_not_staged_gates(self) -> None:
        controller = SwarmController(simulation=True, scenario="local_obstacle")
        try:
            controller.load_half_group_config()
            controller.connect_all()
            can_start, reason = controller.can_start_observation()
            self.assertTrue(can_start, reason)

            controller.run_sensor_check()
            can_start, reason = controller.can_start_observation()
            self.assertTrue(can_start, reason)
        finally:
            controller.close()

    def test_observation_front_block_turns_then_group_moves_new_heading(self) -> None:
        controller = self._ready_controller("local_obstacle")
        try:
            controller.start_full_observation_mode(max_steps=1)

            events = "\n".join(controller.events)
            self.assertIn("front blocked -> AI hover, Ranger re-slot turn RIGHT", events)
            self.assertIn("Ranger re-slot plan RIGHT", events)
            self.assertIn("(32,0) -> (97,0) -> (97,45)", events)
            self.assertIn("turn RIGHT Ranger re-slot DONE", events)
            self.assertIn("post-turn synchronized formation forward", events)
            self.assertIn("continuing heading 90deg until next Ranger obstacle", events)
            self.assertNotIn("bypass finished", events)

            x_front_commands = " ".join(controller.drones["X_FRONT"].commands)
            self.assertIn("vel:0.00,-0.25", x_front_commands)
            self.assertIn("vel:-0.25,0.00", x_front_commands)
            self.assertIn("heading=90.0", x_front_commands)

            o1_commands = " ".join(controller.drones["O1"].commands)
            o2_commands = " ".join(controller.drones["O2"].commands)
            self.assertIn("yaw:180.0,rate=72.0", o1_commands)
            self.assertIn("yaw:270.0,rate=72.0", o2_commands)
            self.assertIn("heading=90.0,body=0.00,0.25,yaw=180.0", o1_commands)
            self.assertIn("heading=90.0,body=0.25,0.00,yaw=270.0", o2_commands)
        finally:
            controller.close()

    def test_observation_resumes_group_forward_after_ranger_reslot(self) -> None:
        controller = self._ready_controller("local_obstacle")
        try:
            controller.start_full_observation_mode(max_steps=2)

            events = "\n".join(controller.events)
            self.assertIn("front blocked -> AI hover, Ranger re-slot turn RIGHT", events)
            self.assertIn("turn RIGHT Ranger re-slot DONE", events)
            self.assertIn("simulation: new heading is clear after Ranger re-slot", events)
            self.assertIn("post-turn AI yaw targets applied: O1=180deg, O2=270deg", events)
            self.assertIn("post-turn next move will be synchronized", events)
            self.assertIn("post-turn synchronized formation forward", events)
            self.assertIn("forward step 1/2 completed", events)
            self.assertIn("forward step 2/2 completed", events)
            self.assertEqual(controller.formation.heading_deg, 90.0)
            self.assertEqual(controller.drones["O1"].state.yaw_deg, 180.0)
            self.assertEqual(controller.drones["O2"].state.yaw_deg, 270.0)

            x_front_commands = " ".join(controller.drones["X_FRONT"].commands)
            o1_commands = " ".join(controller.drones["O1"].commands)
            o2_commands = " ".join(controller.drones["O2"].commands)
            self.assertIn("vel:0.00,-0.25", x_front_commands)
            self.assertIn("vel:-0.25,0.00", x_front_commands)
            self.assertIn("heading=90.0", x_front_commands)
            self.assertIn("heading=90.0,body=0.00,0.25,yaw=180.0", o1_commands)
            self.assertIn("heading=90.0,body=0.25,0.00,yaw=270.0", o2_commands)
        finally:
            controller.close()

    def test_observation_front_block_left_clear_turns_west(self) -> None:
        controller = self._ready_controller("local_obstacle_left_clear")
        try:
            controller.start_full_observation_mode(max_steps=1)

            self.assertEqual(controller.formation.heading_deg, 270.0)
            slots = controller.formation.rotated_offsets()
            self.assertEqual((slots["X_FRONT"].x, slots["X_FRONT"].y), (-33.0, 45.0))
            self.assertEqual((slots["O1"].x, slots["O1"].y), (12.0, 45.0))
            self.assertEqual((slots["O2"].x, slots["O2"].y), (52.0, 45.0))
            self.assertEqual(controller.formation.intended_ai_yaws()["O1"], 0.0)
            self.assertEqual(controller.formation.intended_ai_yaws()["O2"], 90.0)
            self.assertIn("Ranger re-slot plan LEFT", "\n".join(controller.events))
        finally:
            controller.close()

    def test_front_critical_copies_backoff_then_turns_with_ai_hovering(self) -> None:
        controller = self._ready_controller("front_critical")
        try:
            controller.start_full_observation_mode(max_steps=1)

            events = "\n".join(controller.events)
            self.assertIn("front critical:", events)
            self.assertIn("front recovery backoff", events)
            self.assertIn("front recovery finished", events)
            self.assertIn("front recovery backoff: X_FRONT,O1,O2 backward", events)
            self.assertIn("front blocked -> AI hover, Ranger re-slot turn RIGHT", events)
            self.assertFalse(controller.emergency)
            self.assertEqual(controller.formation.heading_deg, 90.0)

            x_front_commands = " ".join(controller.drones["X_FRONT"].commands)
            self.assertIn("vel:-0.15,0.00", x_front_commands)
            self.assertIn("vel:0.00,-0.25", x_front_commands)

            self.assertNotIn("FORMATION_FORWARD", "\n".join(controller.events))
            for drone_id in ("O1", "O2"):
                commands = " ".join(controller.drones[drone_id].commands)
                self.assertIn("vel:-0.15,0.00", commands)
                self.assertNotIn("vel:0.00,-0.25", commands)
        finally:
            controller.close()

    def test_front_critical_with_unknown_back_still_turns_without_ai_copy(self) -> None:
        controller = self._ready_controller("front_critical_back_unknown")
        try:
            controller.start_full_observation_mode(max_steps=1)

            events = "\n".join(controller.events)
            self.assertIn("front critical:", events)
            self.assertIn("front recovery skipped: X_FRONT back Ranger unknown", events)
            self.assertIn("front blocked -> AI hover, Ranger re-slot turn RIGHT", events)
            self.assertFalse(controller.emergency)
            self.assertEqual(controller.formation.heading_deg, 90.0)

            x_front_commands = " ".join(controller.drones["X_FRONT"].commands)
            self.assertNotIn("X_FRONT_FRONT_RECOVERY_BACKOFF", x_front_commands)
            self.assertIn("vel:0.00,-0.25", x_front_commands)

            for drone_id in ("O1", "O2"):
                commands = " ".join(controller.drones[drone_id].commands)
                self.assertNotIn("vel:-0.15,0.00", commands)
                self.assertIn("heading=90.0", commands)
        finally:
            controller.close()

    def test_ranger_reslot_ignores_old_front_critical_but_not_escape_sensors(self) -> None:
        controller = self._ready_controller("front_critical")
        try:
            moved = controller._try_reslot_turn("RIGHT")

            events = "\n".join(controller.events)
            self.assertTrue(moved)
            self.assertIn("turn RIGHT safety: FREE RESLOT_PATH_FREE", events)
            self.assertIn("turn RIGHT Ranger re-slot DONE", events)
            self.assertEqual(controller.formation.heading_deg, 90.0)
            self.assertIn("vel:0.00,-0.25", " ".join(controller.drones["X_FRONT"].commands))
        finally:
            controller.close()

    def test_post_turn_front_block_replans_instead_of_sticking_in_safe_hover(self) -> None:
        controller = self._ready_controller("front_critical")
        try:
            controller.takeoff_all()
            assert controller.ranger_monitor is not None and controller.formation is not None
            controller.formation.update_heading(90.0)
            controller._post_reslot_sync_next = True
            controller.ranger_monitor.update_front(
                RangerReading(
                    front=0.50,
                    back=4.0,
                    left=3.0,
                    right=3.0,
                    up=2.5,
                    valid={"front": True, "back": True, "left": True, "right": True, "up": True},
                )
            )
            controller._seq += 1

            moved_or_replanned = controller._move_base_formation_forward()

            events = "\n".join(controller.events)
            self.assertTrue(moved_or_replanned)
            self.assertIn("post-turn synchronized move blocked by Ranger front", events)
            self.assertIn("front recovery", events)
            self.assertNotEqual(controller.state, "SAFE_HOVER")
        finally:
            controller.close()

    def test_side_blocked_enters_safe_hover_without_reslot(self) -> None:
        controller = self._ready_controller("side_blocked")
        try:
            controller.start_full_observation_mode(max_steps=1)

            events = "\n".join(controller.events)
            self.assertNotIn("Ranger re-slot plan", events)
            self.assertEqual(controller.state, "SAFE_HOVER")
            self.assertEqual(controller.formation.heading_deg, 0.0)
        finally:
            controller.close()

    def test_wall_obstacle_probe_test_is_ranger_only(self) -> None:
        controller = SwarmController(simulation=True, scenario="local_obstacle")
        try:
            controller.load_half_group_config()
            controller.connect_all()
            controller.run_sensor_check()

            controller.run_wall_obstacle_probe_test()

            events = "\n".join(controller.events)
            self.assertIn("lateral probe started mode=ranger-only", events)
            self.assertIn("vel:0.00,-0.25", " ".join(controller.drones["X_FRONT"].commands))
            self.assertNotIn("vel:", " ".join(controller.drones["O1"].commands))
            self.assertNotIn("vel:", " ".join(controller.drones["O2"].commands))
            self.assertNotIn("takeoff", " ".join(controller.drones["O1"].commands))
            self.assertNotIn("takeoff", " ".join(controller.drones["O2"].commands))
        finally:
            controller.close()

    def test_manual_swarm_arrows_move_after_takeoff(self) -> None:
        controller = SwarmController(simulation=True, scenario="open_space")
        try:
            controller.load_half_group_config()
            controller.connect_all()
            controller.run_sensor_check()
            controller.takeoff_all()

            controller.manual_swarm_forward()
            controller.manual_swarm_left()
            controller.manual_swarm_yaw_right()

            commands = "\n".join(command for drone in controller.drones.values() for command in drone.commands)
            self.assertIn("vel:0.18,0.00", commands)
            self.assertIn("vel:0.00,0.18", commands)
            self.assertIn("yaw:", commands)
        finally:
            controller.close()

    def test_takeoff_all_aligns_ranger_and_ai_camera_yaws(self) -> None:
        controller = SwarmController(simulation=True, scenario="open_space")
        try:
            controller.load_half_group_config()
            controller.connect_all()
            controller.run_sensor_check()

            controller.takeoff_all()

            self.assertEqual(controller.drones["X_FRONT"].state.yaw_deg, 0.0)
            self.assertEqual(controller.drones["O1"].state.yaw_deg, 270.0)
            self.assertEqual(controller.drones["O2"].state.yaw_deg, 0.0)
            self.assertIn("formation yaw aligned", "\n".join(controller.events))
        finally:
            controller.close()

    def test_left_ai_uses_inverted_body_lateral_when_group_moves_forward(self) -> None:
        controller = SwarmController(simulation=True, scenario="open_space")
        try:
            controller.load_half_group_config()
            controller.connect_all()
            controller.run_sensor_check()
            controller.takeoff_all()

            controller.manual_swarm_forward()

            o1_commands = " ".join(controller.drones["O1"].commands)
            o2_commands = " ".join(controller.drones["O2"].commands)
            self.assertIn("yaw:270.0", o1_commands)
            self.assertIn("vel:0.18,0.00", o1_commands)
            self.assertIn("body=-0.00,0.18", o1_commands)
            self.assertIn("yaw:0.0", o2_commands)
            self.assertIn("body=0.18,0.00", o2_commands)
        finally:
            controller.close()

    def test_formation_yaw_keeps_ai_stream_directions_relative_to_heading(self) -> None:
        controller = SwarmController(simulation=True, scenario="open_space")
        try:
            controller.load_half_group_config()
            controller.connect_all()
            controller.run_sensor_check()
            controller.takeoff_all()

            controller._motion_controller().formation_yaw(1, "RIGHT", 90.0)

            self.assertEqual(controller.formation.heading_deg, 90.0)
            self.assertEqual(controller.drones["X_FRONT"].state.yaw_deg, 270.0)
            self.assertEqual(controller.drones["O1"].state.yaw_deg, 180.0)
            self.assertEqual(controller.drones["O2"].state.yaw_deg, 270.0)
        finally:
            controller.close()

    def test_unknown_side_space_saves_frontier_and_does_not_move_forward(self) -> None:
        controller = self._ready_controller("side_unknown")
        try:
            controller.start_full_observation_mode(max_steps=3)

            mission_events = "\n".join(controller.events)
            self.assertIn("frontier saved: unknown space blocks fixed formation", mission_events)
            self.assertNotIn("base formation move forward", mission_events)
            self.assertFalse(controller.emergency)
            self.assertEqual(controller.state, "MISSION_COMPLETE")
        finally:
            controller.close()

    def test_hard_motor_kill_stops_all_connected_drones_and_blocks_motion(self) -> None:
        controller = SwarmController(simulation=True, scenario="open_space")
        try:
            controller.load_half_group_config()
            controller.connect_all()

            controller.hard_motor_kill()
            controller.mode_pass["MODE_SENSOR_CHECK"] = True
            controller.takeoff_drone("X_FRONT")

            self.assertTrue(controller.emergency_manager.emergency_event.is_set())
            self.assertTrue(controller.emergency_manager.killed)
            self.assertFalse(controller.emergency_manager.hard_kill_armed)
            for drone in controller.drones.values():
                self.assertIn("hard_kill", drone.commands)
            self.assertNotIn("takeoff", " ".join(controller.drones["X_FRONT"].commands))
        finally:
            controller.close()

    def test_safe_hover_land_lands_all_without_spending_hard_kill(self) -> None:
        controller = SwarmController(simulation=True, scenario="open_space")
        try:
            controller.load_half_group_config()
            controller.connect_all()
            for drone in controller.drones.values():
                drone.takeoff(0.2)

            controller.safe_hover_land()

            self.assertTrue(controller.emergency_manager.soft_stop_event.is_set())
            self.assertFalse(controller.emergency_manager.killed)
            self.assertTrue(controller.emergency_manager.hard_kill_armed)
            for drone in controller.drones.values():
                self.assertEqual(drone.state.z, 0.0)
                self.assertIn("safe_hover_land", drone.commands)
        finally:
            controller.close()

    def test_emergency_during_forward_motion_blocks_later_normal_setpoints(self) -> None:
        controller = self._ready_controller("open_space")
        try:
            assert controller.config is not None and controller.formation is not None and controller.logger is not None
            callback_count = 0

            def trip_emergency_after_first_setpoint() -> None:
                nonlocal callback_count
                callback_count += 1
                if callback_count == 2:
                    controller.emergency_manager.hard_kill_all(controller.drones)

            motion = MotionController(
                controller.config,
                controller.formation,
                controller.drones,
                controller.logger,
                controller._emergency_reason,
                setpoint_callback=trip_emergency_after_first_setpoint,
            )

            moved = motion.formation_forward(99, controller.config.step_size)

            self.assertFalse(moved)
            self.assertTrue(controller.emergency_manager.emergency_event.is_set())
            self._assert_no_velocity_after_hard_kill(controller)
        finally:
            controller.close()

    def test_emergency_during_ranger_reslot_stops_mid_reposition(self) -> None:
        controller = self._ready_controller("local_obstacle")
        try:
            assert controller.config is not None and controller.formation is not None and controller.logger is not None
            callback_count = 0

            def trip_emergency_mid_reslot() -> None:
                nonlocal callback_count
                callback_count += 1
                if callback_count == 3:
                    controller.emergency_manager.hard_kill_all(controller.drones)

            motion = MotionController(
                controller.config,
                controller.formation,
                controller.drones,
                controller.logger,
                controller._emergency_reason,
                setpoint_callback=trip_emergency_mid_reslot,
            )

            moved = motion.move_ranger_through_waypoints(
                100,
                "X_FRONT",
                controller.formation.reslot_waypoints("RIGHT")["X_FRONT"],
            )

            self.assertFalse(moved)
            self.assertTrue(controller.emergency_manager.emergency_event.is_set())
            self._assert_no_velocity_after_hard_kill(controller)
            self.assertNotIn("vel:", " ".join(controller.drones["O1"].commands))
            self.assertNotIn("vel:", " ".join(controller.drones["O2"].commands))
        finally:
            controller.close()

    def test_low_battery_auto_safe_lands_observation(self) -> None:
        controller = self._ready_controller("open_space")
        try:
            controller.drones["O1"].get_battery = lambda: 2.95  # type: ignore[method-assign]

            controller.start_full_observation_mode(max_steps=1)

            self.assertFalse(controller.emergency)
            self.assertIn("BATTERY_LAND_O1", controller._auto_land_reason)
            for drone in controller.drones.values():
                self.assertIn("safe_hover_land", drone.commands)
        finally:
            controller.close()

    def test_mission_timeout_auto_safe_lands_observation(self) -> None:
        controller = self._ready_controller("open_space")
        try:
            assert controller.config is not None
            controller.config.max_mission_time_s = 0.0

            controller.start_full_observation_mode(max_steps=1)

            self.assertFalse(controller.emergency)
            self.assertIn("MAX_MISSION_TIME", controller._auto_land_reason)
            for drone in controller.drones.values():
                self.assertIn("safe_hover_land", drone.commands)
        finally:
            controller.close()

    def test_flow_missing_blocks_real_preflight(self) -> None:
        config = default_half_group_config()
        drones = self._snapshot_drones(config)
        drones["X_FRONT"].params["deck.bcFlow2"] = "0"

        result = PreflightManager(config, EmergencyManager()).run(drones, real_flight_confirm=True, simulation=False)

        self.assertFalse(result.passed)
        self.assertIn("deck_bcFlow2", result.reason)

    def test_front_ranger_invalid_blocks_real_preflight(self) -> None:
        config = default_half_group_config()
        drones = self._snapshot_drones(config)
        drones["X_FRONT"].values["range.front"] = 0.0

        result = PreflightManager(config, EmergencyManager()).run(drones, real_flight_confirm=True, simulation=False)

        self.assertFalse(result.passed)
        self.assertIn("x_front_ranger_valid", result.reason)

    def test_real_ground_sensor_values_from_gui_pass_preflight(self) -> None:
        config = default_half_group_config()
        drones = {
            "X_FRONT": SnapshotDrone(
                config.drones[0],
                {
                    "pm.vbat": 4.09,
                    "stateEstimate.z": 0.01,
                    "range.zrange": 10.0,
                    "range.front": 1670.0,
                    "range.left": 880.0,
                    "range.right": 32767.0,
                    "range.back": 32767.0,
                    "range.up": 3090.0,
                },
            ),
            "O1": SnapshotDrone(
                config.drones[1],
                {
                    "pm.vbat": 4.12,
                    "stateEstimate.z": 0.01,
                    "range.zrange": 10.0,
                },
            ),
            "O2": SnapshotDrone(
                config.drones[2],
                {
                    "pm.vbat": 4.06,
                    "stateEstimate.z": 0.01,
                    "range.zrange": 10.0,
                },
            ),
        }
        result = PreflightManager(config, EmergencyManager()).run(drones, real_flight_confirm=True, simulation=False)

        self.assertTrue(result.passed, result.reason)
        self.assertIsNotNone(result.front_ranger)
        self.assertTrue(result.front_ranger.valid["right"])
        self.assertTrue(result.front_ranger.valid["back"])

    def test_zrange_zero_is_warning_not_sensor_check_failure(self) -> None:
        config = default_half_group_config()
        drones = {
            "X_FRONT": SnapshotDrone(
                config.drones[0],
                {
                    "pm.vbat": 4.09,
                    "stateEstimate.z": 0.00,
                    "range.zrange": 0.0,
                    "range.front": 1670.0,
                    "range.left": 880.0,
                    "range.right": 32767.0,
                    "range.back": 32767.0,
                    "range.up": 3090.0,
                },
            ),
            "O1": SnapshotDrone(config.drones[1], {"pm.vbat": 4.12, "stateEstimate.z": 0.01, "range.zrange": 10.0}),
            "O2": SnapshotDrone(config.drones[2], {"pm.vbat": 4.06, "stateEstimate.z": 0.00, "range.zrange": 0.0}),
        }

        result = PreflightManager(config, EmergencyManager()).run(drones, real_flight_confirm=True, simulation=False)

        self.assertTrue(result.passed, result.reason)
        x_front = next(drone for drone in result.drones if drone.drone_id == "X_FRONT")
        flow_check = next(check for check in x_front.checks if check.name == "flow_zrange")
        self.assertTrue(flow_check.passed)
        self.assertIn("warning", flow_check.reason)

    def _assert_no_velocity_after_hard_kill(self, controller: SwarmController) -> None:
        for drone in controller.drones.values():
            if "hard_kill" not in drone.commands:
                continue
            kill_index = drone.commands.index("hard_kill")
            after_kill = drone.commands[kill_index + 1 :]
            self.assertFalse(any(command.startswith("vel:") for command in after_kill), drone.commands)

    def _snapshot_drones(self, config):
        return {
            "X_FRONT": SnapshotDrone(
                config.drones[0],
                {
                    "pm.vbat": 4.09,
                    "stateEstimate.z": 0.01,
                    "range.zrange": 10.0,
                    "range.front": 1670.0,
                    "range.left": 880.0,
                    "range.right": 32767.0,
                    "range.back": 32767.0,
                    "range.up": 3090.0,
                },
            ),
            "O1": SnapshotDrone(config.drones[1], {"pm.vbat": 4.12, "stateEstimate.z": 0.01, "range.zrange": 10.0}),
            "O2": SnapshotDrone(config.drones[2], {"pm.vbat": 4.06, "stateEstimate.z": 0.01, "range.zrange": 10.0}),
        }

    def _ready_controller(self, scenario: str) -> SwarmController:
        controller = SwarmController(simulation=True, scenario=scenario)
        controller.load_half_group_config()
        controller.connect_all()
        controller.start_ai_streams()
        controller.run_sensor_check()
        can_start, reason = controller.can_start_observation()
        self.assertTrue(can_start, reason)
        return controller


class SnapshotDrone:
    def __init__(self, config, values: dict[str, float], params: dict[str, str] | None = None) -> None:
        self.config = config
        self.values = values
        self.params = {"deck.bcFlow2": "1"}
        if params is not None:
            self.params.update(params)

    def read_log_snapshot(self, variables: list[str]) -> dict[str, float]:
        return {variable: self.values[variable] for variable in variables if variable in self.values}

    def get_battery(self) -> float:
        return self.values.get("pm.vbat", 0.0)

    def read_param(self, name: str) -> str | None:
        return self.params.get(name)


if __name__ == "__main__":
    unittest.main()
