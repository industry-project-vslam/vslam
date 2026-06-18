from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QGridLayout, QGroupBox, QPushButton

from swarm_controller.controller import SwarmStatusSnapshot


class MissionControlPanel(QGroupBox):
    command_requested = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__("Mission Controls")
        self.buttons: dict[str, QPushButton] = {}
        layout = QGridLayout(self)
        labels = [
            ("load_config", "Load Config"),
            ("load_half_group_config", "Load Half Group (3)"),
            ("run_real_mode", "Use Real Crazyflies"),
            ("connect_all", "Connect All"),
            ("run_sensor_check", "Sensor Check"),
            ("start_ai_streams", "Start AI Streams"),
            ("stop_ai_streams", "Stop AI Streams"),
            ("takeoff_all", "Takeoff All"),
            ("takeoff_x_front", "Test Takeoff X_FRONT"),
            ("takeoff_o1", "Test Takeoff O1"),
            ("takeoff_o2", "Test Takeoff O2"),
            ("run_scout_sweep", "Run Scout Sweep"),
            ("run_wall_obstacle_probe_test", "Wall/Obstacle Probe Test"),
            ("run_formation_hover_only", "Formation Hover Only"),
            ("run_formation_micro_step", "Formation Micro Step"),
            ("start_observation_demo", "Start Video Demo (Top-3)"),
            ("start_full_observation_mode", "Start Full Observation Mode"),
            ("manual_forward", "Forward"),
            ("manual_back", "Back"),
            ("manual_left", "Left"),
            ("manual_right", "Right"),
            ("manual_yaw_left", "Yaw Left"),
            ("manual_yaw_right", "Yaw Right"),
            ("pause_hover", "Pause / Hover"),
            ("resume", "Resume"),
            ("land_all", "Land All"),
            ("emergency_stop", "EMERGENCY STOP / HARD KILL"),
            ("safe_hover_land", "SAFE HOVER / LAND"),
            ("hard_motor_kill", "HARD MOTOR KILL"),
            ("save_logs", "Save Logs"),
            ("run_sim_open_space", "Sim Open"),
            ("run_sim_obstacle", "Sim Obstacle"),
            ("run_sim_wall", "Sim Wall"),
            ("run_simulation_mode", "Run Simulation Mode"),
        ]
        for index, (command, text) in enumerate(labels):
            button = QPushButton(text)
            button.clicked.connect(lambda _checked=False, name=command: self.command_requested.emit(name))
            self.buttons[command] = button
            layout.addWidget(button, index // 2, index % 2)

        self.buttons["emergency_stop"].setEnabled(True)
        self.buttons["emergency_stop"].setStyleSheet("background:#d62828;color:white;font-weight:bold;")
        self.buttons["hard_motor_kill"].setEnabled(True)
        self.buttons["hard_motor_kill"].setStyleSheet("background:#b00020;color:white;font-weight:bold;")
        self.buttons["safe_hover_land"].setEnabled(True)
        self.buttons["safe_hover_land"].setStyleSheet("background:#ffb703;color:#111;font-weight:bold;")

    def update_snapshot(self, snapshot: SwarmStatusSnapshot) -> None:
        start = self.buttons["start_full_observation_mode"]
        start.setEnabled(snapshot.mission.start_observation_enabled)
        start.setToolTip(snapshot.mission.disabled_reason or "Ready")
        demo = self.buttons["start_observation_demo"]
        demo.setEnabled(True)
        demo.setToolTip(
            "Runs the filmable half-swarm sequence: staged takeoff, AI yaw alignment, "
            "Ranger scout step, AI copy step, re-slot turn, then continued movement "
            "in the new heading until the next Ranger obstacle or demo budget."
        )
        self.buttons["emergency_stop"].setEnabled(True)
        self.buttons["hard_motor_kill"].setEnabled(True)
        self.buttons["safe_hover_land"].setEnabled(True)
        self.buttons["land_all"].setEnabled(bool(snapshot.drones))
        self.buttons["resume"].setEnabled(snapshot.mission.mode == "PAUSE_HOVER")
