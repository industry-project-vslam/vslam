from __future__ import annotations

from PyQt6.QtWidgets import QFormLayout, QGroupBox, QLabel

from swarm_controller.config import default_swarm_config


class ThresholdsPanel(QGroupBox):
    def __init__(self) -> None:
        super().__init__("Safety Thresholds")
        config = default_swarm_config()
        layout = QFormLayout(self)
        layout.addRow("Test takeoff height", QLabel(f"{config.test_takeoff_height:.2f} m"))
        layout.addRow("Mission height", QLabel(f"{config.flight_height:.2f} m"))
        layout.addRow("Initial proof step", QLabel(f"{config.initial_step:.2f} m"))
        layout.addRow("Step size", QLabel(f"{config.step_size:.2f} m"))
        layout.addRow("Initial proof speed", QLabel(f"{config.initial_speed:.2f} m/s"))
        layout.addRow("Max mission speed", QLabel(f"{config.speed:.2f} m/s"))
        layout.addRow("Target wall offset", QLabel(f"{config.target_wall_offset:.2f} m"))
        layout.addRow("Critical front", QLabel(f"{config.critical_front:.2f} m"))
        layout.addRow("Critical side", QLabel(f"{config.critical_side:.2f} m"))
        layout.addRow("Critical back", QLabel(f"{config.critical_back:.2f} m"))
        layout.addRow("Critical up", QLabel(f"{config.critical_up:.2f} m"))
        layout.addRow("Formation margin", QLabel(f"{config.formation_margin:.2f} m"))
        layout.addRow("Unknown space", QLabel("BLOCKS movement"))
        layout.addRow("Line mode", QLabel("Disabled in MVP"))
