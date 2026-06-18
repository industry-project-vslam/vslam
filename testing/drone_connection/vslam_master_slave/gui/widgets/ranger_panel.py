from __future__ import annotations

import math

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGroupBox, QLabel, QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout

from swarm_controller.controller import SwarmStatusSnapshot


class RangerPanel(QGroupBox):
    def __init__(self) -> None:
        super().__init__("Ranger Readings and Safety Envelope")
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["Drone", "Front", "Back", "Left", "Right", "Up", "ZRange", "Min", "Corrected Wall"])
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.table.setMinimumWidth(0)
        self.envelope = QLabel("Envelope: NOT_EVALUATED")
        self.envelope.setWordWrap(True)
        self.setMinimumWidth(220)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addWidget(self.envelope)

    def update_snapshot(self, snapshot: SwarmStatusSnapshot) -> None:
        self.table.setRowCount(len(snapshot.rangers))
        for row, ranger in enumerate(snapshot.rangers):
            values = [
                ranger.drone_id,
                _fmt(ranger.front),
                _fmt(ranger.back),
                _fmt(ranger.left),
                _fmt(ranger.right),
                _fmt(ranger.up),
                _fmt(ranger.zrange),
                _fmt(ranger.min_clearance),
                _fmt(ranger.corrected_front_wall_distance),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setBackground(_health_color(ranger.health))
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()
        self.envelope.setText(f"Envelope: {snapshot.envelope.state} | {snapshot.envelope.reason}")
        self.envelope.setStyleSheet(_envelope_style(snapshot.envelope.state))


def _fmt(value: float) -> str:
    return "inf" if not math.isfinite(value) else f"{value:.2f}"


def _health_color(health: str):
    from PyQt6.QtGui import QColor

    return {
        "SAFE": QColor("#d7f5dd"),
        "WARNING": QColor("#fff0b3"),
        "CRITICAL": QColor("#ffc4c4"),
    }.get(health, QColor("#eeeeee"))


def _envelope_style(state: str) -> str:
    return {
        "FREE": "background:#1b8f3a;color:white;padding:6px;",
        "UNKNOWN": "background:#c9b458;color:black;padding:6px;",
        "OCCUPIED": "background:#e8792e;color:white;padding:6px;",
        "CRITICAL": "background:#b00020;color:white;padding:6px;",
    }.get(state, "background:#eeeeee;color:black;padding:6px;")
