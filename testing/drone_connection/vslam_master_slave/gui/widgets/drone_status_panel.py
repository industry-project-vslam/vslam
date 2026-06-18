from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGroupBox, QTableWidget, QTableWidgetItem, QVBoxLayout

from swarm_controller.controller import SwarmStatusSnapshot


class DroneStatusPanel(QGroupBox):
    def __init__(self) -> None:
        super().__init__("Drone Connections")
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["ID", "Role", "URI", "Connected", "Airborne", "Battery"])
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.table.setMinimumWidth(0)
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)

    def update_snapshot(self, snapshot: SwarmStatusSnapshot) -> None:
        self.table.setRowCount(len(snapshot.drones))
        for row, drone in enumerate(snapshot.drones):
            values = [
                drone.drone_id,
                drone.role,
                drone.uri or "(simulation/blank)",
                "yes" if drone.connected else "no",
                "yes" if drone.airborne else "no",
                f"{drone.battery_v:.2f} V" if drone.battery_v else "n/a",
            ]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()
