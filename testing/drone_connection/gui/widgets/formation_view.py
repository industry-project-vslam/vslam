from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QGroupBox, QSizePolicy, QVBoxLayout, QWidget

from swarm_controller.controller import SwarmStatusSnapshot


class FormationView(QGroupBox):
    def __init__(self) -> None:
        super().__init__("Fixed Formation View")
        self.canvas = _FormationCanvas()
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def update_snapshot(self, snapshot: SwarmStatusSnapshot) -> None:
        self.canvas.snapshot = snapshot
        self.canvas.update()


class _FormationCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.snapshot: SwarmStatusSnapshot | None = None
        self.setMinimumSize(280, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#111820"))
        if self.snapshot is None:
            painter.setPen(QColor("white"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load config to show formation")
            return

        pad_w = 120.0
        pad_h = 118.0
        all_x = [0.0, pad_w] + [point[0] for point in self.snapshot.formation_slots.values()]
        all_y = [0.0, pad_h] + [point[1] for point in self.snapshot.formation_slots.values()]
        min_x, max_x = min(all_x) - 8.0, max(all_x) + 8.0
        min_y, max_y = min(all_y) - 8.0, max(all_y) + 8.0
        scale = min((self.width() - 40.0) / max(1.0, max_x - min_x), (self.height() - 50.0) / max(1.0, max_y - min_y))

        def map_point(x: float, y: float) -> QPointF:
            return QPointF(20.0 + (x - min_x) * scale, 30.0 + (y - min_y) * scale)

        pad_rect = QRectF(map_point(0.0, 0.0), map_point(pad_w, pad_h))
        painter.setPen(QPen(QColor("#4f5d75"), 1))
        painter.drawRect(pad_rect)

        center = map_point(32.0, 45.0)
        heading = math.radians(self.snapshot.mission.heading_deg)
        arrow = QPointF(center.x() + math.sin(heading) * 45.0, center.y() - math.cos(heading) * 45.0)
        painter.setPen(QPen(QColor("#7bdff2"), 3))
        painter.drawLine(center, arrow)
        painter.drawText(arrow + QPointF(6, -6), "heading")

        for drone_id, (x, y) in self.snapshot.formation_slots.items():
            point = map_point(x, y)
            px = point.x()
            py = point.y()
            is_ranger = drone_id.startswith("X_")
            color = QColor("#ffb703") if is_ranger else QColor("#8ecae6")
            painter.setBrush(color)
            painter.setPen(QPen(QColor("white"), 2))
            painter.drawEllipse(QRectF(px - 18, py - 18, 36, 36))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(QRectF(px - 30, py - 34, 60, 18), Qt.AlignmentFlag.AlignCenter, drone_id)

            if drone_id in self.snapshot.ai_yaws:
                yaw = math.radians(self.snapshot.ai_yaws[drone_id])
                end = QPointF(px + math.sin(yaw) * 34.0, py - math.cos(yaw) * 34.0)
                painter.setPen(QPen(QColor("#b9fbc0"), 2))
                painter.drawLine(QPointF(px, py), end)

        painter.setPen(QColor("#d8dee9"))
        painter.drawText(12, 22, "Top-down launch model: x right, y down, NORTH toward smaller y. AI arrows show stream directions.")
