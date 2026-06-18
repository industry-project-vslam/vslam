from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.ai_stream_preview import StreamPreviewConfig
from swarm_controller.controller import SwarmStatusSnapshot


STREAM_ROWS = [
    ("O1", "left"),
    ("O2", "forward"),
    ("O3", "backward"),
    ("O4", "right"),
]


class AIStreamPanel(QGroupBox):
    start_real_streams_requested = pyqtSignal(object)
    stop_real_streams_requested = pyqtSignal()
    scan_requested = pyqtSignal(str, int)

    def __init__(self) -> None:
        super().__init__("AI-deck Streams")
        self.host_edits: dict[str, QLineEdit] = {}
        self.port_edits: dict[str, QSpinBox] = {}
        self.enabled_checks: dict[str, QCheckBox] = {}
        self.preview_labels: dict[str, QLabel] = {}
        self.preview_status: dict[str, QLabel] = {}
        self.preview_images: dict[str, QImage] = {}

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Drone", "Direction", "Backend Active", "FPS", "Last Frame"])
        self.table.setMaximumHeight(150)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.expected_wifi = QLineEdit("swarming")
        self.default_port = QSpinBox()
        self.default_port.setRange(1, 65535)
        self.default_port.setValue(5000)
        self.max_fps = QDoubleSpinBox()
        self.max_fps.setRange(0.0, 60.0)
        self.max_fps.setSingleStep(0.5)
        self.max_fps.setValue(0.0)
        self.timeout_s = QDoubleSpinBox()
        self.timeout_s.setRange(1.0, 30.0)
        self.timeout_s.setSingleStep(1.0)
        self.timeout_s.setValue(8.0)
        self.output_root = QLineEdit("stream_out/gui_ai_streams")
        self.save_frames = QCheckBox("Save frames")
        self.save_frames.setChecked(True)
        self.debug_stream = QCheckBox("Debug stream")

        self.scan_button = QPushButton("Scan swarming Wi-Fi for AI decks")
        self.start_button = QPushButton("Start Real AI-deck Streams")
        self.stop_button = QPushButton("Stop Real AI-deck Streams")
        self.scan_status = QLabel("Connect laptop and AI decks to swarming, then scan or enter IPs manually.")
        self.scan_status.setWordWrap(True)

        self.scan_button.clicked.connect(
            lambda: self.scan_requested.emit(self.expected_wifi.text().strip(), int(self.default_port.value()))
        )
        self.start_button.clicked.connect(lambda: self.start_real_streams_requested.emit(self.build_stream_configs()))
        self.stop_button.clicked.connect(self.stop_real_streams_requested.emit)

        self._build_layout()

    def build_stream_configs(self) -> list[StreamPreviewConfig]:
        output_root = Path(self.output_root.text().strip() or "stream_out/gui_ai_streams")
        configs: list[StreamPreviewConfig] = []
        for drone_id, direction in STREAM_ROWS:
            configs.append(
                StreamPreviewConfig(
                    drone_id=drone_id,
                    stream_direction=direction,
                    host=self.host_edits[drone_id].text().strip(),
                    port=int(self.port_edits[drone_id].value()),
                    max_fps=float(self.max_fps.value()),
                    timeout=float(self.timeout_s.value()),
                    save_frames=self.save_frames.isChecked(),
                    output_dir=output_root,
                    debug=self.debug_stream.isChecked(),
                    enabled=self.enabled_checks[drone_id].isChecked(),
                )
            )
        return configs

    def apply_discovered_hosts(self, hosts: list[str]) -> None:
        free = list(hosts)
        for drone_id, _direction in STREAM_ROWS:
            if not self.enabled_checks[drone_id].isChecked():
                continue
            if self.host_edits[drone_id].text().strip():
                continue
            if not free:
                break
            self.host_edits[drone_id].setText(free.pop(0))
            self.port_edits[drone_id].setValue(int(self.default_port.value()))
        self.scan_status.setText(
            "No AI-deck streams found on this Wi-Fi."
            if not hosts
            else f"Found candidates: {', '.join(hosts)}. Verify O1/O2 mapping before flight."
        )

    def append_scan_status(self, message: str) -> None:
        self.scan_status.setText(message)

    def update_preview_status(self, drone_id: str, message: str) -> None:
        label = self.preview_status.get(drone_id)
        if label is not None:
            label.setText(message)

    def update_preview_frame(self, drone_id: str, image: QImage, fps: float, frame_id: int, frame_type: str) -> None:
        label = self.preview_labels.get(drone_id)
        if label is None:
            return
        self.preview_images[drone_id] = image
        pixmap = QPixmap.fromImage(image)
        label.setPixmap(
            pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.update_preview_status(drone_id, f"live {frame_type} frame={frame_id} fps={fps:.1f}")

    def update_snapshot(self, snapshot: SwarmStatusSnapshot) -> None:
        self.table.setRowCount(len(snapshot.ai_streams))
        for row, stream in enumerate(snapshot.ai_streams):
            last = time.strftime("%H:%M:%S", time.localtime(stream.last_frame_timestamp)) if stream.last_frame_timestamp else "n/a"
            values = [
                stream.drone_id,
                stream.stream_direction,
                "active" if stream.active else "inactive",
                f"{stream.fps:.1f}",
                last,
            ]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self._connection_controls())
        stream_scroll = QScrollArea()
        stream_scroll.setWidgetResizable(True)
        stream_scroll.setMinimumHeight(220)
        stream_scroll.setWidget(self._stream_grid())
        layout.addWidget(stream_scroll, 1)
        layout.addWidget(QLabel("Backend mission stream markers"))
        layout.addWidget(self.table)

    def _connection_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        form = QFormLayout()
        form.addRow("Expected Wi-Fi SSID", self.expected_wifi)
        form.addRow("Default stream port", self.default_port)
        form.addRow("Preview max FPS (0 = no local drop)", self.max_fps)
        form.addRow("Socket timeout s", self.timeout_s)
        form.addRow("Output root", self.output_root)
        layout.addLayout(form)

        checks = QHBoxLayout()
        checks.addWidget(self.save_frames)
        checks.addWidget(self.debug_stream)
        checks.addStretch(1)
        layout.addLayout(checks)

        buttons = QHBoxLayout()
        buttons.addWidget(self.scan_button)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        layout.addLayout(buttons)
        layout.addWidget(self.scan_status)
        return panel

    def _stream_grid(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        headers = ["Use", "Drone", "Direction", "AI-deck IP", "Port", "Preview"]
        for col, header in enumerate(headers):
            label = QLabel(header)
            label.setStyleSheet("font-weight:bold;")
            layout.addWidget(label, 0, col)

        for row, (drone_id, direction) in enumerate(STREAM_ROWS, start=1):
            enabled = QCheckBox()
            enabled.setChecked(drone_id in {"O1", "O2"})
            host = QLineEdit()
            host.setPlaceholderText("AI-deck IP, e.g. 192.168.x.x")
            host.setMinimumWidth(110)
            port = QSpinBox()
            port.setRange(1, 65535)
            port.setValue(5000)
            preview_box = self._preview_box(drone_id)

            self.enabled_checks[drone_id] = enabled
            self.host_edits[drone_id] = host
            self.port_edits[drone_id] = port

            layout.addWidget(enabled, row, 0)
            layout.addWidget(QLabel(drone_id), row, 1)
            layout.addWidget(QLabel(direction), row, 2)
            layout.addWidget(host, row, 3)
            layout.addWidget(port, row, 4)
            layout.addWidget(preview_box, row, 5)
        return panel

    def _preview_box(self, drone_id: str) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        image = QLabel("no stream")
        image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image.setMinimumSize(140, 90)
        image.setMaximumSize(220, 140)
        image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        image.setStyleSheet("background:#111;color:#bbb;border:1px solid #555;")
        status = QLabel("inactive")
        status.setWordWrap(True)
        self.preview_labels[drone_id] = image
        self.preview_status[drone_id] = status
        layout.addWidget(image)
        layout.addWidget(status)
        return panel
