from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from swarm_controller.controller import SwarmStatusSnapshot

from .ai_stream_preview import AIStreamPreviewManager
from .widgets.ai_stream_panel import AIStreamPanel
from .widgets.drone_status_panel import DroneStatusPanel
from .widgets.event_log_panel import EventLogPanel
from .widgets.formation_view import FormationView
from .widgets.mission_control_panel import MissionControlPanel
from .widgets.ranger_panel import RangerPanel
from .widgets.thresholds_panel import ThresholdsPanel
from .worker import WorkerThread


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Fixed-Formation Ranger-Guided Streaming Swarm")
        self.setMinimumSize(980, 650)

        self.worker_thread = WorkerThread()
        self.worker_thread.worker.status_updated.connect(self.update_snapshot)
        self.worker_thread.worker.log_event.connect(self.append_worker_event)
        self.ai_preview_manager = AIStreamPreviewManager()
        self.ai_preview_manager.frame_received.connect(self.ai_stream_panel_frame_received)
        self.ai_preview_manager.status_changed.connect(self.ai_stream_panel_status_changed)
        self.ai_preview_manager.scan_progress.connect(self.append_worker_event)
        self.ai_preview_manager.scan_finished.connect(self.ai_stream_scan_finished)

        self.mode_label = QLabel("Mode: IDLE")
        self.state_label = QLabel("State: INIT")
        self.emergency_label = QLabel("Emergency: no")
        self.battery_label = QLabel("Battery: n/a")
        self.radio_label = QLabel("Radio: disconnected")
        self.safety_label = QLabel("Safety: inactive")
        self.setpoint_label = QLabel("Setpoint age: n/a")
        self.time_label = QLabel("Mission time: 0.0s")
        self.chunk_label = QLabel("Chunk: idle")
        self.disabled_label = QLabel("")

        self.drone_panel = DroneStatusPanel()
        self.control_panel = MissionControlPanel()
        self.thresholds_panel = ThresholdsPanel()
        self.formation_view = FormationView()
        self.ranger_panel = RangerPanel()
        self.ai_stream_panel = AIStreamPanel()
        self.event_log_panel = EventLogPanel()

        self.control_panel.command_requested.connect(self.send_command)
        self.ai_stream_panel.start_real_streams_requested.connect(self.start_real_ai_streams)
        self.ai_stream_panel.stop_real_streams_requested.connect(self.stop_real_ai_streams)
        self.ai_stream_panel.scan_requested.connect(self.scan_ai_streams)
        self._build_layout()
        self._fit_to_screen()
        self.worker_thread.start()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.ai_preview_manager.stop_streams()
        self.worker_thread.stop()
        super().closeEvent(event)

    def send_command(self, command: str) -> None:
        if command == "start_ai_streams":
            self.start_real_ai_streams(self.ai_stream_panel.build_stream_configs())
            return
        if command == "stop_ai_streams":
            self.stop_real_ai_streams()
            return
        if command == "emergency_stop":
            self.worker_thread.emergency_stop_direct()
            return
        if command == "safe_hover_land":
            self.worker_thread.safe_hover_land_direct()
            return
        if command == "hard_motor_kill":
            self.worker_thread.hard_motor_kill_direct()
            return
        self.worker_thread.enqueue(command)

    def start_real_ai_streams(self, configs: object) -> None:
        self.worker_thread.enqueue("start_ai_streams")
        stream_configs = list(configs) if isinstance(configs, list) else self.ai_stream_panel.build_stream_configs()
        self.append_worker_event("[AI streams] starting real AI-deck previews")
        self.ai_preview_manager.start_streams(stream_configs)

    def stop_real_ai_streams(self) -> None:
        self.worker_thread.enqueue("stop_ai_streams")
        self.ai_preview_manager.stop_streams()
        self.append_worker_event("[AI streams] stopped real AI-deck previews")

    def scan_ai_streams(self, expected_ssid: str, port: int) -> None:
        self.append_worker_event(f"[AI streams] scanning {expected_ssid or 'current Wi-Fi'} for port {port}")
        self.ai_preview_manager.scan_for_streams(expected_ssid, port)

    def ai_stream_scan_finished(self, hosts: object) -> None:
        host_list = list(hosts) if isinstance(hosts, list) else []
        self.ai_stream_panel.apply_discovered_hosts(host_list)
        if host_list:
            self.append_worker_event(f"[AI streams] scan finished: {', '.join(host_list)}")
        else:
            self.append_worker_event("[AI streams] scan finished: no stream endpoints found")

    def ai_stream_panel_frame_received(self, drone_id: str, image: object, fps: float, frame_id: int, frame_type: str) -> None:
        self.ai_stream_panel.update_preview_frame(drone_id, image, fps, frame_id, frame_type)

    def ai_stream_panel_status_changed(self, drone_id: str, message: str) -> None:
        self.ai_stream_panel.update_preview_status(drone_id, message)
        self.append_worker_event(f"[AI streams] {drone_id}: {message}")

    def update_snapshot(self, snapshot: SwarmStatusSnapshot) -> None:
        self.mode_label.setText(f"Mission mode: {snapshot.mission.mode}")
        self.state_label.setText(f"Current state: {snapshot.mission.state}")
        self.emergency_label.setText(f"Emergency: {'YES' if snapshot.mission.emergency else 'no'}")
        self.battery_label.setText(f"Battery: {snapshot.mission.battery_summary}")
        self.radio_label.setText(f"Radio: {snapshot.mission.radio_status}")
        self.safety_label.setText(
            f"EmergencyManager: {'ACTIVE' if snapshot.mission.emergency_manager_active else 'INACTIVE'} | "
            f"Hard kill: {'armed' if snapshot.mission.hard_kill_armed else 'spent'} | "
            f"Watchdog: {snapshot.mission.watchdog_status} | "
            f"Preflight: {'PASS' if snapshot.mission.preflight_passed else 'no'}"
        )
        age = snapshot.mission.last_setpoint_age_s
        self.setpoint_label.setText(f"Setpoint age: {'n/a' if age == float('inf') else f'{age:.2f}s'}")
        self.time_label.setText(
            f"Mission time: {snapshot.mission.mission_elapsed_s:.1f}s | "
            f"remaining {snapshot.mission.mission_remaining_s:.1f}s"
        )
        self.chunk_label.setText(f"Chunk: {snapshot.mission.chunk_progress or 'idle'}")
        reason = snapshot.mission.disabled_reason
        if snapshot.mission.auto_land_reason:
            reason = f"Auto-land: {snapshot.mission.auto_land_reason}"
        self.disabled_label.setText(reason)
        self.emergency_label.setStyleSheet("color:#b00020;font-weight:bold;" if snapshot.mission.emergency else "")

        self.drone_panel.update_snapshot(snapshot)
        self.control_panel.update_snapshot(snapshot)
        self.formation_view.update_snapshot(snapshot)
        self.ranger_panel.update_snapshot(snapshot)
        self.ai_stream_panel.update_snapshot(snapshot)
        self.event_log_panel.update_snapshot(snapshot)

    def append_worker_event(self, event: str) -> None:
        self.event_log_panel.append_event(event)

    def _build_layout(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.addLayout(self._status_bar())
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.addWidget(self.drone_panel)
        left_layout.addWidget(self.control_panel)
        left_layout.addWidget(self.thresholds_panel)
        left_layout.addStretch(1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_content)
        left_scroll.setMinimumWidth(260)
        left_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        center_tabs = QTabWidget()
        center_tabs.addTab(self.formation_view, "Formation")
        center_tabs.addTab(self.ai_stream_panel, "AI Streams")
        center_tabs.setMinimumWidth(360)
        center_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        horizontal = QSplitter(Qt.Orientation.Horizontal)
        horizontal.addWidget(left_scroll)
        horizontal.addWidget(center_tabs)
        horizontal.addWidget(self.ranger_panel)
        horizontal.setChildrenCollapsible(True)
        horizontal.setSizes([300, 620, 280])

        vertical = QSplitter(Qt.Orientation.Vertical)
        vertical.addWidget(horizontal)
        vertical.addWidget(self.event_log_panel)
        vertical.setChildrenCollapsible(False)
        vertical.setSizes([500, 210])

        root_layout.addWidget(vertical)
        self.setCentralWidget(root)

    def _status_bar(self) -> QGridLayout:
        layout = QGridLayout()
        labels = (
            self.mode_label,
            self.state_label,
            self.emergency_label,
            self.battery_label,
            self.radio_label,
            self.safety_label,
            self.setpoint_label,
            self.time_label,
            self.chunk_label,
        )
        for index, label in enumerate(labels):
            label.setStyleSheet("padding:6px;font-weight:bold;")
            label.setWordWrap(True)
            label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            layout.addWidget(label, index // 5, index % 5)
        self.disabled_label.setStyleSheet("padding:6px;color:#8a5a00;")
        self.disabled_label.setWordWrap(True)
        layout.addWidget(self.disabled_label, 1, 4)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 1)
        layout.setColumnStretch(4, 2)
        return layout

    def _fit_to_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1180, 760)
            return
        available = screen.availableGeometry()
        width = max(980, min(1280, available.width() - 60))
        height = max(650, min(820, available.height() - 70))
        self.resize(width, height)
