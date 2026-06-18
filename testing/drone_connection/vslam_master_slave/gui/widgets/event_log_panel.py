from __future__ import annotations

from PyQt6.QtWidgets import QGroupBox, QPlainTextEdit, QVBoxLayout

from swarm_controller.controller import SwarmStatusSnapshot


class EventLogPanel(QGroupBox):
    def __init__(self) -> None:
        super().__init__("Event Log and Command Log")
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.text)

    def update_snapshot(self, snapshot: SwarmStatusSnapshot) -> None:
        self.text.setPlainText("\n".join(snapshot.events))
        self.text.verticalScrollBar().setValue(self.text.verticalScrollBar().maximum())

    def append_event(self, event: str) -> None:
        self.text.appendPlainText(event)

