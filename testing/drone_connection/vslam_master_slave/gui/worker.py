from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from swarm_controller.controller import SwarmController, SwarmStatusSnapshot


@dataclass(frozen=True)
class GuiCommand:
    name: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] | None = None


class SwarmWorker(QObject):
    status_updated = pyqtSignal(object)
    log_event = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, controller: SwarmController | None = None) -> None:
        super().__init__()
        self.controller = controller or SwarmController(simulation=True)
        self.commands: queue.Queue[GuiCommand] = queue.Queue()
        self.running = True

    def enqueue(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.commands.put(GuiCommand(name, args, kwargs))

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        while self.running:
            self._drain_commands()
            self.status_updated.emit(self.controller.get_status_snapshot())
            time.sleep(0.20)
        self.controller.close()
        self.finished.emit()

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                self._execute(command)
            except Exception as exc:  # GUI must stay alive even if backend rejects a command
                self.log_event.emit(f"ERROR {command.name}: {exc}")

    def _execute(self, command: GuiCommand) -> None:
        kwargs = command.kwargs or {}
        name = command.name
        if name == "run_simulation_mode":
            self.controller.set_simulation_mode(True, kwargs.get("scenario", "open_space"))
            self.controller.load_half_group_config()
        elif name == "run_sim_open_space":
            self.controller.set_simulation_mode(True, "open_space")
            self.controller.load_half_group_config()
        elif name == "run_sim_obstacle":
            self.controller.set_simulation_mode(True, "local_obstacle")
            self.controller.load_half_group_config()
        elif name == "run_sim_wall":
            self.controller.set_simulation_mode(True, "wall_ahead")
            self.controller.load_half_group_config()
        elif name == "run_real_mode":
            self.controller.set_simulation_mode(False)
        elif name == "load_config":
            self.controller.load_config()
        elif name == "load_half_group_config":
            self.controller.load_half_group_config()
        elif name == "connect_all":
            self.controller.connect_all()
        elif name == "run_sensor_check":
            self.controller.run_sensor_check()
        elif name == "start_ai_streams":
            self.controller.start_ai_streams()
        elif name == "stop_ai_streams":
            self.controller.stop_ai_streams()
        elif name == "takeoff_all":
            self.controller.takeoff_all()
        elif name == "takeoff_x_front":
            self.controller.takeoff_drone("X_FRONT")
        elif name == "takeoff_o1":
            self.controller.takeoff_drone("O1")
        elif name == "takeoff_o2":
            self.controller.takeoff_drone("O2")
        elif name == "run_scout_sweep":
            self.controller.run_scout_sweep()
        elif name == "run_wall_obstacle_probe_test":
            self.controller.run_wall_obstacle_probe_test()
        elif name == "run_formation_hover_only":
            self.controller.run_formation_hover_only()
        elif name == "run_formation_micro_step":
            self.controller.run_formation_micro_step()
        elif name == "start_full_observation_mode":
            self.controller.start_full_observation_mode()
        elif name == "start_observation_demo":
            self.controller.start_observation_demo()
        elif name == "pause_hover":
            self.controller.pause_hover()
        elif name == "resume":
            self.controller.resume()
        elif name == "manual_forward":
            self.controller.manual_swarm_forward()
        elif name == "manual_back":
            self.controller.manual_swarm_back()
        elif name == "manual_left":
            self.controller.manual_swarm_left()
        elif name == "manual_right":
            self.controller.manual_swarm_right()
        elif name == "manual_yaw_left":
            self.controller.manual_swarm_yaw_left()
        elif name == "manual_yaw_right":
            self.controller.manual_swarm_yaw_right()
        elif name == "land_all":
            self.controller.land_all()
        elif name == "emergency_stop":
            self.controller.hard_motor_kill()
        elif name == "safe_hover_land":
            self.controller.safe_hover_land()
        elif name == "hard_motor_kill":
            self.controller.hard_motor_kill()
        elif name == "save_logs":
            path = self.controller.save_logs()
            self.log_event.emit(f"logs saved: {path}")
        else:
            self.log_event.emit(f"unknown command: {name}")


class WorkerThread:
    def __init__(self) -> None:
        self.thread = QThread()
        self.worker = SwarmWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.worker.stop()
        self.thread.quit()
        self.thread.wait(3000)

    def enqueue(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.worker.enqueue(name, *args, **kwargs)

    def hard_motor_kill_direct(self) -> None:
        self.worker.controller.hard_motor_kill()

    def emergency_stop_direct(self) -> None:
        self.worker.controller.hard_motor_kill()

    def safe_hover_land_direct(self) -> None:
        self.worker.controller.safe_hover_land()
