from __future__ import annotations

import queue
import shutil
import socket
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from .aideck_stream import AIDeckStreamClient, StreamSettings


def find_cfclient() -> str | None:
    executable = shutil.which("cfclient")
    if executable is not None:
        return executable

    python_dir = Path(sys.executable).resolve().parent
    local_executable = python_dir / "cfclient.exe"
    if local_executable.exists():
        return str(local_executable)

    return None


class AIDeckApp(ttk.Frame):
    def __init__(
        self,
        master: tk.Tk,
        host: str = "172.20.10.7",
        port: int = 5000,
        max_fps: float = 0.0,
        timeout: float = 30.0,
        save_frames: bool = False,
        raw_view: str = "color",
        display_scale: int = 2,
        debug_stream: bool = False,
        auto_open_cfclient: bool = False,
        auto_start_stream: bool = False,
    ):
        super().__init__(master, padding=16)
        self.master = master
        self.master.title("VSLAM Flight + AI-deck Camera")
        self.master.minsize(760, 420)

        self.status_queue: queue.Queue[str] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None

        self.host = tk.StringVar(value=host)
        self.port = tk.IntVar(value=port)
        self.max_fps = tk.DoubleVar(value=max_fps)
        self.timeout = tk.DoubleVar(value=timeout)
        self.save_frames = tk.BooleanVar(value=save_frames)
        self.raw_view = tk.StringVar(value=raw_view)
        self.display_scale = tk.IntVar(value=display_scale)
        self.debug_stream = tk.BooleanVar(value=debug_stream)
        self.status = tk.StringVar(value="Ready")
        self.frame_status = tk.StringVar(value="Frames: 0")

        self._build()
        self._poll_status()

        if auto_open_cfclient:
            self.after(300, self.open_crazyflie_client)
        if auto_start_stream:
            self.after(800, self.start_stream)

    def _build(self) -> None:
        self.grid(row=0, column=0, sticky="nsew")
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        self.columnconfigure(1, weight=1)

        ttk.Label(self, text="AI-deck IP").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(self, textvariable=self.host).grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(self, text="Port").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(self, textvariable=self.port).grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(self, text="Max FPS").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Spinbox(self, from_=0, to=60, textvariable=self.max_fps, increment=1).grid(
            row=2, column=1, sticky="ew", pady=4
        )

        ttk.Label(self, text="Timeout").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Spinbox(self, from_=1, to=120, textvariable=self.timeout, increment=1).grid(
            row=3, column=1, sticky="ew", pady=4
        )

        ttk.Label(self, text="Raw preview").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Combobox(
            self,
            textvariable=self.raw_view,
            values=("color", "raw", "both"),
            state="readonly",
        ).grid(row=4, column=1, sticky="ew", pady=4)

        ttk.Label(self, text="Display scale").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Spinbox(self, from_=1, to=4, textvariable=self.display_scale, increment=1).grid(
            row=5, column=1, sticky="ew", pady=4
        )

        ttk.Checkbutton(self, text="Save frames to stream_out/", variable=self.save_frames).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=8
        )
        ttk.Checkbutton(self, text="Debug stream in terminal", variable=self.debug_stream).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=4
        )

        buttons = ttk.Frame(self)
        buttons.grid(row=8, column=0, columnspan=2, sticky="ew", pady=10)
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        buttons.columnconfigure(2, weight=1)
        buttons.columnconfigure(3, weight=1)
        buttons.columnconfigure(4, weight=1)

        ttk.Button(buttons, text="Open flight GUI + Start camera", command=self.open_flight_gui_and_camera).grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(buttons, text="Start camera", command=self.start_stream).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(buttons, text="Stop camera", command=self.stop_stream).grid(
            row=0, column=2, sticky="ew", padx=6
        )
        ttk.Button(buttons, text="Open flight GUI", command=self.open_crazyflie_client).grid(
            row=0, column=3, sticky="ew", padx=6
        )
        ttk.Button(buttons, text="Test stream IP", command=self.test_stream_ip).grid(
            row=0, column=4, sticky="ew", padx=(6, 0)
        )

        ttk.Separator(self).grid(row=9, column=0, columnspan=2, sticky="ew", pady=10)
        ttk.Label(self, textvariable=self.status).grid(row=10, column=0, columnspan=2, sticky="w")
        ttk.Label(self, textvariable=self.frame_status).grid(row=11, column=0, columnspan=2, sticky="w", pady=(4, 0))

        note = (
            "Use the Crazyflie Client window for flight control. "
            "Use this window for the AI-deck camera; press q in the OpenCV image window or Stop camera here."
        )
        ttk.Label(self, text=note, wraplength=650).grid(row=12, column=0, columnspan=2, sticky="w", pady=(16, 0))

    def open_flight_gui_and_camera(self) -> None:
        self.open_crazyflie_client()
        self.start_stream()

    def start_stream(self) -> None:
        if self.worker and self.worker.is_alive():
            self.status.set("Stream is already running")
            return

        self.stop_event.clear()
        settings = StreamSettings(
            host=self.host.get(),
            port=int(self.port.get()),
            max_fps=float(self.max_fps.get()),
            timeout=float(self.timeout.get()),
            save_frames=bool(self.save_frames.get()),
            raw_view=self.raw_view.get(),
            display_scale=int(self.display_scale.get()),
            output_dir=Path("stream_out"),
            debug=bool(self.debug_stream.get()),
        )

        self.worker = threading.Thread(target=self._run_stream, args=(settings,), daemon=True)
        self.worker.start()
        self.status.set("Connecting...")

    def stop_stream(self) -> None:
        self.stop_event.set()
        self.status.set("Stopping...")

    def open_crazyflie_client(self) -> None:
        executable = find_cfclient()
        if executable is None:
            self.status.set("cfclient is not installed. Run: pip install -r requirements.txt")
            return

        try:
            subprocess.Popen([executable], cwd=Path.cwd())
            self.status.set("Crazyflie Client opened")
        except OSError as exc:
            self.status.set(f"Could not open Crazyflie Client: {exc}")

    def test_stream_ip(self) -> None:
        host = self.host.get()
        port = int(self.port.get())
        self.status.set(f"Testing {host}:{port}...")

        def worker() -> None:
            try:
                with socket.create_connection((host, port), timeout=float(self.timeout.get())):
                    self.status_queue.put(f"OK: {host}:{port} is reachable")
            except OSError as exc:
                self.status_queue.put(f"Cannot reach {host}:{port}: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _run_stream(self, settings: StreamSettings) -> None:
        client = AIDeckStreamClient(settings)

        def on_frame(count: int, fps: float, frame_type: str) -> None:
            self.status_queue.put(f"Frames: {count} | {fps:.2f} fps | {frame_type}")

        try:
            client.run(stop_event=self.stop_event, on_frame=on_frame)
            self.status_queue.put("Stopped")
        except Exception as exc:
            self.status_queue.put(f"Error: {exc}")

    def _poll_status(self) -> None:
        while not self.status_queue.empty():
            message = self.status_queue.get_nowait()
            if message.startswith("Frames:"):
                self.frame_status.set(message)
                self.status.set("Streaming")
            else:
                self.status.set(message)

        self.after(200, self._poll_status)


def launch_gui(
    host: str = "172.20.10.7",
    port: int = 5000,
    max_fps: float = 0.0,
    timeout: float = 30.0,
    save_frames: bool = False,
    raw_view: str = "color",
    display_scale: int = 2,
    debug_stream: bool = False,
    auto_open_cfclient: bool = False,
    auto_start_stream: bool = False,
) -> None:
    root = tk.Tk()
    AIDeckApp(
        root,
        host=host,
        port=port,
        max_fps=max_fps,
        timeout=timeout,
        save_frames=save_frames,
        raw_view=raw_view,
        display_scale=display_scale,
        debug_stream=debug_stream,
        auto_open_cfclient=auto_open_cfclient,
        auto_start_stream=auto_start_stream,
    )
    root.mainloop()
