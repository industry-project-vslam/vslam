from __future__ import annotations

import concurrent.futures
import contextlib
import ipaddress
import socket
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QImage

from src.aideck_stream import AIDeckStreamClient, StreamSettings


@dataclass(frozen=True)
class StreamPreviewConfig:
    drone_id: str
    stream_direction: str
    host: str
    port: int = 5000
    max_fps: float = 0.0
    timeout: float = 8.0
    save_frames: bool = True
    output_dir: Path = Path("stream_out/gui_ai_streams")
    debug: bool = False
    enabled: bool = True


class AIStreamPreviewManager(QObject):
    frame_received = pyqtSignal(str, object, float, int, str)
    status_changed = pyqtSignal(str, str)
    scan_progress = pyqtSignal(str)
    scan_finished = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._stop_events: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._clients: dict[str, AIDeckStreamClient] = {}

    def start_streams(self, configs: list[StreamPreviewConfig]) -> None:
        self.stop_streams()
        started = 0
        for config in configs:
            if not config.enabled:
                continue
            if not config.host.strip():
                self.status_changed.emit(config.drone_id, "IP empty; stream not started")
                continue
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._run_stream,
                args=(config, stop_event),
                name=f"ai-stream-{config.drone_id}",
                daemon=True,
            )
            with self._lock:
                self._stop_events[config.drone_id] = stop_event
                self._threads[config.drone_id] = thread
            thread.start()
            started += 1
        if started == 0:
            self.scan_progress.emit("No AI stream started. Fill at least one AI-deck IP or run scan first.")

    def stop_streams(self) -> None:
        with self._lock:
            events = list(self._stop_events.items())
            clients = list(self._clients.items())
            threads = list(self._threads.items())
            self._stop_events.clear()
            self._clients.clear()
            self._threads.clear()

        for _drone_id, stop_event in events:
            stop_event.set()
        for _drone_id, client in clients:
            with contextlib.suppress(Exception):
                client.close()
        for _drone_id, thread in threads:
            if thread.is_alive():
                thread.join(timeout=0.2)

    def scan_for_streams(self, expected_ssid: str, port: int) -> None:
        thread = threading.Thread(
            target=self._scan_worker,
            args=(expected_ssid, port),
            name="ai-stream-scan",
            daemon=True,
        )
        thread.start()

    def _run_stream(self, config: StreamPreviewConfig, stop_event: threading.Event) -> None:
        output_dir = config.output_dir / config.drone_id
        settings = StreamSettings(
            host=config.host.strip(),
            port=config.port,
            max_fps=config.max_fps,
            timeout=config.timeout,
            save_frames=config.save_frames,
            output_dir=output_dir,
            raw_view="color",
            debug=config.debug,
            display_scale=1,
            window_name=f"{config.drone_id} AI-deck stream",
            show_window=False,
        )
        client = AIDeckStreamClient(settings)
        with self._lock:
            self._clients[config.drone_id] = client
        self.status_changed.emit(config.drone_id, f"connecting {config.host}:{config.port}")

        def on_image(image: np.ndarray, frame_id: int, fps: float, frame_type: str) -> np.ndarray:
            self.frame_received.emit(config.drone_id, _qimage_from_numpy(image), fps, frame_id, frame_type)
            return image

        def on_frame(frame_id: int, fps: float, frame_type: str) -> None:
            self.status_changed.emit(config.drone_id, f"streaming {frame_type} frame={frame_id} fps={fps:.1f}")

        try:
            count = client.run(stop_event=stop_event, on_frame=on_frame, on_image=on_image)
            if stop_event.is_set():
                self.status_changed.emit(config.drone_id, "stopped")
            else:
                self.status_changed.emit(config.drone_id, f"finished after {count} frames")
        except Exception as exc:
            self.status_changed.emit(config.drone_id, f"ERROR: {exc}")
        finally:
            with self._lock:
                self._clients.pop(config.drone_id, None)
                self._stop_events.pop(config.drone_id, None)
                self._threads.pop(config.drone_id, None)
            with contextlib.suppress(Exception):
                client.close()

    def _scan_worker(self, expected_ssid: str, port: int) -> None:
        ssid = _current_wifi_ssid()
        if ssid:
            self.scan_progress.emit(f"PC Wi-Fi SSID: {ssid}")
            if expected_ssid and ssid.lower() != expected_ssid.lower():
                self.scan_progress.emit(f"WARNING: PC is not connected to {expected_ssid}")
        else:
            self.scan_progress.emit("PC Wi-Fi SSID could not be detected")

        networks = _local_scan_subnets()
        if not networks:
            self.scan_progress.emit("Could not determine local subnet for AI-deck scan")
            self.scan_finished.emit([])
            return

        candidates: list[str] = []
        seen: set[str] = set()
        for network in networks[:3]:
            hosts = [str(ip) for ip in network.hosts()]
            self.scan_progress.emit(f"Scanning {network} for AI-deck streams on port {port}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
                for result in executor.map(lambda ip: _open_stream_port(ip, port), hosts):
                    if result and result not in seen:
                        seen.add(result)
                        candidates.append(result)
                        self.scan_progress.emit(f"Found AI-deck stream candidate: {result}:{port}")

        candidates = sorted(candidates, key=lambda value: tuple(int(part) for part in value.split(".")))
        self.scan_finished.emit(candidates)


def _qimage_from_numpy(image: np.ndarray) -> QImage:
    if image.ndim == 2:
        frame = np.ascontiguousarray(image)
        return QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format.Format_Grayscale8).copy()

    if image.ndim != 3:
        raise ValueError(f"Unsupported AI-deck image shape: {image.shape}")

    channels = image.shape[2]
    if channels == 4:
        frame = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        frame = np.ascontiguousarray(frame)
        return QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format.Format_RGBA8888).copy()
    if channels == 3:
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = np.ascontiguousarray(frame)
        return QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format.Format_RGB888).copy()

    raise ValueError(f"Unsupported AI-deck channel count: {channels}")


def _current_wifi_ssid() -> str | None:
    with contextlib.suppress(Exception):
        result = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        for line in result.stdout.splitlines():
            stripped = line.strip()
            lower = stripped.lower()
            if lower.startswith("ssid") and not lower.startswith("bssid") and ":" in stripped:
                ssid = stripped.split(":", 1)[1].strip()
                if ssid:
                    return ssid
    return None


def _local_scan_subnets() -> list[ipaddress.IPv4Network]:
    addresses: set[str] = set()
    with contextlib.suppress(Exception):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            addresses.add(sock.getsockname()[0])
        finally:
            sock.close()
    with contextlib.suppress(Exception):
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            addresses.add(ip)

    networks: list[ipaddress.IPv4Network] = []
    for ip in sorted(addresses):
        with contextlib.suppress(ValueError):
            addr = ipaddress.IPv4Address(ip)
            if addr.is_loopback or addr.is_link_local or addr.is_multicast:
                continue
            networks.append(ipaddress.IPv4Network(f"{ip}/24", strict=False))
    return networks


def _open_stream_port(ip: str, port: int) -> str | None:
    try:
        with socket.create_connection((ip, port), timeout=0.08):
            return ip
    except OSError:
        return None
