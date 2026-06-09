from __future__ import annotations

import os
import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

import cv2
import numpy as np


FrameCallback = Callable[[int, float, str], None]
FrameProcessor = Callable[[np.ndarray], np.ndarray]


@dataclass
class StreamSettings:
    host: str = "192.168.4.1"
    port: int = 5000
    max_fps: float = 0.0
    timeout: float = 5.0
    save_frames: bool = False
    output_dir: Path = Path("stream_out")
    raw_view: str = "color"
    max_frames: int = 0
    debug: bool = False
    display_scale: int = 2
    display_fullscreen: bool = False
    frame_processor: FrameProcessor | None = None


class AIDeckStreamClient:
    def __init__(self, settings: StreamSettings):
        self.settings = settings
        self._socket: socket.socket | None = None
        self._debug_headers_seen = 0
        self._stream_windows: set[str] = set()
        self._screen_size = self._get_screen_size()

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        cv2.destroyAllWindows()

    def run(
        self,
        stop_event: Event | None = None,
        on_frame: FrameCallback | None = None,
    ) -> int:
        stop_event = stop_event or Event()
        frame_count = 0
        received_count = 0
        stream_started = time.time()
        last_frame_time = 0.0
        frame_interval = 0.0 if self.settings.max_fps <= 0 else 1.0 / self.settings.max_fps

        self._connect()

        try:
            while not stop_event.is_set():
                magic, width, height, depth, image_format, image_size = self._read_header()
                if magic != 0xBC:
                    continue

                image_data = self._read_image(image_size)
                received_count += 1

                now = time.time()
                if frame_interval > 0 and now - last_frame_time < frame_interval:
                    self._debug(
                        f"received frame {received_count}, dropped locally to keep viewer at "
                        f"{self.settings.max_fps:.2f} fps"
                    )
                    continue

                frame_count += 1
                last_frame_time = now
                average_fps = frame_count / max(now - stream_started, 0.001)

                if image_format == 0:
                    self._show_raw(image_data, width, height, frame_count)
                    frame_type = "raw"
                else:
                    self._show_jpeg(image_data, frame_count)
                    frame_type = "jpeg"

                if on_frame is not None:
                    on_frame(frame_count, average_fps, frame_type)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if self.settings.max_frames and frame_count >= self.settings.max_frames:
                    break
        finally:
            self.close()

        return frame_count

    def _connect(self) -> None:
        self._debug(f"connecting to {self.settings.host}:{self.settings.port}")
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(self.settings.timeout)
        self._socket.connect((self.settings.host, self.settings.port))
        self._debug("socket connected")

    def _read_exact(self, size: int) -> bytes:
        if self._socket is None:
            raise ConnectionError("Stream socket is not connected")

        data = bytearray()
        while len(data) < size:
            try:
                chunk = self._socket.recv(size - len(data))
            except socket.timeout as exc:
                raise TimeoutError(
                    f"Timed out after {self.settings.timeout:.1f}s while reading "
                    f"{size} byte(s); got {len(data)} byte(s)"
                ) from exc
            if not chunk:
                raise ConnectionError(f"AI-deck stream closed while reading {size} byte(s)")
            data.extend(chunk)
        return bytes(data)

    def _read_header(self) -> tuple[int, int, int, int, int, int]:
        packet_info = self._read_exact(4)
        length, routing, function = struct.unpack("<HBB", packet_info)
        self._debug(f"CPX header packet: length={length} routing=0x{routing:02x} function=0x{function:02x}")
        header = self._read_exact(length - 2)
        image_header = struct.unpack("<BHHBBI", header)
        self._debug_headers_seen += 1
        if self._debug_headers_seen <= 10 or self._debug_headers_seen % 30 == 0:
            magic, width, height, depth, image_format, image_size = image_header
            self._debug(
                "image header: "
                f"magic=0x{magic:02x} width={width} height={height} "
                f"depth={depth} format={image_format} size={image_size}"
            )
        return image_header

    def _read_image(self, image_size: int) -> bytes:
        image_data = bytearray()
        packet_count = 0
        while len(image_data) < image_size:
            packet_info = self._read_exact(4)
            length, _destination, _source = struct.unpack("<HBB", packet_info)
            image_data.extend(self._read_exact(length - 2))
            packet_count += 1
        if self._debug_headers_seen <= 10 or self._debug_headers_seen % 30 == 0:
            self._debug(f"image payload complete: bytes={len(image_data)} packets={packet_count}")
        return bytes(image_data)

    def _show_raw(self, image_data: bytes, width: int, height: int, frame_count: int) -> None:
        raw_image = np.frombuffer(image_data, dtype=np.uint8)
        raw_image.shape = (height, width)
        color_image = cv2.cvtColor(raw_image, cv2.COLOR_BayerBG2BGRA)
        color_image = self._process_frame(color_image)
        raw_preview = self._scale_for_display(raw_image)
        color_preview = self._scale_for_display(color_image)

        if self.settings.raw_view in ["raw", "both"]:
            cv2.imshow("AI-deck raw", raw_preview)
        if self.settings.raw_view in ["color", "both"]:
            self._show_stream_window(color_image)

        if self.settings.save_frames:
            raw_dir = self.settings.output_dir / "raw"
            color_dir = self.settings.output_dir / "debayer"
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(color_dir, exist_ok=True)
            cv2.imwrite(str(raw_dir / f"img_{frame_count:06d}.png"), raw_image)
            cv2.imwrite(str(color_dir / f"img_{frame_count:06d}.png"), color_image)

    def _show_jpeg(self, image_data: bytes, frame_count: int) -> None:
        encoded = np.frombuffer(image_data, np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Received invalid JPEG frame")

        decoded = self._process_frame(decoded)
        self._show_stream_window(decoded)

        if self.settings.save_frames:
            jpeg_dir = self.settings.output_dir / "jpeg"
            os.makedirs(jpeg_dir, exist_ok=True)
            cv2.imwrite(str(jpeg_dir / f"img_{frame_count:06d}.jpg"), decoded)

    def _scale_for_display(self, image: np.ndarray) -> np.ndarray:
        scale = max(int(self.settings.display_scale), 1)
        if scale == 1:
            return image
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    def _show_stream_window(self, image: np.ndarray) -> None:
        window_name = "AI-deck stream"
        self._ensure_stream_window(window_name)
        cv2.imshow(window_name, self._fit_to_window(image, window_name))

    def _ensure_stream_window(self, window_name: str) -> None:
        if window_name in self._stream_windows:
            return

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if self.settings.display_fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            if self._screen_size is not None:
                cv2.moveWindow(window_name, 0, 0)
                cv2.resizeWindow(window_name, *self._screen_size)
        else:
            initial = self._initial_window_size()
            cv2.resizeWindow(window_name, *initial)

        self._stream_windows.add(window_name)

    def _initial_window_size(self) -> tuple[int, int]:
        scale = max(int(self.settings.display_scale), 1)
        return 324 * scale, 244 * scale

    def _fit_to_window(self, image: np.ndarray, window_name: str) -> np.ndarray:
        target = self._current_window_size(window_name)
        if target is None:
            return self._scale_for_display(image)

        target_width, target_height = target
        image_height, image_width = image.shape[:2]
        if target_width <= 0 or target_height <= 0 or image_width <= 0 or image_height <= 0:
            return image

        scale = min(target_width / image_width, target_height / image_height)
        resized_width = max(1, int(round(image_width * scale)))
        resized_height = max(1, int(round(image_height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        channels = 1 if resized.ndim == 2 else resized.shape[2]
        if channels == 1:
            canvas = np.zeros((target_height, target_width), dtype=resized.dtype)
        else:
            canvas = np.zeros((target_height, target_width, channels), dtype=resized.dtype)

        x_offset = max((target_width - resized_width) // 2, 0)
        y_offset = max((target_height - resized_height) // 2, 0)
        canvas[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized
        return canvas

    def _current_window_size(self, window_name: str) -> tuple[int, int] | None:
        if self.settings.display_fullscreen and self._screen_size is not None:
            return self._screen_size

        try:
            _x, _y, width, height = cv2.getWindowImageRect(window_name)
        except cv2.error:
            return None

        if width <= 1 or height <= 1:
            return None
        return int(width), int(height)

    def _process_frame(self, image: np.ndarray) -> np.ndarray:
        if self.settings.frame_processor is None:
            return image
        return self.settings.frame_processor(image)

    def _debug(self, message: str) -> None:
        if self.settings.debug:
            print(f"[aideck-debug {time.strftime('%H:%M:%S')}] {message}", flush=True)

    @staticmethod
    def _get_screen_size() -> tuple[int, int] | None:
        try:
            import ctypes

            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            width = int(user32.GetSystemMetrics(0))
            height = int(user32.GetSystemMetrics(1))
            if width > 0 and height > 0:
                return width, height
        except Exception:
            return None
        return None
