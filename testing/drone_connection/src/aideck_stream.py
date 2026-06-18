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
ImageCallback = Callable[[np.ndarray, int, float, str], np.ndarray | None]


@dataclass
class StreamSettings:
    host: str = "172.20.10.7"
    port: int = 5000
    max_fps: float = 0.0
    timeout: float = 5.0
    save_frames: bool = False
    output_dir: Path = Path("stream_out")
    raw_view: str = "color"
    max_frames: int = 0
    debug: bool = False
    display_scale: int = 2
    window_name: str = "AI-deck stream"
    show_window: bool = True


class AIDeckStreamClient:
    def __init__(self, settings: StreamSettings):
        self.settings = settings
        self._socket: socket.socket | None = None
        self._debug_headers_seen = 0

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self.settings.show_window:
            cv2.destroyAllWindows()

    def run(
        self,
        stop_event: Event | None = None,
        on_frame: FrameCallback | None = None,
        on_image: ImageCallback | None = None,
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
                    self._show_raw(image_data, width, height, frame_count, average_fps, on_image)
                    frame_type = "raw"
                else:
                    self._show_jpeg(image_data, frame_count, average_fps, on_image)
                    frame_type = "jpeg"

                if on_frame is not None:
                    on_frame(frame_count, average_fps, frame_type)

                if self.settings.show_window:
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

    def _show_raw(
        self,
        image_data: bytes,
        width: int,
        height: int,
        frame_count: int,
        average_fps: float,
        on_image: ImageCallback | None,
    ) -> None:
        raw_image = np.frombuffer(image_data, dtype=np.uint8)
        raw_image.shape = (height, width)
        color_image = cv2.cvtColor(raw_image, cv2.COLOR_BayerBG2BGRA)
        display_image = color_image
        if on_image is not None:
            callback_image = on_image(color_image, frame_count, average_fps, "raw")
            display_image = callback_image if callback_image is not None else color_image

        raw_preview = self._scale_for_display(raw_image)
        color_preview = self._scale_for_display(display_image)

        if self.settings.show_window:
            if self.settings.raw_view in ["raw", "both"]:
                cv2.imshow(f"{self.settings.window_name} raw", raw_preview)
            if self.settings.raw_view in ["color", "both"]:
                cv2.imshow(self.settings.window_name, color_preview)

        if self.settings.save_frames:
            raw_dir = self.settings.output_dir / "raw"
            color_dir = self.settings.output_dir / "debayer"
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(color_dir, exist_ok=True)
            cv2.imwrite(str(raw_dir / f"img_{frame_count:06d}.png"), raw_image)
            cv2.imwrite(str(color_dir / f"img_{frame_count:06d}.png"), color_image)

    def _show_jpeg(
        self,
        image_data: bytes,
        frame_count: int,
        average_fps: float,
        on_image: ImageCallback | None,
    ) -> None:
        encoded = np.frombuffer(image_data, np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError("Received invalid JPEG frame")

        display_image = decoded
        if on_image is not None:
            callback_image = on_image(decoded, frame_count, average_fps, "jpeg")
            display_image = callback_image if callback_image is not None else decoded

        if self.settings.show_window:
            cv2.imshow(self.settings.window_name, self._scale_for_display(display_image))

        if self.settings.save_frames:
            jpeg_dir = self.settings.output_dir / "jpeg"
            os.makedirs(jpeg_dir, exist_ok=True)
            cv2.imwrite(str(jpeg_dir / f"img_{frame_count:06d}.jpg"), decoded)

    def _scale_for_display(self, image: np.ndarray) -> np.ndarray:
        scale = max(int(self.settings.display_scale), 1)
        if scale == 1:
            return image
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    def _debug(self, message: str) -> None:
        if self.settings.debug:
            print(f"[aideck-debug {time.strftime('%H:%M:%S')}] {message}", flush=True)
