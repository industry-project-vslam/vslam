"""Connect to the Crazyflie, then show images from the AI-deck.

Before running this script:
1. Power the Crazyflie with the AI-deck attached.
2. Connect this computer to the AI-deck Wi-Fi network.
3. Connect the Crazyflie over USB, or set CFLIB_URI for radio.

Run:
    python main3.py

Run with Crazyradio:
    python main3.py --radio

Close the stream with q in the OpenCV image window, or Ctrl+C in the terminal.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

from src.aideck_stream import AIDeckStreamClient, StreamSettings


DEFAULT_URI = uri_helper.uri_from_env(default="usb://0")
DEFAULT_RADIO_URI = "radio://0/82/2M/E7E7E7E7E8"
DEFAULT_AI_DECK_HOST = "192.168.4.1"
DEFAULT_AI_DECK_PORT = 5000

logging.basicConfig(level=logging.ERROR)
AI_DECK_IP_RE = re.compile(r"WIFI:\s+got ip:\s+(\d{1,3}(?:\.\d{1,3}){3})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect to a Crazyflie and display the AI-deck camera stream."
    )
    parser.add_argument(
        "--uri",
        default=DEFAULT_URI,
        help="Crazyflie URI. Default: CFLIB_URI env var, otherwise usb://0",
    )
    parser.add_argument(
        "--radio",
        action="store_true",
        help=f"Use the default Crazyradio URI: {DEFAULT_RADIO_URI}",
    )
    parser.add_argument(
        "--host",
        default=None,
        help=f"AI-deck Wi-Fi IP address. Default: auto-detect from console, fallback {DEFAULT_AI_DECK_HOST}",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_AI_DECK_PORT,
        help="AI-deck image stream TCP port",
    )
    parser.add_argument("--max-fps", type=float, default=0.5, help="Viewer FPS limit")
    parser.add_argument("--timeout", type=float, default=30.0, help="Socket timeout in seconds")
    parser.add_argument("--save", action="store_true", help="Save frames to stream_out/")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames, 0 means unlimited")
    parser.add_argument(
        "--raw-view",
        choices=("color", "raw", "both"),
        default="color",
        help="Raw frame display mode. Default is color-only.",
    )
    parser.add_argument(
        "--require-ai-deck-param",
        action="store_true",
        help="Stop if the Crazyflie does not report deck.bcAI=1",
    )
    parser.add_argument(
        "--debug-stream",
        action="store_true",
        help="Print low-level AI-deck stream diagnostics",
    )
    parser.add_argument(
        "--auto-host-timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for a WIFI got ip console log when --host is not set",
    )
    parser.add_argument(
        "--fallback-ap-host",
        action="store_true",
        help=f"Use {DEFAULT_AI_DECK_HOST} if auto-detect does not find a station-mode IP",
    )
    parser.add_argument(
        "--show-console",
        action="store_true",
        help="Print Crazyflie console logs while waiting for the AI-deck IP",
    )
    parser.add_argument(
        "--no-reconnect",
        action="store_true",
        help="Exit instead of reconnecting when the AI-deck TCP stream stalls",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=3.0,
        help="Seconds to wait before reconnecting to the AI-deck stream",
    )
    parser.add_argument(
        "--reconnect-attempts",
        type=int,
        default=0,
        help="Reconnect attempts after a stream error. 0 means keep trying.",
    )
    return parser.parse_args()


class ConsoleIpDetector:
    def __init__(self, show_console: bool = False):
        self.show_console = show_console
        self.ip_address: str | None = None
        self.ip_found = Event()
        self._buffer = ""

    def on_console_text(self, text: str) -> None:
        if self.show_console:
            print(text, end="", flush=True)

        self._buffer += text
        match = AI_DECK_IP_RE.search(self._buffer)
        if match is not None:
            self.ip_address = match.group(1)
            self.ip_found.set()

        if len(self._buffer) > 2048:
            self._buffer = self._buffer[-2048:]


def print_console_text(text: str) -> None:
    print(text, end="", flush=True)


def resolve_ai_deck_host(cf: Crazyflie, args: argparse.Namespace) -> str:
    if args.host is not None:
        return args.host

    detector = ConsoleIpDetector(show_console=args.show_console)
    cf.console.receivedChar.add_callback(detector.on_console_text)

    print(f"Waiting up to {args.auto_host_timeout:.1f}s for AI-deck IP in console logs...")
    try:
        if detector.ip_found.wait(timeout=args.auto_host_timeout) and detector.ip_address is not None:
            print(f"Using AI-deck IP from console: {detector.ip_address}")
            return detector.ip_address
    finally:
        cf.console.receivedChar.remove_callback(detector.on_console_text)

    if args.fallback_ap_host:
        print(f"No AI-deck IP found in console logs; using AP fallback {DEFAULT_AI_DECK_HOST}")
        return DEFAULT_AI_DECK_HOST

    raise RuntimeError(
        "No AI-deck IP found in console logs. "
        "Power-cycle the Crazyflie/AI-deck and rerun, or pass --host with the IP from the Crazyflie console."
    )


def wait_for_ai_deck_param(cf: Crazyflie, timeout: float = 3.0) -> bool:
    """Return True when the Crazyflie parameter system reports an AI-deck."""
    deck_found = Event()
    deck_missing = Event()

    def callback(_name: str, value: str) -> None:
        if int(value):
            deck_found.set()
        else:
            deck_missing.set()

    cf.param.add_update_callback(group="deck", name="bcAI", cb=callback)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if deck_found.is_set():
            return True
        if deck_missing.is_set():
            return False
        time.sleep(0.05)

    return False


def run_stream(args: argparse.Namespace) -> int:
    settings = StreamSettings(
        host=args.host,
        port=args.port,
        max_fps=args.max_fps,
        timeout=args.timeout,
        save_frames=args.save,
        output_dir=Path("stream_out"),
        raw_view=args.raw_view,
        max_frames=args.max_frames,
        debug=args.debug_stream,
    )

    def on_frame(count: int, fps: float, frame_type: str) -> None:
        print(f"frame={count} fps={fps:.2f} type={frame_type}")

    total_frames = 0
    attempt = 0

    while True:
        attempt += 1
        client = AIDeckStreamClient(settings)

        print(f"Opening AI-deck stream at {args.host}:{args.port}")
        print("Press q in the image window to stop.")

        try:
            total_frames += client.run(on_frame=on_frame)
            return total_frames
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if args.no_reconnect:
                raise

            print(f"Stream error: {exc}")
            if args.reconnect_attempts and attempt >= args.reconnect_attempts:
                raise RuntimeError(f"stream failed after {attempt} attempt(s)") from exc

            print(f"Reconnecting in {args.reconnect_delay:.1f}s... attempt {attempt + 1}")
            time.sleep(args.reconnect_delay)


def main() -> int:
    args = parse_args()
    if args.radio:
        args.uri = DEFAULT_RADIO_URI

    print("Starting Crazyflie drivers")
    cflib.crtp.init_drivers()

    print(f"Connecting to Crazyflie: {args.uri}")

    try:
        with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache="./cache")) as scf:
            console_logging_enabled = False
            print("Crazyflie connected")

            ai_deck_detected = wait_for_ai_deck_param(scf.cf)
            if ai_deck_detected:
                print("AI-deck detected by Crazyflie")
            elif args.require_ai_deck_param:
                print("AI-deck was not reported by deck.bcAI. Stop.")
                return 1
            else:
                print("AI-deck parameter was not confirmed; trying the Wi-Fi stream anyway")

            args.host = resolve_ai_deck_host(scf.cf, args)
            if args.show_console:
                scf.cf.console.receivedChar.add_callback(print_console_text)
                console_logging_enabled = True

            try:
                frame_count = run_stream(args)
                print(f"Stream stopped after {frame_count} frame(s)")
                return 0
            finally:
                if console_logging_enabled:
                    scf.cf.console.receivedChar.remove_callback(print_console_text)

    except KeyboardInterrupt:
        print("\nStopped by user")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        print("Check that the Crazyflie is connected and that your PC is on the AI-deck Wi-Fi.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
