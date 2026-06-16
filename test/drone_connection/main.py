from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.aideck_stream import AIDeckStreamClient, StreamSettings
from src.gui import launch_gui


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VSLAM AI-deck stream manager")
    parser.add_argument("--cli", action="store_true", help="Run without the GUI")
    parser.add_argument("--host", default="192.168.4.1", help="AI-deck IP address")
    parser.add_argument("--port", type=int, default=5000, help="AI-deck stream port")
    parser.add_argument("--max-fps", type=float, default=0.0, help="Viewer FPS limit. 0 disables local throttling")
    parser.add_argument("--timeout", type=float, default=30.0, help="Socket timeout in seconds")
    parser.add_argument("--save", action="store_true", help="Save frames to stream_out/")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames, 0 means unlimited")
    parser.add_argument("--raw-view", choices=("color", "raw", "both"), default="color")
    parser.add_argument("--display-scale", type=int, default=2, help="OpenCV preview scale factor")
    parser.add_argument("--debug-stream", action="store_true", help="Print low-level AI-deck stream diagnostics")
    parser.add_argument("--with-cfclient", action="store_true", help="Open the Crazyflie Client flight GUI")
    parser.add_argument("--start-camera", action="store_true", help="Start the AI-deck camera stream when the GUI opens")
    parser.add_argument(
        "--both",
        action="store_true",
        help="Open Crazyflie Client and start the AI-deck camera stream",
    )
    parser.add_argument("--no-reconnect", action="store_true", help="Exit instead of reconnecting after stream errors")
    parser.add_argument("--reconnect-delay", type=float, default=3.0, help="Seconds to wait before reconnecting")
    parser.add_argument(
        "--reconnect-attempts",
        type=int,
        default=0,
        help="Reconnect attempts after a stream error. 0 means keep trying.",
    )
    return parser.parse_args()


def run_cli(args: argparse.Namespace) -> None:
    settings = StreamSettings(
        host=args.host,
        port=args.port,
        max_fps=args.max_fps,
        timeout=args.timeout,
        save_frames=args.save,
        max_frames=args.max_frames,
        raw_view=args.raw_view,
        display_scale=args.display_scale,
        output_dir=Path("stream_out"),
        debug=args.debug_stream,
    )

    def show_status(count: int, fps: float, frame_type: str) -> None:
        print(f"frame={count} fps={fps:.2f} type={frame_type}")

    attempt = 0
    while True:
        attempt += 1
        client = AIDeckStreamClient(settings)
        try:
            client.run(on_frame=show_status)
            return
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


def main() -> None:
    args = parse_args()
    if args.cli:
        run_cli(args)
    else:
        launch_gui(
            host=args.host,
            port=args.port,
            max_fps=args.max_fps,
            timeout=args.timeout,
            save_frames=args.save,
            raw_view=args.raw_view,
            display_scale=args.display_scale,
            debug_stream=args.debug_stream,
            auto_open_cfclient=args.with_cfclient or args.both,
            auto_start_stream=args.start_camera or args.both,
        )


if __name__ == "__main__":
    main()
