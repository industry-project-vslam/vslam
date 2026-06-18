from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import websockets


async def send_image(url: str, image_path: Path) -> None:
    image_bytes = image_path.read_bytes()
    async with websockets.connect(url, max_size=10_000_000) as websocket:
        await websocket.send(image_bytes)
        response = await websocket.recv()
        print(json.dumps(json.loads(response), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send one image to the object detection WebSocket API.")
    parser.add_argument("image", type=Path, help="JPEG or PNG image path")
    parser.add_argument("--url", default="ws://localhost:8000/stream", help="WebSocket URL")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(send_image(args.url, args.image))
