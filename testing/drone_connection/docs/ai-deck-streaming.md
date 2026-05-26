# AI-deck streaming notes

## Current fixes

- The GAP8 streamer now defaults to JPEG frames instead of raw frames. This lowers the Wi-Fi traffic a lot and avoids opening separate raw/color preview windows in the normal viewer path.
- The OpenCV viewer now uses one window named `AI-deck stream` by default.
- The project GUI has a button to open the standard Crazyflie Client (`cfclient`).
- If the stream disconnects or times out, the viewer stops with a clear message instead of waiting forever.
- Generated captures and flash images are ignored by Git.

## Run the viewer

GUI from `testing/drone_connection`:

```powershell
python main.py
```

Command line from `testing/drone_connection`:

```powershell
python main.py --cli --max-fps 2
```

Useful options:

```powershell
python main.py --cli --save --max-fps 2
python main.py --cli --raw-view both
python main.py --cli --max-frames 50
```

## What to push to GitHub

Push source code and documentation:

```powershell
git add .gitignore testing/drone_connection/
git status
git commit -m "Simplify AI-deck streaming workspace"
git push
```

Do not push generated files:

- `.venv/`
- `__pycache__/`
- `stream_out/`
- `img.jpeg`
- `*.img`
- `*.bin`
- `target.board.devices.flash.img`

These are local environment, capture, or build/flash artifacts. They can be recreated and make the repository noisy or too large.

## If Wi-Fi still stops

Use JPEG mode, keep the viewer at `--max-fps 2`, and restart the AI-deck AP after a crash. If it still fails around the same frame count, test with:

```powershell
python main.py --cli --max-fps 1 --max-frames 100
```

If `--max-fps 1` is stable but `--max-fps 2` is not, the bottleneck is the AI-deck Wi-Fi/CPX transfer rate rather than the Python viewer.

## Connect AI-deck to an existing Wi-Fi network

The current streamer firmware is configured for access point mode. The AI-deck creates a network and the laptop connects to it.

For station mode, configure and flash the Crazyflie firmware instead:

```text
Expansion deck configuration
Support AI-deck
WiFi setup at startup
Connect to a WiFi network
Credentials for access point
```

Do not save real Wi-Fi credentials in Git. After flashing, connect with the IP address shown in the Crazyflie Client logs.
