# AI-deck streaming notes

## Current fixes

- The OpenCV viewer uses one window named `AI-deck stream` by default.
- The viewer can show real-time YOLO human detection and MobileNet posture labels.
- Fullscreen/maximized display scales the stream into the actual window area.
- The project GUI has a button to open the standard Crazyflie Client (`cfclient`).
- If the stream disconnects or times out, the viewer stops with a clear message instead of waiting forever.
- Generated captures are ignored by Git.

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
python main.py --cli --fullscreen
python main.py --cli --no-analysis
python main.py --cli --save --max-fps 2
python main.py --cli --raw-view both
python main.py --cli --max-frames 50
```

## What to push to GitHub

Push source code and documentation:

```powershell
git add testing/drone_connection/
git status
git commit -m "Add real-time AI-deck posture detection"
git push
```

Do not push generated files:

- `.venv/`
- `__pycache__/`
- `stream_out/`
- `img.jpeg`

These are local environment or capture artifacts. They can be recreated and make
the repository noisy or too large.

## If Wi-Fi still stops

Keep the viewer at `--max-fps 2`, and restart the AI-deck AP after a crash. If
it still fails around the same frame count, test with:

```powershell
python main.py --cli --max-fps 1 --max-frames 100
```

If `--max-fps 1` is stable but `--max-fps 2` is not, the bottleneck is the AI-deck Wi-Fi/CPX transfer rate rather than the Python viewer.

## AI-deck network mode

This app expects the AI-deck camera streamer to already be running. The default
IP is `192.168.4.1`, which matches AI-deck access point mode. If the AI-deck is
already configured for another network, enter that IP in the GUI.
