# VSLAM AI-deck Stream

Small project wrapper for receiving images from the Crazyflie AI-deck over Wi-Fi.

The important entry point is:

```powershell
python main.py
```

That opens a simple GUI where you can start/stop the stream, set the AI-deck
IP, limit FPS, save frames, and open the standard Crazyflie Client.

## Setup

```powershell
cd code\drone_connection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If posture analysis was added after the virtual environment already existed,
activate `.venv` and run `pip install -r requirements.txt` again.

Connect your computer to the AI-deck Wi-Fi before starting the stream.

Default connection settings:

- IP: `192.168.4.1`
- Port: `5000`
- FPS limit: `0`, meaning no local viewer throttling

## Run

GUI:

```powershell
python main.py
```

The camera window runs real-time human posture analysis by default. It uses the
human detector and posture classifier from:

```text
code/object_detection/runs/human_detection/weights/best.pt
code/object_detection/runs/posture_classification/mobilenet_v3_small/best.pt
code/object_detection/runs/posture_classification/mobilenet_v3_small/class_to_idx.json
```

The stream is analyzed in memory and is not saved unless `Save frames to
stream_out/` is enabled.

Human detection defaults to confidence `0.23`. Posture classification uses one
unknown rule: if the top posture confidence is below `0.50`, the label is shown
as `unknown`. There is no top1/top2 margin rule in the live pipeline.

Use `Fullscreen camera window` in the GUI when you want the main OpenCV preview
to cover the screen. In CLI mode, use:

```powershell
python main.py --cli --fullscreen
```

Run without posture analysis:

```powershell
python main.py --no-analysis
```

Command line:

```powershell
python main.py --cli --max-fps 2
```

Save frames:

```powershell
python main.py --cli --save --max-fps 2
```

Saved images go to `stream_out/`, which is ignored by Git.

## Wi-Fi Mode

This app expects the AI-deck camera streamer to already be running. No drone
firmware rebuild or reprogramming is needed for the real-time detection
pipeline.

The default setup is AI-deck access point mode: the AI-deck creates a Wi-Fi
network and your laptop connects to it. If the AI-deck is configured for another
network, enter its actual IP address in the GUI.

## Project Layout

```text
main.py                          Main entry point for GUI or CLI
src/aideck_stream.py             AI-deck socket stream reader
src/gui.py                       Simple Tkinter GUI
src/posture_pipeline.py          YOLO detection + MobileNet classification
requirements.txt                 Python dependencies
docs/ai-deck-streaming.md        Streaming and Git notes
```

## What To Push

Push source code and docs:

```powershell
cd ..\..
git add code/drone_connection/
git status
git commit -m "Add real-time AI-deck posture detection"
git push
```

Do not push generated/local files:

- `.venv/`
- `stream_out/`
- `__pycache__/`
- `*.img`
- `*.bin`
- `target.board.devices.flash.img`

These are already ignored by `.gitignore`.

## Notes For Teammates

1. Install dependencies with `pip install -r requirements.txt`.
2. Connect to the AI-deck Wi-Fi.
3. Run `python main.py`.
4. Click `Test stream IP` first.
5. Click `Start stream` if the IP test is OK.
6. Click `Open Crazyflie Client` if you need the normal Crazyflie GUI.
7. Press `q` in the image window or click `Stop stream` to stop.

If the stream stops after many frames, lower `Max FPS` to `1` in the GUI and test again.
