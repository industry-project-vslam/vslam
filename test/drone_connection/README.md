# VSLAM AI-deck Stream

Small project wrapper for receiving images from the Crazyflie AI-deck over Wi-Fi.

The important entry point is:

```powershell
python main.py
```

That opens a simple GUI where you can start/stop the stream, set the AI-deck IP, limit FPS, save frames, and open the standard Crazyflie Client.

## Setup

```powershell
cd testing\drone_connection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Connect your computer to the AI-deck Wi-Fi before starting the stream.

Default settings:

- IP: `192.168.4.1`
- Port: `5000`
- FPS limit: `2`

## Run

GUI:

```powershell
python main.py
```

Only open the standard Crazyflie Client:

```powershell
python main2.py
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

The current GAP8 streamer is built for AI-deck access point mode. That means the AI-deck creates a Wi-Fi network and your laptop connects to it.

To make the AI-deck connect to your own router/hotspot instead, configure the Crazyflie firmware:

```text
Expansion deck configuration
Support AI-deck
WiFi setup at startup
Connect to a WiFi network
Credentials for access point
```

Then rebuild and flash the Crazyflie firmware. Do not commit Wi-Fi passwords into this repository.

After the AI-deck joins your network, get its IP address from the Crazyflie Client logs and use that IP in this GUI.

## Project Layout

```text
main.py                          Main entry point for GUI or CLI
src/aideck_stream.py             AI-deck socket stream reader
src/gui.py                       Simple Tkinter GUI
requirements.txt                 Python dependencies
docs/ai-deck-streaming.md        Streaming and Git notes
drone_connection_default.py      Safe Crazyflie connection/takeoff test
aideck-gap8-examples/            Required Bitcraze GAP8 streamer subset
```

The firmware file we changed is:

```text
aideck-gap8-examples/examples/other/wifi-img-streamer/wifi-img-streamer.c
```

It now defaults to JPEG streaming to reduce Wi-Fi traffic.

## What To Push

Push source code, docs, and the AI-deck example source:

```powershell
cd ..\..
git add .gitignore testing/drone_connection/
git status
git commit -m "Simplify AI-deck streaming workspace"
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
