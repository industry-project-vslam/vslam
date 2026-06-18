# Installation And Usage Guide

## Project

This guide explains how to install, run, test, and present the Ranger-guided AI-deck streaming swarm project.

The guide is folder-independent. Replace the placeholders below with paths from your own computer.

### Path placeholders

```text
<PROJECT_ROOT>          Folder where the main project is located
<DRONE_CONNECTION_DIR>  <PROJECT_ROOT>/testing/drone_connection
<VENV_DIR>              <PROJECT_ROOT>/.venv
<AI_DECK_IP>            IP address of an AI-deck stream server
```

`<PROJECT_ROOT>` must be the folder that contains the `testing` directory.

Example:

```text
<PROJECT_ROOT>/testing/drone_connection/gui/main.py
```

means: open your own project folder, then go to `testing/drone_connection/gui/main.py`.

On Windows, PowerShell usually accepts both `/` and `\` in paths. In this guide, project-relative paths use `/` to avoid duplicating the same command for every operating system.

Main GUI entry point:

```text
testing/drone_connection/gui/main.py
```

Main backend controller:

```text
testing/drone_connection/swarm_controller/controller.py
```

---

## 1. Hardware Setup

For the current half-swarm presentation, use three drones:

| Role | Radio URI | Decks | Purpose |
| --- | --- | --- | --- |
| `X_FRONT` | `radio://0/82/2M/E7E7E7E701` | Ranger + Flow | Scout and safety leader |
| `O1` | `radio://0/82/2M/E7E7E7E702` | AI deck + Flow | Left visual observer |
| `O2` | `radio://0/82/2M/E7E7E7E703` | AI deck + Flow | Forward/right visual observer |

Use the launch mat coordinates:

```text
X_FRONT: x=32, y=20
O1:      x=12, y=45
O2:      x=52, y=45
```

All drones should physically face the same forward direction before takeoff.

After takeoff, the controller aligns camera yaw:

```text
X_FRONT -> movement heading
O1      -> 270 deg, left-looking stream
O2      -> 0 deg, forward-looking stream
```

---

## 2. Wi-Fi Setup

The AI decks should connect to the existing Wi-Fi network:

```text
SSID: swarming
Password: swarming
AI-deck stream port: 5000
```

The laptop should also be connected to `swarming` when using live AI-deck previews.

AI-deck streaming code:

```text
testing/drone_connection/src/aideck_stream.py
testing/drone_connection/gui/ai_stream_preview.py
testing/drone_connection/gui/widgets/ai_stream_panel.py
testing/drone_connection/aideck-gap8-examples/examples/other/wifi-img-streamer/wifi-img-streamer.c
```

---

## 3. Install Python Environment

### 3.1. Open a terminal

Use:

- **Windows:** PowerShell
- **macOS/Linux:** Terminal

### 3.2. Go to the project root

```bash
cd "<PROJECT_ROOT>"
```

### 3.3. Create a virtual environment

Use the command for your operating system:

**Windows PowerShell**

```powershell
py -3 -m venv .venv
```

**macOS/Linux**

```bash
python3 -m venv .venv
```

### 3.4. Activate the virtual environment

Use the command for your operating system:

**Windows PowerShell**

```powershell
& ".\.venv\Scripts\Activate.ps1"
```

If PowerShell blocks activation, run this once for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& ".\.venv\Scripts\Activate.ps1"
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

After activation, the terminal should show something like:

```text
(.venv)
```

### 3.5. Install dependencies

From `<PROJECT_ROOT>`, run:

```bash
python -m pip install --upgrade pip
pip install -r testing/drone_connection/requirements.txt
```

The requirements file is:

```text
testing/drone_connection/requirements.txt
```

It installs the main project dependencies, including:

```text
numpy
opencv-python
cflib
cfclient
h5py
PyQt6
```

---

## 4. Verify Installation

Go to the drone connection directory:

```bash
cd "<PROJECT_ROOT>/testing/drone_connection"
```

Verify Python dependencies:

```bash
python -c "import PyQt6, cflib, cv2, numpy; print('installation OK')"
```

Check Crazyradio / Crazyflie interfaces:

```bash
python -c "import cflib.crtp; cflib.crtp.init_drivers(); print(cflib.crtp.scan_interfaces())"
```

Expected result when Crazyradio and/or USB Crazyflie are visible:

```text
[('radio://...', '...')]
```

or:

```text
[('usb://0', '')]
```

If the result is empty, reconnect Crazyradio, check drivers, close other Crazyflie programs, and try again.

---

## 5. Run The Final Swarm GUI

The GUI is the preferred way to run the project during testing and presentation.

Make sure the virtual environment is activated, then run:

```bash
cd "<PROJECT_ROOT>/testing/drone_connection"
python -m gui.main
```

Main GUI files:

```text
testing/drone_connection/gui/main.py
testing/drone_connection/gui/main_window.py
testing/drone_connection/gui/worker.py
```

GUI widget files:

```text
testing/drone_connection/gui/widgets/drone_status_panel.py
testing/drone_connection/gui/widgets/mission_control_panel.py
testing/drone_connection/gui/widgets/ranger_panel.py
testing/drone_connection/gui/widgets/formation_view.py
testing/drone_connection/gui/widgets/ai_stream_panel.py
testing/drone_connection/gui/widgets/event_log_panel.py
testing/drone_connection/gui/widgets/thresholds_panel.py
```

---

## 6. GUI Usage Workflow

For a presentation or real test, use this order:

1. Place the drones on the launch mat.
2. Connect Crazyradio to the laptop.
3. Turn on the drones.
4. Connect the laptop to Wi-Fi `swarming`.
5. Start the GUI.
6. Click `Load Half Group (3)`.
7. Click `Use Real Crazyflies`.
8. Click `Connect All`.
9. Click `Sensor Check`.
10. Open the `AI Streams` tab.
11. Click `Scan swarming Wi-Fi for AI decks`.
12. Confirm that O1/O2 IPs are filled or enter them manually.
13. Click `Start AI Streams`.
14. Click `Formation Hover Only` for a safe hover demonstration.
15. Click `Formation Micro Step` for a small movement proof.
16. Click `Start Full Observation Mode` for the full autonomous observation demo.
17. Use `SAFE HOVER / LAND` or `Land All` to finish.

Emergency buttons are always active:

```text
EMERGENCY STOP / HARD KILL
SAFE HOVER / LAND
HARD MOTOR KILL
```

---

## 7. Simulation Mode

Simulation is useful before a real flight demonstration.

Start the GUI and click:

```text
Run Simulation Mode
Sim Open
Sim Obstacle
Sim Wall
Start Top-3 Observation Demo
```

Command-line simulation from `<DRONE_CONNECTION_DIR>`:

```bash
cd "<PROJECT_ROOT>/testing/drone_connection"
python -m swarm_controller.main --simulate --half-group --scenario open_space --steps 5
```

Other useful scenarios:

```bash
python -m swarm_controller.main --simulate --half-group --scenario obstacle --steps 5
python -m swarm_controller.main --simulate --half-group --scenario wall --steps 5
```

Simulation code:

```text
testing/drone_connection/swarm_controller/simulation_stub.py
```

---

## 8. Real Drone Backend Run

The GUI is the preferred presentation interface. The backend can also be run directly.

From `<DRONE_CONNECTION_DIR>`:

```bash
cd "<PROJECT_ROOT>/testing/drone_connection"
python -m swarm_controller.main --half-group --mission-id final_demo_half_group --steps 5 --x-front-uri radio://0/82/2M/E7E7E7E701 --o1-uri radio://0/82/2M/E7E7E7E702 --o2-uri radio://0/82/2M/E7E7E7E703
```

Backend files:

```text
testing/drone_connection/swarm_controller/main.py
testing/drone_connection/swarm_controller/controller.py
testing/drone_connection/swarm_controller/state_machine.py
testing/drone_connection/swarm_controller/drones.py
testing/drone_connection/swarm_controller/motion.py
testing/drone_connection/swarm_controller/safety.py
testing/drone_connection/swarm_controller/ranger.py
testing/drone_connection/swarm_controller/formation.py
testing/drone_connection/swarm_controller/logs.py
```

---

## 9. AI-Deck Stream Viewer

Standalone AI-deck stream GUI from `<DRONE_CONNECTION_DIR>`:

```bash
cd "<PROJECT_ROOT>/testing/drone_connection"
python main.py
```

Command-line stream viewer:

```bash
python main.py --cli --host <AI_DECK_IP> --port 5000 --max-fps 2
```

Save frames:

```bash
python main.py --cli --host <AI_DECK_IP> --port 5000 --save --max-fps 2
```

Saved frames go under:

```text
testing/drone_connection/stream_out/
```

---

## 10. Mission Logs

Every mission writes evidence logs to:

```text
testing/drone_connection/stream_out/fixed_formation_missions/<mission_id>_<timestamp>/
```

Important files:

```text
README_LOGS.md
run_summary.json
formation_config.json
mission_log.csv
command_log.csv
ranger_log.csv
safety_grid_log.csv
capture_manifest.csv
classification_log.csv
reslot_path_log.csv
turn_reslot_log.csv
breadcrumb_log.csv
preflight_log.csv
battery_log.csv
mission_timing_log.csv
error_log.csv
```

Log implementation:

```text
testing/drone_connection/swarm_controller/logs.py
```

---

## 11. Tests

Run the focused project tests from `<DRONE_CONNECTION_DIR>`:

```bash
cd "<PROJECT_ROOT>/testing/drone_connection"
python -m unittest gui.tests.test_button_wiring swarm_controller.tests.test_simulation swarm_controller.tests.test_controller_behavior swarm_controller.tests.test_top3_logic
```

These tests check:

- GUI button wiring;
- simulation behavior;
- controller behavior;
- top-3 formation logic.

Expected result:

```text
Ran 51 tests
OK
```

---

## 12. Presentation Run Checklist

Before the final presentation:

1. Charge all drone batteries.
2. Check propellers.
3. Confirm Crazyradio is connected.
4. Confirm laptop is connected to `swarming`.
5. Confirm drones are placed on the launch mat:

```text
X_FRONT: x=32, y=20
O1:      x=12, y=45
O2:      x=52, y=45
```

6. Confirm all drones face the same physical forward direction.
7. Start the GUI.
8. Run `Load Half Group (3)`.
9. Run `Use Real Crazyflies`.
10. Run `Connect All`.
11. Run `Sensor Check`.
12. Start AI streams.
13. Run `Formation Hover Only`.
14. Run `Formation Micro Step`.
15. Run `Start Full Observation Mode`.
16. Finish with `SAFE HOVER / LAND` or `Land All`.

---

## 13. Troubleshooting

### GUI does not start

Check PyQt6:

```bash
python -c "import PyQt6; print('PyQt6 OK')"
```

If PyQt6 is missing:

```bash
pip install PyQt6
```

### Crazyradio not found

Check Crazyradio / Crazyflie interfaces:

```bash
python -c "import cflib.crtp; cflib.crtp.init_drivers(); print(cflib.crtp.scan_interfaces())"
```

Reconnect Crazyradio, close other Crazyflie programs, then restart the GUI.

### AI-deck stream IP unknown

Use the GUI:

```text
AI Streams tab -> Scan swarming Wi-Fi for AI decks
```

or read the IP from Crazyflie Client console logs.

### Flow/zrange warning

Use a matte floor with visible texture and good light. Flow decks need floor texture to stabilize correctly.

### Mission behavior should be checked first

Use simulation before real flight:

```text
Run Simulation Mode -> Sim Open -> Start Top-3 Observation Demo
```

---

## 14. Related Documentation

Presentation document:

```text
testing/drone_connection/docs/final-presentation-documentation.md
```

System concept:

```text
testing/drone_connection/docs/adaptive-360-mapping-swarm.md
```

Half-swarm behavior:

```text
testing/drone_connection/docs/half-swarm-safe-mvp-logic.md
```

AI-deck streaming:

```text
testing/drone_connection/docs/ai-deck-streaming.md
```

Backend README:

```text
testing/drone_connection/swarm_controller/README.md
```
