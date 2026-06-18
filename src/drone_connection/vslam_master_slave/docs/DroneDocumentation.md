# Ranger-Guided AI-Deck Streaming Swarm - Project Documentation

## 1. Project Overview

This project is an indoor drone-swarm exploration system built with Crazyflie drones. The system combines different drone roles in one fixed formation:

- **Ranger drone** - works as the safety scout and checks obstacles before the group moves.
- **AI-deck drones** - work as visual observers and stream camera data to the laptop.
- **Flow decks** - support stable hover and short local movements.
- **PC mission controller** - coordinates the mission, makes safety decisions, controls movement, shows the GUI, and saves logs.

The current deliverable is a **half-group MVP** with three drones:

| Role | Address | Hardware | Main purpose |
|---|---|---|---|
| `X_FRONT` | `radio://0/82/2M/E7E7E7E701` | Crazyflie + Flow deck + Multi-ranger deck | Front safety scout and navigation leader |
| `O1` | `radio://0/82/2M/E7E7E7E702` | Crazyflie + Flow deck + AI deck | Left AI-deck visual observer |
| `O2` | `radio://0/82/2M/E7E7E7E703` | Crazyflie + Flow deck + AI deck | Forward/right AI-deck visual observer |

The future full version scales naturally to six drones: two Ranger safety drones and four AI-deck streamers for 360-degree visual coverage.

Directory placeholders used in this documentation:

| Placeholder | Meaning |
|---|---|
| `<PROJECT_ROOT>` | The folder where the full project or repository is located |
| `<DRONE_CONNECTION_DIR>` | `<PROJECT_ROOT>/testing/drone_connection` |
| `<VENV_DIR>` | `<PROJECT_ROOT>/.venv` |

Most code paths below use `<DRONE_CONNECTION_DIR>` so the documentation is not tied to one local folder.

Main GUI entry point:

```text
<DRONE_CONNECTION_DIR>/gui/main.py
```

Main backend controller:

```text
<DRONE_CONNECTION_DIR>/swarm_controller/controller.py
```

---

## 2. Main Idea

The swarm does not move blindly. The system follows a safety-first behavior:

```text
Scout -> Verify -> Move -> Capture -> Re-check -> Continue
```

The Ranger drone always checks the environment first. If the path is safe, it performs a short movement. After that, the PC checks Ranger data again. Only if the movement is still safe, the AI-deck drones copy the approved movement while streaming visual data.

If an obstacle or wall is detected, the AI-deck drones hover in place. The Ranger drone then looks for the clearer direction, moves around the formation, becomes the scout for the new direction, and the swarm continues observation.

This makes the system controlled, explainable, and ready for step-by-step lab testing and presentation.

Core implementation files:

```text
<DRONE_CONNECTION_DIR>/swarm_controller/controller.py
<DRONE_CONNECTION_DIR>/swarm_controller/state_machine.py
<DRONE_CONNECTION_DIR>/swarm_controller/motion.py
<DRONE_CONNECTION_DIR>/swarm_controller/safety.py
<DRONE_CONNECTION_DIR>/swarm_controller/ranger.py
<DRONE_CONNECTION_DIR>/swarm_controller/formation.py
```

---

## 3. System Architecture

```text
AI Deck Streams over Wi-Fi              Flight commands over Crazyradio

O1 AI Deck  ---- Wi-Fi stream ----\
                                   \
O2 AI Deck  ---- Wi-Fi stream ------> Laptop / PC Mission Controller / PyQt6 GUI
                                    /        |
X_FRONT Ranger telemetry ----------/         |
                                             |
                                             v
                                      Crazyradio commands
                                      to Crazyflie drones
```

### Component responsibilities

| Component | Responsibility | Project file or folder |
|---|---|---|
| **PC / laptop** | Mission state machine, safety decisions, GUI, logs, command sequencing | `<DRONE_CONNECTION_DIR>/swarm_controller/controller.py` |
| **Crazyradio** | Sends flight commands and receives telemetry | `<DRONE_CONNECTION_DIR>/swarm_controller/drones.py` |
| **Multi-ranger deck** | Measures front, left, right, back, and up distances | `<DRONE_CONNECTION_DIR>/swarm_controller/ranger.py` |
| **Flow deck** | Helps stable hover and short local movement | telemetry handled through `swarm_controller/drones.py` and `preflight.py` |
| **AI deck** | Streams camera frames over Wi-Fi | `<DRONE_CONNECTION_DIR>/src/aideck_stream.py` |
| **PyQt6 GUI** | Gives the operator control, monitoring, simulation, and emergency tools | `<DRONE_CONNECTION_DIR>/gui/` |
| **Logs** | Store evidence for decisions, movement, safety checks, and errors | `<DRONE_CONNECTION_DIR>/swarm_controller/logs.py` |

---

## 4. Why the Architecture Works This Way

### Ranger-led navigation

The Ranger drone is responsible for local safety. It reads distance values from several directions and lets the PC decide whether the formation may move.

This is stronger than relying only on camera images, because the Ranger deck gives direct physical distance measurements.

Implementation files:

```text
<DRONE_CONNECTION_DIR>/swarm_controller/ranger.py
<DRONE_CONNECTION_DIR>/swarm_controller/safety.py
<DRONE_CONNECTION_DIR>/swarm_controller/controller.py
```

### AI-deck visual observation

The AI-deck drones are used for camera streaming and visual observation. Their role is to collect image data for mapping evidence and future VSLAM or 3D reconstruction.

Implementation files:

```text
<DRONE_CONNECTION_DIR>/src/aideck_stream.py
<DRONE_CONNECTION_DIR>/gui/ai_stream_preview.py
<DRONE_CONNECTION_DIR>/gui/widgets/ai_stream_panel.py
<DRONE_CONNECTION_DIR>/swarm_controller/ai_streams.py
```

### PC-centered control

The laptop is the mission brain. This makes the system easier to debug, safer to test, and clearer to present. All decisions can be shown in the GUI and saved in logs.

Implementation files:

```text
<DRONE_CONNECTION_DIR>/gui/main.py
<DRONE_CONNECTION_DIR>/gui/main_window.py
<DRONE_CONNECTION_DIR>/gui/worker.py
<DRONE_CONNECTION_DIR>/swarm_controller/controller.py
```

### Short movement primitives

The system uses short controlled movements instead of long autonomous flight commands. This is important because Flow-deck odometry is best for stable local movement and controlled short primitives.

A short primitive usually includes:

1. Command movement.
2. Stabilize or hover.
3. Check Ranger readings.
4. Check emergency state.
5. Save logs.
6. Decide next state.

Implementation files:

```text
<DRONE_CONNECTION_DIR>/swarm_controller/motion.py
<DRONE_CONNECTION_DIR>/swarm_controller/emergency.py
<DRONE_CONNECTION_DIR>/swarm_controller/logs.py
```

---

## 5. Launch Formation

The launch pad is `120 x 118 cm`. The current half-group uses the front part of the full formation.

```text
+------------------------------------------------+
|                                                |
|                   X_FRONT                      |
|                  (32, 20)                      |
|                                                |
|      O1                           O2           |
|   (12, 45)                    (52, 45)         |
|                                                |
|                                                |
|                                                |
|                                                |
+------------------------------------------------+
```

All drones start facing the same physical forward direction. After takeoff, the controller aligns camera roles:

- `X_FRONT` faces the movement direction.
- `O1` rotates to `270 deg` for left-looking image capture.
- `O2` stays at `0 deg` for forward/right-side coverage.

The controller compensates movement after yaw alignment. This means an AI drone can rotate for camera coverage and still move correctly with the formation.

Launch formation and yaw logic are implemented in:

```text
<DRONE_CONNECTION_DIR>/swarm_controller/config.py
<DRONE_CONNECTION_DIR>/swarm_controller/formation.py
<DRONE_CONNECTION_DIR>/swarm_controller/top3_logic.py
<DRONE_CONNECTION_DIR>/swarm_controller/geometry.py
<DRONE_CONNECTION_DIR>/gui/widgets/formation_view.py
```

---

## 6. Mission Behavior

### Stage 1 - Connect and check

The operator opens the GUI and connects all drones through Crazyradio.

The GUI displays:

- drone connection state;
- battery voltage;
- Ranger readings;
- Flow/zrange state;
- AI stream status;
- mission mode;
- emergency state;
- command and event logs.

Before real flight, the operator can run sensor checks and simulation modes.

Implementation files:

```text
<DRONE_CONNECTION_DIR>/gui/widgets/drone_status_panel.py
<DRONE_CONNECTION_DIR>/gui/widgets/ranger_panel.py
<DRONE_CONNECTION_DIR>/gui/widgets/event_log_panel.py
<DRONE_CONNECTION_DIR>/swarm_controller/preflight.py
<DRONE_CONNECTION_DIR>/swarm_controller/drones.py
```

### Stage 2 - Takeoff and stabilize

The drones take off to the mission height and stabilize. The current observation height is designed for safe lab testing.

Current implemented values:

| Parameter | Value |
|---|---:|
| Test takeoff height | `0.30 m` |
| Mission flight height | `0.40 m` |
| Takeoff velocity | `0.30 m/s` |
| Landing velocity | `0.20 m/s` |
| Normal speed | `0.25 m/s` |
| Max speed clamp | `0.28 m/s` |
| Normal observation step | `0.30 m` |
| Motion segment | `0.45 s` |
| Hover after primitive | `0.20 s` |
| Hover after turn | `0.40 s` |

These values are defined in:

```text
<DRONE_CONNECTION_DIR>/swarm_controller/config.py
```

### Stage 3 - AI yaw alignment

The AI-deck drones rotate to their assigned camera directions. This gives the swarm multi-angle visual coverage while it still moves as one formation.

Current half-group yaw targets:

```text
O1 = 270 deg
O2 = 0 deg
```

### Stage 4 - Ranger-first observation step

The main loop is:

```text
READ RANGER
CHECK FORMATION ENVELOPE
MOVE X_FRONT FIRST
READ RANGER AGAIN
MOVE AI DRONES IF SAFE
CAPTURE / LOG
REPEAT
```

The Ranger drone moves first because it can directly sense nearby obstacles. The AI drones copy the movement only after the Ranger step is accepted.

### Stage 5 - Obstacle and wall handling

If the front path is blocked:

```text
AI drones hover
Ranger compares left and right clearance
Ranger chooses the more open side
Ranger moves to the front of the new direction
AI camera yaws are updated
Swarm continues observation
```

This is the main adaptive behavior of the system.

### Stage 6 - Re-slot turn behavior

When the swarm needs to turn, the AI drones stay stable in hover. `X_FRONT` moves around the outside of the formation and becomes the front scout for the new direction.

Example behavior for a right turn:

1. `O1` and `O2` hover.
2. `X_FRONT` moves to the outside side position.
3. `X_FRONT` moves into the new front position.
4. The formation heading is updated.
5. AI camera yaws are updated.
6. The observation loop continues.

This avoids a chaotic group turn and keeps the movement explainable.

Implementation files for the mission behavior:

```text
<DRONE_CONNECTION_DIR>/swarm_controller/controller.py
<DRONE_CONNECTION_DIR>/swarm_controller/motion.py
<DRONE_CONNECTION_DIR>/swarm_controller/safety.py
<DRONE_CONNECTION_DIR>/swarm_controller/classifier.py
<DRONE_CONNECTION_DIR>/swarm_controller/frontiers.py
<DRONE_CONNECTION_DIR>/swarm_controller/scout_sweep.py
```

---

## 7. AI-Deck Streaming

The AI-deck drones connect to the Wi-Fi network:

```text
SSID: swarming
Password: swarming
Port: 5000
```

The AI decks stream visual data to the laptop while the swarm moves. The GUI can scan the `swarming` network for AI-deck stream servers and display live previews.

In the current MVP, AI streams are used for observation and mapping evidence. Navigation decisions remain Ranger-led.

Implementation files:

```text
<DRONE_CONNECTION_DIR>/main.py
<DRONE_CONNECTION_DIR>/src/aideck_stream.py
<DRONE_CONNECTION_DIR>/gui/ai_stream_preview.py
<DRONE_CONNECTION_DIR>/gui/widgets/ai_stream_panel.py
<DRONE_CONNECTION_DIR>/docs/ai-deck-streaming.md
```

GAP8 streamer source:

```text
<DRONE_CONNECTION_DIR>/aideck-gap8-examples/examples/other/wifi-img-streamer/wifi-img-streamer.c
```

---

## 8. GUI Features

The PyQt6 GUI is the main control dashboard for testing and demonstration.

Main GUI files:

```text
<DRONE_CONNECTION_DIR>/gui/main.py
<DRONE_CONNECTION_DIR>/gui/main_window.py
<DRONE_CONNECTION_DIR>/gui/worker.py
```

Widget files:

```text
<DRONE_CONNECTION_DIR>/gui/widgets/drone_status_panel.py
<DRONE_CONNECTION_DIR>/gui/widgets/mission_control_panel.py
<DRONE_CONNECTION_DIR>/gui/widgets/ranger_panel.py
<DRONE_CONNECTION_DIR>/gui/widgets/formation_view.py
<DRONE_CONNECTION_DIR>/gui/widgets/ai_stream_panel.py
<DRONE_CONNECTION_DIR>/gui/widgets/event_log_panel.py
<DRONE_CONNECTION_DIR>/gui/widgets/thresholds_panel.py
```

Main features:

- connect all drones;
- run simulation mode;
- run sensor checks;
- test individual takeoff;
- run scout sweep;
- run formation hover;
- run formation micro step;
- start full observation mode;
- pause / hover;
- resume;
- land all;
- emergency stop / hard kill;
- safe hover / land;
- save logs;
- start AI-deck streams;
- stop AI-deck streams;
- scan Wi-Fi for AI-deck IP addresses.

The GUI also displays:

- formation visualization;
- drone roles and addresses;
- Ranger readings: front, back, left, right, up;
- formation envelope state;
- current mission state;
- battery summary;
- radio status;
- AI stream previews;
- command and event logs.

---

## 9. How to Run the Project

### 9.1. Recommended hardware setup

Before running the system, prepare:

- Crazyflie drones with correct addresses;
- one Ranger drone with Flow deck + Multi-ranger deck;
- AI-deck drones with Flow deck + AI deck;
- charged batteries;
- Crazyradio connected to the PC;
- `swarming` Wi-Fi network active for AI-deck streams;
- textured floor and good lighting;
- enough open space for first tests;
- emergency stop visible in the GUI.

### 9.2. Open the project directory

Go to the drone connection directory. Replace `<DRONE_CONNECTION_DIR>` with your local path.

```bash
cd "<DRONE_CONNECTION_DIR>"
```

Example meaning:

```text
<DRONE_CONNECTION_DIR> = <PROJECT_ROOT>/testing/drone_connection
```

### 9.3. Create and activate the virtual environment

If the virtual environment does not exist yet, create it from `<PROJECT_ROOT>`:

```bash
cd "<PROJECT_ROOT>"
python -m venv .venv
```

Activate the virtual environment.

Windows PowerShell:

```powershell
& "<VENV_DIR>\Scripts\Activate.ps1"
```

If activation is blocked in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& "<VENV_DIR>\Scripts\Activate.ps1"
```

macOS / Linux:

```bash
source "<VENV_DIR>/bin/activate"
```

Install dependencies from `<DRONE_CONNECTION_DIR>`:

```bash
cd "<DRONE_CONNECTION_DIR>"
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 9.4. Run the GUI

From `<DRONE_CONNECTION_DIR>`:

```bash
python -m gui.main
```

### 9.5. Run simulation from command line

From `<DRONE_CONNECTION_DIR>`:

```bash
python -m swarm_controller.main --simulate --half-group --scenario open_space --steps 5
```

### 9.6. Run real half-group backend from command line

The GUI is preferred for presentation, but the backend can also be started directly.

From `<DRONE_CONNECTION_DIR>`:

```bash
python -m swarm_controller.main --half-group --mission-id final_demo_half_group --steps 5 --x-front-uri radio://0/82/2M/E7E7E7E701 --o1-uri radio://0/82/2M/E7E7E7E702 --o2-uri radio://0/82/2M/E7E7E7E703
```

---

## 10. Recommended Testing Workflow

Use staged testing.

Recommended order:

1. Place drones correctly on the launch pad.
2. Open the GUI.
3. Click `Load Half Group (3)`.
4. Click `Use Real Crazyflies`.
5. Connect all drones.
6. Check battery state.
7. Run sensor checks.
8. Confirm `X_FRONT` Ranger values are reasonable.
9. Start or verify AI-deck streams.
10. Run simulation modes first.
11. Test takeoff for `X_FRONT` only.
12. Test takeoff for `O1` and `O2` individually if needed.
13. Run formation hover.
14. Run formation micro step.
15. Start full observation mode.
16. Use safe hover / land for normal stop.
17. Use emergency stop / hard kill if needed.

---

## 11. Simulation and Unit Tests

The project includes simulation and controller behavior tests. They check formation geometry, re-slot behavior, AI yaw compensation, emergency handling, observation logic, sensor checks, and log behavior.

Run tests from `<DRONE_CONNECTION_DIR>`:

```bash
cd "<DRONE_CONNECTION_DIR>"
python -m unittest gui.tests.test_button_wiring swarm_controller.tests.test_simulation swarm_controller.tests.test_controller_behavior swarm_controller.tests.test_top3_logic
```

Expected latest result from the project:

```text
Ran 51 tests
OK
```

---

## 12. Logging and Evidence

Every mission creates logs for debugging, analysis, and presentation evidence.

Mission log directory:

```text
<DRONE_CONNECTION_DIR>/stream_out/fixed_formation_missions/<mission_id>_<timestamp>/
```

| File | Purpose |
|---|---|
| `README_LOGS.md` | Human-readable guide for reviewing a mission run |
| `run_summary.json` | Final mission summary |
| `formation_config.json` | Formation slots and yaw configuration for the run |
| `mission_log.csv` | Mission phases, command sequence, and state changes |
| `command_log.csv` | Command ids, targets, and command results |
| `ranger_log.csv` | Live Ranger readings and safety context |
| `safety_grid_log.csv` | Formation envelope checks |
| `capture_manifest.csv` | AI-deck capture events |
| `classification_log.csv` | Wall/obstacle decisions |
| `reslot_path_log.csv` | Ranger path checks during re-slot turns |
| `turn_reslot_log.csv` | Ranger re-slot turn behavior |
| `breadcrumb_log.csv` | Movement history |
| `preflight_log.csv` | Preflight and sensor-check evidence |
| `battery_log.csv` | Battery values during the mission |
| `mission_timing_log.csv` | Mission phase timing |
| `error_log.csv` | Errors with context |

These logs are important because they explain why the swarm made each movement decision.

Log implementation:

```text
<DRONE_CONNECTION_DIR>/swarm_controller/logs.py
```

---

## 13. File Map

```text
<DRONE_CONNECTION_DIR>/gui/main.py
    GUI entry point.

<DRONE_CONNECTION_DIR>/gui/main_window.py
    Main PyQt6 window.

<DRONE_CONNECTION_DIR>/gui/worker.py
    Worker thread and command queue.

<DRONE_CONNECTION_DIR>/gui/ai_stream_preview.py
    AI-deck live preview manager.

<DRONE_CONNECTION_DIR>/gui/widgets/
    GUI panels for controls, drones, Ranger readings, formation, AI streams, logs, and thresholds.

<DRONE_CONNECTION_DIR>/swarm_controller/config.py
    Current drone addresses, formation mode, movement parameters, battery thresholds.

<DRONE_CONNECTION_DIR>/swarm_controller/controller.py
    Main SwarmController, mission workflow, observation behavior, emergency paths.

<DRONE_CONNECTION_DIR>/swarm_controller/state_machine.py
    Mission state definitions and transitions.

<DRONE_CONNECTION_DIR>/swarm_controller/motion.py
    Formation-frame velocity primitives, yaw alignment, hover and land helpers.

<DRONE_CONNECTION_DIR>/swarm_controller/formation.py
    Fixed formation geometry, X_FRONT target slots, turn waypoints, yaw targets.

<DRONE_CONNECTION_DIR>/swarm_controller/safety.py
    Ranger threshold checks, envelope checks, re-slot path safety.

<DRONE_CONNECTION_DIR>/swarm_controller/ranger.py
    Ranger reading model and monitor.

<DRONE_CONNECTION_DIR>/swarm_controller/ai_streams.py
    AI stream status, capture metadata, stream lifecycle helpers.

<DRONE_CONNECTION_DIR>/swarm_controller/logs.py
    CSV mission logs and diagnostics.

<DRONE_CONNECTION_DIR>/swarm_controller/tests/
    Simulation and behavior tests.

<DRONE_CONNECTION_DIR>/src/aideck_stream.py
    AI-deck TCP image stream reader.

<DRONE_CONNECTION_DIR>/aideck-gap8-examples/examples/other/wifi-img-streamer/wifi-img-streamer.c
    GAP8 AI-deck Wi-Fi image streamer source.

<DRONE_CONNECTION_DIR>/docs/
    Research, installation, and presentation documentation.
```

---

## 14. Safety Notes

Important safety rules:

- Test with propellers off until connection and sensor checks pass.
- Start with simulation mode before real flight.
- Use fresh batteries.
- Use good lighting and a matte textured floor.
- Keep the first real flight very small.
- Always keep emergency stop visible.
- Run hover and micro-step tests before full observation mode.
- Stop immediately if a drone drifts, tumbles, loses connection, or behaves unexpectedly.

---

## 15. Current Scope

The current system is a controlled half-group MVP and the architecture is designed to scale.

Current implemented scope:

- three-drone half-group: `X_FRONT`, `O1`, `O2`;
- Ranger-led movement decisions;
- Flow-assisted short local movement;
- AI-deck camera streaming;
- PyQt6 control GUI;
- simulation mode;
- mission logs and evidence;
- emergency stop / hard kill;
- safe hover / land;
- formation re-slot turn behavior.

Planned expansion:

- add `X_BACK` Ranger safety drone;
- expand to four AI stream drones;
- add more complete VSLAM / reconstruction processing;
- use saved image and mission logs for mapping evidence.

---

## 16. Troubleshooting

| Problem | Possible cause | Recommended action |
|---|---|---|
| Drone does not connect | Wrong URI, busy Crazyradio, another process holds the link | Close cfclient/Python processes, check URI, reconnect Crazyradio |
| Too many packets lost | Weak radio link, wrong address, interference | Move Crazyradio closer, use correct address, connect one drone at a time |
| AI stream not visible | AI deck not connected to `swarming` Wi-Fi or wrong IP | Check Wi-Fi, scan from GUI, power-cycle AI deck |
| Ranger values invalid | Surface issue, bad angle, low reflection, deck problem | Test in open space, check deck placement, verify readings before flight |
| Flow/zrange warning | Low-texture floor or poor lighting | Use matte textured floor and better lighting |
| Drone drifts during takeoff | Bad calibration, uneven surface, wrong propellers, weak motor | Restart on flat surface, test individual hover, inspect hardware |
| Re-slot too close | Formation spacing or real hardware spacing mismatch | Increase clearance and test in simulation first |
| Emergency lock | Crash/tumble/safety lock | Reboot drone and test single-drone hover before swarm flight |

---

## 187 Future Work

Planned next steps:

1. Stabilize half-group observation in open space and wall scenarios.
2. Improve AI stream capture manifest with frame metadata.
3. Add `X_BACK` Ranger drone as rear/right safety veto.
4. Expand to four AI streams for left, forward, backward, and right coverage.
5. Integrate VSLAM or 3D reconstruction using captured frames.
6. Add compact P2P command/status messages as an optional future layer.

---

## 18. Final Status

The current codebase is ready for controlled half-group testing with one Ranger scout and two AI-deck streamers.

It includes:

- PC-centered swarm control;
- PyQt6 GUI;
- simulation modes;
- Ranger-led safety checks;
- AI-deck streaming support;
- formation geometry;
- re-slot turn behavior;
- safe hover and land paths;
- emergency stop / hard kill;
- mission logs;
- scalable architecture for the future six-drone swarm.

The system is ready to present as a practical and explainable MVP for Ranger-guided AI-deck indoor swarm exploration.
