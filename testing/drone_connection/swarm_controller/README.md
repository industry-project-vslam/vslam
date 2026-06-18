# Ranger-Led Fixed-Formation Streaming Swarm

This package is a realistic MVP for a PC-centered Crazyflie swarm:

- `X_FRONT`: front Ranger scout
- `X_BACK`: rear Ranger safety veto
- `O1`-`O4`: passive AI-deck streaming drones
- Flow decks are assumed on all drones for stable hover and short local motion

The MVP does not use direct P2P for flight control. Crazyradio/cflib sends
commands from the PC, and AI-deck image streams are logged as passive metadata.
AI frames do not influence navigation decisions.

## Formation

The local formation model uses the 120 x 118 launch-pad drawing coordinates:

- `x` grows to the right
- `y` grows downward
- initial heading `NORTH` points toward smaller `y`
- formation center is `(32, 60)`
- default scale is `1 drawing unit = 0.02 m`

| Drone | x | y | Role |
|---|---:|---:|---|
| `X_FRONT` | `32` | `20` | front Ranger |
| `O1` | `12` | `45` | left stream |
| `O2` | `52` | `45` | forward stream |
| `O3` | `12` | `80` | backward stream |
| `O4` | `52` | `80` | right stream |
| `X_BACK` | `32` | `100` | rear Ranger |

The AI core stays fixed. On turns, only `X_FRONT` and `X_BACK` re-slot:

| Heading | X_FRONT | X_BACK |
|---|---|---|
| `NORTH` | `(32, 20)` | `(32, 100)` |
| `EAST` | `(72, 60)` | `(-8, 60)` |
| `SOUTH` | `(32, 100)` | `(32, 20)` |
| `WEST` | `(-8, 60)` | `(72, 60)` |

For a `NORTH -> EAST` turn the Ranger paths are:

```text
X_FRONT: (32,20)  -> (72,20)  -> (72,60)
X_BACK:  (32,100) -> (-8,100) -> (-8,60)
```

These outside-corner paths avoid moving Ranger drones through the AI core.
In the current 3-drone half-group test (`X_FRONT`, `O1`, `O2`), only
`X_FRONT` is re-slotted. For a `NORTH -> EAST` turn the half-group target is
`(32,0) -> (97,0) -> (97,45)`. This keeps the Ranger about `0.45 m`
outside the right AI drone, and uses a larger `0.45 m` front/back separation
from the AI row. The safety check uses the known re-slot leg length plus the
formation margin, so a turn is allowed only when the target side has enough
clearance. If that clearance is not available, the controller saves a frontier
and does not force the turn.

Commands are formation-frame primitives:

- `FORMATION_FORWARD`
- `FORMATION_LEFT`
- `FORMATION_RIGHT`
- `TURN_LEFT_90`
- `TURN_RIGHT_90`
- `HOVER`
- `LAND`

No line mode is implemented in this MVP. If the full fixed formation does not
fit, the swarm saves a frontier and chooses another heading or lands.

## Safety Thresholds

Defaults are in `config.py`:

- initial proof step: `0.20 m`
- normal movement step: `0.30 m`
- single-drone test height: `0.30 m`
- MVP observation height: `0.40 m`
- takeoff velocity: `0.30 m/s`
- landing velocity: `0.20 m/s`
- initial proof horizontal velocity: `0.18 m/s`
- normal MVP horizontal velocity: `0.25 m/s`
- maximum MVP horizontal velocity: `0.28 m/s`
- takeoff stagger delay in real mode: `2.0 s`
- yaw rate: `72 deg/s`
- minimum battery for staged tests: `3.05 V`
- target wall offset: `3.5 m`
- critical front: `0.70 m`
- critical side/back: `0.50 m`
- critical up: `0.40 m`
- front personal-space recovery target: `0.85 m`
- front personal-space recovery max backoff: `0.30 m`
- probe step: `0.25 m`
- max probe shift: `1.25 m`
- formation margin: `0.50 m`

Unknown space is unsafe. AI streams never override Ranger safety.

If a person or obstacle enters the front critical zone, the controller does not
blindly keep moving forward. It first holds the AI drones, lets `X_FRONT` back
away up to `0.30 m` only if its back Ranger reading is clear, then chooses the
turn/re-slot direction from the live `left`/`right` Ranger distances. If back
clearance is not safe, it skips the backoff and tries the side re-slot decision
directly. For that immediate escape only, an unknown `X_FRONT` back leg does not
freeze the side re-slot; side, up, emergency, battery, and finite critical back
readings still stop motion.

## Simulation

Run the fake-room simulation first:

```powershell
cd "D:\2024-2025\MTS4-MCTE Industry Project\vslam\testing\drone_connection"
& "D:\2024-2025\MTS4-MCTE Industry Project\vslam\.venv\Scripts\python.exe" -m swarm_controller.main --simulate --scenario open_space --steps 3
```

Available scenarios:

- `open_space`
- `wall_ahead`
- `local_obstacle`
- `ambiguous_wide_obstacle`
- `side_unknown`
- `turn_reslot_safe`
- `turn_reslot_one_side_blocked`

Run tests:

```powershell
& "D:\2024-2025\MTS4-MCTE Industry Project\vslam\.venv\Scripts\python.exe" -m unittest swarm_controller.tests.test_simulation
```

## Real Crazyflie Run

Use the GUI staged workflow first:

1. `Connect All`
2. `Sensor Check`
3. `Test Takeoff X_FRONT`
4. `Run Scout Sweep`
5. `Wall/Obstacle Probe Test`
6. `Formation Hover Only`
7. `Formation Micro Step`
8. `Start Full Observation Mode`

Full observation is blocked until the staged safety tests pass. After those
checks pass, Full Observation sends real guarded movement primitives: short
forward formation steps when the Ranger envelope is clear, lateral probes for
surface candidates, and Ranger re-slot turns when a wall/boundary is confirmed.
The first real motion proof uses `0.20 m` at `0.18 m/s`; the observation
mission normally uses `0.25 m/s` and remains capped at `0.28 m/s`.

CLI simulation example:

```powershell
& "D:\2024-2025\MTS4-MCTE Industry Project\vslam\.venv\Scripts\python.exe" -m swarm_controller.main --steps 1 --x-front-uri radio://0/82/2M/E7E7E7E701 --x-back-uri radio://0/82/2M/E7E7E7E713 --o1-uri radio://0/82/2M/E7E7E7E702 --o2-uri radio://0/82/2M/E7E7E7E703
```

Only provide URIs for drones you actually want to connect.

## Logs

Each run creates a mission folder with:

- `formation_config.json`
- `README_LOGS.md`
- `run_summary.json`
- `event_log.csv`
- `decision_log.csv`
- `state_snapshot_log.csv`
- `reslot_path_log.csv`
- `error_log.csv`
- `ranger_log.csv`
- `scout_sweep_log.csv`
- `classification_log.csv`
- `bypass_log.csv`
- `turn_reslot_log.csv`
- `frontier_log.csv`
- `breadcrumb_log.csv`
- `ai_stream_log.csv`
- `command_log.csv`

To summarize the newest run after a test:

```powershell
cd "D:\2024-2025\MTS4-MCTE Industry Project\vslam\testing\drone_connection"
& "D:\2024-2025\MTS4-MCTE Industry Project\vslam\.venv\Scripts\python.exe" -m swarm_controller.diagnose_logs --latest
```

For debugging, send the whole newest mission folder. The most useful files are
`event_log.csv`, `decision_log.csv`, `state_snapshot_log.csv`,
`command_log.csv`, `reslot_path_log.csv`, and `ranger_log.csv`.

## Limitations

- Multi-ranger is not 360 degree LiDAR.
- This is conservative and safety-first, not guaranteed collision-proof.
- Flow deck odometry is local and drifts.
- Narrow passages are blocked/frontier because line mode is intentionally not used.
- Firmware should remain passive for MVP; do not run onboard autonomous logic that fights PC commands.
