# Passive Swarm App-Layer Skeleton

This folder is an isolated Crazyflie app-layer skeleton for the fixed-formation
swarm project.

It is intentionally not a full autonomous swarm firmware.

## MVP Scope

Implemented:

- `swarm.role` parameter:
  - `1` = `FRONT_RANGER`
  - `2` = `BACK_RANGER`
  - `3` = `AI_STREAM`
- `swarm.passive` parameter, default `1`
- `swarm.emergency` parameter
- log variables:
  - `swarm.state`
  - `swarm.seq`
  - `swarm.role`
  - `swarm.lastCmd`
  - `swarm.emergency`
- compact future command struct in `swarm_protocol.h`

Not implemented in MVP:

- autonomous onboard swarm routing
- P2P-dependent motion
- image transport
- direct motor control

## Integration

Copy these files into a Crazyflie firmware app layer folder, for example:

```text
crazyflie-firmware/examples/app_swarm_passive/
```

Then build the app using the normal Crazyflie firmware app-layer flow for your
repository version.

The PC-centered Python controller remains the source of truth for movement
decisions during MVP testing.

## Safety

Keep `swarm.passive=1` while testing the PC-centered MVP. Do not run an onboard
executor that can fight cflib setpoints from the PC.

