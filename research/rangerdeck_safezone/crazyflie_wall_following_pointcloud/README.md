# Crazyflie wall following + Multi-ranger point cloud

This merged example combines Bitcraze's `multiranger_pointcloud` demo with the `multiranger_wall_following` demo.

It will:

1. connect to the Crazyflie,
2. arm and hover at `HOVER_HEIGHT`,
3. follow a wall automatically using the Multi-ranger deck,
4. continuously add Multi-ranger measurements to the live Vispy point-cloud map.

## Hardware

- Crazyflie 2.x
- Crazyradio
- Flow deck
- Multi-ranger deck

## Install/run

Install the dependencies used by the original examples, typically:

```bash
pip install cflib vispy PyQt6 numpy
```

Run with the default URI:

```bash
python multiranger_wall_following_pointcloud.py
```

Or pass a URI:

```bash
python multiranger_wall_following_pointcloud.py radio://0/80/2M/E7E7E7E701
```

## Controls and tuning

Edit the constants at the top of `multiranger_wall_following_pointcloud.py`:

- `WALL_FOLLOWING_DIRECTION`: `LEFT` or `RIGHT`
- `REFERENCE_DISTANCE_FROM_WALL`: desired wall distance in meters
- `MAX_FORWARD_SPEED`: forward speed in m/s
- `MAX_TURN_RATE`: yaw rate limit in rad/s
- `HOVER_HEIGHT`: flight height in meters

Cover/trigger the up-facing sensor or press `Esc` to stop and land.
