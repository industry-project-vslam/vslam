from utils import address
from utils.drone import DroneCommand, MultiDroneCommand

import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.swarm import Swarm, CachedCfFactory

import time


# available = address.scan_drones()
available=["radio://0/80/2M/E7E7E7E70F"]

# HARDCODED_DELTA = 1

# print("Available drones:", available)

# starting_positions = {uri: [(1 * i, 0, 0, 0)] for i, uri in enumerate(available)}
# moves = {uri: [((1,0), HARDCODED_DELTA)] for uri in available}

# print("Starting positions:", starting_positions)

# cflib.crtp.init_drivers()

# factory = CachedCfFactory(rw_cache='./cache')
# with Swarm(available, factory=factory) as swarm:
#     # swarm.parallel_safe(DroneCommand.set_position, starting_positions)
#     # time.sleep(1)
#     # swarm.parallel_safe(DroneCommand.reset_estimator)
#     swarm.parallel_safe(DroneCommand.activate_led_bit_mask)
#     time.sleep(2)
#     swarm.parallel_safe(DroneCommand.deactivate_led_bit_mask)

#     swarm.parallel_safe(DroneCommand.take_off)
#     time.sleep(1)

#     # Get estimated positions after takeoff
#     positions_after_takeoff = swarm.get_estimated_positions()
#     print("\nPositions after takeoff:")
#     for uri, pos in positions_after_takeoff.items():
#         print(f"  {uri} -> (x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f})")

#     swarm.parallel_safe(DroneCommand.run_step, moves)
#     time.sleep(3)

#     # Get estimated positions after moves
#     positions_after_moves = swarm.get_estimated_positions()
#     print("\nPositions after moves:")
#     for uri, pos in positions_after_moves.items():
#         print(f"  {uri} -> (x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f})")

#     swarm.parallel_safe(DroneCommand.land)

# -*- coding: utf-8 -*-
"""
Single Crazyflie flying in a square using the high-level commander.
Requires:
  - Flow deck (or other positioning system)
  - Positioning enabled in firmware
  - crazyflie-lib-python installed

pip install crazyflie-lib-python
"""

import time
import sys
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crtp import init_drivers

# Configuration
URI = 'radio://0/80/2M/E7E7E7E702'  # Change to your drone's URI
SIDE_LENGTH = 1.0  # meters
TAKEOFF_HEIGHT = 0.5  # meters
SPEED = 0.5  # m/s

# High-level commander setpoint: (x, y, z, yaw)
# We'll fly a square in the XY plane at constant height.

def square_trajectory(cf: SyncCrazyflie):
    """
    Send a sequence of high-level setpoints to fly a square.
    The high-level commander on the Crazyflie will execute the path.
    """

    hl_commander = cf.cf.high_level_commander

    # Define square corners (starting at origin, going CCW)
    # Start -> (SIDE_LENGTH, 0) -> (SIDE_LENGTH, SIDE_LENGTH) -> (0, SIDE_LENGTH) -> (0, 0)
    corners = [
        (SIDE_LENGTH, 0, TAKEOFF_HEIGHT, 0.0),
        (SIDE_LENGTH, SIDE_LENGTH, TAKEOFF_HEIGHT, 0.0),
        (0.0, SIDE_LENGTH, TAKEOFF_HEIGHT, 0.0),
        (0.0, 0.0, TAKEOFF_HEIGHT, 0.0),
    ]

    # Use take_off first
    hl_commander.takeoff(TAKEOFF_HEIGHT, 1.0)  # height, duration
    time.sleep(1.2)

    # Fly each side of the square
    for x, y, z, yaw in corners:
        # go_to(x, y, z, yaw, relative=False, duration)
        # duration is in seconds, choose based on distance / speed
        distance = (x**2 + y**2)**0.5 if x != 0 or y != 0 else 0
        duration = max(2.0, distance / SPEED)  # at least 2s per segment
        hl_commander.go_to(x, y, z, yaw, duration_s=duration, relative=False)
        time.sleep(duration + 0.2)

    # Land
    hl_commander.land(0.0, 1.5)
    time.sleep(1.7)


def main():
    init_drivers()

    print("Connecting to Crazyflie at", URI)

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as cf:
        # Optional: wait for position estimate to be ready
        # print("Waiting for position estimate...")
        # for _ in range(30):
        #     pos = cf.positioning.get_position()
        #     if pos is not None:
        #         print(f"Position estimate ready: {pos}")
        #         break
        #     time.sleep(0.5)
        # else:
        #     print("Warning: No position estimate received, but continuing...")

        # Fly the square
        print("Flying square trajectory...")
        square_trajectory(cf)
        print("Square flight complete.")


if __name__ == '__main__':
    main()