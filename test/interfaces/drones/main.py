from utils import address
from utils.drone import DroneCommand, MultiDroneCommand

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.swarm import Swarm, CachedCfFactory


import time

available = address.scan_drones()

HARDCODED_DELTA = 1
POSITION_ORIGIN = (0,0)

print(available)

starting_positions = {available[i]: [(1*i, 0,0,0)] for i in range(len(available))}
moves = {available[i]: [(POSITION_ORIGIN,HARDCODED_DELTA)] for i in range(len(available))}

print(starting_positions)

factory = CachedCfFactory(rw_cache='./cache')
with Swarm(available, factory=factory) as swarm:
    swarm.parallel_safe(DroneCommand.set_position, starting_positions)
    time.sleep(1)
    swarm.parallel_safe(DroneCommand.reset_estimator)
    swarm.parallel_safe(DroneCommand.activate_led_bit_mask)
    time.sleep(2)
    swarm.parallel_safe(DroneCommand.deactivate_led_bit_mask)

    swarm.parallel_safe(DroneCommand.take_off)
    time.sleep(1)
    swarm.parallel_safe(DroneCommand.run_step, moves)
    time.sleep(3)
    swarm.parallel_safe(DroneCommand.land)