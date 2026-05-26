from utils import address
from utils.drone import DroneCommand, MultiDroneCommand

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.swarm import Swarm, CachedCfFactory


import time
import threading

available = address.scan_drones()

print(available)

# swarm: list[SyncCrazyflie] = []
# events = []

# for uri in available:
#     cf = Crazyflie(rw_cache="./cache")
#     scf = SyncCrazyflie(uri, cf=cf)
#     evt = threading.Event()

#     def connected_cb(link_uri, evt=evt):
#         print(f"connected {link_uri}")
#         evt.set()

#     def failed_cb(link_uri, msg, evt=evt):
#         print(f"failed {link_uri}")
#         print(msg)
#         evt.set()

#     cf.connected.add_callback(connected_cb)
#     cf.connection_failed.add_callback(failed_cb)
#     cf.connection_lost.add_callback(failed_cb)

#     scf.open_link()
#     evt.wait()

#     if cf.is_connected:
#         swarm.append(scf)
#         events.append(evt)

# MultiDroneCommand.deactivate_led_bit_mask(swarm)
# print("lights off")
# MultiDroneCommand.activate_led_bit_mask(swarm)
# print("lights on")
# MultiDroneCommand.deactivate_led_bit_mask(swarm)
# print("lights off")

factory = CachedCfFactory(rw_cache='./cache')
with Swarm(available, factory=factory) as swarm:
    swarm.parallel_safe(DroneCommand.activate_led_bit_mask)
    time.sleep(2)
    swarm.parallel_safe(DroneCommand.deactivate_led_bit_mask)
    swarm.parallel_safe(DroneCommand.take_off)
    time.sleep(3)
    swarm.parallel_safe(DroneCommand.land)