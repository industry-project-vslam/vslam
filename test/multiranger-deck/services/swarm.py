from utils.address import scan_drones
from services.drone import DroneService
from services.navigation import NavigationService

import time
import cflib.crtp

START_PLACEHOLDER = {"x": 0.0, "y": 0.0, "yaw": 0.0, "z": 0.0}

class SwarmService:
    def __init__(self):
        self.navigation = NavigationService()
        self.drones: dict[str, DroneService] = {}

        cflib.crtp.init_drivers(enable_debug_driver=False)
        # drone_uris = scan_drones()
        drone_uris = [
            "radio://0/80/2M/E7E7E7E70A"
        ]

        drone_start_positions = {uri: {"x": 0.0, "y": 1*i, "yaw": 0.0, "z": 0.0} for i, uri in enumerate(drone_uris)}

        for uri, position in drone_start_positions.items():

            self.drones[uri] = DroneService(uri, position, self.receive_meas_update)
    
    def receive_meas_update(self, uri: str, meas: dict):
        self.drones[uri].position.update({"x": meas["x"], "y": meas["y"], "yaw": meas["yaw"], "z": meas["z"]})
        self.navigation.update_obstacles(meas)

    def take_step(self):
        drone_steps = self.navigation.step()

        for uri, drone in self.drones.items():
            drone.send_hover_command(drone_steps[uri])

    def close_links(self):
        self.navigation.save_plot()
        drones = self.drones.values()

        for drone in drones:
            drone.cf.commander.send_hover_setpoint(0,0,0,0)

        time.sleep(0.5)

        for drone in drones:
            drone.cf.high_level_commander.stop()

        time.sleep(0.5)

        for drone in drones:
            drone.cf.close_link()
        
