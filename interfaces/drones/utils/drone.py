import time
import cflib
from cflib.crazyflie.swarm import SwarmPosition
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger

FLIGHT_HEIGHT = 0.75 # Height at which a drone will fly
Z_MIN = 0.0 # The lowest height a drone can go (ground level)

V_XY_DRONE = 0.20 # the speed of a drone (in meters/second)
V_Z_DRONE = 0.5 # the speed of a drone (in meters/second)

T_TAKEOFF = FLIGHT_HEIGHT / V_Z_DRONE # time it takes a drone to go from Z_MIN to FLIGHT_HEIGHT
T_LANDING = FLIGHT_HEIGHT / V_Z_DRONE # time it takes a drone to go from FLIGHT_HEIGHT to Z_MIN

class MultiDroneCommand:
    @staticmethod
    def get_uris():
        cflib.crtp.init_drivers()
        available = cflib.crtp.scan_interfaces()

    @staticmethod
    def activate_led_bit_mask(swarm: list[SyncCrazyflie]):
        for scf in swarm:
            DroneCommand.activate_led_bit_mask(scf)

    @staticmethod
    def deactivate_led_bit_mask(swarm: list[SyncCrazyflie]):
        for scf in swarm:
            DroneCommand.deactivate_led_bit_mask(scf)

    @staticmethod
    def light_check(swarm: list[SyncCrazyflie]):
        MultiDroneCommand.activate_led_bit_mask(swarm)
        time.sleep(2)
        MultiDroneCommand.deactivate_led_bit_mask(swarm)

    @staticmethod
    def get_positions(swarm: list[SyncCrazyflie]):
        return {scf._link_uri: DroneCommand.get_position(scf) for scf in swarm}

    @staticmethod
    def take_off(swarm: list[SyncCrazyflie]):
        for scf in swarm:
            DroneCommand.take_off(scf)
        time.sleep(T_TAKEOFF)

    @staticmethod
    def land(swarm: list[SyncCrazyflie]):
        for scf in swarm:
            DroneCommand.land(scf)
        time.sleep(T_LANDING)
        MultiDroneCommand.stop_rotors(swarm)

    @staticmethod
    def stop_rotors(swarm: list[SyncCrazyflie]):
        for scf in swarm:
            DroneCommand.stop_rotors(scf)

    @staticmethod
    def run_step(swarm: list[SyncCrazyflie], args_dict, max_swarm_delta):
        for scf in swarm:
            drone_move = args_dict[scf._link_uri][0]
            DroneCommand.run_step(scf, drone_move)
        wait_time = max_swarm_delta / V_XY_DRONE
        time.sleep(wait_time)

    @staticmethod
    def reset_estimators(swarm: list[SyncCrazyflie]):
        print('Waiting for estimators to find positions...', end='\r')
        for scf in swarm:
            DroneCommand.reset_estimator(scf)
        print('Waiting for estimators to find positions...success!')

class DroneCommand:
    @staticmethod
    def activate_led_bit_mask(scf: SyncCrazyflie):
        scf.cf.param.set_value('led.bitmask', 255)

    @staticmethod
    def deactivate_led_bit_mask(scf: SyncCrazyflie):
        scf.cf.param.set_value('led.bitmask', 0)

    @staticmethod
    def get_position(scf: SyncCrazyflie):
        log_config = LogConfig(name='stateEstimate', period_in_ms=10)
        log_config.add_variable('stateEstimate.x', 'float')
        log_config.add_variable('stateEstimate.y', 'float')
        log_config.add_variable('stateEstimate.z', 'float')

        with SyncLogger(scf, log_config) as logger:
            for entry in logger:
                x = entry[1]['stateEstimate.x']
                y = entry[1]['stateEstimate.y']
                z = entry[1]['stateEstimate.z']
                return SwarmPosition(x, y, z)

    @staticmethod
    def take_off(scf: SyncCrazyflie):
        commander = scf.cf.high_level_commander
        commander.takeoff(FLIGHT_HEIGHT, T_TAKEOFF)

    @staticmethod
    def land(scf: SyncCrazyflie):
        commander = scf.cf.high_level_commander
        commander.land(Z_MIN, T_LANDING)

    @staticmethod
    def stop_rotors(scf: SyncCrazyflie):
        commander = scf.cf.high_level_commander
        commander.stop()

    @staticmethod
    def run_step(scf: SyncCrazyflie, move: tuple[tuple[float, float], float]):
        (x_pn, y_pn), delta_p = move
        duration = delta_p / V_XY_DRONE

        commander = scf.cf.high_level_commander
        commander.go_to(x_pn,y_pn,FLIGHT_HEIGHT,0,duration)

    @staticmethod
    def wait_for_position_estimator(scf: SyncCrazyflie):
            log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
            log_config.add_variable('kalman.varPX', 'float')
            log_config.add_variable('kalman.varPY', 'float')
            log_config.add_variable('kalman.varPZ', 'float')

            var_y_history = [1000] * 10
            var_x_history = [1000] * 10
            var_z_history = [1000] * 10

            threshold = 0.001

            with SyncLogger(scf, log_config) as logger:
                for log_entry in logger:
                    data = log_entry[1]

                    var_x_history.append(data['kalman.varPX'])
                    var_x_history.pop(0)
                    var_y_history.append(data['kalman.varPY'])
                    var_y_history.pop(0)
                    var_z_history.append(data['kalman.varPZ'])
                    var_z_history.pop(0)

                    min_x = min(var_x_history)
                    max_x = max(var_x_history)
                    min_y = min(var_y_history)
                    max_y = max(var_y_history)
                    min_z = min(var_z_history)
                    max_z = max(var_z_history)

                    if (max_x - min_x) < threshold and (
                            max_y - min_y) < threshold and (
                            max_z - min_z) < threshold:
                        break

    @staticmethod
    def reset_estimator(scf: SyncCrazyflie):
        cf = scf.cf
        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0')
        DroneCommand.wait_for_position_estimator(scf)