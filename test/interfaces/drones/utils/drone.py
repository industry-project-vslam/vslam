import time
from cflib.crazyflie.swarm import SwarmPosition
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger

class DroneCommand:
    @staticmethod
    def activate_led_bit_mask(scf: SyncCrazyflie):
        scf.cf.param.set_value('led.bitmask', 255)

    @staticmethod
    def deactivate_led_bit_mask(scf: SyncCrazyflie):
        scf.cf.param.set_value('led.bitmask', 0)

    @staticmethod
    def light_check(scf: SyncCrazyflie):
        DroneCommand.activate_led_bit_mask(scf)
        time.sleep(1.00)
        DroneCommand.deactivate_led_bit_mask(scf)

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
    def take_off(scf: SyncCrazyflie, take_off: tuple[float, float]):
        z, t_take_off = take_off
        commander = scf.cf.high_level_commander
        commander.takeoff(z, t_take_off)
        time.sleep(t_take_off)

    @staticmethod
    def land(scf: SyncCrazyflie, landing: tuple[float, float]):
        z, t_landing = landing
        commander = scf.cf.high_level_commander
        commander.land(z, t_landing)
        time.sleep(t_landing)

    @staticmethod
    def stop_rotors(scf: SyncCrazyflie):
        commander = scf.cf.high_level_commander
        commander.stop()

    @staticmethod
    def run_step(scf: SyncCrazyflie, move: tuple[float, float, float, float, float]):
        x, y, z, yaw, duration = move
        commander = scf.cf.high_level_commander
        commander.go_to(x, y, z, yaw, duration)
        time.sleep(duration)

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
    def set_position(scf: SyncCrazyflie, position):
        x, y, z, yaw_radians = position
        scf.cf.param.set_value('kalman.initialX', x)
        scf.cf.param.set_value('kalman.initialY', y)
        scf.cf.param.set_value('kalman.initialZ', z)
        scf.cf.param.set_value('kalman.initialYaw', yaw_radians)

    @staticmethod
    def reset_estimator(scf: SyncCrazyflie):
        cf = scf.cf
        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0')
        DroneCommand.wait_for_position_estimator(scf)