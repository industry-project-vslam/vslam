from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
import time
import math

class DroneService:
    def __init__(self, uri: str, position: dict[str, float], meas_cb):
        self.uri = uri
        self.position = position
        self.meas_cb = meas_cb

        self.cf = Crazyflie(ro_cache=None, rw_cache='cache')

        self.cf.connected.add_callback(self.connected)
        self.cf.open_link(uri)

        self.cf.supervisor.send_arming_request(True)
        time.sleep(1.0)

    def connected(self, uri):
        lmeas = LogConfig(name='Meas', period_in_ms=100)
        lmeas.add_variable('range.front')
        lmeas.add_variable('range.back')
        lmeas.add_variable('range.left')
        lmeas.add_variable('range.right')
        
        lmeas.add_variable('stateEstimate.yaw')

        lmeas.add_variable('stateEstimate.x')
        lmeas.add_variable('stateEstimate.y')
        lmeas.add_variable('stateEstimate.z')

        try:
            self.cf.log.add_config(lmeas)
            lmeas.data_received_cb.add_callback(self.meas_data)
            lmeas.start()
        except Exception as e:
            print(f'Could not start Measurement log config for {uri}: {e}')

    def meas_data(self, timestamp, data, logconf):
        try:
            measurement = {
                'yaw': data['stateEstimate.yaw'],
                'front': data['range.front'],
                'back': data['range.back'],
                'left': data['range.left'],
                'right': data['range.right'],
                'x': float(data['stateEstimate.x']),
                'y': float(data['stateEstimate.y']),
                'z': float(data['stateEstimate.z']),
                'yaw': math.radians(float(data['stateEstimate.yaw'])),
            }
            self.meas_cb(self.uri, measurement)
        except:
            pass

    def send_hover_command(self, p):
        self.cf.commander.send_hover_setpoint(p['x'], p['y'], p['yaw'], p['z'])