#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import math
import sys
import time
import random

import cflib
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

logging.basicConfig(level=logging.INFO)

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E70A')
if len(sys.argv) > 1:
    URI = sys.argv[1]

SENSOR_TH = 2000
SPEED_FACTOR = 0.3
LOOP_DT = 0.1


class Navigator:
    def __init__(self):
        self.hover = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0, 'height': 0.3}
        self.last_meas = {
            'front': 2000, 'back': 2000, 'left': 2000, 'right': 2000, 'up': 2000, 'down': 2000,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        }
        self.rng = random.Random()

    def update_measurement(self, m):
        self.last_meas.update(m)

    def compute(self):
        front = self.last_meas['front']
        left = self.last_meas['left']
        right = self.last_meas['right']
        back = self.last_meas['back']
        yaw = self.last_meas['yaw']

        vx = 0.20
        vy = 0.0
        yawrate = 0.0

        if front < SENSOR_TH:
            vx = -0.10
            yawrate = 60.0 if left > right else -60.0
        else:
            if left < SENSOR_TH and right >= SENSOR_TH:
                yawrate = -35.0
            elif right < SENSOR_TH and left >= SENSOR_TH:
                yawrate = 35.0
            elif left < SENSOR_TH and right < SENSOR_TH:
                yawrate = self.rng.choice([-70.0, 70.0])
            else:
                yawrate = self.rng.uniform(-15.0, 15.0)

        if back < SENSOR_TH and front >= SENSOR_TH:
            vx = max(vx, 0.12)

        self.hover['x'] = vx
        self.hover['y'] = vy
        self.hover['yaw'] = yawrate
        self.hover['height'] = 0.30
        return self.hover


class HeadlessCrazyflieApp:
    def __init__(self, uri):
        cflib.crtp.init_drivers()
        self.cf = Crazyflie(ro_cache=None, rw_cache='cache')
        self.nav = Navigator()
        self.running = True

        self.cf.connected.add_callback(self.connected)
        self.cf.disconnected.add_callback(self.disconnected)
        self.cf.open_link(uri)

        self.cf.platform.send_arming_request(True)
        time.sleep(1.0)

    def disconnected(self, uri):
        print('Disconnected')
        self.running = False

    def connected(self, uri):
        print(f'We are now connected to {uri}')

        lpos = LogConfig(name='Position', period_in_ms=100)
        lpos.add_variable('stateEstimate.x')
        lpos.add_variable('stateEstimate.y')
        lpos.add_variable('stateEstimate.z')

        lmeas = LogConfig(name='Meas', period_in_ms=100)
        lmeas.add_variable('range.front')
        lmeas.add_variable('range.back')
        lmeas.add_variable('range.up')
        lmeas.add_variable('range.left')
        lmeas.add_variable('range.right')
        lmeas.add_variable('range.zrange')
        lmeas.add_variable('stabilizer.roll')
        lmeas.add_variable('stabilizer.pitch')
        lmeas.add_variable('stabilizer.yaw')

        try:
            self.cf.log.add_config(lpos)
            lpos.data_received_cb.add_callback(self.pos_data)
            lpos.start()
        except Exception as e:
            print(f'Could not start Position log config for {uri}: {e}')

        try:
            self.cf.log.add_config(lmeas)
            lmeas.data_received_cb.add_callback(self.meas_data)
            lmeas.start()
        except Exception as e:
            print(f'Could not start Measurement log config for {uri}: {e}')

    def pos_data(self, timestamp, data, logconf):
        pass

    def meas_data(self, timestamp, data, logconf):
        measurement = {
            'roll': data['stabilizer.roll'],
            'pitch': data['stabilizer.pitch'],
            'yaw': data['stabilizer.yaw'],
            'front': data['range.front'],
            'back': data['range.back'],
            'up': data['range.up'],
            'down': data['range.zrange'],
            'left': data['range.left'],
            'right': data['range.right'],
        }
        self.nav.update_measurement(measurement)

    def send_hover_command(self):
        h = self.nav.compute()
        self.cf.commander.send_hover_setpoint(
            h['x'], h['y'], h['yaw'], h['height']
        )

    def run(self):
        try:
            while self.running:
                self.send_hover_command()
                time.sleep(LOOP_DT)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        try:
            self.cf.close_link()
        except Exception:
            pass


if __name__ == '__main__':
    app = HeadlessCrazyflieApp(URI)
    app.run()