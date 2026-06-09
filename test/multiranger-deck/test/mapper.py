#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import math
import sys
import time
import random

from collections import defaultdict

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


class OccupancyGrid:
    """
    2D occupancy grid mapping using Multiranger plus yaw.
    Only uses front, left, right, back + yaw for mapping.
    Stores probabilistic occupancy in memory (no visualization).
    """
    def __init__(self, cell_size=0.05, bounds=10.0):
        self.cell_size = cell_size
        self.bounds = bounds
        self.grid = defaultdict(int)
        self.min_x = 0.0
        self.min_y = 0.0
        self.max_x = 0.0
        self.max_y = 0.0

    def update_from_sensor(self, pose_x, pose_y, yaw, side, distance_mm):
        """
        Add a sensor reading to the occupancy grid.
        side: 'front', 'left', 'right', 'back'
        distance_mm: millimeters from multiranger
        """
        if distance_mm >= SENSOR_TH:
            return  # no obstacle detected
        distance_m = distance_mm / 1000.0

        if side == 'front':
            angle = yaw
        elif side == 'left':
            angle = yaw + math.pi / 2
        elif side == 'right':
            angle = yaw - math.pi / 2
        elif side == 'back':
            angle = yaw + math.pi
        else:
            return

        sensor_x = pose_x + distance_m * math.cos(angle)
        sensor_y = pose_y + distance_m * math.sin(angle)

        # Update bounds
        self.min_x = min(self.min_x, pose_x, sensor_x)
        self.max_x = max(self.max_x, pose_x, sensor_x)
        self.min_y = min(self.min_y, pose_y, sensor_y)
        self.max_y = max(self.max_y, pose_y, sensor_y)

        # Occupied cell at sensor endpoint
        self.grid[(sensor_x, sensor_y)] = min(100, self.grid[(sensor_x, sensor_y)] + 30)

        # Free cells along the ray (simple inverse sensor model)
        for d in [0.0, distance_m * 0.33, distance_m * 0.66]:
            cx = pose_x + d * math.cos(angle)
            cy = pose_y + d * math.sin(angle)
            self.grid[(cx, cy)] = max(-20, self.grid[(cx, cy)] - 10)

    def is_occupied(self, x, y):
        return self.grid[(x, y)] > 50

    def get_approx_bounds(self):
        return (self.min_x, self.min_y, self.max_x, self.max_y)


class Navigator:
    def __init__(self):
        self.hover = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0, 'height': 0.3}
        self.last_meas = {
            'front': 2000, 'back': 2000, 'left': 2000, 'right': 2000, 'up': 2000, 'down': 2000,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        }
        self.last_pose = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.rng = random.Random()
        self.map = OccupancyGrid()

    def update_measurement(self, m):
        self.last_meas.update(m)

    def update_pose(self, x, y, z):
        """
        Update pose and integrate new sensor readings into the 2D map.
        Only front, left, right, back + yaw are used for mapping.
        """
        self.last_pose = {'x': x, 'y': y, 'z': z}
        yaw_rad = math.radians(self.last_meas['yaw'])

        for side in ['front', 'left', 'right', 'back']:
            dist = self.last_meas.get(side, 2000)
            self.map.update_from_sensor(x, y, yaw_rad, side, dist)

    def get_map(self):
        return self.map

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
            print(f'Could not start Position log config: {e}')

        try:
            self.cf.log.add_config(lmeas)
            lmeas.data_received_cb.add_callback(self.meas_data)
            lmeas.start()
        except Exception as e:
            print(f'Could not start Measurement log config: {e}')

    def pos_data(self, timestamp, data, logconf):
        x = data['stateEstimate.x']
        y = data['stateEstimate.y']
        z = data['stateEstimate.z']
        self.nav.update_pose(x, y, z)

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
            self.print_map_summary()
            self.close()

    def print_map_summary(self):
        m = self.nav.get_map()
        bounds = m.get_approx_bounds()
        print("\n=== Map Summary ===")
        print(f"Bounds: X [{bounds[0]:.2f}, {bounds[2]:.2f}], Y [{bounds[1]:.2f}, {bounds[3]:.2f}]")
        print(f"Occupied cells: {sum(1 for v in m.grid.values() if v > 50)}")
        print("==================\n")

    def close(self):
        try:
            self.cf.close_link()
        except Exception:
            pass


if __name__ == '__main__':
    app = HeadlessCrazyflieApp(URI)
    app.run()