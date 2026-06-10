#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import math
import sys
import time
import random
from collections import deque

import pygame
import numpy as np

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

MAP_SIZE = 220
CELL_SIZE = 4
WORLD_SCALE = 1.0
MAX_RANGE_M = 2.0
FRONTIER_TH = 1
OBSTACLE_RANGE_M = 0.35

UNKNOWN = 0
FREE = 1
OBSTACLE = 2
FRONTIER = 3


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


class FrontierMap:
    def __init__(self, size=MAP_SIZE, cell_size=CELL_SIZE, world_scale=WORLD_SCALE):
        self.size = size
        self.cell_size = cell_size
        self.world_scale = world_scale
        self.grid = np.zeros((size, size), dtype=np.uint8)
        self.last_pose = (size // 2, size // 2, 0.0)

    def world_to_grid(self, x, y):
        gx = int(self.size // 2 + x / self.world_scale / self.cell_size)
        gy = int(self.size // 2 - y / self.world_scale / self.cell_size)
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = (gx - self.size // 2) * self.cell_size * self.world_scale
        y = -(gy - self.size // 2) * self.cell_size * self.world_scale
        return x, y

    def in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def bresenham(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return points

    def update_from_pose_and_ranges(self, pose, meas):
        x, y, z, yaw = pose
        self.last_pose = (x, y, yaw)
        cx, cy = self.world_to_grid(x, y)

        ranges = {
            'front': meas['front'],
            'back': meas['back'],
            'left': meas['left'],
            'right': meas['right'],
        }

        angles = {
            'front': yaw,
            'back': yaw + math.pi,
            'left': yaw + math.pi / 2,
            'right': yaw - math.pi / 2,
        }

        self.grid[max(0, cy-1):min(self.size, cy+2), max(0, cx-1):min(self.size, cx+2)] = FREE

        for k, dist_mm in ranges.items():
            dist_m = float(dist_mm) / 1000.0
            if dist_m <= 0.01:
                continue
            ang = angles[k]
            ex = x + math.cos(ang) * min(dist_m, MAX_RANGE_M)
            ey = y + math.sin(ang) * min(dist_m, MAX_RANGE_M)
            gx1, gy1 = self.world_to_grid(ex, ey)
            if not self.in_bounds(gx1, gy1):
                gx1 = max(0, min(self.size - 1, gx1))
                gy1 = max(0, min(self.size - 1, gy1))

            line = self.bresenham(cx, cy, gx1, gy1)
            if len(line) > 2:
                for px, py in line[:-1]:
                    if self.in_bounds(px, py):
                        self.grid[py, px] = FREE
            if dist_m < MAX_RANGE_M and dist_m < OBSTACLE_RANGE_M:
                if self.in_bounds(gx1, gy1):
                    self.grid[gy1, gx1] = OBSTACLE

        self.update_frontiers()

    def update_frontiers(self):
        frontier = np.zeros_like(self.grid)
        for y in range(1, self.size - 1):
            for x in range(1, self.size - 1):
                if self.grid[y, x] == FREE:
                    neigh = self.grid[y-1:y+2, x-1:x+2]
                    if np.any(neigh == UNKNOWN):
                        frontier[y, x] = FRONTIER
        self.grid[self.grid == FRONTIER] = FREE
        self.grid[frontier == FRONTIER] = FRONTIER

    def render(self, surf):
        colors = {
            UNKNOWN: (30, 30, 35),
            FREE: (210, 210, 210),
            OBSTACLE: (200, 70, 70),
            FRONTIER: (80, 180, 255),
        }
        arr = pygame.surfarray.pixels3d(surf)
        for y in range(self.size):
            for x in range(self.size):
                c = colors[int(self.grid[y, x])]
                sx = x * self.cell_size
                sy = y * self.cell_size
                arr[sx:sx+self.cell_size, sy:sy+self.cell_size] = c
        del arr

    def draw_overlay(self, screen, pose, meas):
        x, y, yaw = self.last_pose
        cx, cy = self.world_to_grid(x, y)
        px = cx * self.cell_size + self.cell_size // 2
        py = cy * self.cell_size + self.cell_size // 2

        pygame.draw.circle(screen, (255, 255, 0), (px, py), max(3, self.cell_size // 2))
        ring_r = int(MAX_RANGE_M / self.world_scale / self.cell_size)
        pygame.draw.circle(screen, (120, 120, 120), (px, py), ring_r, 1)

        for ang, col in [(yaw, (0, 255, 0)), (yaw + math.pi/2, (0, 180, 255)), (yaw - math.pi/2, (0, 180, 255))]:
            ex = px + int(math.cos(ang) * ring_r)
            ey = py - int(math.sin(ang) * ring_r)
            pygame.draw.line(screen, col, (px, py), (ex, ey), 1)


class HeadlessCrazyflieApp:
    def __init__(self, uri):
        cflib.crtp.init_drivers()
        self.cf = Crazyflie(ro_cache=None, rw_cache='cache')
        self.nav = Navigator()
        self.map = FrontierMap()
        self.running = True
        self.pose = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}

        self.pygame_ready = False
        self.screen = None
        self.clock = None

        self.cf.connected.add_callback(self.connected)
        self.cf.disconnected.add_callback(self.disconnected)
        self.cf.open_link(uri)

        self.cf.platform.send_arming_request(True)
        time.sleep(1.0)

    def setup_pygame(self):
        pygame.init()
        w = self.map.size * self.map.cell_size
        h = self.map.size * self.map.cell_size
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Crazyflie Frontier Map")
        self.clock = pygame.time.Clock()
        self.pygame_ready = True

    def disconnected(self, uri):
        print('Disconnected')
        self.running = False

    def connected(self, uri):
        print(f'We are now connected to {uri}')
        self.setup_pygame()

        lpos = LogConfig(name='Position', period_in_ms=100)
        lpos.add_variable('stateEstimate.x')
        lpos.add_variable('stateEstimate.y')
        lpos.add_variable('stateEstimate.z')
        lpos.add_variable('stateEstimate.yaw')

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
        self.pose['x'] = float(data.get('stateEstimate.x', 0.0))
        self.pose['y'] = float(data.get('stateEstimate.y', 0.0))
        self.pose['z'] = float(data.get('stateEstimate.z', 0.0))
        self.pose['yaw'] = math.radians(float(data.get('stateEstimate.yaw', 0.0)))

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
        self.map.update_from_pose_and_ranges(
            (self.pose['x'], self.pose['y'], self.pose['z'], self.pose['yaw']),
            measurement
        )

    def send_hover_command(self):
        h = self.nav.compute()
        self.cf.commander.send_hover_setpoint(h['x'], h['y'], h['yaw'], h['height'])

    def handle_pygame(self):
        if not self.pygame_ready:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        self.screen.fill((30, 30, 35))
        self.map.render(self.screen)
        self.map.draw_overlay(self.screen, self.pose, self.nav.last_meas)
        pygame.display.flip()

        self.clock.tick(30)

    def run(self):
        self.setup_pygame()

        try:
            last = time.time()
            count = 0
            while self.running:
                self.send_hover_command()
                count += 1
                now = time.time()
                if now - last >= 1.0:
                    print(f"Hover commands per second: {count}")
                    count = 0
                    last = now
                self.handle_pygame()
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
        try:
            pygame.quit()
        except Exception:
            pass


if __name__ == '__main__':
    app = HeadlessCrazyflieApp(URI)
    app.run()