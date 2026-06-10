#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import math
import numpy as np
import pygame

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig


URI = 'radio://0/80/2M/E7E7E7E70A'

GRID_W, GRID_H = 160, 120
CELL = 6
WIN_W, WIN_H = GRID_W * CELL, GRID_H * CELL

UNKNOWN = 0
FREE = 1
OCCUPIED = 2
FRONTIER = 3

SENSOR_TH = 2000
SCALE = 0.0025
MAX_RAY = 4.0

COLORS = {
    UNKNOWN: (25, 25, 25),
    FREE: (235, 235, 235),
    OCCUPIED: (220, 70, 70),
    FRONTIER: (70, 220, 90),
}

DIRS = {
    'front': (1, 0),
    'back': (-1, 0),
    'left': (0, 1),
    'right': (0, -1),
}


class Mapper:
    def __init__(self):
        self.grid = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
        self.robot = np.array([GRID_W // 2, GRID_H // 2], dtype=np.int32)
        self.meas = None

    def set_pose(self, x, y):
        self.robot[:] = [int(x), int(y)]

    def update_measurement(self, m):
        self.meas = m
        self._integrate()

    def _integrate(self):
        if self.meas is None:
            return

        rx, ry = self.robot
        self.grid[ry, rx] = FREE

        for name, (dx, dy) in DIRS.items():
            dist_mm = self.meas.get(name, SENSOR_TH + 1)
            if dist_mm >= SENSOR_TH:
                dist_m = MAX_RAY
                hit = False
            else:
                dist_m = max(0.0, dist_mm / 1000.0)
                hit = True

            steps = max(1, int(dist_m / SCALE))
            for i in range(1, steps):
                x = rx + dx * i
                y = ry + dy * i
                if 0 <= x < GRID_W and 0 <= y < GRID_H:
                    if self.grid[y, x] == UNKNOWN:
                        self.grid[y, x] = FREE

            if hit:
                x = rx + dx * steps
                y = ry + dy * steps
                if 0 <= x < GRID_W and 0 <= y < GRID_H:
                    self.grid[y, x] = OCCUPIED

        self._mark_frontiers()

    def _mark_frontiers(self):
        frontier = np.zeros_like(self.grid)
        for y in range(1, GRID_H - 1):
            for x in range(1, GRID_W - 1):
                if self.grid[y, x] != FREE:
                    continue
                n = self.grid[y - 1:y + 2, x - 1:x + 2]
                if np.any(n == UNKNOWN):
                    frontier[y, x] = FRONTIER
        self.frontier = frontier


def draw_grid(screen, mapper):
    for y in range(GRID_H):
        for x in range(GRID_W):
            v = mapper.grid[y, x]
            if hasattr(mapper, 'frontier') and mapper.frontier[y, x] == FRONTIER:
                c = COLORS[FRONTIER]
            else:
                c = COLORS[int(v)]
            pygame.draw.rect(screen, c, (x * CELL, y * CELL, CELL, CELL))

    rx, ry = mapper.robot
    pygame.draw.rect(screen, (30, 120, 255), (rx * CELL, ry * CELL, CELL, CELL))


def world_to_grid(dx_m, dy_m):
    return dx_m / SCALE, dy_m / SCALE


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Crazyflie 2D Frontier Map")
    clock = pygame.time.Clock()

    cflib.crtp.init_drivers()
    cf = Crazyflie(ro_cache=None, rw_cache='cache')
    mapper = Mapper()
    connected = False

    def pos_cb(timestamp, data, logconf):
        x_m = data['stateEstimate.x']
        y_m = data['stateEstimate.y']
        gx = GRID_W // 2 + int(x_m / SCALE)
        gy = GRID_H // 2 - int(y_m / SCALE)
        mapper.set_pose(gx, gy)

    def meas_cb(timestamp, data, logconf):
        mapper.update_measurement({
            'front': data['range.front'],
            'back': data['range.back'],
            'left': data['range.left'],
            'right': data['range.right'],
        })

    def connected_cb(uri):
        nonlocal connected
        connected = True
        lp = LogConfig(name='Position', period_in_ms=100)
        lp.add_variable('stateEstimate.x')
        lp.add_variable('stateEstimate.y')
        cf.log.add_config(lp)
        lp.data_received_cb.add_callback(pos_cb)
        lp.start()

        lm = LogConfig(name='Meas', period_in_ms=100)
        lm.add_variable('range.front')
        lm.add_variable('range.back')
        lm.add_variable('range.left')
        lm.add_variable('range.right')
        cf.log.add_config(lm)
        lm.data_received_cb.add_callback(meas_cb)
        lm.start()

    def disconnected_cb(uri):
        pass

    cf.connected.add_callback(connected_cb)
    cf.disconnected.add_callback(disconnected_cb)
    cf.open_link(URI)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))
        draw_grid(screen, mapper)
        pygame.display.flip()
        clock.tick(30)

    cf.close_link()
    pygame.quit()


if __name__ == "__main__":
    main()