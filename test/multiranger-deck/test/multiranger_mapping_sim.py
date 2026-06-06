#!/usr/bin/env python3
import math
import sys
import numpy as np
import pygame

GRID_W, GRID_H = 160, 120
CELL = 6
WIN_W, WIN_H = GRID_W * CELL, GRID_H * CELL

UNKNOWN, FREE, OCCUPIED, FRONTIER = 0, 1, 2, 3

SCALE = 0.05          # meters per grid cell
SENSOR_MAX = 4.0      # meters
SENSOR_STEP = 0.02    # ray march step in meters
MOVE_SPEED = 1.5      # m/s
TURN_SPEED = 120.0    # deg/s
DT = 1.0 / 30.0

COLORS = {
    UNKNOWN: (25, 25, 25),
    FREE: (235, 235, 235),
    OCCUPIED: (220, 70, 70),
    FRONTIER: (70, 220, 90),
}

DIRS = {
    'front': 0.0,
    'left': 90.0,
    'back': 180.0,
    'right': -90.0,
}


class SimWorld:
    def __init__(self):
        self.true_map = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
        self.occ = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
        self.frontier = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
        self.robot_x = GRID_W // 2
        self.robot_y = GRID_H // 2
        self.robot_yaw = 0.0
        self._build_world()

    def _build_world(self):
        self.true_map[:, :] = 0
        self.true_map[0, :] = 1
        self.true_map[-1, :] = 1
        self.true_map[:, 0] = 1
        self.true_map[:, -1] = 1

        self.true_map[20:100, 35] = 1
        self.true_map[20, 35:110] = 1
        self.true_map[60:61, 60:130] = 1
        self.true_map[35:90, 110] = 1
        self.true_map[80:110, 80:81] = 1
        self.true_map[30:31, 15:55] = 1

    def world_to_grid(self, x_m, y_m):
        gx = int(GRID_W // 2 + x_m / SCALE)
        gy = int(GRID_H // 2 - y_m / SCALE)
        return gx, gy

    def grid_to_world(self, gx, gy):
        x_m = (gx - GRID_W // 2) * SCALE
        y_m = -(gy - GRID_H // 2) * SCALE
        return x_m, y_m

    def _in_bounds(self, x, y):
        return 0 <= x < GRID_W and 0 <= y < GRID_H

    def simulate_sensor(self, rel_angle_deg):
        angle = math.radians(self.robot_yaw + rel_angle_deg)
        x0, y0 = self.robot_x, self.robot_y
        dist = 0.0
        while dist < SENSOR_MAX:
            dist += SENSOR_STEP
            x = x0 + (dist / SCALE) * math.cos(angle)
            y = y0 - (dist / SCALE) * math.sin(angle)
            gx, gy = int(round(x)), int(round(y))
            if not self._in_bounds(gx, gy):
                return SENSOR_MAX
            if self.true_map[gy, gx] == 1:
                return dist
        return SENSOR_MAX

    def integrate_scan(self, readings):
        rx, ry = self.robot_x, self.robot_y
        self.occ[ry, rx] = FREE

        for name, rel_angle in DIRS.items():
            dist_m = readings[name]
            hit = dist_m < SENSOR_MAX
            steps = max(1, int(dist_m / SCALE))

            ang = math.radians(self.robot_yaw + rel_angle)
            dx = math.cos(ang)
            dy = -math.sin(ang)

            for i in range(1, steps):
                x = int(round(rx + (i * SCALE / SCALE) * dx))
                y = int(round(ry + (i * SCALE / SCALE) * dy))
                if self._in_bounds(x, y) and self.occ[y, x] == UNKNOWN:
                    self.occ[y, x] = FREE

            if hit:
                hx = int(round(rx + (dist_m / SCALE) * dx))
                hy = int(round(ry + (dist_m / SCALE) * dy))
                if self._in_bounds(hx, hy):
                    self.occ[hy, hx] = OCCUPIED

        self._detect_frontiers()

    def _detect_frontiers(self):
        self.frontier[:, :] = 0
        for y in range(1, GRID_H - 1):
            for x in range(1, GRID_W - 1):
                if self.occ[y, x] != FREE:
                    continue
                n = self.occ[y - 1:y + 2, x - 1:x + 2]
                if np.any(n == UNKNOWN):
                    self.frontier[y, x] = FRONTIER

    def move_robot(self, forward_mps, strafe_mps, yaw_dps, dt):
        yaw_rad = math.radians(self.robot_yaw)
        vx = forward_mps * math.cos(yaw_rad) - strafe_mps * math.sin(yaw_rad)
        vy = forward_mps * math.sin(yaw_rad) + strafe_mps * math.cos(yaw_rad)

        new_x = self.robot_x + int(round((vx * dt) / SCALE))
        new_y = self.robot_y - int(round((vy * dt) / SCALE))
        new_yaw = self.robot_yaw + yaw_dps * dt

        if self._in_bounds(new_x, new_y) and self.true_map[new_y, new_x] == 0:
            self.robot_x = new_x
            self.robot_y = new_y
        self.robot_yaw = new_yaw % 360.0


def draw(screen, world):
    for y in range(GRID_H):
        for x in range(GRID_W):
            v = FRONTIER if world.frontier[y, x] == FRONTIER else int(world.occ[y, x])
            pygame.draw.rect(screen, COLORS[v], (x * CELL, y * CELL, CELL, CELL))

    rx, ry = world.robot_x, world.robot_y
    pygame.draw.rect(screen, (30, 120, 255), (rx * CELL, ry * CELL, CELL, CELL))

    yaw = math.radians(world.robot_yaw)
    tip = (
        int((rx + 4 * math.cos(yaw)) * CELL),
        int((ry - 4 * math.sin(yaw)) * CELL),
    )
    pygame.draw.line(screen, (0, 90, 255), (rx * CELL + CELL // 2, ry * CELL + CELL // 2), tip, 2)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Simulated Frontier Mapping")
    clock = pygame.time.Clock()

    world = SimWorld()
    running = True

    while running:
        dt = DT
        keys = pygame.key.get_pressed()

        forward = 0.0
        strafe = 0.0
        yaw = 0.0

        if keys[pygame.K_UP]:
            forward += MOVE_SPEED
        if keys[pygame.K_DOWN]:
            forward -= MOVE_SPEED
        if keys[pygame.K_LEFT]:
            strafe += MOVE_SPEED
        if keys[pygame.K_RIGHT]:
            strafe -= MOVE_SPEED
        if keys[pygame.K_a]:
            yaw += TURN_SPEED
        if keys[pygame.K_d]:
            yaw -= TURN_SPEED

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        world.move_robot(forward, strafe, yaw, dt)

        readings = {
            'front': world.simulate_sensor(0.0),
            'left': world.simulate_sensor(90.0),
            'back': world.simulate_sensor(180.0),
            'right': world.simulate_sensor(-90.0),
        }
        world.integrate_scan(readings)

        screen.fill((0, 0, 0))
        draw(screen, world)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()