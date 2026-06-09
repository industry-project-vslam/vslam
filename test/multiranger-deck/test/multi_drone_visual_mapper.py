#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pygame visualizer for multi-drone Crazyflie system.
Displays all drones and shared occupancy map.
"""

import sys
import math
from collections import defaultdict

import pygame
from pygame.locals import *

SENSOR_TH = 2000


class SharedOccupancyGrid:
    """Same occupancy grid as in the multi-drone script."""
    
    def __init__(self, cell_size=0.05):
        self.cell_size = cell_size
        self.grid = defaultdict(int)
        self.min_x = 0.0
        self.min_y = 0.0
        self.max_x = 0.0
        self.max_y = 0.0

    def update_from_sensor(self, drone_id, pose_x, pose_y, yaw, side, distance_mm):
        if distance_mm >= SENSOR_TH:
            return
        distance_m = distance_mm / 1000.0

        angles = {
            'front': yaw,
            'left': yaw + math.pi / 2,
            'right': yaw - math.pi / 2,
            'back': yaw + math.pi
        }
        if side not in angles:
            return

        angle = angles[side]
        sensor_x = pose_x + distance_m * math.cos(angle)
        sensor_y = pose_y + distance_m * math.sin(angle)

        self.min_x = min(self.min_x, pose_x, sensor_x)
        self.max_x = max(self.max_x, pose_x, sensor_x)
        self.min_y = min(self.min_y, pose_y, sensor_y)
        self.max_y = max(self.max_y, pose_y, sensor_y)

        self.grid[(sensor_x, sensor_y)] = min(100, self.grid[(sensor_x, sensor_y)] + 30)
        for d in [0.0, distance_m * 0.33, distance_m * 0.66]:
            cx = pose_x + d * math.cos(angle)
            cy = pose_y + d * math.sin(angle)
            self.grid[(cx, cy)] = max(-20, self.grid[(cx, cy)] - 10)

    def get_approx_bounds(self):
        return (self.min_x, self.min_y, self.max_x, self.max_y)


class MultiDroneVisualizer:
    """Pygame visualizer displaying multiple drones and shared map."""
    
    def __init__(self, width=1000, height=700, cell_size=0.05):
        pygame.init()
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Multi-Drone Crazyflie Swarm Map')
        self.clock = pygame.time.Clock()

        self.map = SharedOccupancyGrid(cell_size=cell_size)
        
        # Drone colors (different for each drone)
        self.DRONE_COLORS = [
            (255, 100, 100),  # Red
            (100, 255, 100),  # Green
            (100, 100, 255),  # Blue
            (255, 255, 100),  # Yellow
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Cyan
        ]
        
        self.COLOR_BG = (30, 30, 35)
        self.COLOR_OCCUPIED = (220, 60, 60)
        self.COLOR_FREE = (80, 180, 80)
        self.COLOR_DARK = (40, 40, 45)
        self.COLOR_SENSOR = (100, 200, 255)
        self.COLOR_TEXT = (255, 255, 255)

        self.map_offset_x = width / 2
        self.map_offset_y = height / 2
        self.map_scale = 10.0

        self.drones = {}  # drone_id -> {'x', 'y', 'yaw', 'measurements'}
        self.font = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 28)
        self.running = True

    def world_to_screen(self, x, y):
        screen_x = self.map_offset_x + x * self.map_scale
        screen_y = self.map_offset_y - y * self.map_scale
        return (int(screen_x), int(screen_y))

    def update_drone_data(self, drone_id, pose_x, pose_y, yaw, measurements):
        self.drones[drone_id] = {
            'x': pose_x, 'y': pose_y, 'yaw': yaw,
            'measurements': measurements
        }

        yaw_rad = math.radians(yaw)
        for side in ['front', 'left', 'right', 'back']:
            dist = measurements.get(side, 2000)
            self.map.update_from_sensor(drone_id, pose_x, pose_y, yaw_rad, side, dist)

    def draw_map(self):
        bounds = self.map.get_approx_bounds()
        min_x, min_y, max_x, max_y = bounds

        if max_x > min_x and max_y > min_y:
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            self.map_offset_x = self.width / 2 - center_x * self.map_scale
            self.map_offset_y = self.height / 2 + center_y * self.map_scale

        min_screen_x, min_screen_y = self.world_to_screen(min_x - 2, min_y - 2)
        max_screen_x, max_screen_y = self.world_to_screen(max_x + 2, max_y + 2)

        for px in range(min_screen_x, max_screen_x + 1):
            for py in range(min_screen_y, max_screen_y + 1):
                world_x = (px - self.map_offset_x) / self.map_scale
                world_y = -(py - self.map_offset_y) / self.map_scale
                occ = self.map.grid[(world_x, world_y)]
                
                if occ > 50:
                    color = self.COLOR_OCCUPIED
                elif occ < -10:
                    color = self.COLOR_FREE
                else:
                    color = self.COLOR_DARK
                pygame.draw.rect(self.screen, color, (px, py, 1, 1))

    def draw_drone(self, drone_id, data):
        color = self.DRONE_COLORS[drone_id % len(self.DRONE_COLORS)]
        screen_pos = self.world_to_screen(data['x'], data['y'])

        # Drone body
        pygame.draw.circle(self.screen, color, screen_pos, 7)
        pygame.draw.circle(self.screen, (200, 200, 200), screen_pos, 4)

        # Orientation arrow
        yaw_rad = math.radians(data['yaw'])
        arrow_len = 20
        arrow_x = screen_pos[0] + int(arrow_len * math.cos(yaw_rad))
        arrow_y = screen_pos[1] - int(arrow_len * math.sin(yaw_rad))
        pygame.draw.line(self.screen, color, screen_pos, (arrow_x, arrow_y), 3)
        pygame.draw.circle(self.screen, color, (arrow_x, arrow_y), 3)

        # Drone ID label
        label = self.font.render(f'D{drone_id}', True, self.COLOR_TEXT)
        self.screen.blit(label, (screen_pos[0] + 10, screen_pos[1] - 10))

    def draw_sensors(self, drone_id, data):
        color = self.DRONE_COLORS[drone_id % len(self.DRONE_COLORS)]
        screen_pos = self.world_to_screen(data['x'], data['y'])
        yaw_rad = math.radians(data['yaw'])

        angles = {
            'front': yaw_rad,
            'left': yaw_rad + math.pi / 2,
            'right': yaw_rad - math.pi / 2,
            'back': yaw_rad + math.pi
        }

        for side, angle in angles.items():
            dist_mm = data['measurements'].get(side, 2000)
            ray_len = min(100, dist_mm / 1000.0 * self.map_scale) if dist_mm < 2000 else 100
            end_x = screen_pos[0] + int(ray_len * math.cos(angle))
            end_y = screen_pos[1] - int(ray_len * math.sin(angle))
            sensor_color = color if dist_mm < 2000 else (100, 100, 100)
            pygame.draw.line(self.screen, sensor_color, screen_pos, (end_x, end_y), 2)

    def draw_ui(self):
        title = self.font_large.render('Multi-Drone Crazyflie Swarm', True, self.COLOR_TEXT)
        self.screen.blit(title, (10, 10))

        pos_text = self.font.render(f'Drones: {len(self.drones)}', True, self.COLOR_TEXT)
        self.screen.blit(pos_text, (10, 40))

        bounds = self.map.get_approx_bounds()
        bounds_text = self.font.render(
            f'Bounds: X [{bounds[0]:.2f}, {bounds[2]:.2f}], Y [{bounds[1]:.2f}, {bounds[3]:.2f}]',
            True, self.COLOR_TEXT
        )
        self.screen.blit(bounds_text, (10, 60))

        occupied = sum(1 for v in self.map.grid.values() if v > 50)
        occ_text = self.font.render(f'Occupied cells: {occupied}', True, self.COLOR_OCCUPIED)
        self.screen.blit(occ_text, (10, 80))

        # Drone info
        for i, (did, data) in enumerate(self.drones.items()):
            y = 110 + i * 25
            color = self.DRONE_COLORS[did % len(self.DRONE_COLORS)]
            drone_text = self.font.render(
                f'D{did}: ({data["x"]:.2f}, {data["y"]:.2f}) Yaw:{data["yaw"]:.0f}°',
                True, color
            )
            self.screen.blit(drone_text, (10, y))

        instr = self.font.render('ESC: Quit | Mouse: Pan | Wheel: Zoom', True, (200, 200, 200))
        self.screen.blit(instr, (10, self.height - 25))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                self.running = False
            elif event.type == MOUSEMOTION and event.buttons[0]:
                self.map_offset_x += event.rel[0]
                self.map_offset_y += event.rel[1]
            elif event.type == MOUSEWHEEL:
                zoom = 1.0 + event.y * 0.1
                self.map_scale = max(2.0, min(50.0, self.map_scale * zoom))

    def run_standalone(self):
        """Run with simulated multi-drone data."""
        import random
        rng = random.Random(42)
        n_drones = 3
        
        for i in range(n_drones):
            self.drones[i] = {
                'x': rng.uniform(-1, 1),
                'y': rng.uniform(-1, 1),
                'yaw': rng.uniform(0, 360),
                'measurements': {}
            }

        print('Multi-drone visualizer (standalone mode)')
        print('Press ESC to quit')

        while self.running:
            self.clock.tick(60)
            self.handle_events()

            for i in range(n_drones):
                drone = self.drones[i]
                drone['yaw'] += rng.uniform(-5, 5)
                speed = 0.08
                drone['x'] += speed * math.cos(math.radians(drone['yaw']))
                drone['y'] += speed * math.sin(math.radians(drone['yaw']))

                drone['measurements'] = {
                    'front': rng.uniform(500, 1800) if rng.random() > 0.3 else 2000,
                    'back': rng.uniform(800, 2000),
                    'left': rng.uniform(600, 1900) if rng.random() > 0.4 else 2000,
                    'right': rng.uniform(700, 2000) if rng.random() > 0.35 else 2000
                }

                self.update_drone_data(
                    i, drone['x'], drone['y'], drone['yaw'],
                    drone['measurements']
                )

            self.screen.fill(self.COLOR_BG)
            self.draw_map()
            for did, data in self.drones.items():
                self.draw_drone(did, data)
                self.draw_sensors(did, data)
            self.draw_ui()
            pygame.display.flip()
            pygame.time.sleep(0.05)

        pygame.quit()


if __name__ == '__main__':
    visualizer = MultiDroneVisualizer(width=1000, height=700)
    visualizer.run_standalone()