#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pygame 2D Occupancy Grid Visualizer for Crazyflie Multi-ranger mapping.
Displays the map in real-time without any Crazyflie connection.
Run standalone for testing, or integrate with your headless Crazyflie app.
"""

import sys
import math
from collections import defaultdict

import pygame
from pygame.locals import *

import time

class OccupancyGrid:
    """
    Same occupancy grid as in the headless script.
    Can be used standalone or shared with the Crazyflie app.
    """
    def __init__(self, cell_size=0.05, bounds=10.0):
        self.cell_size = cell_size
        self.bounds = bounds
        self.grid = defaultdict(int)
        self.min_x = 0.0
        self.min_y = 0.0
        self.max_x = 0.0
        self.max_y = 0.0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0

    def update_from_sensor(self, pose_x, pose_y, yaw, side, distance_mm):
        """Add a sensor reading to the occupancy grid."""
        SENSOR_TH = 2000
        if distance_mm >= SENSOR_TH:
            return
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

        self.min_x = min(self.min_x, pose_x, sensor_x)
        self.max_x = max(self.max_x, pose_x, sensor_x)
        self.min_y = min(self.min_y, pose_y, sensor_y)
        self.max_y = max(self.max_y, pose_y, sensor_y)

        self.grid[(sensor_x, sensor_y)] = min(100, self.grid[(sensor_x, sensor_y)] + 30)

        for d in [0.0, distance_m * 0.33, distance_m * 0.66]:
            cx = pose_x + d * math.cos(angle)
            cy = pose_y + d * math.sin(angle)
            self.grid[(cx, cy)] = max(-20, self.grid[(cx, cy)] - 10)

    def set_pose(self, x, y, yaw):
        self.pose_x = x
        self.pose_y = y
        self.pose_yaw = yaw

    def is_occupied(self, x, y):
        return self.grid[(x, y)] > 50

    def get_approx_bounds(self):
        return (self.min_x, self.min_y, self.max_x, self.max_y)


class MapVisualizer:
    """
    Pygame-based 2D occupancy grid visualizer.
    Displays obstacles, drone position, and sensor rays.
    """
    def __init__(self, width=800, height=600, cell_size=0.05):
        pygame.init()
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Crazyflie 2D Occupancy Map')
        self.clock = pygame.time.Clock()

        self.map = OccupancyGrid(cell_size=cell_size)

        # Colors
        self.COLOR_BG = (30, 30, 30)
        self.COLOR_OCCUPIED = (220, 60, 60)
        self.COLOR_FREE = (80, 180, 80)
        self.COLOR_DARK = (40, 40, 40)
        self.COLOR_DRONE = (255, 215, 0)
        self.COLOR_SENSOR = (100, 200, 255)
        self.COLOR_TEXT = (255, 255, 255)

        # Map-to-screen transform
        self.map_offset_x = width / 2
        self.map_offset_y = height / 2
        self.map_scale = 10.0  # pixels per meter

        # Drone state
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_yaw = 0.0

        # Font
        self.font = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 28)

        # Running flag
        self.running = True

    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates."""
        screen_x = self.map_offset_x + x * self.map_scale
        screen_y = self.map_offset_y - y * self.map_scale
        return (int(screen_x), int(screen_y))

    def update_map_data(self, pose_x, pose_y, yaw, measurements):
        """
        Update map with new sensor data.
        pose_x, pose_y, yaw: drone position and orientation (yaw in degrees)
        measurements: dict with 'front', 'left', 'right', 'back' in mm
        """
        yaw_rad = math.radians(yaw)
        self.map.set_pose(pose_x, pose_y, yaw_rad)
        self.drone_x = pose_x
        self.drone_y = pose_y
        self.drone_yaw = yaw

        for side in ['front', 'left', 'right', 'back']:
            dist = measurements.get(side, 2000)
            self.map.update_from_sensor(pose_x, pose_y, yaw_rad, side, dist)

    def draw_map(self):
        """Draw the occupancy grid on the screen."""
        bounds = self.map.get_approx_bounds()
        min_x, min_y, max_x, max_y = bounds

        # Adjust view to center on explored area
        if max_x > min_x and max_y > min_y:
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            self.map_offset_x = self.width / 2 - center_x * self.map_scale
            self.map_offset_y = self.height / 2 + center_y * self.map_scale

        # Draw grid cells (1x1 pixel per cell)
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

    def draw_drone(self):
        """Draw the drone at its current position."""
        screen_pos = self.world_to_screen(self.drone_x, self.drone_y)

        # Drone body (circle)
        pygame.draw.circle(self.screen, self.COLOR_DRONE, screen_pos, 6)
        pygame.draw.circle(self.screen, (200, 170, 0), screen_pos, 4)

        # Drone orientation (arrow)
        yaw_rad = math.radians(self.drone_yaw)
        arrow_len = 20
        arrow_x = screen_pos[0] + int(arrow_len * math.cos(yaw_rad))
        arrow_y = screen_pos[1] - int(arrow_len * math.sin(yaw_rad))

        pygame.draw.line(self.screen, self.COLOR_DRONE, screen_pos, (arrow_x, arrow_y), 3)
        pygame.draw.circle(self.screen, self.COLOR_DRONE, (arrow_x, arrow_y), 3)

    def draw_sensors(self, measurements):
        """Draw sensor rays from the drone."""
        screen_pos = self.world_to_screen(self.drone_x, self.drone_y)
        yaw_rad = math.radians(self.drone_yaw)

        sensor_angles = {
            'front': yaw_rad,
            'left': yaw_rad + math.pi / 2,
            'right': yaw_rad - math.pi / 2,
            'back': yaw_rad + math.pi
        }

        for side, angle in sensor_angles.items():
            dist_mm = measurements.get(side, 2000)
            if dist_mm >= 2000:
                ray_len = 100  # max display length
            else:
                ray_len = min(100, dist_mm / 1000.0 * self.map_scale)

            end_x = screen_pos[0] + int(ray_len * math.cos(angle))
            end_y = screen_pos[1] - int(ray_len * math.sin(angle))

            color = self.COLOR_SENSOR if dist_mm < 2000 else (150, 150, 150)
            pygame.draw.line(self.screen, color, screen_pos, (end_x, end_y), 2)

    def draw_ui(self, measurements):
        """Draw UI text with information."""
        # Title
        title = self.font_large.render('Crazyflie 2D Occupancy Map', True, self.COLOR_TEXT)
        self.screen.blit(title, (10, 10))

        # Drone position
        pos_text = self.font.render(f'Pose: ({self.drone_x:.2f}, {self.drone_y:.2f}), Yaw: {self.drone_yaw:.1f}deg', True, self.COLOR_TEXT)
        self.screen.blit(pos_text, (10, 40))

        # Sensor readings
        sensors_text = self.font.render('Sensors (mm):', True, self.COLOR_TEXT)
        self.screen.blit(sensors_text, (10, 65))

        for i, side in enumerate(['front', 'back', 'left', 'right']):
            dist = measurements.get(side, 2000)
            text = self.font.render(f'  {side}: {dist}', True, self.COLOR_SENSOR if dist < 2000 else (150, 150, 150))
            self.screen.blit(text, (10, 85 + i * 20))

        # Map bounds
        bounds = self.map.get_approx_bounds()
        bounds_text = self.font.render(f'Bounds: X [{bounds[0]:.2f}, {bounds[2]:.2f}], Y [{bounds[1]:.2f}, {bounds[3]:.2f}]', True, self.COLOR_TEXT)
        self.screen.blit(bounds_text, (10, 175))

        # Occupied cells
        occupied = sum(1 for v in self.map.grid.values() if v > 50)
        occ_text = self.font.render(f'Occupied cells: {occupied}', True, self.COLOR_OCCUPIED)
        self.screen.blit(occ_text, (10, 195))

        # Instructions
        instr = self.font.render('ESC: Quit | Mouse: Pan view | Wheel: Zoom', True, (200, 200, 200))
        self.screen.blit(instr, (10, self.height - 25))

    def handle_events(self, measurements):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
            elif event.type == MOUSEMOTION:
                if event.buttons[0]:  # Left mouse button
                    self.map_offset_x += event.rel[0]
                    self.map_offset_y += event.rel[1]
            elif event.type == MOUSEWHEEL:
                zoom_factor = 1.0 + event.y * 0.1
                self.map_scale *= zoom_factor
                self.map_scale = max(2.0, min(50.0, self.map_scale))

    def run_standalone(self):
        """
        Run the visualizer in standalone mode with simulated data.
        Useful for testing without a drone.
        """
        print('Running in standalone mode with simulated data...')
        print('Press ESC to quit')

        import random
        rng = random.Random(42)

        pose_x, pose_y = 0.0, 0.0
        yaw = 0.0

        while self.running:
            self.clock.tick(60)

            # Simulate drone movement
            yaw += rng.uniform(-5, 5)
            speed = 0.1
            pose_x += speed * math.cos(math.radians(yaw))
            pose_y += speed * math.sin(math.radians(yaw))

            # Simulate sensor readings
            measurements = {
                'front': rng.uniform(500, 1800) if rng.random() > 0.3 else 2000,
                'back': rng.uniform(800, 2000),
                'left': rng.uniform(600, 1900) if rng.random() > 0.4 else 2000,
                'right': rng.uniform(700, 2000) if rng.random() > 0.35 else 2000
            }

            self.update_map_data(pose_x, pose_y, yaw, measurements)

            # Draw
            self.screen.fill(self.COLOR_BG)
            self.draw_map()
            self.draw_drone()
            self.draw_sensors(measurements)
            self.draw_ui(measurements)

            pygame.display.flip()

            # Slow down simulation
            time.sleep(0.05)

        pygame.quit()

    def run(self):
        """Main loop - override for real integration."""
        measurements = {
            'front': 2000,
            'back': 2000,
            'left': 2000,
            'right': 2000
        }

        while self.running:
            self.clock.tick(60)
            self.handle_events(measurements)

            self.screen.fill(self.COLOR_BG)
            self.draw_map()
            self.draw_drone()
            self.draw_sensors(measurements)
            self.draw_ui(measurements)

            pygame.display.flip()

        pygame.quit()


if __name__ == '__main__':
    visualizer = MapVisualizer(width=800, height=600, cell_size=0.05)
    visualizer.run_standalone()