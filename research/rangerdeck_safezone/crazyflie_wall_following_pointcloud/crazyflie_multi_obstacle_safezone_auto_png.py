# beste code 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crazyflie Multiranger wall following + point cloud

Controls:
  Esc   = smooth land
  Space = smooth land
  L     = hard emergency motor stop/disarm

This version:
  - follows the wall
  - draws the point cloud
  - detects one full lap using the estimated x/y position
  - after the wall lap, looks for multiple interior obstacle candidates in the point cloud
  - slowly flies around each detected obstacle one by one to collect more points
  - returns close to the starting position
  - lands automatically
  - keeps the point-cloud window open after landing
  - exports safe-zone JSON, cleaned CSV, and automatic PNG preview

Important:
  This is an experimental multi-obstacle scanner.
  Test with soft/light tubes first and keep your hand ready on Esc/Space/L.
"""

import logging
import json
import os
import math
import time
from math import degrees
from math import radians

import numpy as np
from vispy import scene
from vispy.scene import visuals
from vispy.scene.cameras import TurntableCamera

from wall_following import WallFollowing

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from cflib.utils.multiranger import Multiranger

from PyQt6 import QtCore, QtWidgets


URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E701')

# ---------- Flight settings ----------
TAKEOFF_HEIGHT = 0.30
SENSOR_TH = 2000

# Your setup: wall is on the LEFT side of the Crazyflie.
# This keeps the same side logic that worked for your drone:
#   direction RIGHT + left sensor
WALL_ON_LEFT_SIDE = True

REFERENCE_DISTANCE_FROM_WALL = 0.7
MAX_FORWARD_SPEED = 0.22
MAX_TURN_RATE = 0.50
EMERGENCY_DISTANCE = 0.15

# Safety around corners / front wall
FRONT_SLOW_DISTANCE = 0.45
FRONT_STOP_DISTANCE = 0.30
FRONT_EMERGENCY_DISTANCE = 0.18

# Top sensor stop
TOP_CLEARANCE_STOP = 0.20

# Point cloud
PLOT_SENSOR_DOWN = False
MAX_POINTCLOUD_POINTS = 25000

# ---------- One-lap detection ----------
# The drone will NOT check if it is back at the start until this time has passed.
LAP_MIN_TIME_SECONDS = 25.0

# It also must have flown at least this much total distance.
LAP_MIN_TRAVEL_DISTANCE = 1.8

# If it comes within this radius of the start point after LAP_MIN_TIME_SECONDS,
# it counts as being back at the start.
LAP_START_RADIUS = 0.35

# Optional heading check. If True, it also checks if yaw is close to start yaw.
USE_YAW_FOR_LAP_CHECK = False
LAP_YAW_TOLERANCE_DEG = 60.0

# Maximum flight time safety. After this time it lands even if lap was not detected.
MAX_FLIGHT_TIME_SECONDS = 180.0

# ---------- Mission modes ----------
MISSION_WALL_LAP = 1
MISSION_APPROACH_OBSTACLE = 2
MISSION_ORBIT_OBSTACLE = 3
MISSION_RETURN_HOME = 4
MISSION_LAND = 5

# ---------- Multi-obstacle scan settings ----------
# These values are intentionally slow/conservative.
# The drone first makes the wall lap, then tries to find multiple obstacle point clusters
# that are inside the room, not on the outer wall.
INTERIOR_MARGIN_FROM_WALL = 0.40       # ignore points close to outer wall boundary
OBSTACLE_GRID_SIZE = 0.25              # clustering grid size in meters
OBSTACLE_MIN_POINTS = 6                # minimum points in a cluster to count as obstacle
MAX_OBSTACLES_TO_SCAN = 4            # safety limit: maximum number of obstacles to visit
OBSTACLE_MIN_SEPARATION = 0.65       # ignore duplicate obstacle centers closer than this

OBSTACLE_APPROACH_SPEED = 0.08         # m/s, slow approach toward estimated obstacle
OBSTACLE_RETURN_SPEED = 0.10           # m/s, slow return to start
OBSTACLE_ORBIT_SPEED = 0.08            # m/s, slow around-obstacle waypoint speed
OBSTACLE_ORBIT_RADIUS = 0.85           # radius around estimated obstacle center; bigger = safer around obstacles
OBSTACLE_WAYPOINT_RADIUS = 0.18        # waypoint is reached when closer than this
OBSTACLE_ORBIT_POINTS = 8              # number of waypoints around the tube
OBSTACLE_ORBIT_TIMEOUT = 45.0          # max seconds spent orbiting each obstacle

OBSTACLE_DETECT_DISTANCE = 0.95        # if front sees tube closer than this, start orbit
OBSTACLE_STOP_DISTANCE = 0.55          # if too close, stop/side-step instead of approach
OBSTACLE_TOO_CLOSE = 0.35              # emergency distance around obstacle
RETURN_HOME_RADIUS = 0.30              # distance from start where it lands

NAV_YAW_GAIN = 1.2
NAV_MAX_YAW_RATE = 0.45

# ---------- Safe-zone export / outlier filtering ----------
# These do not control the flight directly. They clean the final point cloud and
# save a simple safe-zone map for other drones/controllers to use.
SAFE_ZONE_EXPORT_DIR = 'safe_zone_output'
POINTCLOUD_DOT_SIZE = 2               # smaller dots make the map less blob-like
FILTER_GRID_SIZE = 0.10               # meters; grid used to remove isolated outliers
FILTER_MIN_POINTS_PER_CELL = 2        # isolated cells with fewer points are removed
SAFE_WALL_MARGIN = 0.35               # keep other drones this far away from walls
SAFE_OBSTACLE_MARGIN = 0.55           # keep other drones this far away from obstacle centers


def handle_range_measurement(range_m):
    if range_m is None:
        return 999.0
    try:
        return float(range_m)
    except Exception:
        return 999.0


def safe_m2mm(x_m):
    """Convert meters to mm for renderer; return >SENSOR_TH if invalid."""
    if x_m is None or x_m >= 999:
        return SENSOR_TH + 1
    try:
        return float(x_m) * 1000.0
    except Exception:
        return SENSOR_TH + 1


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def distance_2d(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def yaw_difference_deg(a, b):
    """Smallest difference between two yaw angles in degrees."""
    diff = (a - b + 180.0) % 360.0 - 180.0
    return abs(diff)


def wrap_to_pi(angle):
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def make_body_velocity_to_target(position, target_position, yaw_rad, max_speed):
    """
    Convert world-frame target error into Crazyflie body-frame velocity commands.

    velocity_x = forward/backward in drone body frame
    velocity_y = left/right in drone body frame
    yaw_rate   = turn toward travel direction
    """
    dx = float(target_position[0] - position[0])
    dy = float(target_position[1] - position[1])
    dist = math.sqrt(dx * dx + dy * dy)

    if dist < 0.01:
        return 0.0, 0.0, 0.0, dist

    # World error -> body frame.
    # Body x is forward, body y is left.
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)

    body_x = cos_y * dx + sin_y * dy
    body_y = -sin_y * dx + cos_y * dy

    gain = 0.7
    velocity_x = clamp(gain * body_x, -max_speed, max_speed)
    velocity_y = clamp(gain * body_y, -max_speed, max_speed)

    wanted_yaw = math.atan2(dy, dx)
    yaw_error = wrap_to_pi(wanted_yaw - yaw_rad)
    yaw_rate = clamp(NAV_YAW_GAIN * yaw_error, -NAV_MAX_YAW_RATE, NAV_MAX_YAW_RATE)

    return velocity_x, velocity_y, yaw_rate, dist


def find_all_interior_obstacle_candidates(point_cloud, start_position=None):
    """
    Simple point-cloud clustering for multiple obstacles:
    - Estimate the outer room boundary from the full point cloud.
    - Ignore points close to that boundary.
    - Grid-cluster remaining interior points.
    - Merge connected grid cells into obstacle clusters.
    - Return several obstacle centers, sorted in a useful order.

    This is not real SLAM. It is a practical version for a few tubes/obstacles
    in the middle of the room.
    """
    if point_cloud is None or len(point_cloud) < 30:
        print('Obstacle search: not enough point-cloud points yet.')
        return []

    pts = np.asarray(point_cloud, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return []

    # Keep reasonable height points only.
    z = pts[:, 2]
    mask_z = (z > 0.05) & (z < 1.20)
    pts = pts[mask_z]

    if len(pts) < 30:
        print('Obstacle search: not enough valid-height points.')
        return []

    xy = pts[:, 0:2]

    # Robust room boundary estimate.
    min_x, max_x = np.percentile(xy[:, 0], [5, 95])
    min_y, max_y = np.percentile(xy[:, 1], [5, 95])

    if (max_x - min_x) < 1.0 or (max_y - min_y) < 1.0:
        print('Obstacle search: room boundary estimate too small/uncertain.')
        return []

    interior = (
        (xy[:, 0] > min_x + INTERIOR_MARGIN_FROM_WALL) &
        (xy[:, 0] < max_x - INTERIOR_MARGIN_FROM_WALL) &
        (xy[:, 1] > min_y + INTERIOR_MARGIN_FROM_WALL) &
        (xy[:, 1] < max_y - INTERIOR_MARGIN_FROM_WALL)
    )

    candidates = pts[interior]

    if len(candidates) < OBSTACLE_MIN_POINTS:
        print('Obstacle search: no clear interior obstacle candidate found.')
        return []

    # Put candidate points into a 2D grid.
    grid_xy = np.floor(candidates[:, 0:2] / OBSTACLE_GRID_SIZE).astype(int)

    # Map grid cells to point indices.
    cell_to_indices = {}
    for idx, cell in enumerate(grid_xy):
        key = (int(cell[0]), int(cell[1]))
        cell_to_indices.setdefault(key, []).append(idx)

    visited = set()
    clusters = []

    # Connected-component clustering in the grid using 8-neighbour cells.
    for cell in list(cell_to_indices.keys()):
        if cell in visited:
            continue

        stack = [cell]
        visited.add(cell)
        cluster_indices = []

        while stack:
            c = stack.pop()
            cluster_indices.extend(cell_to_indices[c])

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbour = (c[0] + dx, c[1] + dy)
                    if neighbour in cell_to_indices and neighbour not in visited:
                        visited.add(neighbour)
                        stack.append(neighbour)

        if len(cluster_indices) >= OBSTACLE_MIN_POINTS:
            cluster = candidates[cluster_indices]
            center = np.mean(cluster[:, 0:3], axis=0)
            center[2] = TAKEOFF_HEIGHT
            clusters.append({
                'center': center,
                'count': len(cluster_indices)
            })

    if len(clusters) == 0:
        print('Obstacle search: no clusters strong enough.')
        return []

    # Sort strongest clusters first.
    clusters.sort(key=lambda item: item['count'], reverse=True)

    # Remove duplicate/very close cluster centers.
    selected = []
    for cluster in clusters:
        center = cluster['center']
        duplicate = False
        for chosen in selected:
            if distance_2d(center, chosen['center']) < OBSTACLE_MIN_SEPARATION:
                duplicate = True
                break
        if not duplicate:
            selected.append(cluster)
        if len(selected) >= MAX_OBSTACLES_TO_SCAN:
            break

    if len(selected) == 0:
        print('Obstacle search: all clusters were duplicates/too close.')
        return []

    # If start position is known, order them with a simple nearest-neighbour route.
    if start_position is not None:
        ordered = []
        current = np.array(start_position, dtype=float)
        remaining = selected[:]

        while remaining:
            best_i = min(
                range(len(remaining)),
                key=lambda i: distance_2d(current, remaining[i]['center'])
            )
            next_cluster = remaining.pop(best_i)
            ordered.append(next_cluster)
            current = next_cluster['center']

        selected = ordered

    print('')
    print(f'Obstacle search: found {len(selected)} obstacle candidate(s).')
    for i, cluster in enumerate(selected):
        c = cluster['center']
        print(f'  obstacle {i + 1}: x={c[0]:.2f}, y={c[1]:.2f}, points={cluster["count"]}')
    print('')

    return [cluster['center'] for cluster in selected]


def filter_pointcloud_outliers(point_cloud):
    """
    Remove isolated random dots from the final map.

    This is NOT used for live obstacle avoidance during flight.
    It is only used for exporting a cleaner map/safe-zone after the mission.
    """
    if point_cloud is None or len(point_cloud) == 0:
        return np.zeros((0, 3), dtype=float)

    pts = np.asarray(point_cloud, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return np.zeros((0, 3), dtype=float)

    # Keep only reasonable flight/map height points.
    z = pts[:, 2]
    mask_z = (z > 0.03) & (z < 1.30)
    pts = pts[mask_z]

    if len(pts) == 0:
        return np.zeros((0, 3), dtype=float)

    # Count points per 2D grid cell.
    grid_xy = np.floor(pts[:, 0:2] / FILTER_GRID_SIZE).astype(int)
    unique_cells, inverse, counts = np.unique(
        grid_xy,
        axis=0,
        return_inverse=True,
        return_counts=True
    )

    keep = counts[inverse] >= FILTER_MIN_POINTS_PER_CELL
    return pts[keep]


def estimate_room_boundary(point_cloud):
    """Estimate rectangular room boundary from the cleaned point cloud."""
    pts = np.asarray(point_cloud, dtype=float)
    if len(pts) < 20:
        return None

    xy = pts[:, 0:2]
    min_x, max_x = np.percentile(xy[:, 0], [3, 97])
    min_y, max_y = np.percentile(xy[:, 1], [3, 97])

    if (max_x - min_x) < 0.5 or (max_y - min_y) < 0.5:
        return None

    return {
        'min_x': float(min_x),
        'max_x': float(max_x),
        'min_y': float(min_y),
        'max_y': float(max_y)
    }


def export_safe_zone(point_cloud, obstacle_targets, start_position):
    """
    Save a cleaned point cloud and a simple 2D safe-zone JSON file.

    The safe zone is conservative:
      - wall boundary is shrunk inward by SAFE_WALL_MARGIN
      - each scanned obstacle gets a circular keep-out zone with SAFE_OBSTACLE_MARGIN

    Other drones should only fly inside safe_zone_polygon and outside obstacle_keepouts.
    """
    try:
        os.makedirs(SAFE_ZONE_EXPORT_DIR, exist_ok=True)

        filtered = filter_pointcloud_outliers(point_cloud)
        boundary = estimate_room_boundary(filtered)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(SAFE_ZONE_EXPORT_DIR, f'filtered_pointcloud_{timestamp}.csv')
        json_path = os.path.join(SAFE_ZONE_EXPORT_DIR, f'safe_zone_{timestamp}.json')
        png_path = os.path.join(SAFE_ZONE_EXPORT_DIR, f'safe_zone_{timestamp}.png')

        if len(filtered) > 0:
            np.savetxt(
                csv_path,
                filtered,
                delimiter=',',
                header='x,y,z',
                comments=''
            )

        obstacle_keepouts = []
        if obstacle_targets is not None:
            for i, obstacle in enumerate(obstacle_targets):
                c = np.asarray(obstacle, dtype=float)
                obstacle_keepouts.append({
                    'id': int(i + 1),
                    'center': [float(c[0]), float(c[1])],
                    'radius': float(SAFE_OBSTACLE_MARGIN)
                })

        safe_zone_polygon = []
        room_boundary_polygon = []
        if boundary is not None:
            min_x = boundary['min_x']
            max_x = boundary['max_x']
            min_y = boundary['min_y']
            max_y = boundary['max_y']

            room_boundary_polygon = [
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y]
            ]

            safe_min_x = min_x + SAFE_WALL_MARGIN
            safe_max_x = max_x - SAFE_WALL_MARGIN
            safe_min_y = min_y + SAFE_WALL_MARGIN
            safe_max_y = max_y - SAFE_WALL_MARGIN

            if safe_min_x < safe_max_x and safe_min_y < safe_max_y:
                safe_zone_polygon = [
                    [safe_min_x, safe_min_y],
                    [safe_max_x, safe_min_y],
                    [safe_max_x, safe_max_y],
                    [safe_min_x, safe_max_y]
                ]

        start = None
        if start_position is not None:
            s = np.asarray(start_position, dtype=float)
            start = [float(s[0]), float(s[1])]

        output = {
            'description': '2D safe-zone map generated from Crazyflie wall/obstacle scan',
            'units': 'meters',
            'filtered_pointcloud_csv': csv_path,
            'raw_point_count': int(0 if point_cloud is None else len(point_cloud)),
            'filtered_point_count': int(len(filtered)),
            'room_boundary_polygon': room_boundary_polygon,
            'safe_zone_polygon': safe_zone_polygon,
            'obstacle_keepouts': obstacle_keepouts,
            'start_position_xy': start,
            'safe_zone_png': png_path,
            'rules_for_other_drones': {
                'allowed': 'inside safe_zone_polygon',
                'not_allowed': 'outside safe_zone_polygon or inside any obstacle_keepout circle',
                'wall_margin_m': float(SAFE_WALL_MARGIN),
                'obstacle_margin_m': float(SAFE_OBSTACLE_MARGIN)
            }
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

        save_safe_zone_png(filtered, output, png_path)

        print('')
        print('Safe-zone export complete.')
        print(f'  raw points: {output["raw_point_count"]}')
        print(f'  filtered points: {output["filtered_point_count"]}')
        print(f'  pointcloud csv: {csv_path}')
        print(f'  safe-zone json: {json_path}')
        print('')

    except Exception as e:
        print(f'Safe-zone export failed: {e}')


def save_safe_zone_png(filtered_points, safe_zone_data, png_path):
    """
    Save a PNG image showing:
      - filtered point cloud
      - room boundary
      - safe-zone polygon
      - obstacle keep-out circles
      - start position

    This is only for visualization after the flight. It does not affect flight control.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, Circle

        pts = np.asarray(filtered_points, dtype=float)

        fig, ax = plt.subplots(figsize=(10, 8))

        if pts.ndim == 2 and pts.shape[1] >= 2 and len(pts) > 0:
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=5,
                c='blue',
                alpha=0.70,
                label='filtered pointcloud'
            )

        room_boundary = safe_zone_data.get('room_boundary_polygon', [])
        if room_boundary:
            room_poly = Polygon(
                room_boundary,
                closed=True,
                fill=False,
                edgecolor='black',
                linewidth=2.0,
                label='room boundary'
            )
            ax.add_patch(room_poly)

        safe_zone = safe_zone_data.get('safe_zone_polygon', [])
        if safe_zone:
            safe_poly = Polygon(
                safe_zone,
                closed=True,
                fill=True,
                facecolor='green',
                edgecolor='green',
                alpha=0.18,
                linewidth=2.0,
                label='safe zone'
            )
            ax.add_patch(safe_poly)

        obstacle_keepouts = safe_zone_data.get('obstacle_keepouts', [])
        for obstacle in obstacle_keepouts:
            center = obstacle.get('center', None)
            radius = obstacle.get('radius', None)
            obstacle_id = obstacle.get('id', '?')

            if center is None or radius is None:
                continue

            circle = Circle(
                (float(center[0]), float(center[1])),
                float(radius),
                fill=True,
                facecolor='red',
                edgecolor='red',
                alpha=0.25,
                linewidth=2.0
            )
            ax.add_patch(circle)
            ax.text(
                float(center[0]),
                float(center[1]),
                f'obstacle {obstacle_id}',
                ha='center',
                va='center',
                fontsize=9
            )

        start = safe_zone_data.get('start_position_xy', None)
        if start is not None:
            ax.scatter(
                [float(start[0])],
                [float(start[1])],
                s=140,
                c='orange',
                marker='*',
                edgecolors='black',
                label='start'
            )

        ax.set_title('Crazyflie safe-zone map')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.axis('equal')
        ax.grid(True)
        ax.legend(loc='best')

        fig.tight_layout()
        fig.savefig(png_path, dpi=200)
        plt.close(fig)

        print(f'  safe-zone png: {png_path}')

    except Exception as e:
        print(f'Could not save safe-zone PNG: {e}')


class ObstacleOrbitPlanner:
    def __init__(self):
        self.center = None
        self.waypoints = []
        self.index = 0
        self.start_time = None
        self.finished = False

    def start(self, obstacle_center, current_position):
        self.center = np.array(obstacle_center, dtype=float)
        current_position = np.array(current_position, dtype=float)

        start_angle = math.atan2(
            current_position[1] - self.center[1],
            current_position[0] - self.center[0]
        )

        self.waypoints = []
        for i in range(OBSTACLE_ORBIT_POINTS):
            angle = start_angle + 2.0 * math.pi * (i / OBSTACLE_ORBIT_POINTS)
            wp = np.array([
                self.center[0] + OBSTACLE_ORBIT_RADIUS * math.cos(angle),
                self.center[1] + OBSTACLE_ORBIT_RADIUS * math.sin(angle),
                TAKEOFF_HEIGHT
            ])
            self.waypoints.append(wp)

        self.index = 0
        self.start_time = time.time()
        self.finished = False

        print('')
        print('Obstacle orbit started.')
        print(f'  center x={self.center[0]:.2f}, y={self.center[1]:.2f}')
        print(f'  orbit radius={OBSTACLE_ORBIT_RADIUS:.2f} m')
        print('')

    def current_waypoint(self):
        if self.finished or self.index >= len(self.waypoints):
            return None
        return self.waypoints[self.index]

    def update(self, current_position):
        if self.finished:
            return True

        if self.start_time is None:
            return True

        if time.time() - self.start_time > OBSTACLE_ORBIT_TIMEOUT:
            print('Obstacle orbit timeout reached.')
            self.finished = True
            return True

        wp = self.current_waypoint()
        if wp is None:
            self.finished = True
            return True

        if distance_2d(current_position, wp) < OBSTACLE_WAYPOINT_RADIUS:
            self.index += 1
            print(f'Obstacle orbit waypoint {self.index}/{len(self.waypoints)} reached.')

            if self.index >= len(self.waypoints):
                print('Obstacle orbit complete.')
                self.finished = True
                return True

        return False


class LapDetector:
    def __init__(self):
        self.start_position = None
        self.start_yaw = None
        self.last_position = None
        self.start_time = None
        self.travel_distance = 0.0
        self.lap_complete = False

    def start(self, position, yaw_deg):
        self.start_position = np.array(position, dtype=float)
        self.last_position = np.array(position, dtype=float)
        self.start_yaw = float(yaw_deg)
        self.start_time = time.time()
        self.travel_distance = 0.0
        self.lap_complete = False

        print('')
        print('Lap detector started.')
        print(f'  Start position: x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f}')
        print(f'  Start yaw: {yaw_deg:.1f} deg')
        print('')

    def update(self, position, yaw_deg):
        if self.start_position is None:
            self.start(position, yaw_deg)
            return False

        position = np.array(position, dtype=float)

        step_distance = distance_2d(position, self.last_position)
        if step_distance < 0.50:
            self.travel_distance += step_distance

        self.last_position = position

        elapsed = time.time() - self.start_time
        distance_to_start = distance_2d(position, self.start_position)

        if elapsed < LAP_MIN_TIME_SECONDS:
            return False

        if self.travel_distance < LAP_MIN_TRAVEL_DISTANCE:
            return False

        if distance_to_start > LAP_START_RADIUS:
            return False

        if USE_YAW_FOR_LAP_CHECK:
            yaw_error = yaw_difference_deg(float(yaw_deg), self.start_yaw)
            if yaw_error > LAP_YAW_TOLERANCE_DEG:
                return False

        self.lap_complete = True
        print('')
        print('ONE FULL LAP DETECTED.')
        print(f'  elapsed={elapsed:.1f}s')
        print(f'  travelled={self.travel_distance:.2f}m')
        print(f'  distance_to_start={distance_to_start:.2f}m')
        print('Wall lap complete. Next: obstacle scan mode.')
        print('')

        return True


class Canvas(scene.SceneCanvas):
    def __init__(self):
        super().__init__(keys=None)
        self.unfreeze()

        self.view = self.central_widget.add_view()
        self.view.bgcolor = '#ffffff'
        self.view.camera = TurntableCamera(
            fov=10.0,
            distance=8.0,
            up='+z',
            center=(0, 0, 0)
        )

        self.pos_markers = visuals.Markers()
        self.meas_markers = visuals.Markers()
        self.lines = []

        self.cf_position = np.array([0.0, 0.0, 0.0])
        self.meas_data = np.zeros((0, 3), dtype=float)

        self.land_requested = False
        self.hard_stop_requested = False

        self.view.add(self.pos_markers)
        self.view.add(self.meas_markers)

        for _ in range(6):
            line = visuals.Line(color='black', width=1)
            self.lines.append(line)
            self.view.add(line)

        scene.visuals.XYZAxis(parent=self.view.scene)

        self.pos_markers.set_data(
            np.array([[0.0, 0.0, 0.0]]),
            face_color='red',
            size=8
        )

        self.meas_markers.set_data(
            np.zeros((0, 3)),
            face_color='blue',
            size=POINTCLOUD_DOT_SIZE
        )

        self.freeze()

    def on_key_press(self, event):
        if event.native.isAutoRepeat():
            return

        key = event.native.key()

        if key in (QtCore.Qt.Key.Key_Escape, QtCore.Qt.Key.Key_Space):
            print('Emergency land requested.')
            self.land_requested = True

        elif key == QtCore.Qt.Key.Key_L:
            print('HARD EMERGENCY STOP requested.')
            self.hard_stop_requested = True

    def set_position(self, position):
        self.cf_position = np.array(position, dtype=float)
        self.pos_markers.set_data(
            np.array([self.cf_position]),
            face_color='red',
            size=8
        )

    def rot(self, roll, pitch, yaw, origin, point):
        cosr = math.cos(math.radians(roll))
        cosp = math.cos(math.radians(pitch))
        cosy = math.cos(math.radians(yaw))

        sinr = math.sin(math.radians(roll))
        sinp = math.sin(math.radians(pitch))
        siny = math.sin(math.radians(yaw))

        roty = np.array([
            [cosy, -siny, 0],
            [siny, cosy, 0],
            [0, 0, 1]
        ])

        rotp = np.array([
            [cosp, 0, sinp],
            [0, 1, 0],
            [-sinp, 0, cosp]
        ])

        rotr = np.array([
            [1, 0, 0],
            [0, cosr, -sinr],
            [0, sinr, cosr]
        ])

        rot = np.dot(np.dot(rotr, rotp), roty)
        return np.add(np.dot(rot, np.subtract(point, origin)), origin)

    def rotate_and_create_points(self, m_mm, origin):
        data = []

        roll = m_mm.get('roll', 0.0)
        pitch = -m_mm.get('pitch', 0.0)
        yaw = m_mm.get('yaw', 0.0)

        def is_valid(meas):
            if meas is None:
                return False
            try:
                return float(meas) < SENSOR_TH
            except Exception:
                return False

        if is_valid(m_mm.get('up')):
            up = [
                origin[0],
                origin[1],
                origin[2] + float(m_mm['up']) / 1000.0
            ]
            data.append(self.rot(roll, pitch, yaw, origin, up))

        if is_valid(m_mm.get('down')) and PLOT_SENSOR_DOWN:
            down = [
                origin[0],
                origin[1],
                origin[2] - float(m_mm['down']) / 1000.0
            ]
            data.append(self.rot(roll, pitch, yaw, origin, down))

        if is_valid(m_mm.get('left')):
            left = [
                origin[0],
                origin[1] + float(m_mm['left']) / 1000.0,
                origin[2]
            ]
            data.append(self.rot(roll, pitch, yaw, origin, left))

        if is_valid(m_mm.get('right')):
            right = [
                origin[0],
                origin[1] - float(m_mm['right']) / 1000.0,
                origin[2]
            ]
            data.append(self.rot(roll, pitch, yaw, origin, right))

        if is_valid(m_mm.get('front')):
            front = [
                origin[0] + float(m_mm['front']) / 1000.0,
                origin[1],
                origin[2]
            ]
            data.append(self.rot(roll, pitch, yaw, origin, front))

        if is_valid(m_mm.get('back')):
            back = [
                origin[0] - float(m_mm['back']) / 1000.0,
                origin[1],
                origin[2]
            ]
            data.append(self.rot(roll, pitch, yaw, origin, back))

        return data

    def set_measurement(self, measurements_mm, position):
        position = np.array(position, dtype=float)
        points = self.rotate_and_create_points(measurements_mm, position)

        if len(points) > 0:
            self.meas_data = np.append(self.meas_data, points, axis=0)

            if len(self.meas_data) > MAX_POINTCLOUD_POINTS:
                self.meas_data = self.meas_data[-MAX_POINTCLOUD_POINTS:]

            self.meas_markers.set_data(
                self.meas_data,
                face_color='blue',
                size=POINTCLOUD_DOT_SIZE
            )

        for idx in range(len(self.lines)):
            if idx < len(points):
                self.lines[idx].set_data(np.array([position, points[idx]]))
            else:
                self.lines[idx].set_data(np.array([position, position]))


def choose_wall_following_side(left_range, right_range):
    if WALL_ON_LEFT_SIDE:
        wall_following_direction = WallFollowing.WallFollowingDirection.RIGHT
        side_range = left_range
    else:
        wall_following_direction = WallFollowing.WallFollowingDirection.LEFT
        side_range = right_range

    return wall_following_direction, side_range


def apply_corner_safety(velocity_x, velocity_y, yaw_rate, front_range):
    """
    Prevents the drone from continuing forward into a wall after corners.
    It keeps the wall follower's yaw command, but limits forward motion.
    """

    if front_range < FRONT_EMERGENCY_DISTANCE:
        velocity_x = -0.03
        velocity_y = 0.0
        print('FRONT EMERGENCY: backing up slightly.')

    elif front_range < FRONT_STOP_DISTANCE:
        velocity_x = 0.0
        velocity_y = 0.0
        print('FRONT STOP: stopping forward motion and turning.')

    elif front_range < FRONT_SLOW_DISTANCE:
        velocity_x = min(velocity_x, 0.06)
        velocity_y = clamp(velocity_y, -0.05, 0.05)
        print('FRONT SLOW: slowing down before wall/corner.')

    return velocity_x, velocity_y, yaw_rate


def arm_cf(cf, do_arm):
    try:
        if hasattr(cf, 'supervisor'):
            cf.supervisor.send_arming_request(do_arm)
        else:
            cf.platform.send_arming_request(do_arm)
    except Exception as e:
        print(f'Arming request failed: {e}')


def smooth_land(motion_commander):
    print('Landing smoothly...')
    try:
        motion_commander.stop()
        motion_commander.land(velocity=0.15)
        print('Landed.')
    except Exception as e:
        print(f'Smooth land failed: {e}')


def hard_stop(scf, motion_commander):
    print('HARD STOP: motors off now.')
    try:
        motion_commander.stop()
    except Exception:
        pass

    try:
        scf.cf.commander.send_stop_setpoint()
    except Exception:
        pass

    arm_cf(scf.cf, False)


if __name__ == '__main__':
    cflib.crtp.init_drivers()
    logging.basicConfig(level=logging.ERROR)

    app = QtWidgets.QApplication([])
    canvas = Canvas()
    canvas.show()

    print('Controls:')
    print('  Esc   = smooth land')
    print('  Space = smooth land')
    print('  L     = HARD motor stop/disarm')
    print('  Cover top sensor = smooth land')
    print('')
    print('Wall setup:')
    print(f'  WALL_ON_LEFT_SIDE = {WALL_ON_LEFT_SIDE}')
    print('')
    print('One-lap mode:')
    print(f'  LAP_MIN_TIME_SECONDS = {LAP_MIN_TIME_SECONDS}')
    print(f'  LAP_MIN_TRAVEL_DISTANCE = {LAP_MIN_TRAVEL_DISTANCE}')
    print(f'  LAP_START_RADIUS = {LAP_START_RADIUS}')
    print('')

    keep_flying = True
    flight_finished = False
    flight_start_time = None

    mission_mode = MISSION_WALL_LAP
    obstacle_targets = []
    current_obstacle_index = 0
    obstacle_target = None
    orbit_planner = ObstacleOrbitPlanner()

    lap_detector = LapDetector()

    wall_following = WallFollowing(
        angle_value_buffer=0.1,
        reference_distance_from_wall=REFERENCE_DISTANCE_FROM_WALL,
        max_forward_speed=MAX_FORWARD_SPEED,
        max_turn_rate=MAX_TURN_RATE,
        emergency_distance=EMERGENCY_DISTANCE,
        init_state=WallFollowing.StateWallFollowing.FORWARD
    )

    lg = LogConfig(name='PosStab', period_in_ms=100)
    lg.add_variable('stateEstimate.x', 'float')
    lg.add_variable('stateEstimate.y', 'float')
    lg.add_variable('stateEstimate.z', 'float')
    lg.add_variable('stabilizer.roll', 'float')
    lg.add_variable('stabilizer.pitch', 'float')
    lg.add_variable('stabilizer.yaw', 'float')

    cf = Crazyflie(rw_cache='./cache')

    try:
        with SyncCrazyflie(URI, cf=cf) as scf:
            arm_cf(scf.cf, True)
            time.sleep(1.0)

            with MotionCommander(scf, default_height=TAKEOFF_HEIGHT) as motion_commander:
                print('Taking off...')
                time.sleep(1.0)

                flight_start_time = time.time()

                with Multiranger(scf) as multiranger:
                    with SyncLogger(scf, lg) as logger:
                        print('Wall following started.')
                        print('Mission: wall lap -> find obstacles -> fly around each obstacle -> return home -> land.')
                        print('The point cloud window will stay open after landing.')
                        print('')

                        while keep_flying:
                            app.processEvents()

                            if canvas.hard_stop_requested:
                                hard_stop(scf, motion_commander)
                                keep_flying = False
                                flight_finished = True
                                break

                            if canvas.land_requested:
                                smooth_land(motion_commander)
                                keep_flying = False
                                flight_finished = True
                                break

                            log_entry = logger.next()
                            data = log_entry[1]

                            pos_x = data.get('stateEstimate.x', 0.0)
                            pos_y = data.get('stateEstimate.y', 0.0)
                            pos_z = data.get('stateEstimate.z', TAKEOFF_HEIGHT)

                            roll = data.get('stabilizer.roll', 0.0)
                            pitch = data.get('stabilizer.pitch', 0.0)
                            actual_yaw = data.get('stabilizer.yaw', 0.0)
                            actual_yaw_rad = radians(actual_yaw)

                            position = np.array([pos_x, pos_y, pos_z])

                            front_range = handle_range_measurement(multiranger.front)
                            back_range = handle_range_measurement(multiranger.back)
                            top_range = handle_range_measurement(multiranger.up)
                            down_range = handle_range_measurement(multiranger.down)
                            left_range = handle_range_measurement(multiranger.left)
                            right_range = handle_range_measurement(multiranger.right)

                            wall_following_direction, side_range = choose_wall_following_side(
                                left_range,
                                right_range
                            )

                            # ---------------- Mission behavior ----------------
                            state_wf = 'NONE'

                            if mission_mode == MISSION_WALL_LAP:
                                velocity_x, velocity_y, yaw_rate, state_wf = wall_following.wall_follower(
                                    front_range,
                                    side_range,
                                    actual_yaw_rad,
                                    wall_following_direction,
                                    time.time()
                                )

                                velocity_x, velocity_y, yaw_rate = apply_corner_safety(
                                    velocity_x,
                                    velocity_y,
                                    yaw_rate,
                                    front_range
                                )

                            elif mission_mode == MISSION_APPROACH_OBSTACLE:
                                state_wf = 'APPROACH_OBSTACLE'

                                if obstacle_target is None:
                                    print('No obstacle target available. Returning home.')
                                    mission_mode = MISSION_RETURN_HOME
                                    velocity_x = 0.0
                                    velocity_y = 0.0
                                    yaw_rate = 0.0
                                else:
                                    velocity_x, velocity_y, yaw_rate, dist_to_obstacle = make_body_velocity_to_target(
                                        position,
                                        obstacle_target,
                                        actual_yaw_rad,
                                        OBSTACLE_APPROACH_SPEED
                                    )

                                    # Live sensor safety while approaching the estimated tube.
                                    if front_range < OBSTACLE_TOO_CLOSE:
                                        velocity_x = -0.04
                                        velocity_y = 0.05
                                        yaw_rate = 0.35
                                        print('Obstacle too close: backing/side-stepping.')

                                    elif front_range < OBSTACLE_STOP_DISTANCE:
                                        velocity_x = 0.0
                                        velocity_y = 0.05
                                        yaw_rate = 0.35
                                        print('Obstacle close: starting scan/orbit soon.')

                                    # Start orbit either when the front sensor sees the tube,
                                    # or when position estimate says we are near it.
                                    if front_range < OBSTACLE_DETECT_DISTANCE or dist_to_obstacle < OBSTACLE_ORBIT_RADIUS:
                                        orbit_planner.start(obstacle_target, position)
                                        mission_mode = MISSION_ORBIT_OBSTACLE

                            elif mission_mode == MISSION_ORBIT_OBSTACLE:
                                state_wf = 'ORBIT_OBSTACLE'

                                if orbit_planner.finished:
                                    current_obstacle_index += 1
                                    if current_obstacle_index < len(obstacle_targets):
                                        obstacle_target = obstacle_targets[current_obstacle_index]
                                        orbit_planner = ObstacleOrbitPlanner()
                                        mission_mode = MISSION_APPROACH_OBSTACLE
                                        print(f'Going to next obstacle {current_obstacle_index + 1}/{len(obstacle_targets)}.')
                                    else:
                                        mission_mode = MISSION_RETURN_HOME
                                        print('All detected obstacles scanned. Returning to start position.')
                                    velocity_x = 0.0
                                    velocity_y = 0.0
                                    yaw_rate = 0.0
                                else:
                                    wp = orbit_planner.current_waypoint()

                                    if wp is None:
                                        current_obstacle_index += 1
                                        if current_obstacle_index < len(obstacle_targets):
                                            obstacle_target = obstacle_targets[current_obstacle_index]
                                            orbit_planner = ObstacleOrbitPlanner()
                                            mission_mode = MISSION_APPROACH_OBSTACLE
                                            print(f'Going to next obstacle {current_obstacle_index + 1}/{len(obstacle_targets)}.')
                                        else:
                                            mission_mode = MISSION_RETURN_HOME
                                            print('All detected obstacles scanned. Returning to start position.')
                                        velocity_x = 0.0
                                        velocity_y = 0.0
                                        yaw_rate = 0.0
                                    else:
                                        velocity_x, velocity_y, yaw_rate, dist_to_wp = make_body_velocity_to_target(
                                            position,
                                            wp,
                                            actual_yaw_rad,
                                            OBSTACLE_ORBIT_SPEED
                                        )

                                        # Live obstacle safety during orbit.
                                        if front_range < OBSTACLE_TOO_CLOSE:
                                            velocity_x = -0.04
                                            velocity_y = 0.06
                                            yaw_rate = 0.35
                                            print('Orbit safety: too close in front.')

                                        elif front_range < OBSTACLE_STOP_DISTANCE:
                                            velocity_x = 0.0
                                            velocity_y = 0.06
                                            yaw_rate = 0.35
                                            print('Orbit safety: front close, side-stepping.')

                                        if orbit_planner.update(position):
                                            current_obstacle_index += 1
                                            if current_obstacle_index < len(obstacle_targets):
                                                obstacle_target = obstacle_targets[current_obstacle_index]
                                                orbit_planner = ObstacleOrbitPlanner()
                                                mission_mode = MISSION_APPROACH_OBSTACLE
                                                print(f'Going to next obstacle {current_obstacle_index + 1}/{len(obstacle_targets)}.')
                                            else:
                                                mission_mode = MISSION_RETURN_HOME
                                                print('All detected obstacles scanned. Returning to start position.')

                            elif mission_mode == MISSION_RETURN_HOME:
                                state_wf = 'RETURN_HOME'

                                if lap_detector.start_position is None:
                                    print('No start position known. Landing.')
                                    mission_mode = MISSION_LAND
                                    velocity_x = 0.0
                                    velocity_y = 0.0
                                    yaw_rate = 0.0
                                else:
                                    velocity_x, velocity_y, yaw_rate, dist_home = make_body_velocity_to_target(
                                        position,
                                        lap_detector.start_position,
                                        actual_yaw_rad,
                                        OBSTACLE_RETURN_SPEED
                                    )

                                    # Simple live safety while returning.
                                    if front_range < OBSTACLE_STOP_DISTANCE:
                                        velocity_x = 0.0
                                        velocity_y = 0.05
                                        yaw_rate = 0.35
                                        print('Return safety: front close, side-stepping.')

                                    if dist_home < RETURN_HOME_RADIUS:
                                        print('Returned close to start. Landing.')
                                        mission_mode = MISSION_LAND
                                        velocity_x = 0.0
                                        velocity_y = 0.0
                                        yaw_rate = 0.0

                            elif mission_mode == MISSION_LAND:
                                state_wf = 'LAND'
                                velocity_x = 0.0
                                velocity_y = 0.0
                                yaw_rate = 0.0

                            else:
                                state_wf = 'UNKNOWN'
                                velocity_x = 0.0
                                velocity_y = 0.0
                                yaw_rate = 0.0

                            yaw_rate_deg = degrees(yaw_rate)

                            motion_commander.start_linear_motion(
                                velocity_x,
                                velocity_y,
                                0.0,
                                rate_yaw=yaw_rate_deg
                            )

                            meas_mm = {
                                'roll': roll,
                                'pitch': pitch,
                                'yaw': actual_yaw,
                                'front': safe_m2mm(front_range),
                                'back': safe_m2mm(back_range),
                                'up': safe_m2mm(top_range),
                                'down': safe_m2mm(down_range),
                                'left': safe_m2mm(left_range),
                                'right': safe_m2mm(right_range)
                            }

                            canvas.set_position(position)
                            canvas.set_measurement(meas_mm, position)

                            lap_complete = False
                            if mission_mode == MISSION_WALL_LAP:
                                lap_complete = lap_detector.update(position, actual_yaw)

                            elapsed_flight_time = time.time() - flight_start_time

                            if hasattr(state_wf, 'name'):
                                state_text = state_wf.name
                            else:
                                state_text = str(state_wf)

                            print(
                                f'mode={mission_mode}, obstacle={current_obstacle_index + 1}/{len(obstacle_targets)}, '
                                f'vx={velocity_x:.2f}, vy={velocity_y:.2f}, '
                                f'yaw={yaw_rate_deg:.1f}, state={state_text}, '
                                f'front={front_range:.2f}, side={side_range:.2f}, '
                                f'z={pos_z:.2f}, travelled={lap_detector.travel_distance:.2f}'
                            )

                            if lap_complete:
                                print('Wall lap complete. Looking for interior obstacles.')
                                obstacle_targets = find_all_interior_obstacle_candidates(
                                    canvas.meas_data,
                                    lap_detector.start_position
                                )

                                if len(obstacle_targets) == 0:
                                    print('No interior obstacles found. Returning home/landing.')
                                    mission_mode = MISSION_RETURN_HOME
                                else:
                                    current_obstacle_index = 0
                                    obstacle_target = obstacle_targets[current_obstacle_index]
                                    orbit_planner = ObstacleOrbitPlanner()
                                    print(f'Starting approach to obstacle {current_obstacle_index + 1}/{len(obstacle_targets)}.')
                                    mission_mode = MISSION_APPROACH_OBSTACLE

                            if mission_mode == MISSION_LAND:
                                smooth_land(motion_commander)
                                keep_flying = False
                                flight_finished = True
                                break

                            if elapsed_flight_time > MAX_FLIGHT_TIME_SECONDS:
                                print('Maximum flight time reached. Landing.')
                                smooth_land(motion_commander)
                                keep_flying = False
                                flight_finished = True
                                break

                            if top_range < TOP_CLEARANCE_STOP:
                                print('Top sensor triggered: landing.')
                                smooth_land(motion_commander)
                                keep_flying = False
                                flight_finished = True
                                break

    except KeyboardInterrupt:
        print('Interrupted by user.')

    except Exception as e:
        print(f'Error: {e}')

    finally:
        print('')
        print('Flight code finished.')
        print('The Crazyflie link should now be closed.')
        print('The point cloud window will stay open.')
        print('Close the point cloud window manually when you are done inspecting it.')
        print('')

        try:
            arm_cf(cf, False)
        except Exception:
            pass

        # Export a cleaned point cloud + a simple safe-zone JSON for other drones.
        try:
            export_safe_zone(canvas.meas_data, obstacle_targets, lap_detector.start_position)
        except Exception as e:
            print(f'Could not export safe zone: {e}')

        # Important:
        # Do NOT call app.exit() here.
        # This keeps the point cloud visible after landing.
        app.exec()



































