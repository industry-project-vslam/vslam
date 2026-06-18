#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crazyflie AI-deck safe-zone perimeter + object inspection navigator with grid planner

Drone URI: radio://0/80/2M/E7E7E7E702

Mission:
  1. Load newest safe-zone JSON from safe_zone_output
  2. Take off and wait for Flow deck / stabilizer to settle
  3. Go to nearest safe-zone entry point
  4. Fly a perimeter lap inside the safe zone and visit all 4 safe-zone corners
  5. If obstacle keepouts exist, visit safe viewpoints around each obstacle and look at it for 5 seconds
  6. Fly a coarse interior sweep
  7. Return to start and land

This does not yet run AI object/person detection. It is a camera-coverage flight plan.
"""

import glob
import json
import logging
import math
import os
import time
from math import radians, degrees

import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E702')

SAFE_ZONE_DIR = 'safe_zone_output'
SAFE_ZONE_JSON = None

# Offset support for combined mission:
# If the AI-deck drone is not placed at exactly the same physical start spot
# as the multiranger mapping drone, set this offset.
#
# Positive X = in front of the multiranger start, assuming both drones face the same direction.
# Positive Y = left of the multiranger start.
#
# The combined launcher can also set these with environment variables:
#   AI_START_OFFSET_X
#   AI_START_OFFSET_Y
AI_START_OFFSET_X = float(os.environ.get('AI_START_OFFSET_X', '0.0'))
AI_START_OFFSET_Y = float(os.environ.get('AI_START_OFFSET_Y', '0.0'))

TAKEOFF_HEIGHT = 0.30
ENTRY_SPEED = 0.06
SCAN_SPEED = 0.065
INSPECT_SPEED = 0.045
RETURN_SPEED = 0.07
RECOVERY_SPEED = 0.05

WAYPOINT_RADIUS = 0.18
RECOVERY_WAYPOINT_RADIUS = 0.18
PATH_EDGE_MARGIN = 0.22
RUNTIME_EDGE_MARGIN = 0.16
AI_EXTRA_SAFETY_MARGIN = 0.20
RECOVERY_INNER_MARGIN = 0.45

OBJECT_LOOK_SECONDS = 5.0
CORNER_LOOK_SECONDS = 2.0
CORNER_YAW_SCAN = True
CORNER_YAW_RATE_DEG = 12.0
YAW_LOOK_GAIN = 1.0
YAW_LOOK_MAX_RATE_DEG = 30.0

PERIMETER_SPACING = 0.45
INTERIOR_ROW_SPACING = 0.55
INTERIOR_POINT_SPACING = 0.55

# Grid planner:
# This helps the drone go around obstacle keepouts instead of getting stuck at them.
PLANNER_GRID_STEP = 0.20
PLANNER_WAYPOINT_MIN_DIST = 0.18

MAX_FLIGHT_TIME_SECONDS = 420.0
MAX_RECOVERY_ATTEMPTS = 50
MAX_RECOVERY_ATTEMPTS_PER_TARGET = 4
TAKEOFF_SETTLE_SECONDS = 2.5
LOOP_SLEEP_SECONDS = 0.10
PRINT_EVERY_N_LOOPS = 4

MISSION_GO_TO_ENTRY = 1
MISSION_ROUTE = 2
MISSION_LOOK = 3
MISSION_RECOVER_SAFE = 4
MISSION_RETURN_HOME = 5
MISSION_LAND = 6


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def distance_2d(a, b):
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)


def wrap_to_pi(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def load_latest_safezone_json():
    if SAFE_ZONE_JSON is not None:
        path = SAFE_ZONE_JSON
    else:
        files = sorted(glob.glob(os.path.join(SAFE_ZONE_DIR, 'safe_zone_*.json')), key=os.path.getmtime)
        if not files:
            raise FileNotFoundError(f'No safe-zone JSON found in {SAFE_ZONE_DIR}. Run the multiranger mapping drone first.')
        path = files[-1]
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('\nLoaded safe-zone file:')
    print(f'  {path}\n')
    return path, data


def polygon_bbox(poly):
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return min(xs), max(xs), min(ys), max(ys)


def point_in_rect_polygon(point, polygon):
    if not polygon:
        return False
    min_x, max_x, min_y, max_y = polygon_bbox(polygon)
    x = float(point[0])
    y = float(point[1])
    return min_x <= x <= max_x and min_y <= y <= max_y


def is_inside_obstacle_keepout(point, obstacles, extra_margin=0.0):
    x = float(point[0])
    y = float(point[1])
    for obstacle in obstacles:
        center = obstacle.get('center', None)
        radius = obstacle.get('radius', None)
        if center is None or radius is None:
            continue
        cx = float(center[0])
        cy = float(center[1])
        r = float(radius) + float(extra_margin)
        if math.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= r:
            return True
    return False


def is_safe_point(point, safe_polygon, obstacles, extra_margin=AI_EXTRA_SAFETY_MARGIN):
    return point_in_rect_polygon(point, safe_polygon) and not is_inside_obstacle_keepout(point, obstacles, extra_margin)


def is_inside_runtime_safe_area(point, safe_polygon, obstacles):
    if not safe_polygon:
        return False
    min_x, max_x, min_y, max_y = polygon_bbox(safe_polygon)
    x = float(point[0])
    y = float(point[1])
    if x < min_x + RUNTIME_EDGE_MARGIN or x > max_x - RUNTIME_EDGE_MARGIN:
        return False
    if y < min_y + RUNTIME_EDGE_MARGIN or y > max_y - RUNTIME_EDGE_MARGIN:
        return False
    if is_inside_obstacle_keepout(point, obstacles, extra_margin=AI_EXTRA_SAFETY_MARGIN):
        return False
    return True


def clip_point_to_safe_rect(point, safe_polygon, edge_margin=0.0):
    min_x, max_x, min_y, max_y = polygon_bbox(safe_polygon)
    return np.array([
        clamp(float(point[0]), min_x + edge_margin, max_x - edge_margin),
        clamp(float(point[1]), min_y + edge_margin, max_y - edge_margin),
        TAKEOFF_HEIGHT
    ], dtype=float)


def nearest_safe_entry_point(start_map_xy, safe_polygon, obstacles):
    clipped = clip_point_to_safe_rect(start_map_xy, safe_polygon, PATH_EDGE_MARGIN)
    if is_safe_point(clipped, safe_polygon, obstacles):
        return clipped
    min_x, max_x, min_y, max_y = polygon_bbox(safe_polygon)
    candidates = []
    for y in np.arange(min_y + PATH_EDGE_MARGIN, max_y - PATH_EDGE_MARGIN + 1e-6, 0.10):
        for x in np.arange(min_x + PATH_EDGE_MARGIN, max_x - PATH_EDGE_MARGIN + 1e-6, 0.10):
            p = np.array([x, y, TAKEOFF_HEIGHT], dtype=float)
            if is_safe_point(p, safe_polygon, obstacles):
                candidates.append(p)
    if not candidates:
        raise RuntimeError('Could not find a valid safe-zone entry point.')
    candidates.sort(key=lambda p: distance_2d(p, start_map_xy))
    return candidates[0]


def nearest_runtime_safe_point(point, safe_polygon, obstacles):
    min_x, max_x, min_y, max_y = polygon_bbox(safe_polygon)
    p = np.array([
        clamp(float(point[0]), min_x + RECOVERY_INNER_MARGIN, max_x - RECOVERY_INNER_MARGIN),
        clamp(float(point[1]), min_y + RECOVERY_INNER_MARGIN, max_y - RECOVERY_INNER_MARGIN),
        TAKEOFF_HEIGHT
    ], dtype=float)
    if is_inside_runtime_safe_area(p, safe_polygon, obstacles) and distance_2d(p, point) > 0.20:
        return p
    candidates = []
    for y in np.arange(min_y + RECOVERY_INNER_MARGIN, max_y - RECOVERY_INNER_MARGIN + 1e-6, 0.15):
        for x in np.arange(min_x + RECOVERY_INNER_MARGIN, max_x - RECOVERY_INNER_MARGIN + 1e-6, 0.15):
            c = np.array([x, y, TAKEOFF_HEIGHT], dtype=float)
            if is_inside_runtime_safe_area(c, safe_polygon, obstacles):
                candidates.append(c)
    if not candidates:
        return np.array([0.5 * (min_x + max_x), 0.5 * (min_y + max_y), TAKEOFF_HEIGHT], dtype=float)
    candidates.sort(key=lambda c: distance_2d(c, point))
    return candidates[0]


def segment_is_safe(p1, p2, safe_polygon, obstacles):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    steps = max(4, int(distance_2d(p1, p2) / 0.10))
    for i in range(steps + 1):
        t = i / float(steps)
        p = p1 * (1.0 - t) + p2 * t
        p[2] = TAKEOFF_HEIGHT
        if not is_inside_runtime_safe_area(p, safe_polygon, obstacles):
            return False
    return True


def safe_hub_point(safe_polygon, obstacles, map_start_xy):
    min_x, max_x, min_y, max_y = polygon_bbox(safe_polygon)
    center = np.array([0.5 * (min_x + max_x), 0.5 * (min_y + max_y), TAKEOFF_HEIGHT], dtype=float)
    candidates = [
        np.array([float(map_start_xy[0]), float(map_start_xy[1]), TAKEOFF_HEIGHT], dtype=float),
        center,
        np.array([center[0], max_y - RECOVERY_INNER_MARGIN, TAKEOFF_HEIGHT], dtype=float),
        np.array([min_x + RECOVERY_INNER_MARGIN, max_y - RECOVERY_INNER_MARGIN, TAKEOFF_HEIGHT], dtype=float),
        np.array([max_x - RECOVERY_INNER_MARGIN, max_y - RECOVERY_INNER_MARGIN, TAKEOFF_HEIGHT], dtype=float),
    ]
    for cand in candidates:
        cand = clip_point_to_safe_rect(cand, safe_polygon, RECOVERY_INNER_MARGIN)
        if is_inside_runtime_safe_area(cand, safe_polygon, obstacles):
            return cand
    return clip_point_to_safe_rect(center, safe_polygon, RECOVERY_INNER_MARGIN)


def points_along_segment(a, b, spacing):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    n = max(1, int(math.ceil(distance_2d(a, b) / spacing)))
    pts = []
    for i in range(n + 1):
        t = i / float(n)
        p = a * (1.0 - t) + b * t
        p[2] = TAKEOFF_HEIGHT
        pts.append(p)
    return pts


def make_target(pos, label='scan', look_at=None, hold_seconds=0.0, yaw_scan=False, speed=None):
    return {
        'pos': np.array(pos, dtype=float),
        'label': label,
        'look_at': None if look_at is None else np.array(look_at, dtype=float),
        'hold_seconds': float(hold_seconds),
        'yaw_scan': bool(yaw_scan),
        'speed': speed,
    }


def append_unique(route, target):
    if route and distance_2d(route[-1]['pos'], target['pos']) < 0.10:
        return
    route.append(target)


def build_perimeter_route(safe_polygon, obstacles):
    min_x, max_x, min_y, max_y = polygon_bbox(safe_polygon)
    left = min_x + PATH_EDGE_MARGIN
    right = max_x - PATH_EDGE_MARGIN
    bottom = min_y + PATH_EDGE_MARGIN
    top = max_y - PATH_EDGE_MARGIN
    corners = [
        np.array([left, top, TAKEOFF_HEIGHT], dtype=float),
        np.array([right, top, TAKEOFF_HEIGHT], dtype=float),
        np.array([right, bottom, TAKEOFF_HEIGHT], dtype=float),
        np.array([left, bottom, TAKEOFF_HEIGHT], dtype=float),
        np.array([left, top, TAKEOFF_HEIGHT], dtype=float),
    ]
    route = []
    for si in range(len(corners) - 1):
        seg = points_along_segment(corners[si], corners[si + 1], PERIMETER_SPACING)
        for i, p in enumerate(seg):
            if not is_safe_point(p, safe_polygon, obstacles):
                continue
            is_corner = (i == 0 or i == len(seg) - 1)
            append_unique(route, make_target(
                p,
                label='corner/perimeter' if is_corner else 'perimeter',
                hold_seconds=CORNER_LOOK_SECONDS if is_corner else 0.0,
                yaw_scan=CORNER_YAW_SCAN if is_corner else False,
                speed=SCAN_SPEED,
            ))
    return route


def build_object_inspection_route(safe_polygon, obstacles):
    route = []
    for obstacle in obstacles:
        center = obstacle.get('center', None)
        radius = obstacle.get('radius', None)
        obstacle_id = obstacle.get('id', '?')
        if center is None or radius is None:
            continue
        cx = float(center[0])
        cy = float(center[1])
        r = float(radius) + AI_EXTRA_SAFETY_MARGIN + 0.25
        for angle in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
            p = np.array([cx + r * math.cos(angle), cy + r * math.sin(angle), TAKEOFF_HEIGHT], dtype=float)
            p = clip_point_to_safe_rect(p, safe_polygon, PATH_EDGE_MARGIN)
            if not is_safe_point(p, safe_polygon, obstacles):
                continue
            append_unique(route, make_target(
                p,
                label=f'obstacle {obstacle_id} view',
                look_at=np.array([cx, cy, TAKEOFF_HEIGHT], dtype=float),
                hold_seconds=OBJECT_LOOK_SECONDS,
                yaw_scan=False,
                speed=INSPECT_SPEED,
            ))
    return route


def build_interior_route(safe_polygon, obstacles):
    min_x, max_x, min_y, max_y = polygon_bbox(safe_polygon)
    min_x += PATH_EDGE_MARGIN
    max_x -= PATH_EDGE_MARGIN
    min_y += PATH_EDGE_MARGIN
    max_y -= PATH_EDGE_MARGIN
    route = []
    y = max_y
    reverse = False
    while y >= min_y - 1e-6:
        xs = list(np.arange(min_x, max_x + 1e-6, INTERIOR_POINT_SPACING))
        if len(xs) == 0 or abs(xs[-1] - max_x) > 0.15:
            xs.append(float(max_x))
        if reverse:
            xs.reverse()
        for x in xs:
            p = np.array([x, y, TAKEOFF_HEIGHT], dtype=float)
            if is_safe_point(p, safe_polygon, obstacles):
                append_unique(route, make_target(p, label='interior sweep', speed=SCAN_SPEED))
        y -= INTERIOR_ROW_SPACING
        reverse = not reverse
    return route


def build_full_route(safe_polygon, obstacles):
    perimeter = build_perimeter_route(safe_polygon, obstacles)
    objects = build_object_inspection_route(safe_polygon, obstacles)
    interior = build_interior_route(safe_polygon, obstacles)
    route = []
    for target in perimeter + objects + interior:
        append_unique(route, target)
    return route, perimeter, objects, interior



def grid_key_from_point(point, min_x, min_y, step):
    ix = int(round((float(point[0]) - min_x) / step))
    iy = int(round((float(point[1]) - min_y) / step))
    return ix, iy


def nearest_planner_node(point, nodes):
    best = None
    best_d = 999999.0
    for node_key, node_point in nodes.items():
        d = distance_2d(point, node_point)
        if d < best_d:
            best = node_key
            best_d = d
    return best


def build_planner_nodes(safe_polygon, obstacles):
    min_x, max_x, min_y, max_y = polygon_bbox(safe_polygon)
    nodes = {}

    xs = np.arange(min_x + RUNTIME_EDGE_MARGIN, max_x - RUNTIME_EDGE_MARGIN + 1e-6, PLANNER_GRID_STEP)
    ys = np.arange(min_y + RUNTIME_EDGE_MARGIN, max_y - RUNTIME_EDGE_MARGIN + 1e-6, PLANNER_GRID_STEP)

    for y in ys:
        for x in xs:
            p = np.array([float(x), float(y), TAKEOFF_HEIGHT], dtype=float)
            if is_inside_runtime_safe_area(p, safe_polygon, obstacles):
                key = grid_key_from_point(p, min_x, min_y, PLANNER_GRID_STEP)
                nodes[key] = p

    return nodes


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar_grid_path(start_point, goal_point, safe_polygon, obstacles):
    nodes = build_planner_nodes(safe_polygon, obstacles)
    if not nodes:
        return []

    start_key = nearest_planner_node(start_point, nodes)
    goal_key = nearest_planner_node(goal_point, nodes)

    if start_key is None or goal_key is None:
        return []

    if start_key == goal_key:
        return [nodes[goal_key]]

    open_set = {start_key}
    came_from = {}
    g_score = {start_key: 0.0}
    f_score = {start_key: distance_2d(nodes[start_key], nodes[goal_key])}

    neighbours = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    while open_set:
        current = min(open_set, key=lambda k: f_score.get(k, 999999.0))

        if current == goal_key:
            keys = reconstruct_path(came_from, current)
            return [nodes[k] for k in keys]

        open_set.remove(current)

        for dx, dy in neighbours:
            nb = (current[0] + dx, current[1] + dy)
            if nb not in nodes:
                continue

            if not segment_is_safe(nodes[current], nodes[nb], safe_polygon, obstacles):
                continue

            tentative_g = g_score[current] + distance_2d(nodes[current], nodes[nb])

            if tentative_g < g_score.get(nb, 999999.0):
                came_from[nb] = current
                g_score[nb] = tentative_g
                f_score[nb] = tentative_g + distance_2d(nodes[nb], nodes[goal_key])
                open_set.add(nb)

    return []


def simplify_planned_points(points):
    if not points:
        return []

    simplified = [points[0]]
    for p in points[1:]:
        if distance_2d(p, simplified[-1]) >= PLANNER_WAYPOINT_MIN_DIST:
            simplified.append(p)

    if distance_2d(points[-1], simplified[-1]) > 0.05:
        simplified.append(points[-1])

    return simplified


def expand_route_with_grid_planner(raw_route, safe_polygon, obstacles, entry_point_map):
    expanded = []
    current = np.array(entry_point_map, dtype=float)

    for target in raw_route:
        goal = np.array(target['pos'], dtype=float)

        if segment_is_safe(current, goal, safe_polygon, obstacles):
            append_unique(expanded, target)
            current = goal
            continue

        planned = astar_grid_path(current, goal, safe_polygon, obstacles)
        planned = simplify_planned_points(planned)

        if not planned:
            print(f'Planner could not reach target "{target["label"]}". Skipping it.')
            continue

        for p in planned[1:-1]:
            append_unique(
                expanded,
                make_target(
                    p,
                    label='planned path',
                    hold_seconds=0.0,
                    yaw_scan=False,
                    speed=SCAN_SPEED
                )
            )

        append_unique(expanded, target)
        current = goal

    return expanded


def make_body_velocity_to_target(local_position, local_target, yaw_rad, max_speed):
    dx = float(local_target[0] - local_position[0])
    dy = float(local_target[1] - local_position[1])
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 0.01:
        return 0.0, 0.0, 0.0, dist
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    body_x = cos_y * dx + sin_y * dy
    body_y = -sin_y * dx + cos_y * dy
    gain = 0.7
    velocity_x = clamp(gain * body_x, -max_speed, max_speed)
    velocity_y = clamp(gain * body_y, -max_speed, max_speed)
    wanted_yaw = math.atan2(dy, dx)
    yaw_error = wrap_to_pi(wanted_yaw - yaw_rad)
    yaw_rate = clamp(1.2 * yaw_error, -0.45, 0.45)
    return velocity_x, velocity_y, yaw_rate, dist


def yaw_rate_to_face_target(map_position, look_at_map, yaw_rad):
    dx = float(look_at_map[0] - map_position[0])
    dy = float(look_at_map[1] - map_position[1])
    wanted_yaw = math.atan2(dy, dx)
    yaw_error = wrap_to_pi(wanted_yaw - yaw_rad)
    return clamp(YAW_LOOK_GAIN * yaw_error, -radians(YAW_LOOK_MAX_RATE_DEG), radians(YAW_LOOK_MAX_RATE_DEG))


def map_to_local_target(map_target, map_start_xy, local_start_xy):
    target = np.array(map_target, dtype=float)
    map_start = np.array([map_start_xy[0], map_start_xy[1], TAKEOFF_HEIGHT], dtype=float)
    local_start = np.array([local_start_xy[0], local_start_xy[1], TAKEOFF_HEIGHT], dtype=float)
    local = local_start + (target - map_start)
    local[2] = TAKEOFF_HEIGHT
    return local


def local_to_map_position(local_position, map_start_xy, local_start_xy):
    local = np.array(local_position, dtype=float)
    map_start = np.array([map_start_xy[0], map_start_xy[1], TAKEOFF_HEIGHT], dtype=float)
    local_start = np.array([local_start_xy[0], local_start_xy[1], TAKEOFF_HEIGHT], dtype=float)
    map_pos = map_start + (local - local_start)
    map_pos[2] = TAKEOFF_HEIGHT
    return map_pos


def arm_cf(cf, do_arm):
    return


def smooth_land(motion_commander):
    print('Landing smoothly...')
    try:
        motion_commander.stop()
        motion_commander.land(velocity=0.15)
        print('Landed.')
    except Exception as e:
        print(f'Smooth land failed: {e}')


def get_log_config():
    lg = LogConfig(name='PosStab', period_in_ms=200)
    lg.add_variable('stateEstimate.x', 'float')
    lg.add_variable('stateEstimate.y', 'float')
    lg.add_variable('stateEstimate.z', 'float')
    lg.add_variable('stabilizer.yaw', 'float')
    return lg


def main():
    logging.basicConfig(level=logging.ERROR)
    cflib.crtp.init_drivers()
    print('\nAI-deck perimeter/object inspection navigator starting...')
    print(f'  URI = {URI}')
    print(f'  Working directory = {os.getcwd()}')
    print(f'  Looking for safe-zone files in: {SAFE_ZONE_DIR}\n')

    safezone_path, safezone = load_latest_safezone_json()
    safe_polygon = safezone.get('safe_zone_polygon', [])
    obstacles = safezone.get('obstacle_keepouts', [])
    mapper_start_xy = safezone.get('start_position_xy', None)
    if not safe_polygon:
        raise RuntimeError('The safe-zone JSON does not contain safe_zone_polygon.')
    if mapper_start_xy is None:
        print('WARNING: safe-zone JSON has no start_position_xy. Using [0, 0].')
        mapper_start_xy = [0.0, 0.0]

    # The safe-zone map is made in the multiranger drone's coordinate frame.
    # If the AI drone starts a little in front of the multiranger start position,
    # its map-frame start is mapper_start + offset.
    map_start_xy = [
        float(mapper_start_xy[0]) + float(AI_START_OFFSET_X),
        float(mapper_start_xy[1]) + float(AI_START_OFFSET_Y)
    ]

    entry_point_map = nearest_safe_entry_point(map_start_xy, safe_polygon, obstacles)
    route, perimeter_route, object_route, interior_route = build_full_route(safe_polygon, obstacles)
    hub_map = safe_hub_point(safe_polygon, obstacles, map_start_xy)
    if not route:
        raise RuntimeError('No route targets generated.')

    planned_route = expand_route_with_grid_planner(route, safe_polygon, obstacles, entry_point_map)
    full_route = [make_target(entry_point_map, label='safe-zone entry', speed=ENTRY_SPEED)] + planned_route

    print('Mission plan:')
    print(f'  safe-zone file: {safezone_path}')
    print(f'  obstacle keepouts from map: {len(obstacles)}')
    print(f'  mapper start from map: x={mapper_start_xy[0]:.2f}, y={mapper_start_xy[1]:.2f}')
    print(f'  AI start offset: x={AI_START_OFFSET_X:.2f}, y={AI_START_OFFSET_Y:.2f}')
    print(f'  AI start used by script: x={map_start_xy[0]:.2f}, y={map_start_xy[1]:.2f}')
    print(f'  entry point: x={entry_point_map[0]:.2f}, y={entry_point_map[1]:.2f}')
    print(f'  safe hub: x={hub_map[0]:.2f}, y={hub_map[1]:.2f}')
    print(f'  perimeter targets: {len(perimeter_route)}')
    print(f'  object inspection targets: {len(object_route)}')
    print(f'  interior sweep targets: {len(interior_route)}')
    print(f'  total route targets after planner: {len(full_route)}')
    print(f'  planner grid step: {PLANNER_GRID_STEP:.2f} m')
    print(f'  object look seconds: {OBJECT_LOOK_SECONDS:.1f}')
    print(f'  corner look seconds: {CORNER_LOOK_SECONDS:.1f}')
    print(f'  scan speed: {SCAN_SPEED:.3f} m/s')
    print('')
    print('IMPORTANT START SETUP:')
    print('  Face the AI-deck drone in the SAME direction as the multiranger mapping drone.')
    print('  If the AI drone is in front of the mapper start, set AI_START_OFFSET_X correctly.')
    print('')
    print('Press Ctrl+C to stop and land.\n')

    cf = Crazyflie(rw_cache='./cache')
    motion_commander = None
    mission_mode = MISSION_GO_TO_ENTRY
    target_index = 0
    current_subtarget_map = None
    recovery_target_map = None
    recovery_attempts = 0
    recovery_attempts_for_target = 0
    look_start_time = None
    loops = 0
    mission_start_time = None

    try:
        with SyncCrazyflie(URI, cf=cf) as scf:
            arm_cf(scf.cf, True)
            time.sleep(1.0)
            with MotionCommander(scf, default_height=TAKEOFF_HEIGHT) as mc:
                motion_commander = mc
                print('Taking off...')
                print(f'Waiting {TAKEOFF_SETTLE_SECONDS:.1f}s for stabilizer/Flow deck to settle...')
                time.sleep(TAKEOFF_SETTLE_SECONDS)

                with SyncLogger(scf, get_log_config()) as logger:
                    first_entry = logger.next()
                    first_data = first_entry[1]
                    local_start_xy = [
                        float(first_data.get('stateEstimate.x', 0.0)),
                        float(first_data.get('stateEstimate.y', 0.0))
                    ]
                    print('Local start locked:')
                    print(f'  local_start x={local_start_xy[0]:.2f}, y={local_start_xy[1]:.2f}')
                    print(f'  map_start   x={map_start_xy[0]:.2f}, y={map_start_xy[1]:.2f}\n')
                    mission_start_time = time.time()

                    while True:
                        loops += 1
                        log_entry = logger.next()
                        data = log_entry[1]
                        local_position = np.array([
                            float(data.get('stateEstimate.x', 0.0)),
                            float(data.get('stateEstimate.y', 0.0)),
                            float(data.get('stateEstimate.z', TAKEOFF_HEIGHT))
                        ], dtype=float)
                        yaw_deg = float(data.get('stabilizer.yaw', 0.0))
                        yaw_rad = radians(yaw_deg)
                        map_position = local_to_map_position(local_position, map_start_xy, local_start_xy)
                        elapsed = time.time() - mission_start_time
                        vx = vy = yaw_rate = 0.0

                        if elapsed > MAX_FLIGHT_TIME_SECONDS:
                            print('Maximum mission time reached. Returning home.')
                            mission_mode = MISSION_RETURN_HOME

                        if mission_mode == MISSION_GO_TO_ENTRY:
                            target = full_route[0]
                            target_local = map_to_local_target(target['pos'], map_start_xy, local_start_xy)
                            vx, vy, yaw_rate, dist = make_body_velocity_to_target(local_position, target_local, yaw_rad, ENTRY_SPEED)
                            if dist < WAYPOINT_RADIUS:
                                print('Reached safe-zone entry. Starting perimeter/object route.')
                                mission_mode = MISSION_ROUTE
                                target_index = 1
                                current_subtarget_map = None

                        elif mission_mode == MISSION_ROUTE:
                            if target_index >= len(full_route):
                                print('Full camera-coverage route complete. Returning home.')
                                mission_mode = MISSION_RETURN_HOME
                            else:
                                target = full_route[target_index]
                                target_map = target['pos']
                                if not is_safe_point(target_map, safe_polygon, obstacles):
                                    print(f'Skipping unsafe target {target_index + 1}/{len(full_route)}: {target["label"]}')
                                    target_index += 1
                                    current_subtarget_map = None
                                else:
                                    speed = target.get('speed', None) or SCAN_SPEED
                                    target_local = map_to_local_target(target_map, map_start_xy, local_start_xy)
                                    vx, vy, yaw_rate, dist = make_body_velocity_to_target(local_position, target_local, yaw_rad, speed)
                                    if dist < WAYPOINT_RADIUS:
                                        print(f'Reached target {target_index + 1}/{len(full_route)}: {target["label"]}')
                                        recovery_attempts_for_target = 0
                                        current_subtarget_map = None
                                        if target.get('hold_seconds', 0.0) > 0.0:
                                            mission_mode = MISSION_LOOK
                                            look_start_time = time.time()
                                            print(f'Camera look: {target["label"]} for {target["hold_seconds"]:.1f}s')
                                        else:
                                            target_index += 1
                                        vx = vy = yaw_rate = 0.0

                        elif mission_mode == MISSION_LOOK:
                            target = full_route[target_index]
                            vx = vy = 0.0
                            if target.get('look_at', None) is not None:
                                yaw_rate = yaw_rate_to_face_target(map_position, target['look_at'], yaw_rad)
                            elif target.get('yaw_scan', False):
                                yaw_rate = radians(CORNER_YAW_RATE_DEG)
                            else:
                                yaw_rate = 0.0
                            if look_start_time is not None and (time.time() - look_start_time) >= target['hold_seconds']:
                                print('Camera look complete.')
                                target_index += 1
                                mission_mode = MISSION_ROUTE
                                look_start_time = None
                                vx = vy = yaw_rate = 0.0

                        elif mission_mode == MISSION_RECOVER_SAFE:
                            if recovery_target_map is None:
                                recovery_target_map = nearest_runtime_safe_point(map_position, safe_polygon, obstacles)
                                print(f'Recovery target: x={recovery_target_map[0]:.2f}, y={recovery_target_map[1]:.2f}')
                            recovery_local = map_to_local_target(recovery_target_map, map_start_xy, local_start_xy)
                            vx, vy, yaw_rate, dist_recovery = make_body_velocity_to_target(local_position, recovery_local, yaw_rad, RECOVERY_SPEED)
                            if dist_recovery < RECOVERY_WAYPOINT_RADIUS:
                                print('Recovered inward. Continuing route.')
                                mission_mode = MISSION_ROUTE
                                recovery_target_map = None
                                current_subtarget_map = None
                                vx = vy = yaw_rate = 0.0

                        elif mission_mode == MISSION_RETURN_HOME:
                            home_map = np.array([map_start_xy[0], map_start_xy[1], TAKEOFF_HEIGHT], dtype=float)
                            home_local = map_to_local_target(home_map, map_start_xy, local_start_xy)
                            vx, vy, yaw_rate, dist_home = make_body_velocity_to_target(local_position, home_local, yaw_rad, RETURN_SPEED)
                            if dist_home < WAYPOINT_RADIUS:
                                print('Returned to start. Landing.')
                                mission_mode = MISSION_LAND
                                vx = vy = yaw_rate = 0.0

                        elif mission_mode == MISSION_LAND:
                            mc.stop()
                            smooth_land(mc)
                            break
                        else:
                            print('Unknown mission mode. Returning home.')
                            mission_mode = MISSION_RETURN_HOME
                            vx = vy = yaw_rate = 0.0

                        if mission_mode in (MISSION_ROUTE, MISSION_LOOK):
                            if not is_inside_runtime_safe_area(map_position, safe_polygon, obstacles):
                                recovery_attempts += 1
                                recovery_attempts_for_target += 1
                                print('WARNING: estimate too close to edge/obstacle. Recovering inward, then continuing route.')
                                if recovery_attempts_for_target > MAX_RECOVERY_ATTEMPTS_PER_TARGET:
                                    print(f'Target {target_index + 1} caused too many recoveries. Skipping that target and continuing.')
                                    target_index += 1
                                    current_subtarget_map = None
                                    recovery_attempts_for_target = 0
                                    mission_mode = MISSION_ROUTE
                                elif recovery_attempts > MAX_RECOVERY_ATTEMPTS:
                                    print('Too many total recovery attempts. Returning home for safety.')
                                    mission_mode = MISSION_RETURN_HOME
                                else:
                                    recovery_target_map = nearest_runtime_safe_point(map_position, safe_polygon, obstacles)
                                    mission_mode = MISSION_RECOVER_SAFE
                                vx = vy = yaw_rate = 0.0

                        mc.start_linear_motion(vx, vy, 0.0, rate_yaw=degrees(yaw_rate))
                        if loops % PRINT_EVERY_N_LOOPS == 0:
                            label = 'none'
                            if 0 <= target_index < len(full_route):
                                label = full_route[target_index]['label']
                            print(f'mode={mission_mode}, target={target_index + 1}/{len(full_route)}, label={label}, map=({map_position[0]:.2f},{map_position[1]:.2f}), vx={vx:.2f}, vy={vy:.2f}, yaw_rate={degrees(yaw_rate):.1f}')
                        time.sleep(LOOP_SLEEP_SECONDS)

    except KeyboardInterrupt:
        print('\nCtrl+C pressed. Trying to land safely...')
        try:
            if motion_commander is not None:
                motion_commander.stop()
                motion_commander.land(velocity=0.15)
        except Exception:
            pass
    except Exception as e:
        print('\nERROR: mission could not continue.')
        print(f'  {type(e).__name__}: {e}\n')
    finally:
        try:
            arm_cf(cf, False)
        except Exception:
            pass
        print('AI-deck perimeter/object inspection mission finished.')


if __name__ == '__main__':
    main()
