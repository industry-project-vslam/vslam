#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual AI-deck front/back safe-zone inspection drone V10 STABLE soft-safety parallel start gate

This script is meant to be launched twice by:
  ai_deck_dual_front_back_single_radio_v10_parallel_safe.py

Drone A1:
  - lower height
  - left-side / lower-work route
  - URI default radio://0/84/2M/E7E7E7E702

Drone A2:
  - higher height
  - right-side / balanced helper route
  - URI default radio://0/86/2M/E7E7E7E704

Both drones:
  - load the newest safe-zone JSON from safe_zone_output
  - split perimeter/interior targets by front/back half
  - split obstacle camera viewpoints by front/back half
  - use A* grid planner for safe intermediate waypoints
  - return to their own start and land

Important:
  - This is camera-coverage / inspection planning.
  - It does not run onboard AI detection yet.
  - It assumes both AI drones face the same direction as the mapping drone.
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


# -------------------------------------------------------------------------
# Which AI drone is this process?
# -------------------------------------------------------------------------
AI_DRONE_ID = int(os.environ.get('AI_DRONE_ID', '1'))

if AI_DRONE_ID == 1:
    DEFAULT_URI = 'radio://0/84/2M/E7E7E7E702'
else:
    DEFAULT_URI = 'radio://0/86/2M/E7E7E7E704'

URI = uri_helper.uri_from_env(default=os.environ.get('AI_URI', DEFAULT_URI))

SAFE_ZONE_DIR = 'safe_zone_output'
SAFE_ZONE_JSON = os.environ.get('SAFE_ZONE_JSON', None)

MISSION_CONTROL_DIR = os.environ.get('MISSION_CONTROL_DIR', 'mission_control')
LAND_NOW_FILE = os.path.join(MISSION_CONTROL_DIR, 'land_now.flag')
EMERGENCY_FILE = os.path.join(MISSION_CONTROL_DIR, 'emergency_stop.flag')

# Optional parallel-start gate files.
# The AI launcher uses these to make both AI drones connect first, then take off together.
AI_READY_FILE = os.environ.get('AI_READY_FILE', None)
AI_START_FILE = os.environ.get('AI_START_FILE', None)
AI_START_TIMEOUT_SECONDS = float(os.environ.get('AI_START_TIMEOUT_SECONDS', '45.0'))

# -------------------------------------------------------------------------
# Start offsets in the multiranger map frame
# -------------------------------------------------------------------------
# Positive X = in front of mapper start, assuming same yaw.
# Positive Y = left of mapper start.
if AI_DRONE_ID == 1:
    AI_START_OFFSET_X = float(os.environ.get('AI1_START_OFFSET_X', '0.20'))
    AI_START_OFFSET_Y = float(os.environ.get('AI1_START_OFFSET_Y', '-0.20'))
    TAKEOFF_HEIGHT = float(os.environ.get('AI1_TAKEOFF_HEIGHT', '0.30'))
else:
    AI_START_OFFSET_X = float(os.environ.get('AI2_START_OFFSET_X', '0.20'))
    AI_START_OFFSET_Y = float(os.environ.get('AI2_START_OFFSET_Y', '0.20'))
    TAKEOFF_HEIGHT = float(os.environ.get('AI2_TAKEOFF_HEIGHT', '0.45'))


# -------------------------------------------------------------------------
# Flight settings
# -------------------------------------------------------------------------
ENTRY_SPEED = float(os.environ.get('AI_ENTRY_SPEED', '0.06'))
SCAN_SPEED = float(os.environ.get('AI_SCAN_SPEED', '0.07'))
INSPECT_SPEED = float(os.environ.get('AI_INSPECT_SPEED', '0.055'))
RETURN_SPEED = float(os.environ.get('AI_RETURN_SPEED', '0.07'))
RECOVERY_SPEED = float(os.environ.get('AI_RECOVERY_SPEED', '0.06'))

WAYPOINT_RADIUS = 0.18
RECOVERY_WAYPOINT_RADIUS = 0.18

PATH_EDGE_MARGIN = float(os.environ.get('AI_PATH_EDGE_MARGIN', '0.20'))
RUNTIME_EDGE_MARGIN = float(os.environ.get('AI_RUNTIME_EDGE_MARGIN', '0.16'))
AI_EXTRA_SAFETY_MARGIN = 0.20
RECOVERY_INNER_MARGIN = 0.45

OBJECT_LOOK_SECONDS = 2.5
CORNER_LOOK_SECONDS = 0.8

# With two AI drones, avoid lots of spinning because it causes Flow-deck drift
# and two drones turning at once is harder to observe. Obstacle look points still
# face the object.
CORNER_YAW_SCAN = False
CORNER_YAW_RATE_DEG = 10.0

YAW_LOOK_GAIN = 1.0
YAW_LOOK_MAX_RATE_DEG = 30.0

PERIMETER_SPACING = float(os.environ.get('AI_PERIMETER_SPACING', '0.75'))
INTERIOR_ROW_SPACING = 0.65
INTERIOR_POINT_SPACING = 0.65

# V10_CORNERS: sparse corner-first route. The drone tries to visit
# the safe inner corners of its own half first, then obstacle view points,
# then a few broad interior scan points.
AI_CORNER_MARGIN = float(os.environ.get('AI_CORNER_MARGIN', '0.32'))
AI_SPARSE_INTERIOR = os.environ.get('AI_SPARSE_INTERIOR', '1').lower() in ('1', 'true', 'yes')

PLANNER_GRID_STEP = 0.22
PLANNER_WAYPOINT_MIN_DIST = 0.20

MAX_FLIGHT_TIME_SECONDS = 360.0
MAX_RECOVERY_ATTEMPTS = 120
MAX_RECOVERY_ATTEMPTS_PER_TARGET = 2
TAKEOFF_SETTLE_SECONDS = 2.5

# V10_LIFTCHECK: prove that the drone actually left the ground.
# The old code printed 'Taking off' as soon as MotionCommander started, but it did
# not check stateEstimate.z. If a drone stayed on the floor, the route code still ran.
TAKEOFF_VERIFY_SECONDS = float(os.environ.get('AI_TAKEOFF_VERIFY_SECONDS', '2.5'))
TAKEOFF_RETRY_SECONDS = float(os.environ.get('AI_TAKEOFF_RETRY_SECONDS', '3.0'))
TAKEOFF_MIN_Z_RATIO = float(os.environ.get('AI_TAKEOFF_MIN_Z_RATIO', '0.55'))

LOOP_SLEEP_SECONDS = 0.10
PRINT_EVERY_N_LOOPS = 4

# V10_FIXED: do not let one AI drone hover forever on the same waypoint.
# In the last test A2 took off, but stayed at target 2/19 for a long time.
# These guards skip a target when the estimate does not make progress.
AI_TARGET_TIMEOUT_SECONDS = float(os.environ.get('AI_TARGET_TIMEOUT_SECONDS', '16.0'))
AI_TARGET_STUCK_SECONDS = float(os.environ.get('AI_TARGET_STUCK_SECONDS', '6.0'))
AI_TARGET_PROGRESS_EPS_M = float(os.environ.get('AI_TARGET_PROGRESS_EPS_M', '0.07'))

# V10_CORNERS soft runtime safety.
# The route generator still stays inside each half, but runtime drift near the split/edge
# should not make the drone panic and keep recovering. Only real danger triggers recovery.
AI_RUNTIME_SPLIT_BUFFER = float(os.environ.get('AI_RUNTIME_SPLIT_BUFFER', '0.20'))
AI_HARD_EDGE_MARGIN = float(os.environ.get('AI_HARD_EDGE_MARGIN', '0.04'))
AI_RUNTIME_OBSTACLE_MARGIN = float(os.environ.get('AI_RUNTIME_OBSTACLE_MARGIN', '0.05'))
AI_MIN_ALTITUDE_RATIO = float(os.environ.get('AI_MIN_ALTITUDE_RATIO', '0.72'))
AI_LOW_ALTITUDE_CLIMB_SPEED = float(os.environ.get('AI_LOW_ALTITUDE_CLIMB_SPEED', '0.06'))

# Front/back split:
# The multiranger start position is the split line.
# A1 is the FRONT drone and may only scan x >= mapper_start_x.
# A2 is the BACK drone and may only scan x <= mapper_start_x.
#
# No overlap means the drones do not cross the multiranger start line.
# A small buffer keeps them slightly away from the split line.
FRONT_BACK_SPLIT_BUFFER = float(os.environ.get('FRONT_BACK_SPLIT_BUFFER', '0.20'))

# This value is set after the safe-zone JSON is loaded.
ACTIVE_SPLIT_X = None

MISSION_GO_TO_ENTRY = 1
MISSION_ROUTE = 2
MISSION_LOOK = 3
MISSION_RECOVER_SAFE = 4
MISSION_RETURN_HOME = 5
MISSION_LAND = 6


# -------------------------------------------------------------------------
# Basic geometry helpers
# -------------------------------------------------------------------------
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


def polygon_bbox(poly):
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return min(xs), max(xs), min(ys), max(ys)


def load_latest_safezone_json():
    if SAFE_ZONE_JSON:
        path = SAFE_ZONE_JSON
    else:
        files = sorted(glob.glob(os.path.join(SAFE_ZONE_DIR, 'safe_zone_*.json')), key=os.path.getmtime)
        if not files:
            raise FileNotFoundError(f'No safe-zone JSON found in {SAFE_ZONE_DIR}. Run the multiranger mapping drone first.')
        path = files[-1]
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('')
    print(f'[A{AI_DRONE_ID}] Loaded safe-zone file:')
    print(f'  {path}')
    print('')
    return path, data


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


def is_in_this_drone_half(point):
    """
    True if the map point is allowed for this AI drone.

    A1/front drone: x must stay in front of the mapper start line.
    A2/back drone:  x must stay behind the mapper start line.
    """
    if ACTIVE_SPLIT_X is None:
        return True

    x = float(point[0])
    if AI_DRONE_ID == 1:
        return x >= float(ACTIVE_SPLIT_X) + FRONT_BACK_SPLIT_BUFFER
    return x <= float(ACTIVE_SPLIT_X) - FRONT_BACK_SPLIT_BUFFER


def is_inside_runtime_safe_area(point, safe_polygon, obstacles):
    if not safe_polygon:
        return False

    if not is_in_this_drone_half(point):
        return False

    min_x, max_x, min_y, max_y = polygon_bbox(safe_polygon)
    x = float(point[0])
    y = float(point[1])

    if x < min_x + RUNTIME_EDGE_MARGIN:
        return False
    if x > max_x - RUNTIME_EDGE_MARGIN:
        return False
    if y < min_y + RUNTIME_EDGE_MARGIN:
        return False
    if y > max_y - RUNTIME_EDGE_MARGIN:
        return False
    if is_inside_obstacle_keepout(point, obstacles, extra_margin=AI_EXTRA_SAFETY_MARGIN):
        return False
    return True



def is_runtime_hard_danger(point, safe_polygon, obstacles):
    """
    Runtime guard used while flying.

    This is intentionally softer than is_inside_runtime_safe_area(). The old
    version used the strict planner area at runtime. Small Flow-deck drift near
    the split line then caused endless recoveries and early landing. This guard
    only reacts when the drone is truly outside the safe rectangle, deeply in the
    other half, or inside an obstacle keepout.
    """
    if not safe_polygon:
        return True

    min_x, max_x, min_y, max_y = polygon_bbox(safe_polygon)
    x = float(point[0])
    y = float(point[1])

    # Hard wall/safe-zone boundary. The safe-zone polygon already has wall margin,
    # so only a small extra margin is needed at runtime.
    if x < min_x + AI_HARD_EDGE_MARGIN:
        return True
    if x > max_x - AI_HARD_EDGE_MARGIN:
        return True
    if y < min_y + AI_HARD_EDGE_MARGIN:
        return True
    if y > max_y - AI_HARD_EDGE_MARGIN:
        return True

    # Soft center split: do not allow deep crossing into the other drone's half,
    # but allow small estimator drift near the split line.
    if ACTIVE_SPLIT_X is not None:
        split = float(ACTIVE_SPLIT_X)
        if AI_DRONE_ID == 1 and x < split + AI_RUNTIME_SPLIT_BUFFER:
            return True
        if AI_DRONE_ID == 2 and x > split - AI_RUNTIME_SPLIT_BUFFER:
            return True

    # Obstacles remain real no-fly zones, but use a smaller runtime margin than
    # the planner. The planner route already avoids the larger keepout.
    if is_inside_obstacle_keepout(point, obstacles, extra_margin=AI_RUNTIME_OBSTACLE_MARGIN):
        return True

    return False

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


def make_target(pos, label='scan', look_at=None, hold_seconds=0.0, yaw_scan=False, speed=None, obstacle_id=None):
    return {
        'pos': np.array(pos, dtype=float),
        'label': label,
        'look_at': None if look_at is None else np.array(look_at, dtype=float),
        'hold_seconds': float(hold_seconds),
        'yaw_scan': bool(yaw_scan),
        'speed': speed,
        'obstacle_id': obstacle_id,
    }


def append_unique(route, target):
    if route and distance_2d(route[-1]['pos'], target['pos']) < 0.10:
        return
    route.append(target)


# -------------------------------------------------------------------------
# Route generation and splitting
# -------------------------------------------------------------------------
def split_x_value(safe_polygon):
    # The split line is NOT the room center. It is the multiranger start x.
    if ACTIVE_SPLIT_X is not None:
        return float(ACTIVE_SPLIT_X)
    min_x, max_x, _, _ = polygon_bbox(safe_polygon)
    return 0.5 * (min_x + max_x)


def half_bounds(safe_polygon):
    """
    Return the rectangular scan bounds for only this drone's half.
    A1/front: x from split line to room max.
    A2/back:  x from room min to split line.
    """
    min_x, max_x, min_y, max_y = polygon_bbox(safe_polygon)
    split_x = split_x_value(safe_polygon)

    if AI_DRONE_ID == 1:
        left = max(min_x + PATH_EDGE_MARGIN, split_x + FRONT_BACK_SPLIT_BUFFER)
        right = max_x - PATH_EDGE_MARGIN
    else:
        left = min_x + PATH_EDGE_MARGIN
        right = min(max_x - PATH_EDGE_MARGIN, split_x - FRONT_BACK_SPLIT_BUFFER)

    bottom = min_y + PATH_EDGE_MARGIN
    top = max_y - PATH_EDGE_MARGIN

    if left >= right:
        raise RuntimeError(
            f'[A{AI_DRONE_ID}] This half is too narrow for the current split/margins. '
            f'left={left:.2f}, right={right:.2f}, split={split_x:.2f}'
        )

    return left, right, bottom, top


def belongs_to_this_drone_by_zone(point, safe_polygon):
    return is_in_this_drone_half(point)


def nearest_runtime_safe_near(point, safe_polygon, obstacles, max_radius=0.45, step=0.12):
    """
    Try to keep the corner/edge intent, but move the target slightly inward if the
    exact point is too close to a wall, split line, or obstacle keepout.
    Returns None when no nearby safe point exists.
    """
    p0 = np.array(point, dtype=float)
    p0[2] = TAKEOFF_HEIGHT

    if is_inside_runtime_safe_area(p0, safe_polygon, obstacles):
        return p0

    candidates = []
    rings = int(max_radius / step)
    for ring in range(1, rings + 1):
        radius = ring * step
        for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
            p = np.array([
                p0[0] + radius * math.cos(angle),
                p0[1] + radius * math.sin(angle),
                TAKEOFF_HEIGHT,
            ], dtype=float)
            if is_inside_runtime_safe_area(p, safe_polygon, obstacles):
                candidates.append(p)

    if not candidates:
        return None

    candidates.sort(key=lambda p: distance_2d(p, p0))
    return candidates[0]


def add_sparse_target(route, point, safe_polygon, obstacles, label, hold_seconds=0.0, speed=None):
    safe_point = nearest_runtime_safe_near(point, safe_polygon, obstacles)
    if safe_point is None:
        print(f'[A{AI_DRONE_ID}] Skipping {label}: no nearby safe point found.')
        return

    append_unique(route, make_target(
        safe_point,
        label=label,
        hold_seconds=hold_seconds,
        yaw_scan=False,
        speed=speed or SCAN_SPEED
    ))


def build_perimeter_targets_all(safe_polygon, obstacles):
    """
    V10_CORNERS route:
      - fewer points
      - more spread out
      - explicitly tries to visit the corners of this drone's safe-zone half
      - if a corner/edge point is blocked by an obstacle or too close to a wall,
        it moves the target slightly inward or skips it.
    """
    left, right, bottom, top = half_bounds(safe_polygon)

    width = max(0.01, right - left)
    height = max(0.01, top - bottom)

    # Pull the visible "corners" inward. This still scans the whole half but
    # avoids forcing the drone right against a wall/split boundary.
    mx = min(AI_CORNER_MARGIN, max(0.05, width * 0.30))
    my = min(AI_CORNER_MARGIN, max(0.05, height * 0.30))

    l = left + mx
    r = right - mx
    b = bottom + my
    t = top - my

    if l >= r:
        l = left + 0.10 * width
        r = right - 0.10 * width
    if b >= t:
        b = bottom + 0.10 * height
        t = top - 0.10 * height

    cx = 0.5 * (l + r)
    cy = 0.5 * (b + t)

    # Full half loop: corners + edge middle points, then back to first corner.
    sparse_loop = [
        ([l, t, TAKEOFF_HEIGHT], 'corner/top-left'),
        ([cx, t, TAKEOFF_HEIGHT], 'top edge'),
        ([r, t, TAKEOFF_HEIGHT], 'corner/top-right'),
        ([r, cy, TAKEOFF_HEIGHT], 'right edge'),
        ([r, b, TAKEOFF_HEIGHT], 'corner/bottom-right'),
        ([cx, b, TAKEOFF_HEIGHT], 'bottom edge'),
        ([l, b, TAKEOFF_HEIGHT], 'corner/bottom-left'),
        ([l, cy, TAKEOFF_HEIGHT], 'left edge'),
        ([l, t, TAKEOFF_HEIGHT], 'corner/top-left return'),
    ]

    route = []
    for point, label in sparse_loop:
        add_sparse_target(
            route,
            point,
            safe_polygon,
            obstacles,
            label=label,
            hold_seconds=CORNER_LOOK_SECONDS if 'corner' in label else 0.0,
            speed=SCAN_SPEED,
        )

    return route


def build_interior_targets_all(safe_polygon, obstacles):
    """
    Sparse interior coverage after the perimeter loop.
    This avoids the old dense zig-zag route while still looking through the half.
    """
    left, right, bottom, top = half_bounds(safe_polygon)

    if not AI_SPARSE_INTERIOR:
        route = []
        y = top
        reverse = False

        while y >= bottom - 1e-6:
            xs = list(np.arange(left, right + 1e-6, INTERIOR_POINT_SPACING))
            if len(xs) == 0 or abs(xs[-1] - right) > 0.15:
                xs.append(float(right))
            if reverse:
                xs.reverse()

            for x in xs:
                p = np.array([x, y, TAKEOFF_HEIGHT], dtype=float)
                if is_inside_runtime_safe_area(p, safe_polygon, obstacles):
                    append_unique(route, make_target(p, label='interior sweep', speed=SCAN_SPEED))

            y -= INTERIOR_ROW_SPACING
            reverse = not reverse

        return route

    width = max(0.01, right - left)
    height = max(0.01, top - bottom)
    mx = min(AI_CORNER_MARGIN, max(0.05, width * 0.30))
    my = min(AI_CORNER_MARGIN, max(0.05, height * 0.30))

    l = left + mx
    r = right - mx
    b = bottom + my
    t = top - my
    cx = 0.5 * (l + r)
    cy = 0.5 * (b + t)

    # Broad scan points, not many small points.
    sparse_points = [
        ([cx, cy, TAKEOFF_HEIGHT], 'center scan'),
        ([0.5 * (l + cx), 0.5 * (t + cy), TAKEOFF_HEIGHT], 'upper interior'),
        ([0.5 * (r + cx), 0.5 * (b + cy), TAKEOFF_HEIGHT], 'lower interior'),
    ]

    route = []
    for point, label in sparse_points:
        add_sparse_target(route, point, safe_polygon, obstacles, label=label, speed=SCAN_SPEED)

    return route


def build_obstacle_view_targets_all(safe_polygon, obstacles):
    targets = []

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
            p = np.array([
                cx + r * math.cos(angle),
                cy + r * math.sin(angle),
                TAKEOFF_HEIGHT
            ], dtype=float)

            p = clip_point_to_safe_rect(p, safe_polygon, PATH_EDGE_MARGIN)
            if not is_inside_runtime_safe_area(p, safe_polygon, obstacles):
                continue

            targets.append(make_target(
                p,
                label=f'obstacle {obstacle_id} view',
                look_at=np.array([cx, cy, TAKEOFF_HEIGHT], dtype=float),
                hold_seconds=OBJECT_LOOK_SECONDS,
                yaw_scan=False,
                speed=INSPECT_SPEED,
                obstacle_id=obstacle_id
            ))

    return targets


def assign_obstacle_views_balanced(all_view_targets, safe_polygon):
    """
    In this strict front/back version, obstacle view targets are already filtered
    to this drone's half. We still keep this function name so the rest of the
    code structure stays simple.
    """
    assignments = {1: [], 2: []}

    # Every process only builds targets that are legal for its own half.
    assignments[AI_DRONE_ID] = list(all_view_targets)
    assignments[1 if AI_DRONE_ID == 2 else 2] = []

    return assignments


def build_split_route_for_this_drone(safe_polygon, obstacles):
    perimeter_all = build_perimeter_targets_all(safe_polygon, obstacles)
    interior_all = build_interior_targets_all(safe_polygon, obstacles)
    views_all = build_obstacle_view_targets_all(safe_polygon, obstacles)
    view_assignments = assign_obstacle_views_balanced(views_all, safe_polygon)

    perimeter = [t for t in perimeter_all if belongs_to_this_drone_by_zone(t['pos'], safe_polygon)]
    interior = [t for t in interior_all if belongs_to_this_drone_by_zone(t['pos'], safe_polygon)]
    obstacle_views = view_assignments[AI_DRONE_ID]

    # Route order:
    #   own half perimeter/corners -> obstacle views -> sparse interior scan
    route = []
    for t in perimeter + obstacle_views + interior:
        append_unique(route, t)

    counts = {
        'perimeter_all': len(perimeter_all),
        'interior_all': len(interior_all),
        'views_all': len(views_all),
        'perimeter_this': len(perimeter),
        'interior_this': len(interior),
        'views_this': len(obstacle_views),
        'views_other': len(view_assignments[1 if AI_DRONE_ID == 2 else 2]),
    }
    return route, counts


# -------------------------------------------------------------------------
# Grid planner
# -------------------------------------------------------------------------
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
            print(f'[A{AI_DRONE_ID}] Planner could not reach target "{target["label"]}". Skipping it.')
            continue

        for p in planned[1:-1]:
            append_unique(expanded, make_target(p, label='planned path', speed=SCAN_SPEED))

        append_unique(expanded, target)
        current = goal

    return expanded


# -------------------------------------------------------------------------
# Control helpers
# -------------------------------------------------------------------------
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


def smooth_land(motion_commander):
    print(f'[A{AI_DRONE_ID}] Landing smoothly...')
    try:
        motion_commander.stop()
        motion_commander.land(velocity=0.15)
        print(f'[A{AI_DRONE_ID}] Landed.')
    except Exception as e:
        print(f'[A{AI_DRONE_ID}] Smooth land failed: {e}')


def arm_cf(cf, do_arm=True):
    """Explicitly arm the Crazyflie before MotionCommander takeoff."""
    try:
        if hasattr(cf, 'supervisor'):
            cf.supervisor.send_arming_request(do_arm)
        else:
            cf.platform.send_arming_request(do_arm)
        print(f'[A{AI_DRONE_ID}] Arming request sent: {do_arm}')
    except Exception as exc:
        print(f'[A{AI_DRONE_ID}] Arming request failed/ignored: {exc}')


def read_altitude_samples(logger, seconds):
    """Read logger samples for a short time and return the best/last sample."""
    deadline = time.time() + float(seconds)
    max_z = -999.0
    last_data = None

    while time.time() < deadline:
        log_entry = logger.next()
        data = log_entry[1]
        last_data = data
        try:
            z = float(data.get('stateEstimate.z', 0.0))
        except Exception:
            z = 0.0
        max_z = max(max_z, z)
        time.sleep(0.02)

    if last_data is None:
        log_entry = logger.next()
        last_data = log_entry[1]
        try:
            max_z = float(last_data.get('stateEstimate.z', 0.0))
        except Exception:
            max_z = 0.0

    return max_z, last_data


def verify_takeoff_or_raise(mc, logger):
    """
    Verify physical takeoff. If stateEstimate.z stays too low, retry once.

    This fixes the situation where the log says 'Taking off' but the drone is
    still on the ground and then only prints route commands.
    """
    min_ok_z = max(0.12, TAKEOFF_HEIGHT * TAKEOFF_MIN_Z_RATIO)

    max_z, last_data = read_altitude_samples(logger, TAKEOFF_VERIFY_SECONDS)
    print(f'[A{AI_DRONE_ID}] Takeoff check: max z={max_z:.2f} m, required >= {min_ok_z:.2f} m')

    if max_z >= min_ok_z:
        print(f'[A{AI_DRONE_ID}] Takeoff verified.')
        return last_data

    print(f'[A{AI_DRONE_ID}] WARNING: takeoff was commanded but altitude stayed too low.')
    print(f'[A{AI_DRONE_ID}] Retrying takeoff/climb once...')

    try:
        mc.stop()
    except Exception:
        pass

    # Try the strongest available MotionCommander command first.
    try:
        if hasattr(mc, 'take_off'):
            mc.take_off(height=TAKEOFF_HEIGHT, velocity=0.15)
        else:
            mc.up(max(0.15, TAKEOFF_HEIGHT - max(0.0, max_z)), velocity=0.12)
    except TypeError:
        try:
            mc.take_off(TAKEOFF_HEIGHT, velocity=0.15)
        except Exception as exc:
            print(f'[A{AI_DRONE_ID}] Explicit takeoff retry failed: {exc}')
    except Exception as exc:
        print(f'[A{AI_DRONE_ID}] Explicit takeoff retry failed: {exc}')

    max_z_retry, last_data_retry = read_altitude_samples(logger, TAKEOFF_RETRY_SECONDS)
    print(f'[A{AI_DRONE_ID}] Takeoff retry check: max z={max_z_retry:.2f} m, required >= {min_ok_z:.2f} m')

    if max_z_retry < min_ok_z:
        try:
            mc.stop()
        except Exception:
            pass
        raise RuntimeError(
            f'A{AI_DRONE_ID} did not physically take off. ' +
            f'Max z={max(max_z, max_z_retry):.2f} m, required >= {min_ok_z:.2f} m. ' +
            'Check battery, props, deck connection, arming, and URI.'
        )

    print(f'[A{AI_DRONE_ID}] Takeoff verified after retry.')
    return last_data_retry


def get_log_config():
    lg = LogConfig(name='PosStab', period_in_ms=200)
    lg.add_variable('stateEstimate.x', 'float')
    lg.add_variable('stateEstimate.y', 'float')
    lg.add_variable('stateEstimate.z', 'float')
    lg.add_variable('stabilizer.yaw', 'float')
    return lg



def control_land_requested():
    return os.path.exists(LAND_NOW_FILE)


def control_emergency_requested():
    return os.path.exists(EMERGENCY_FILE)


def wait_for_parallel_start_gate(scf):
    """
    Optional start barrier for the two AI drones.

    When AI_READY_FILE and AI_START_FILE are set by the launcher:
      1. This drone writes its ready file after the radio link is open.
      2. It waits on the ground until the launcher creates the shared start file.
      3. Then it continues to MotionCommander/takeoff.
    """
    if AI_READY_FILE:
        try:
            ready_dir = os.path.dirname(AI_READY_FILE)
            if ready_dir:
                os.makedirs(ready_dir, exist_ok=True)
            with open(AI_READY_FILE, 'w', encoding='utf-8') as f:
                f.write(f'A{AI_DRONE_ID} ready\n')
            print(f'[A{AI_DRONE_ID}] Ready file written: {AI_READY_FILE}')
        except Exception as exc:
            print(f'[A{AI_DRONE_ID}] Could not write ready file: {exc}')

    if not AI_START_FILE:
        return

    print(f'[A{AI_DRONE_ID}] Waiting for parallel start gate: {AI_START_FILE}')
    start_wait = time.time()
    while not os.path.exists(AI_START_FILE):
        if control_emergency_requested():
            print(f'[A{AI_DRONE_ID}] Emergency before takeoff. Staying on ground.')
            hard_emergency_stop(scf)
            raise RuntimeError('Emergency requested before AI takeoff gate opened.')

        if control_land_requested():
            print(f'[A{AI_DRONE_ID}] Land/stop requested before takeoff. Staying on ground.')
            raise RuntimeError('Land requested before AI takeoff gate opened.')

        if time.time() - start_wait > AI_START_TIMEOUT_SECONDS:
            raise TimeoutError(f'A{AI_DRONE_ID} timed out waiting for AI start gate.')

        time.sleep(0.05)

    print(f'[A{AI_DRONE_ID}] Parallel start gate opened. Taking off now.')


def hard_emergency_stop(scf):
    print(f'[A{AI_DRONE_ID}] EMERGENCY STOP: cutting setpoints and disarming.')
    try:
        scf.cf.commander.send_stop_setpoint()
    except Exception:
        pass
    try:
        if hasattr(scf.cf, 'supervisor'):
            scf.cf.supervisor.send_arming_request(False)
        else:
            scf.cf.platform.send_arming_request(False)
    except Exception:
        pass

def main():
    logging.basicConfig(level=logging.ERROR)
    cflib.crtp.init_drivers()

    print('')
    print(f'[A{AI_DRONE_ID}] Dual AI-deck front/back inspection V10_CORNERS starting...')
    print(f'[A{AI_DRONE_ID}] URI = {URI}')
    print(f'[A{AI_DRONE_ID}] height = {TAKEOFF_HEIGHT:.2f} m')
    print(f'[A{AI_DRONE_ID}] start offset = x {AI_START_OFFSET_X:.2f}, y {AI_START_OFFSET_Y:.2f}')
    print(f'[A{AI_DRONE_ID}] working directory = {os.getcwd()}')
    print(f'[A{AI_DRONE_ID}] control dir = {MISSION_CONTROL_DIR}')
    print('')

    safezone_path, safezone = load_latest_safezone_json()
    safe_polygon = safezone.get('safe_zone_polygon', [])
    obstacles = safezone.get('obstacle_keepouts', [])
    mapper_start_xy = safezone.get('start_position_xy', None)

    if not safe_polygon:
        raise RuntimeError('The safe-zone JSON does not contain safe_zone_polygon.')
    if mapper_start_xy is None:
        print(f'[A{AI_DRONE_ID}] WARNING: safe-zone JSON has no start_position_xy. Using [0, 0].')
        mapper_start_xy = [0.0, 0.0]

    global ACTIVE_SPLIT_X
    ACTIVE_SPLIT_X = float(mapper_start_xy[0])

    map_start_xy = [
        float(mapper_start_xy[0]) + AI_START_OFFSET_X,
        float(mapper_start_xy[1]) + AI_START_OFFSET_Y
    ]

    entry_point_map = nearest_safe_entry_point(map_start_xy, safe_polygon, obstacles)
    raw_route, counts = build_split_route_for_this_drone(safe_polygon, obstacles)

    if not raw_route:
        raise RuntimeError(f'[A{AI_DRONE_ID}] No route targets generated for this drone.')

    planned_route = expand_route_with_grid_planner(raw_route, safe_polygon, obstacles, entry_point_map)
    full_route = [make_target(entry_point_map, label='safe-zone entry', speed=ENTRY_SPEED)] + planned_route

    print(f'[A{AI_DRONE_ID}] Mission plan:')
    print(f'  safe-zone file: {safezone_path}')
    print(f'  obstacles: {len(obstacles)}')
    print(f'  mapper start: x={mapper_start_xy[0]:.2f}, y={mapper_start_xy[1]:.2f}')
    print(f'  AI start used: x={map_start_xy[0]:.2f}, y={map_start_xy[1]:.2f}')
    print(f'  entry: x={entry_point_map[0]:.2f}, y={entry_point_map[1]:.2f}')
    print(f'  all perimeter targets: {counts["perimeter_all"]}')
    print(f'  this drone perimeter targets: {counts["perimeter_this"]}')
    print(f'  all obstacle view targets: {counts["views_all"]}')
    print(f'  this drone obstacle views: {counts["views_this"]}')
    print(f'  other drone obstacle views: {counts["views_other"]}')
    print(f'  this drone interior targets: {counts["interior_this"]}')
    print(f'  total route targets after planner: {len(full_route)}')
    print(f'  front/back split line x={split_x_value(safe_polygon):.2f}')
    print(f'  split buffer={FRONT_BACK_SPLIT_BUFFER:.2f} m')
    print('  rule: A1 and A2 stay outside the center no-fly barrier')
    print('')

    cf = Crazyflie(rw_cache='./cache')
    motion_commander = None

    mission_mode = MISSION_GO_TO_ENTRY
    target_index = 0
    recovery_target_map = None
    recovery_attempts = 0
    recovery_attempts_for_target = 0
    look_start_time = None
    loops = 0
    mission_start_time = None

    # V10_FIXED target-progress guard state.
    current_target_for_guard = None
    target_guard_start_time = None
    target_guard_last_progress_time = None
    target_guard_last_progress_pos = None

    try:
        with SyncCrazyflie(URI, cf=cf) as scf:
            time.sleep(1.0)

            wait_for_parallel_start_gate(scf)

            # V10_LIFTCHECK: explicitly arm before MotionCommander takeoff.
            arm_cf(scf.cf, True)
            time.sleep(0.5)

            with MotionCommander(scf, default_height=TAKEOFF_HEIGHT) as mc:
                motion_commander = mc
                print(f'[A{AI_DRONE_ID}] Taking off...')
                print(f'[A{AI_DRONE_ID}] Waiting {TAKEOFF_SETTLE_SECONDS:.1f}s for stabilizer/Flow deck to settle...')
                time.sleep(TAKEOFF_SETTLE_SECONDS)

                with SyncLogger(scf, get_log_config()) as logger:
                    # V10_LIFTCHECK: do not start the route until the drone is really airborne.
                    first_data = verify_takeoff_or_raise(mc, logger)
                    local_start_xy = [
                        float(first_data.get('stateEstimate.x', 0.0)),
                        float(first_data.get('stateEstimate.y', 0.0))
                    ]
                    local_start_z = float(first_data.get('stateEstimate.z', 0.0))

                    print(f'[A{AI_DRONE_ID}] Local start locked:')
                    print(f'  local_start x={local_start_xy[0]:.2f}, y={local_start_xy[1]:.2f}, z={local_start_z:.2f}')
                    print(f'  map_start   x={map_start_xy[0]:.2f}, y={map_start_xy[1]:.2f}')
                    print('')

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
                        vx = 0.0
                        vy = 0.0
                        yaw_rate = 0.0

                        if control_emergency_requested():
                            hard_emergency_stop(scf)
                            break

                        if control_land_requested() and mission_mode != MISSION_LAND:
                            print(f'[A{AI_DRONE_ID}] LAND key requested. Landing now.')
                            mission_mode = MISSION_LAND
                            vx = vy = yaw_rate = 0.0

                        if elapsed > MAX_FLIGHT_TIME_SECONDS:
                            print(f'[A{AI_DRONE_ID}] Maximum mission time reached. Returning home.')
                            mission_mode = MISSION_RETURN_HOME

                        if mission_mode == MISSION_GO_TO_ENTRY:
                            target = full_route[0]
                            target_local = map_to_local_target(target['pos'], map_start_xy, local_start_xy)
                            vx, vy, yaw_rate, dist = make_body_velocity_to_target(
                                local_position, target_local, yaw_rad, ENTRY_SPEED
                            )
                            if dist < WAYPOINT_RADIUS:
                                print(f'[A{AI_DRONE_ID}] Reached safe-zone entry. Starting split inspection route.')
                                mission_mode = MISSION_ROUTE
                                target_index = 1

                        elif mission_mode == MISSION_ROUTE:
                            if target_index >= len(full_route):
                                print(f'[A{AI_DRONE_ID}] Route complete. Returning home.')
                                mission_mode = MISSION_RETURN_HOME
                            else:
                                target = full_route[target_index]
                                target_map = target['pos']

                                # V10_FIXED: reset/track the progress guard for this target.
                                now_for_guard = time.time()
                                if current_target_for_guard != target_index:
                                    current_target_for_guard = target_index
                                    target_guard_start_time = now_for_guard
                                    target_guard_last_progress_time = now_for_guard
                                    target_guard_last_progress_pos = np.array(map_position, dtype=float)
                                else:
                                    if target_guard_last_progress_pos is not None:
                                        if distance_2d(map_position, target_guard_last_progress_pos) >= AI_TARGET_PROGRESS_EPS_M:
                                            target_guard_last_progress_time = now_for_guard
                                            target_guard_last_progress_pos = np.array(map_position, dtype=float)

                                target_elapsed = 0.0 if target_guard_start_time is None else now_for_guard - target_guard_start_time
                                stuck_elapsed = 0.0 if target_guard_last_progress_time is None else now_for_guard - target_guard_last_progress_time

                                if target_elapsed > AI_TARGET_TIMEOUT_SECONDS:
                                    print(
                                        f'[A{AI_DRONE_ID}] Target {target_index + 1}/{len(full_route)} timed out '
                                        f'after {target_elapsed:.1f}s. Skipping target: {target["label"]}'
                                    )
                                    target_index += 1
                                    recovery_attempts_for_target = 0
                                    current_target_for_guard = None
                                    target_guard_start_time = None
                                    target_guard_last_progress_time = None
                                    target_guard_last_progress_pos = None
                                    vx = vy = yaw_rate = 0.0

                                elif stuck_elapsed > AI_TARGET_STUCK_SECONDS:
                                    print(
                                        f'[A{AI_DRONE_ID}] No map-position progress for {stuck_elapsed:.1f}s '
                                        f'on target {target_index + 1}/{len(full_route)}. Skipping target: {target["label"]}'
                                    )
                                    target_index += 1
                                    recovery_attempts_for_target = 0
                                    current_target_for_guard = None
                                    target_guard_start_time = None
                                    target_guard_last_progress_time = None
                                    target_guard_last_progress_pos = None
                                    vx = vy = yaw_rate = 0.0

                                elif not is_safe_point(target_map, safe_polygon, obstacles):
                                    print(f'[A{AI_DRONE_ID}] Skipping unsafe target {target_index + 1}/{len(full_route)}: {target["label"]}')
                                    target_index += 1
                                    current_target_for_guard = None
                                    target_guard_start_time = None
                                    target_guard_last_progress_time = None
                                    target_guard_last_progress_pos = None
                                else:
                                    speed = target.get('speed', None) or SCAN_SPEED
                                    target_local = map_to_local_target(target_map, map_start_xy, local_start_xy)
                                    vx, vy, yaw_rate, dist = make_body_velocity_to_target(
                                        local_position, target_local, yaw_rad, speed
                                    )

                                    if dist < WAYPOINT_RADIUS:
                                        print(f'[A{AI_DRONE_ID}] Reached target {target_index + 1}/{len(full_route)}: {target["label"]}')
                                        recovery_attempts_for_target = 0
                                        current_target_for_guard = None
                                        target_guard_start_time = None
                                        target_guard_last_progress_time = None
                                        target_guard_last_progress_pos = None

                                        if target.get('hold_seconds', 0.0) > 0.0:
                                            mission_mode = MISSION_LOOK
                                            look_start_time = time.time()
                                            print(f'[A{AI_DRONE_ID}] Camera look: {target["label"]} for {target["hold_seconds"]:.1f}s')
                                        else:
                                            target_index += 1

                                        vx = vy = yaw_rate = 0.0

                        elif mission_mode == MISSION_LOOK:
                            target = full_route[target_index]
                            vx = 0.0
                            vy = 0.0

                            if target.get('look_at', None) is not None:
                                yaw_rate = yaw_rate_to_face_target(map_position, target['look_at'], yaw_rad)
                            elif target.get('yaw_scan', False):
                                yaw_rate = radians(CORNER_YAW_RATE_DEG)
                            else:
                                yaw_rate = 0.0

                            if look_start_time is not None and (time.time() - look_start_time) >= target['hold_seconds']:
                                print(f'[A{AI_DRONE_ID}] Camera look complete.')
                                target_index += 1
                                mission_mode = MISSION_ROUTE
                                look_start_time = None
                                vx = vy = yaw_rate = 0.0

                        elif mission_mode == MISSION_RECOVER_SAFE:
                            if recovery_target_map is None:
                                recovery_target_map = nearest_runtime_safe_point(map_position, safe_polygon, obstacles)
                                print(f'[A{AI_DRONE_ID}] Recovery target: x={recovery_target_map[0]:.2f}, y={recovery_target_map[1]:.2f}')

                            recovery_local = map_to_local_target(recovery_target_map, map_start_xy, local_start_xy)
                            vx, vy, yaw_rate, dist_recovery = make_body_velocity_to_target(
                                local_position, recovery_local, yaw_rad, RECOVERY_SPEED
                            )

                            if dist_recovery < RECOVERY_WAYPOINT_RADIUS:
                                print(f'[A{AI_DRONE_ID}] Recovered inward. Continuing route.')
                                mission_mode = MISSION_ROUTE
                                recovery_target_map = None
                                vx = vy = yaw_rate = 0.0

                        elif mission_mode == MISSION_RETURN_HOME:
                            home_map = np.array([map_start_xy[0], map_start_xy[1], TAKEOFF_HEIGHT], dtype=float)
                            home_local = map_to_local_target(home_map, map_start_xy, local_start_xy)
                            vx, vy, yaw_rate, dist_home = make_body_velocity_to_target(
                                local_position, home_local, yaw_rad, RETURN_SPEED
                            )

                            if dist_home < WAYPOINT_RADIUS:
                                print(f'[A{AI_DRONE_ID}] Returned to own start. Landing.')
                                mission_mode = MISSION_LAND
                                vx = vy = yaw_rate = 0.0

                        elif mission_mode == MISSION_LAND:
                            mc.stop()
                            smooth_land(mc)
                            break

                        else:
                            print(f'[A{AI_DRONE_ID}] Unknown mode. Returning home.')
                            mission_mode = MISSION_RETURN_HOME
                            vx = vy = yaw_rate = 0.0

                        # V10_CORNERS runtime safety: soft guard, no panic on small drift.
                        # The route is planned inside the half already. At runtime we only recover
                        # from real danger: outside safe rectangle, deep center crossing, or obstacle keepout.
                        if mission_mode in (MISSION_ROUTE, MISSION_LOOK):
                            if is_runtime_hard_danger(map_position, safe_polygon, obstacles):
                                recovery_attempts += 1
                                recovery_attempts_for_target += 1
                                print(f'[A{AI_DRONE_ID}] WARNING: hard boundary/obstacle guard. Recovering inward.')

                                if recovery_attempts_for_target > MAX_RECOVERY_ATTEMPTS_PER_TARGET:
                                    print(f'[A{AI_DRONE_ID}] Target {target_index + 1} caused too many hard-guard recoveries. Skipping target.')
                                    target_index += 1
                                    recovery_attempts_for_target = 0
                                    mission_mode = MISSION_ROUTE
                                elif recovery_attempts > MAX_RECOVERY_ATTEMPTS:
                                    print(f'[A{AI_DRONE_ID}] Too many hard-guard recoveries. Returning home smoothly.')
                                    mission_mode = MISSION_RETURN_HOME
                                else:
                                    recovery_target_map = nearest_runtime_safe_point(map_position, safe_polygon, obstacles)
                                    mission_mode = MISSION_RECOVER_SAFE

                                vx = vy = yaw_rate = 0.0

                        # V10_CORNERS altitude guard. Never emergency-stop because of low z;
                        # slow down and climb gently so the drone does not drop out of the air.
                        vz = 0.0
                        current_z = float(local_position[2])
                        min_flight_z = max(0.12, TAKEOFF_HEIGHT * AI_MIN_ALTITUDE_RATIO)
                        if mission_mode not in (MISSION_LAND,) and current_z < min_flight_z:
                            print(f'[A{AI_DRONE_ID}] WARNING: low altitude z={current_z:.2f} m. Climbing gently.')
                            vx *= 0.25
                            vy *= 0.25
                            yaw_rate *= 0.5
                            vz = AI_LOW_ALTITUDE_CLIMB_SPEED

                        mc.start_linear_motion(vx, vy, vz, rate_yaw=degrees(yaw_rate))

                        if loops % PRINT_EVERY_N_LOOPS == 0:
                            label = 'none'
                            if 0 <= target_index < len(full_route):
                                label = full_route[target_index]['label']

                            print(
                                f'[A{AI_DRONE_ID}] mode={mission_mode}, target={target_index + 1}/{len(full_route)}, '
                                f'label={label}, map=({map_position[0]:.2f},{map_position[1]:.2f}), '
                                f'vx={vx:.2f}, vy={vy:.2f}, yaw_rate={degrees(yaw_rate):.1f}'
                            )

                        time.sleep(LOOP_SLEEP_SECONDS)

    except KeyboardInterrupt:
        print('')
        print(f'[A{AI_DRONE_ID}] Ctrl+C pressed. Trying to land safely...')
        try:
            if motion_commander is not None:
                motion_commander.stop()
                motion_commander.land(velocity=0.15)
        except Exception:
            pass

    except Exception as e:
        print('')
        print(f'[A{AI_DRONE_ID}] ERROR: mission could not continue.')
        print(f'  {type(e).__name__}: {e}')
        print('')

    finally:
        print(f'[A{AI_DRONE_ID}] Dual AI-deck mission finished.')


if __name__ == '__main__':
    main()
