#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V8 MEETING-GUARD - one-process / one-Crazyradio dual Multiranger mapper.

Fixes compared with V2:
  - Both drones must connect before ANY drone takes off.
  - If M2 cannot connect, M1 stays on the ground.
  - Slower logging to reduce one-radio load.
  - Uses actual yaw in the wall-following controller, fixing the endless-corner-spin issue.
  - If one mapper errors, the other mapper lands.

Recommended addresses:
  M1: radio://0/80/2M/E7E7E7E701
  M2: radio://0/80/2M/E7E7E7E703
"""

import json
import logging
import math
import os
import threading
import time
import sys
from pathlib import Path
from math import radians, degrees

import numpy as np

from wall_following import WallFollowing

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils.multiranger import Multiranger


MAPPER1_URI = os.environ.get('MAPPER1_URI', 'radio://0/80/2M/E7E7E7E701')
MAPPER2_URI = os.environ.get('MAPPER2_URI', 'radio://0/80/2M/E7E7E7E703')

MAPPER1_HEIGHT = float(os.environ.get('MAPPER1_HEIGHT', '0.20'))
MAPPER2_HEIGHT = float(os.environ.get('MAPPER2_HEIGHT', '0.32'))

# Slower and safer than V2.
MAPPING_TIME_SECONDS = float(os.environ.get('DUAL_MAPPER_MAPPING_TIME_SECONDS', '110.0'))

# M2 is allowed to take off a bit after M1, but only after BOTH links are connected.
MAPPER2_TAKEOFF_DELAY_SECONDS = float(os.environ.get('MAPPER2_TAKEOFF_DELAY_SECONDS', '2.0'))

# V7 center-start setup:
# M1 and M2 start 30 cm apart around the physical center line.
# M1 faces forward and follows the wall on its LEFT.
# M2 faces backward/opposite and follows the wall on its RIGHT.
START_SEPARATION_M = float(os.environ.get('START_SEPARATION_M', '0.30'))
MAPPER1_START_OFFSET_X = START_SEPARATION_M / 2.0
MAPPER2_START_OFFSET_X = -START_SEPARATION_M / 2.0
MAPPER1_WALL_SIDE = os.environ.get('MAPPER1_WALL_SIDE', 'left').lower()
MAPPER2_WALL_SIDE = os.environ.get('MAPPER2_WALL_SIDE', 'right').lower()

# Ignore back range points during the first seconds, because the two mappers
# are close together and can see each other at takeoff.
BACK_SENSOR_IGNORE_SECONDS = float(os.environ.get('BACK_SENSOR_IGNORE_SECONDS', '5.0'))

# Stop when each mapper reaches the other mapper's start zone, after it has
# clearly left the start and followed the wall for a while.
CENTER_FINISH_STOP_ENABLED = os.environ.get('CENTER_FINISH_STOP_ENABLED', '1').lower() not in ('0', 'false', 'no')
CENTER_FINISH_TARGET_X = -START_SEPARATION_M
CENTER_FINISH_TARGET_Y = 0.0
CENTER_FINISH_RADIUS = float(os.environ.get('CENTER_FINISH_RADIUS', '0.45'))
CENTER_FINISH_MIN_TIME_SECONDS = float(os.environ.get('CENTER_FINISH_MIN_TIME_SECONDS', '30.0'))
CENTER_FINISH_MIN_TRAVEL_M = float(os.environ.get('CENTER_FINISH_MIN_TRAVEL_M', '1.60'))
CENTER_FINISH_MIN_YAW_DEG = float(os.environ.get('CENTER_FINISH_MIN_YAW_DEG', '230.0'))

# V8 extra safety:
# If both mappers have already gone around a large part of the room and their
# transformed positions get close again, land both before they can cross/fly under each other.
MEETING_GUARD_ENABLED = os.environ.get('MEETING_GUARD_ENABLED', '1').lower() not in ('0', 'false', 'no')
MEETING_GUARD_DISTANCE_M = float(os.environ.get('MEETING_GUARD_DISTANCE_M', '1.00'))
MEETING_GUARD_MIN_TIME_SECONDS = float(os.environ.get('MEETING_GUARD_MIN_TIME_SECONDS', '24.0'))
MEETING_GUARD_MIN_PATH_M = float(os.environ.get('MEETING_GUARD_MIN_PATH_M', '2.20'))
MEETING_GUARD_MIN_YAW_DEG = float(os.environ.get('MEETING_GUARD_MIN_YAW_DEG', '180.0'))

# V6: stop each mapper later, when it has done roughly a complete half loop.
# This prevents M1 from continuing into M2's half.
HALF_STOP_ENABLED = os.environ.get('HALF_STOP_ENABLED', '0').lower() in ('1', 'true', 'yes')
HALF_STOP_YAW_DEG = float(os.environ.get('HALF_STOP_YAW_DEG', '320.0'))
HALF_STOP_MIN_TIME_SECONDS = float(os.environ.get('HALF_STOP_MIN_TIME_SECONDS', '32.0'))
HALF_STOP_MIN_POINTS = int(os.environ.get('HALF_STOP_MIN_POINTS', '430'))

# For first tests, do not export automatic obstacles from merged pointcloud.
# The previous black circle was likely a false obstacle caused by map overlap/edge points.
EXPORT_MERGED_OBSTACLES = os.environ.get('EXPORT_MERGED_OBSTACLES', '1').lower() in ('1', 'true', 'yes')

REFERENCE_DISTANCE_FROM_WALL = float(os.environ.get('REFERENCE_DISTANCE_FROM_WALL', '0.55'))
MAX_FORWARD_SPEED = float(os.environ.get('DUAL_MAPPER_MAX_FORWARD_SPEED', '0.15'))
MAX_TURN_RATE = 0.35
EMERGENCY_DISTANCE = 0.15

FRONT_SLOW_DISTANCE = 0.45
FRONT_STOP_DISTANCE = 0.30
FRONT_EMERGENCY_DISTANCE = 0.20
TOP_CLEARANCE_STOP = 0.20

SENSOR_MAX_M = 2.00
MIN_VALID_RANGE_M = 0.08

# Slower logging than V2 to reduce radio load.
LOG_PERIOD_MS = int(os.environ.get('DUAL_MAPPER_LOG_PERIOD_MS', '250'))
PRINT_EVERY_N = 4

SAFE_ZONE_DIR = Path(os.environ.get('SAFE_ZONE_DIR', 'safe_zone_output'))
MAPPER1_DIR = SAFE_ZONE_DIR / 'mapper1'
MAPPER2_DIR = SAFE_ZONE_DIR / 'mapper2'

SAFE_WALL_MARGIN = 0.35
SAFE_OBSTACLE_MARGIN = 0.55
FILTER_GRID_SIZE = 0.10
FILTER_MIN_POINTS_PER_CELL = 2
INTERIOR_MARGIN_FROM_WALL = 0.40
OBSTACLE_GRID_SIZE = 0.25
OBSTACLE_MIN_POINTS = 6
OBSTACLE_MIN_SEPARATION = 0.65
MAX_OBSTACLES_TO_EXPORT = 6

MAPPER2_YAW_DEG = 180.0


def start_keyboard_monitor(stop_event, emergency_event, label='mapper'):
    def monitor_windows():
        try:
            import msvcrt
            print('')
            print(f'[{label}] KEYBOARD CONTROLS:')
            print('  press L = smooth land both drones')
            print('  press E = EMERGENCY stop/disarm both drones')
            print('')
            while not stop_event.is_set() and not emergency_event.is_set():
                if msvcrt.kbhit():
                    ch = msvcrt.getwch().lower()
                    if ch == 'l':
                        print(f'[{label}] LAND key pressed.')
                        stop_event.set()
                        return
                    if ch == 'e':
                        print(f'[{label}] EMERGENCY key pressed.')
                        emergency_event.set()
                        stop_event.set()
                        return
                time.sleep(0.05)
        except Exception as exc:
            print(f'[{label}] Keyboard monitor unavailable: {exc}')

    def monitor_portable():
        print('')
        print(f'[{label}] Keyboard controls: type l + Enter to land, e + Enter for emergency.')
        print('')
        while not stop_event.is_set() and not emergency_event.is_set():
            try:
                line = sys.stdin.readline().strip().lower()
            except Exception:
                return
            if line == 'l':
                print(f'[{label}] LAND command received.')
                stop_event.set()
                return
            if line == 'e':
                print(f'[{label}] EMERGENCY command received.')
                emergency_event.set()
                stop_event.set()
                return

    target = monitor_windows if os.name == 'nt' else monitor_portable
    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def wrap_angle_deg(delta):
    while delta > 180.0:
        delta -= 360.0
    while delta < -180.0:
        delta += 360.0
    return delta


def safe_range_m(value):
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if value < MIN_VALID_RANGE_M or value > SENSOR_MAX_M:
        return None
    return value


def get_wall_left_direction():
    if hasattr(WallFollowing, 'WallFollowingDirection'):
        return WallFollowing.WallFollowingDirection.RIGHT
    raise AttributeError(
        'wall_following.py does not expose WallFollowing.WallFollowingDirection. '
        'Use the same wall_following.py that worked with your original mapper.'
    )


def get_wall_right_direction():
    if hasattr(WallFollowing, 'WallFollowingDirection'):
        return WallFollowing.WallFollowingDirection.LEFT
    raise AttributeError(
        'wall_following.py does not expose WallFollowing.WallFollowingDirection. '
        'Use the same wall_following.py that worked with your original mapper.'
    )


def choose_side_and_direction(wall_side, left_range, right_range):
    if wall_side == 'right':
        return right_range, get_wall_right_direction()
    return left_range, get_wall_left_direction()


def arm_cf(cf, do_arm):
    try:
        if hasattr(cf, 'supervisor'):
            cf.supervisor.send_arming_request(do_arm)
        else:
            cf.platform.send_arming_request(do_arm)
    except Exception as e:
        print(f'Arming request failed: {e}')


def smooth_land(mc, label):
    print(f'[{label}] Landing smoothly...')
    try:
        mc.stop()
        mc.land(velocity=0.15)
        print(f'[{label}] Landed.')
    except Exception as e:
        print(f'[{label}] Smooth land failed: {e}')


def get_log_config():
    lg = LogConfig(name='PosStab', period_in_ms=LOG_PERIOD_MS)
    lg.add_variable('stateEstimate.x', 'float')
    lg.add_variable('stateEstimate.y', 'float')
    lg.add_variable('stateEstimate.z', 'float')
    lg.add_variable('stabilizer.roll', 'float')
    lg.add_variable('stabilizer.pitch', 'float')
    lg.add_variable('stabilizer.yaw', 'float')
    return lg


def add_sensor_point(points, x, y, z, yaw_rad, sensor_name, distance_m):
    d = safe_range_m(distance_m)
    if d is None:
        return

    # Body frame: +x front, +y left
    if sensor_name == 'front':
        bx, by = d, 0.0
    elif sensor_name == 'back':
        bx, by = -d, 0.0
    elif sensor_name == 'left':
        bx, by = 0.0, d
    elif sensor_name == 'right':
        bx, by = 0.0, -d
    else:
        return

    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)

    points.append([
        float(x) + c * bx - s * by,
        float(y) + s * bx + c * by,
        float(z),
    ])


def compute_wall_following_command(wall_following, front, side_range, wall_side, actual_yaw_rad):
    """
    V3 fix:
      pass actual_yaw_rad into wall_follower.
    In V2 this was 0.0, which can make the corner state keep spinning.
    """
    front_value = front if front is not None else 999.0
    side_value = side_range if side_range is not None else 999.0
    wall_direction = get_wall_right_direction() if wall_side == 'right' else get_wall_left_direction()

    velocity_x, velocity_y, yaw_rate, state = wall_following.wall_follower(
        front_value,
        side_value,
        actual_yaw_rad,
        wall_direction,
        time.time()
    )

    # Extra conservative corner safety.
    if front_value < FRONT_EMERGENCY_DISTANCE:
        velocity_x = -0.03
        velocity_y = 0.0
    elif front_value < FRONT_STOP_DISTANCE:
        velocity_x = 0.0
        velocity_y = 0.0
    elif front_value < FRONT_SLOW_DISTANCE:
        velocity_x = min(velocity_x, 0.05)
        velocity_y = clamp(velocity_y, -0.04, 0.04)

    velocity_x = clamp(velocity_x, -MAX_FORWARD_SPEED, MAX_FORWARD_SPEED)
    velocity_y = clamp(velocity_y, -0.08, 0.08)
    yaw_rate = clamp(yaw_rate, -MAX_TURN_RATE, MAX_TURN_RATE)

    return velocity_x, velocity_y, yaw_rate, state


class MapperThread(threading.Thread):
    def __init__(self, label, uri, height, export_dir, both_ready_event, stop_event, emergency_event, wall_side='left', takeoff_delay=0.0):
        super().__init__(daemon=False)
        self.label = label
        self.uri = uri
        self.height = float(height)
        self.export_dir = Path(export_dir)
        self.both_ready_event = both_ready_event
        self.stop_event = stop_event
        self.emergency_event = emergency_event
        self.wall_side = wall_side.lower()
        self.takeoff_delay = float(takeoff_delay)

        self.connected_event = threading.Event()
        self.points = []
        self.error = None
        self.csv_path = None
        self.latest_global_xy = None
        self.latest_elapsed = 0.0
        self.latest_path_m = 0.0
        self.latest_yaw_sum_deg = 0.0
        self.normal_finished = False

    def run(self):
        mc = None
        try:
            self.export_dir.mkdir(parents=True, exist_ok=True)

            wall_following = WallFollowing(
                angle_value_buffer=0.1,
                reference_distance_from_wall=REFERENCE_DISTANCE_FROM_WALL,
                max_forward_speed=MAX_FORWARD_SPEED,
                max_turn_rate=MAX_TURN_RATE,
                emergency_distance=EMERGENCY_DISTANCE,
                init_state=WallFollowing.StateWallFollowing.FORWARD
            )

            print('')
            print(f'[{self.label}] Connecting: {self.uri}')
            print(f'[{self.label}] Height: {self.height:.2f} m')
            print(f'[{self.label}] Log period: {LOG_PERIOD_MS} ms')
            print(f'[{self.label}] Wall on {self.wall_side.upper()}')
            print('')

            cf = Crazyflie(rw_cache='./cache')

            with SyncCrazyflie(self.uri, cf=cf) as scf:
                with Multiranger(scf) as multiranger, SyncLogger(scf, get_log_config()) as logger:
                    print(f'[{self.label}] CONNECTED. Waiting until both mappers are ready...')
                    self.connected_event.set()

                    while not self.both_ready_event.is_set() and not self.stop_event.is_set():
                        time.sleep(0.05)

                    if self.emergency_event.is_set():
                        print(f'[{self.label}] Emergency before takeoff. Staying on ground.')
                        return

                    if self.stop_event.is_set():
                        print(f'[{self.label}] Stop before takeoff. Staying on ground.')
                        return

                    if self.takeoff_delay > 0:
                        print(f'[{self.label}] Takeoff delay {self.takeoff_delay:.1f}s...')
                        end_delay = time.time() + self.takeoff_delay
                        while time.time() < end_delay:
                            if self.stop_event.is_set():
                                print(f'[{self.label}] Stop during takeoff delay.')
                                return
                            time.sleep(0.05)

                    arm_cf(scf.cf, True)
                    time.sleep(0.5)

                    with MotionCommander(scf, default_height=self.height) as mc:
                        print(f'[{self.label}] Taking off...')
                        time.sleep(2.0)

                        start_time = time.time()
                        loops = 0
                        first_yaw_deg = None
                        previous_yaw_deg = None
                        accumulated_abs_yaw_deg = 0.0
                        previous_xy = None
                        accumulated_path_m = 0.0

                        for log_entry in logger:
                            if self.emergency_event.is_set():
                                print(f'[{self.label}] EMERGENCY event received. Cutting commands/disarming.')
                                try:
                                    mc.stop()
                                    scf.cf.commander.send_stop_setpoint()
                                    arm_cf(scf.cf, False)
                                except Exception:
                                    pass
                                return

                            if self.stop_event.is_set():
                                print(f'[{self.label}] LAND event received. Landing now.')
                                break

                            loops += 1
                            data = log_entry[1]

                            x = float(data.get('stateEstimate.x', 0.0))
                            y = float(data.get('stateEstimate.y', 0.0))
                            z = float(data.get('stateEstimate.z', self.height))
                            yaw_deg = float(data.get('stabilizer.yaw', 0.0))
                            yaw_rad = radians(yaw_deg)

                            if first_yaw_deg is None:
                                first_yaw_deg = yaw_deg
                                previous_yaw_deg = yaw_deg
                            else:
                                dyaw = wrap_angle_deg(yaw_deg - previous_yaw_deg)
                                accumulated_abs_yaw_deg += abs(dyaw)
                                previous_yaw_deg = yaw_deg

                            if previous_xy is None:
                                previous_xy = (x, y)
                            else:
                                step_dist = math.hypot(x - previous_xy[0], y - previous_xy[1])
                                if step_dist < 0.40:
                                    accumulated_path_m += step_dist
                                previous_xy = (x, y)

                            front = safe_range_m(multiranger.front)
                            back = safe_range_m(multiranger.back)
                            left = safe_range_m(multiranger.left)
                            right = safe_range_m(multiranger.right)
                            up = safe_range_m(multiranger.up)

                            elapsed = time.time() - start_time

                            gx, gy = local_xy_to_shared_map(self.label, x, y)
                            self.latest_global_xy = (gx, gy)
                            self.latest_elapsed = elapsed
                            self.latest_path_m = accumulated_path_m
                            self.latest_yaw_sum_deg = accumulated_abs_yaw_deg

                            add_sensor_point(self.points, x, y, z, yaw_rad, 'front', front)
                            if elapsed >= BACK_SENSOR_IGNORE_SECONDS:
                                add_sensor_point(self.points, x, y, z, yaw_rad, 'back', back)
                            add_sensor_point(self.points, x, y, z, yaw_rad, 'left', left)
                            add_sensor_point(self.points, x, y, z, yaw_rad, 'right', right)

                            side_range, _wall_direction_for_print = choose_side_and_direction(self.wall_side, left, right)
                            vx, vy, yaw_rate, state = compute_wall_following_command(
                                wall_following, front, side_range, self.wall_side, yaw_rad
                            )
                            mc.start_linear_motion(vx, vy, 0.0, rate_yaw=degrees(yaw_rate))

                            if loops % PRINT_EVERY_N == 0:
                                print(
                                    f'[{self.label}] t={elapsed:.1f}s, '
                                    f'vx={vx:.2f}, vy={vy:.2f}, yaw_cmd={degrees(yaw_rate):.1f}, '
                                    f'actual_yaw={yaw_deg:.1f}, yaw_sum={accumulated_abs_yaw_deg:.0f}, '
                                    f'path={accumulated_path_m:.2f}, '
                                    f'finish_dist={math.hypot(x - CENTER_FINISH_TARGET_X, y - CENTER_FINISH_TARGET_Y):.2f}, '
                                    f'front={front if front is not None else 999:.2f}, '
                                    f'left={left if left is not None else 999:.2f}, '
                                    f'right={right if right is not None else 999:.2f}, '
                                    f'z={z:.2f}, points={len(self.points)}'
                                )

                            if up is not None and up < TOP_CLEARANCE_STOP:
                                print(f'[{self.label}] Top sensor triggered. Landing both mappers for safety.')
                                self.normal_finished = True
                                self.stop_event.set()
                                break

                            finish_dist = math.hypot(x - CENTER_FINISH_TARGET_X, y - CENTER_FINISH_TARGET_Y)
                            if (
                                CENTER_FINISH_STOP_ENABLED
                                and elapsed >= CENTER_FINISH_MIN_TIME_SECONDS
                                and accumulated_path_m >= CENTER_FINISH_MIN_TRAVEL_M
                                and accumulated_abs_yaw_deg >= CENTER_FINISH_MIN_YAW_DEG
                                and finish_dist <= CENTER_FINISH_RADIUS
                            ):
                                print(
                                    f'[{self.label}] Reached opposite center start zone '
                                    f'(distance={finish_dist:.2f} m). Landing both mappers.'
                                )
                                self.normal_finished = True
                                self.stop_event.set()
                                break

                            if (
                                HALF_STOP_ENABLED
                                and elapsed >= HALF_STOP_MIN_TIME_SECONDS
                                and len(self.points) >= HALF_STOP_MIN_POINTS
                                and accumulated_abs_yaw_deg >= HALF_STOP_YAW_DEG
                            ):
                                print(
                                    f'[{self.label}] Half-route yaw reached '
                                    f'({accumulated_abs_yaw_deg:.1f} deg). Landing before crossing into other half.'
                                )
                                break

                            if elapsed >= MAPPING_TIME_SECONDS:
                                print(f'[{self.label}] Mapping time complete. Landing both mappers.')
                                self.normal_finished = True
                                self.stop_event.set()
                                break

                            time.sleep(0.02)

                        smooth_land(mc, self.label)

            timestamp = time.strftime('%Y%m%d_%H%M%S')
            self.csv_path = self.export_dir / f'{self.label.lower()}_filtered_pointcloud_{timestamp}.csv'

            pts = np.asarray(self.points, dtype=float)
            if pts.size == 0:
                pts = np.empty((0, 3), dtype=float)

            np.savetxt(self.csv_path, pts, delimiter=',', header='x,y,z', comments='')
            print(f'[{self.label}] Saved point cloud: {self.csv_path}')

        except Exception as e:
            self.error = e
            self.stop_event.set()
            print('')
            print(f'[{self.label}] ERROR: {type(e).__name__}: {e}')
            print('')

        finally:
            # If an error occurs after takeoff, try to land/stop.
            try:
                if mc is not None and self.stop_event.is_set() and not self.emergency_event.is_set():
                    smooth_land(mc, self.label)
            except Exception:
                pass


def transform_points(points, offset_x, offset_y, yaw_deg):
    yaw = math.radians(yaw_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)

    out = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    out[:, 0] = c * x - s * y + offset_x
    out[:, 1] = s * x + c * y + offset_y
    return out


def local_xy_to_shared_map(label, x, y):
    """Transform each mapper's local estimate into the shared center-start frame."""
    if label == 'MAPPER2':
        # M2 faces opposite M1, so rotate local coordinates by 180 degrees.
        return (-float(x) + MAPPER2_START_OFFSET_X, -float(y))
    # M1 is the reference orientation.
    return (float(x) + MAPPER1_START_OFFSET_X, float(y))


def robust_xy_bounds(points):
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 20:
        raise RuntimeError('Not enough points to estimate bounds for automatic mapper merge.')

    min_x, max_x = np.percentile(pts[:, 0], [2, 98])
    min_y, max_y = np.percentile(pts[:, 1], [2, 98])

    return {
        'min_x': float(min_x),
        'max_x': float(max_x),
        'min_y': float(min_y),
        'max_y': float(max_y),
        'center_x': float(0.5 * (min_x + max_x)),
        'center_y': float(0.5 * (min_y + max_y)),
    }


def estimate_mapper2_transform(pts1, pts2_local):
    pts2_rot = transform_points(pts2_local, 0.0, 0.0, MAPPER2_YAW_DEG)

    b1 = robust_xy_bounds(pts1)
    b2 = robust_xy_bounds(pts2_rot)

    offset_y = b1['center_y'] - b2['center_y']
    offset_x = b1['max_x'] - b2['max_x']

    print('')
    print('Automatic M2 transform estimate:')
    print(f'  M1 bounds: x=[{b1["min_x"]:.2f}, {b1["max_x"]:.2f}], y=[{b1["min_y"]:.2f}, {b1["max_y"]:.2f}]')
    print(f'  M2 rotated bounds: x=[{b2["min_x"]:.2f}, {b2["max_x"]:.2f}], y=[{b2["min_y"]:.2f}, {b2["max_y"]:.2f}]')
    print(f'  estimated offset_x={offset_x:.2f}, offset_y={offset_y:.2f}, yaw={MAPPER2_YAW_DEG:.1f}')
    print('')

    pts2 = pts2_rot.copy()
    pts2[:, 0] += offset_x
    pts2[:, 1] += offset_y

    return pts2, offset_x, offset_y


def filter_pointcloud_outliers(points):
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]

    if len(pts) == 0:
        return np.empty((0, 3), dtype=float)

    z = pts[:, 2]
    pts = pts[(z > 0.02) & (z < 1.50)]

    if len(pts) == 0:
        return np.empty((0, 3), dtype=float)

    grid = np.floor(pts[:, :2] / FILTER_GRID_SIZE).astype(int)
    counts = {}
    for cell in grid:
        key = (int(cell[0]), int(cell[1]))
        counts[key] = counts.get(key, 0) + 1

    keep = []
    for i, cell in enumerate(grid):
        key = (int(cell[0]), int(cell[1]))
        if counts.get(key, 0) >= FILTER_MIN_POINTS_PER_CELL:
            keep.append(i)

    if not keep:
        return pts

    return pts[np.asarray(keep, dtype=int)]


def estimate_boundary(points):
    if points is None or len(points) < 20:
        return None

    xy = points[:, :2]
    min_x, max_x = np.percentile(xy[:, 0], [2, 98])
    min_y, max_y = np.percentile(xy[:, 1], [2, 98])

    if max_x - min_x < 0.5 or max_y - min_y < 0.5:
        return None

    return {
        'min_x': float(min_x),
        'max_x': float(max_x),
        'min_y': float(min_y),
        'max_y': float(max_y),
    }


def find_merged_obstacles(points, boundary):
    if points is None or len(points) < 40 or boundary is None:
        return []

    pts = np.asarray(points, dtype=float)
    z = pts[:, 2]
    pts = pts[(z > 0.05) & (z < 1.20)]

    if len(pts) < OBSTACLE_MIN_POINTS:
        return []

    min_x = boundary['min_x']
    max_x = boundary['max_x']
    min_y = boundary['min_y']
    max_y = boundary['max_y']

    xy = pts[:, :2]
    interior = (
        (xy[:, 0] > min_x + INTERIOR_MARGIN_FROM_WALL) &
        (xy[:, 0] < max_x - INTERIOR_MARGIN_FROM_WALL) &
        (xy[:, 1] > min_y + INTERIOR_MARGIN_FROM_WALL) &
        (xy[:, 1] < max_y - INTERIOR_MARGIN_FROM_WALL)
    )
    candidates = pts[interior]

    if len(candidates) < OBSTACLE_MIN_POINTS:
        return []

    grid_xy = np.floor(candidates[:, :2] / OBSTACLE_GRID_SIZE).astype(int)
    cell_to_indices = {}
    for idx, cell in enumerate(grid_xy):
        key = (int(cell[0]), int(cell[1]))
        cell_to_indices.setdefault(key, []).append(idx)

    visited = set()
    clusters = []

    for cell in list(cell_to_indices.keys()):
        if cell in visited:
            continue

        stack = [cell]
        visited.add(cell)
        cluster_indices = []

        while stack:
            current = stack.pop()
            cluster_indices.extend(cell_to_indices.get(current, []))

            cx, cy = current
            for nx in (cx - 1, cx, cx + 1):
                for ny in (cy - 1, cy, cy + 1):
                    nb = (nx, ny)
                    if nb in cell_to_indices and nb not in visited:
                        visited.add(nb)
                        stack.append(nb)

        if len(cluster_indices) >= OBSTACLE_MIN_POINTS:
            cluster_pts = candidates[np.asarray(cluster_indices, dtype=int)]
            center = np.mean(cluster_pts[:, :2], axis=0)
            clusters.append({'center': center, 'count': len(cluster_indices)})

    clusters.sort(key=lambda item: item['count'], reverse=True)

    selected = []
    for cluster in clusters:
        duplicate = False
        for chosen in selected:
            if math.dist(cluster['center'], chosen['center']) < OBSTACLE_MIN_SEPARATION:
                duplicate = True
                break
        if not duplicate:
            selected.append(cluster)
        if len(selected) >= MAX_OBSTACLES_TO_EXPORT:
            break

    return selected


def write_safezone_png(path, points, safe_polygon, obstacles):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    if points is not None and len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], s=2)

    if safe_polygon:
        poly = np.asarray(safe_polygon + [safe_polygon[0]], dtype=float)
        ax.plot(poly[:, 0], poly[:, 1], linewidth=2)

    for obs in obstacles:
        cx, cy = obs['center']
        circle = plt.Circle((cx, cy), obs['radius'], fill=False, linewidth=2)
        ax.add_patch(circle)
        ax.text(cx, cy, str(obs['id']))

    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Merged safe zone from V8 meeting-guard dual Multiranger mapper')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def merge_and_export(pts1_local, pts2_local):
    # V7 deterministic merge:
    # M1 starts +15 cm from center, yaw 0 deg.
    # M2 starts -15 cm from center, yaw 180 deg.
    # This avoids bad auto-shifting when the half maps are incomplete.
    pts1 = transform_points(pts1_local, MAPPER1_START_OFFSET_X, 0.0, 0.0)
    pts2 = transform_points(pts2_local, MAPPER2_START_OFFSET_X, 0.0, MAPPER2_YAW_DEG)

    offset_x = MAPPER2_START_OFFSET_X
    offset_y = 0.0

    print('')
    print('Deterministic center-start transform:')
    print(f'  M1 offset x={MAPPER1_START_OFFSET_X:.2f}, yaw=0.0 deg')
    print(f'  M2 offset x={MAPPER2_START_OFFSET_X:.2f}, yaw={MAPPER2_YAW_DEG:.1f} deg')
    print(f'  start separation={START_SEPARATION_M:.2f} m')
    print('')

    combined = np.vstack([pts1, pts2])
    filtered = filter_pointcloud_outliers(combined)
    boundary = estimate_boundary(filtered)

    if boundary is None:
        raise RuntimeError('Could not estimate merged boundary. Try a longer mapping time or simpler setup.')

    safe_min_x = boundary['min_x'] + SAFE_WALL_MARGIN
    safe_max_x = boundary['max_x'] - SAFE_WALL_MARGIN
    safe_min_y = boundary['min_y'] + SAFE_WALL_MARGIN
    safe_max_y = boundary['max_y'] - SAFE_WALL_MARGIN

    if safe_min_x >= safe_max_x or safe_min_y >= safe_max_y:
        raise RuntimeError('Merged safe zone is invalid. Check the map preview/transform.')

    room_boundary_polygon = [
        [boundary['min_x'], boundary['min_y']],
        [boundary['max_x'], boundary['min_y']],
        [boundary['max_x'], boundary['max_y']],
        [boundary['min_x'], boundary['max_y']],
    ]

    safe_zone_polygon = [
        [safe_min_x, safe_min_y],
        [safe_max_x, safe_min_y],
        [safe_max_x, safe_max_y],
        [safe_min_x, safe_max_y],
    ]

    obstacle_keepouts = []
    if EXPORT_MERGED_OBSTACLES:
        clusters = find_merged_obstacles(filtered, boundary)
        for i, cluster in enumerate(clusters):
            c = cluster['center']
            obstacle_keepouts.append({
                'id': i + 1,
                'center': [float(c[0]), float(c[1])],
                'radius': float(SAFE_OBSTACLE_MARGIN),
                'points': int(cluster['count']),
            })
    else:
        print('Obstacle export disabled for this test, so no automatic black keepout circles will be created.')

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    SAFE_ZONE_DIR.mkdir(exist_ok=True)

    csv_path = SAFE_ZONE_DIR / f'merged_filtered_pointcloud_{timestamp}.csv'
    json_path = SAFE_ZONE_DIR / f'safe_zone_merged_{timestamp}.json'
    png_path = SAFE_ZONE_DIR / f'safe_zone_merged_{timestamp}.png'

    np.savetxt(csv_path, filtered, delimiter=',', header='x,y,z', comments='')

    # AI start/split center is the center of the merged safe-zone.
    # In the two-mapper layout M1 is on one side, so using M1 as split makes A2's half empty.
    ai_center_xy = [
        float(0.5 * (safe_min_x + safe_max_x)),
        float(0.5 * (safe_min_y + safe_max_y)),
    ]

    output = {
        'description': 'Merged safe-zone map from V8 meeting-guard two simultaneous Multiranger drones controlled by one Crazyradio.',
        'created_at': timestamp,
        'mapper_setup': {
            'mapper1_uri': MAPPER1_URI,
            'mapper2_uri': MAPPER2_URI,
            'mapper1_height': MAPPER1_HEIGHT,
            'mapper2_height': MAPPER2_HEIGHT,
            'log_period_ms': LOG_PERIOD_MS,
            'ai_split_center_xy': ai_center_xy,
            'half_stop_enabled': HALF_STOP_ENABLED,
            'half_stop_yaw_deg': HALF_STOP_YAW_DEG,
            'export_merged_obstacles': EXPORT_MERGED_OBSTACLES,
            'center_start_setup': {
                'start_separation_m': START_SEPARATION_M,
                'mapper1_start_offset_x': MAPPER1_START_OFFSET_X,
                'mapper2_start_offset_x': MAPPER2_START_OFFSET_X,
                'mapper1_wall_side': MAPPER1_WALL_SIDE,
                'mapper2_wall_side': MAPPER2_WALL_SIDE,
                'back_sensor_ignore_seconds': BACK_SENSOR_IGNORE_SECONDS,
                'finish_target_radius': CENTER_FINISH_RADIUS,
                'meeting_guard_enabled': MEETING_GUARD_ENABLED,
                'meeting_guard_distance_m': MEETING_GUARD_DISTANCE_M,
                'meeting_guard_min_time_s': MEETING_GUARD_MIN_TIME_SECONDS,
                'meeting_guard_min_path_m': MEETING_GUARD_MIN_PATH_M,
                'meeting_guard_min_yaw_deg': MEETING_GUARD_MIN_YAW_DEG,
            },
            'mapper2_transform_into_mapper1_frame': {
                'mode': 'fixed_center_start',
                'estimated_offset_x': float(offset_x),
                'estimated_offset_y': float(offset_y),
                'yaw_deg': MAPPER2_YAW_DEG,
            },
        },
        'room_boundary_polygon': room_boundary_polygon,
        'safe_zone_polygon': safe_zone_polygon,
        'obstacle_keepouts': obstacle_keepouts,
        'start_position_xy': ai_center_xy,
        'source_files': {
            'merged_csv': str(csv_path),
            'safe_zone_png': str(png_path),
        },
        'rules_for_other_drones': {
            'stay_inside_safe_zone_polygon': True,
            'stay_outside_obstacle_keepouts': True,
            'note': 'start_position_xy is the center of the merged safe-zone. Place AI1 30 cm in front of this center and AI2 30 cm behind this center.',
        },
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    write_safezone_png(png_path, filtered, safe_zone_polygon, obstacle_keepouts)

    print('')
    print('Merged safe-zone created:')
    print(f'  CSV : {csv_path}')
    print(f'  JSON: {json_path}')
    print(f'  PNG : {png_path}')
    print(f'  obstacles exported: {len(obstacle_keepouts)}')
    print('')

    return str(json_path)


def meeting_guard_should_land(m1, m2):
    if not MEETING_GUARD_ENABLED:
        return False, None

    if m1.latest_global_xy is None or m2.latest_global_xy is None:
        return False, None

    ready = (
        m1.latest_elapsed >= MEETING_GUARD_MIN_TIME_SECONDS and
        m2.latest_elapsed >= MEETING_GUARD_MIN_TIME_SECONDS and
        m1.latest_path_m >= MEETING_GUARD_MIN_PATH_M and
        m2.latest_path_m >= MEETING_GUARD_MIN_PATH_M and
        m1.latest_yaw_sum_deg >= MEETING_GUARD_MIN_YAW_DEG and
        m2.latest_yaw_sum_deg >= MEETING_GUARD_MIN_YAW_DEG
    )

    if not ready:
        return False, None

    dx = m1.latest_global_xy[0] - m2.latest_global_xy[0]
    dy = m1.latest_global_xy[1] - m2.latest_global_xy[1]
    dist = math.hypot(dx, dy)

    if dist <= MEETING_GUARD_DISTANCE_M:
        return True, dist

    return False, dist


def main():
    logging.basicConfig(level=logging.ERROR)
    cflib.crtp.init_drivers()

    print('')
    print('One-radio dual Multiranger mapper - V8_MEETING_GUARD')
    print('SETUP:')
    print(f'  M1 URI: {MAPPER1_URI}')
    print(f'  M2 URI: {MAPPER2_URI}')
    print(f'  M1 height: {MAPPER1_HEIGHT:.2f} m')
    print(f'  M2 height: {MAPPER2_HEIGHT:.2f} m')
    print(f'  max mapping time: {MAPPING_TIME_SECONDS:.1f} s')
    print(f'  start separation: {START_SEPARATION_M:.2f} m')
    print(f'  wall sides: M1={MAPPER1_WALL_SIDE}, M2={MAPPER2_WALL_SIDE}')
    print(f'  finish target radius: {CENTER_FINISH_RADIUS:.2f} m')
    print(f'  meeting guard: {MEETING_GUARD_ENABLED}, distance={MEETING_GUARD_DISTANCE_M:.2f} m')
    print(f'  center-finish stop: {CENTER_FINISH_STOP_ENABLED}, min_yaw={CENTER_FINISH_MIN_YAW_DEG:.0f} deg')
    print(f'  obstacle export: {EXPORT_MERGED_OBSTACLES}')
    print(f'  log period: {LOG_PERIOD_MS} ms')
    print('')
    print('Safety behavior:')
    print('  Both drones connect first.')
    print('  If M2 cannot connect, M1 does NOT take off.')
    print('  If one mapper fails, the other mapper lands.')
    print('')

    MAPPER1_DIR.mkdir(parents=True, exist_ok=True)
    MAPPER2_DIR.mkdir(parents=True, exist_ok=True)

    both_ready_event = threading.Event()
    stop_event = threading.Event()
    emergency_event = threading.Event()

    start_keyboard_monitor(stop_event, emergency_event, label='MAPPER V4')

    m1 = MapperThread('MAPPER1', MAPPER1_URI, MAPPER1_HEIGHT, MAPPER1_DIR, both_ready_event, stop_event, emergency_event, wall_side=MAPPER1_WALL_SIDE, takeoff_delay=0.0)
    m2 = MapperThread('MAPPER2', MAPPER2_URI, MAPPER2_HEIGHT, MAPPER2_DIR, both_ready_event, stop_event, emergency_event, wall_side=MAPPER2_WALL_SIDE, takeoff_delay=MAPPER2_TAKEOFF_DELAY_SECONDS)

    print('Connecting MAPPER1 and MAPPER2 before takeoff...')
    m1.start()
    time.sleep(4.0)
    m2.start()

    deadline = time.time() + 25.0
    while time.time() < deadline:
        if m1.connected_event.is_set() and m2.connected_event.is_set():
            break
        if stop_event.is_set():
            break
        time.sleep(0.1)

    if not (m1.connected_event.is_set() and m2.connected_event.is_set()):
        print('')
        print('Both mappers did not connect. No takeoff will happen.')
        stop_event.set()
        m1.join()
        m2.join()
        if m1.error:
            raise RuntimeError(f'MAPPER1 failed before takeoff: {m1.error}')
        if m2.error:
            raise RuntimeError(f'MAPPER2 failed before takeoff: {m2.error}')
        raise RuntimeError('Dual mapper connection failed before takeoff.')

    print('')
    print('Both mappers connected. Starting takeoff/mapping.')
    both_ready_event.set()

    try:
        while m1.is_alive() or m2.is_alive():
            should_land, meeting_dist = meeting_guard_should_land(m1, m2)
            if should_land and not stop_event.is_set():
                print('')
                print(
                    f'MEETING GUARD: mappers are {meeting_dist:.2f} m apart after both completed enough path. '
                    'Landing both before they can cross/fly under each other.'
                )
                stop_event.set()

            m1.join(timeout=0.5)
            m2.join(timeout=0.5)
            if (m1.error or m2.error) and not stop_event.is_set():
                stop_event.set()
    except KeyboardInterrupt:
        print('')
        print('Ctrl+C pressed. Landing/stopping both mapper threads.')
        stop_event.set()
        raise

    if m1.error is not None:
        raise RuntimeError(f'MAPPER1 failed: {m1.error}')
    if m2.error is not None:
        raise RuntimeError(f'MAPPER2 failed: {m2.error}')

    pts1 = np.asarray(m1.points, dtype=float)
    pts2 = np.asarray(m2.points, dtype=float)

    if len(pts1) < 20:
        raise RuntimeError('MAPPER1 did not collect enough points.')
    if len(pts2) < 20:
        raise RuntimeError('MAPPER2 did not collect enough points.')

    merged_json = merge_and_export(pts1, pts2)

    print(f'MERGED_SAFE_ZONE_JSON={merged_json}')
    return merged_json


if __name__ == '__main__':
    main()
