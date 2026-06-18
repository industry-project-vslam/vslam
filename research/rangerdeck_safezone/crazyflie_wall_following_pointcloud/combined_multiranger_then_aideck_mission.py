#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combined Crazyflie mission launcher

This runs the full two-drone workflow:

1. Multiranger deck drone maps the room:
   crazyflie_multi_obstacle_safezone_auto_png.py
   URI usually: radio://0/80/2M/E7E7E7E701

2. When a new safe_zone_*.json appears in safe_zone_output,
   this launcher assumes the mapping mission is done and the multiranger drone has landed.

3. The launcher closes the mapping script because that script keeps the point-cloud
   window open after landing.

4. AI-deck drone starts its camera-coverage route:
   ai_deck_safezone_perimeter_object_inspection_planner_offset.py
   URI usually: radio://0/80/2M/E7E7E7E702

IMPORTANT SETUP:
- Put the multiranger drone on the mapping start spot.
- Put the AI-deck drone in front of the multiranger start spot.
- Both drones should face the SAME direction.
- Set AI_OFFSET_FORWARD_FROM_MAPPER_M to the real distance between the
  multiranger start spot and AI-deck start spot.
"""

import os
import sys
import time
import glob
import subprocess
from pathlib import Path


# -------------------------------------------------------------------------
# Files this launcher runs
# -------------------------------------------------------------------------

MAPPING_SCRIPT = 'crazyflie_multi_obstacle_safezone_auto_png.py'
AI_SCRIPT = 'ai_deck_safezone_perimeter_object_inspection_planner_offset.py'

SAFE_ZONE_DIR = 'safe_zone_output'


# -------------------------------------------------------------------------
# AI start offset
# -------------------------------------------------------------------------
# Positive X = AI drone is in front of the multiranger drone start position.
# Positive Y = AI drone is left of the multiranger drone start position.
#
# Example:
#   If AI deck drone is 20 cm in front of multiranger start:
#       AI_OFFSET_FORWARD_FROM_MAPPER_M = 0.20
#
# If you place both drones at exactly the same start spot:
#       AI_OFFSET_FORWARD_FROM_MAPPER_M = 0.00
AI_OFFSET_FORWARD_FROM_MAPPER_M = 0.20
AI_OFFSET_LEFT_FROM_MAPPER_M = 0.00


# -------------------------------------------------------------------------
# Wait settings
# -------------------------------------------------------------------------

SAFEZONE_STABLE_SECONDS = 3.0
MAX_MAPPING_WAIT_SECONDS = 360.0
WAIT_BEFORE_AI_TAKEOFF_SECONDS = 8.0


def newest_safezone_json():
    files = glob.glob(os.path.join(SAFE_ZONE_DIR, 'safe_zone_*.json'))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def newest_safezone_png():
    files = glob.glob(os.path.join(SAFE_ZONE_DIR, 'safe_zone_*.png'))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def wait_for_new_safezone(start_time):
    print('')
    print('Waiting for new safe-zone export from multiranger drone...')
    print(f'  folder: {SAFE_ZONE_DIR}')
    print('')

    last_candidate = None
    last_size = None
    stable_since = None
    deadline = time.time() + MAX_MAPPING_WAIT_SECONDS

    while time.time() < deadline:
        candidate = newest_safezone_json()

        if candidate is not None:
            mtime = os.path.getmtime(candidate)
            size = os.path.getsize(candidate)

            if mtime >= start_time:
                if candidate != last_candidate or size != last_size:
                    last_candidate = candidate
                    last_size = size
                    stable_since = time.time()
                else:
                    if stable_since is not None and (time.time() - stable_since) >= SAFEZONE_STABLE_SECONDS:
                        print('New safe-zone JSON detected and stable:')
                        print(f'  {candidate}')
                        png = newest_safezone_png()
                        if png is not None and os.path.getmtime(png) >= start_time:
                            print(f'  PNG preview: {png}')
                        print('')
                        return candidate

        time.sleep(1.0)

    raise TimeoutError('Timed out waiting for a new safe_zone_*.json file.')


def terminate_process(process, name):
    if process is None:
        return

    if process.poll() is not None:
        return

    print(f'Closing {name} process...')
    try:
        process.terminate()
        try:
            process.wait(timeout=8.0)
            print(f'{name} process closed.')
            return
        except subprocess.TimeoutExpired:
            print(f'{name} did not close, killing it.')
            process.kill()
            process.wait(timeout=5.0)
    except Exception as e:
        print(f'Could not terminate {name}: {e}')


def countdown(seconds):
    for remaining in range(int(seconds), 0, -1):
        print(f'AI deck mission starts in {remaining}...')
        time.sleep(1.0)


def main():
    workdir = Path.cwd()

    mapping_path = workdir / MAPPING_SCRIPT
    ai_path = workdir / AI_SCRIPT

    print('')
    print('Combined Crazyflie multiranger -> AI-deck mission')
    print(f'Working directory: {workdir}')
    print('')

    if not mapping_path.exists():
        raise FileNotFoundError(f'Missing mapping script: {mapping_path}')

    if not ai_path.exists():
        raise FileNotFoundError(f'Missing AI script: {ai_path}')

    os.makedirs(SAFE_ZONE_DIR, exist_ok=True)

    print('SETUP CHECK:')
    print('  1. Multiranger drone on the mapping start spot.')
    print('  2. AI-deck drone in front of it, facing same direction.')
    print(f'  3. AI offset forward = {AI_OFFSET_FORWARD_FROM_MAPPER_M:.2f} m')
    print(f'  4. AI offset left    = {AI_OFFSET_LEFT_FROM_MAPPER_M:.2f} m')
    print('  5. Keep your hand ready for emergency stop / Ctrl+C.')
    print('')
    input('Press Enter to start the multiranger mapping mission... ')

    start_time = time.time()

    print('')
    print('Starting multiranger mapping script...')
    print(f'  {MAPPING_SCRIPT}')
    print('')

    mapping_process = subprocess.Popen([sys.executable, str(mapping_path)], cwd=str(workdir))

    try:
        new_safezone = wait_for_new_safezone(start_time)

        # Mapping script keeps the point-cloud window open after landing.
        # Since the safe-zone JSON/PNG exists now, the mapping mission is done.
        terminate_process(mapping_process, 'multiranger mapping')

        print('')
        print('Multiranger part is finished.')
        print(f'Newest safe-zone file: {new_safezone}')
        print('')
        print('Make sure the AI-deck drone area is clear.')
        countdown(WAIT_BEFORE_AI_TAKEOFF_SECONDS)

        env = os.environ.copy()
        env['AI_START_OFFSET_X'] = str(AI_OFFSET_FORWARD_FROM_MAPPER_M)
        env['AI_START_OFFSET_Y'] = str(AI_OFFSET_LEFT_FROM_MAPPER_M)

        print('')
        print('Starting AI-deck perimeter/object inspection script...')
        print(f'  {AI_SCRIPT}')
        print('')

        ai_process = subprocess.Popen([sys.executable, str(ai_path)], cwd=str(workdir), env=env)
        ai_return = ai_process.wait()

        print('')
        print(f'AI-deck script finished with return code {ai_return}.')
        print('Combined mission finished.')

    except KeyboardInterrupt:
        print('')
        print('Ctrl+C pressed. Stopping launcher.')
        terminate_process(mapping_process, 'multiranger mapping')
        print('If the drone is still flying, use its emergency/land control immediately.')

    except Exception as e:
        print('')
        print(f'Combined mission error: {type(e).__name__}: {e}')
        terminate_process(mapping_process, 'multiranger mapping')
        print('Check that both scripts are in this folder and that the multiranger script exported a safe-zone JSON.')


if __name__ == '__main__':
    main()
