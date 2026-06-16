#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-process / one-Crazyradio dual AI-deck front/back inspection launcher V10 PARALLEL SAFE.

Goal:
  - Start AI1 and AI2 in one Python process, using one Crazyradio.
  - Make both AI drones connect first.
  - Keep both drones on the ground until both are ready.
  - Open one shared start gate so both take off and inspect their own half at the same time.

Required companion file in the same folder:
  ai_deck_dual_front_back_inspection_v7_parallel_gate.py

Recommended for one Crazyradio:
  AI1_URI = radio://0/84/2M/E7E7E7E702
  AI2_URI = radio://0/84/2M/E7E7E7E704

Both AI drones should use the same channel/datarate and different addresses.
"""

import importlib.util
import os
import sys
import threading
import time
from pathlib import Path

import cflib.crtp


# -------------------------------------------------------------------------
# Companion AI mission script with the parallel-start gate support
# -------------------------------------------------------------------------
AI_WORKER_SCRIPT = 'ai_worker_v10_corners.py'


# -------------------------------------------------------------------------
# URIs for ONE Crazyradio dongle
# -------------------------------------------------------------------------
AI1_URI = 'radio://0/84/2M/E7E7E7E702'
AI2_URI = 'radio://0/84/2M/E7E7E7E704'


# -------------------------------------------------------------------------
# Physical start offsets relative to the merged safe-zone center
# Positive X = front direction, same direction as MAPPER1
# Negative X = back direction
# -------------------------------------------------------------------------
AI1_START_OFFSET_X = 0.75
AI1_START_OFFSET_Y = 0.00

AI2_START_OFFSET_X = -0.75
AI2_START_OFFSET_Y = 0.00


# -------------------------------------------------------------------------
# Heights
# -------------------------------------------------------------------------
AI1_TAKEOFF_HEIGHT = 0.30
AI2_TAKEOFF_HEIGHT = 0.45


# -------------------------------------------------------------------------
# Speeds
# -------------------------------------------------------------------------
AI_ENTRY_SPEED = 0.055
AI_SCAN_SPEED = 0.06
AI_INSPECT_SPEED = 0.055
AI_RETURN_SPEED = 0.06
AI_RECOVERY_SPEED = 0.06


# -------------------------------------------------------------------------
# Parallel start behavior
# -------------------------------------------------------------------------
AI2_THREAD_CONNECT_DELAY_SECONDS = 2.0
AI_READY_TIMEOUT_SECONDS = 75.0
BOTH_READY_SETTLE_SECONDS = 1.0

# Keep a large center no-fly barrier. With this value:
#   AI1/front may only use x >= center + 0.65 m
#   AI2/back  may only use x <= center - 0.65 m
FRONT_BACK_SPLIT_BUFFER = 0.35
AI_PATH_EDGE_MARGIN = 0.25
AI_RUNTIME_EDGE_MARGIN = 0.08

MISSION_CONTROL_DIR = Path('mission_control')
LAND_NOW_FILE = MISSION_CONTROL_DIR / 'land_now.flag'
EMERGENCY_FILE = MISSION_CONTROL_DIR / 'emergency_stop.flag'
AI1_READY_FILE = MISSION_CONTROL_DIR / 'ai1_ready.flag'
AI2_READY_FILE = MISSION_CONTROL_DIR / 'ai2_ready.flag'
AI_START_FILE = MISSION_CONTROL_DIR / 'ai_parallel_start.flag'


def clear_control_files():
    MISSION_CONTROL_DIR.mkdir(exist_ok=True)
    for path in (LAND_NOW_FILE, EMERGENCY_FILE, AI1_READY_FILE, AI2_READY_FILE, AI_START_FILE):
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def request_land():
    LAND_NOW_FILE.write_text('land now\n', encoding='utf-8')


def request_emergency():
    EMERGENCY_FILE.write_text('emergency stop\n', encoding='utf-8')


def start_keyboard_monitor():
    def monitor_windows():
        try:
            import msvcrt
            print('')
            print('[AI V10] KEYBOARD CONTROLS:')
            print('  press L or SPACE = smooth land both AI drones')
            print('  press E          = EMERGENCY stop/disarm both AI drones')
            print('')
            while True:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch().lower()
                    if ch == 'l' or ch == ' ':
                        request_land()
                        print('[AI V10] LAND key pressed. Both AI drones will land now.')
                        return
                    if ch == 'e':
                        request_emergency()
                        print('[AI V10] EMERGENCY key pressed. Both AI drones will stop/disarm now.')
                        return
                time.sleep(0.05)
        except Exception as exc:
            print(f'[AI V10] Keyboard monitor unavailable: {exc}')

    def monitor_portable():
        print('')
        print('[AI V10] Keyboard controls: type l + Enter to land, e + Enter for emergency.')
        print('')
        while True:
            try:
                line = sys.stdin.readline().strip().lower()
            except Exception:
                return
            if line == 'l' or line == 'land':
                request_land()
                print('[AI V10] LAND command received. Both AI drones will land now.')
                return
            if line == 'e' or line == 'emergency':
                request_emergency()
                print('[AI V10] EMERGENCY command received. Both AI drones will stop/disarm now.')
                return

    target = monitor_windows if os.name == 'nt' else monitor_portable
    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t


def load_worker_module(worker_path, module_name, env_values):
    """
    Load the same worker script twice as two separate module objects.
    This gives AI1 and AI2 separate global variables while staying in one process.
    """
    old_env = os.environ.copy()
    os.environ.update({k: str(v) for k, v in env_values.items()})

    try:
        spec = importlib.util.spec_from_file_location(module_name, str(worker_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Could not load worker module from {worker_path}')

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    finally:
        os.environ.clear()
        os.environ.update(old_env)


def run_module_main(module, label, errors):
    try:
        module.main()
    except BaseException as exc:
        errors.append((label, exc))
        print('')
        print(f'[{label}] THREAD ERROR: {type(exc).__name__}: {exc}')
        print('')


def wait_until_both_ai_ready(t1, t2, errors):
    deadline = time.time() + AI_READY_TIMEOUT_SECONDS

    while time.time() < deadline:
        if errors:
            raise RuntimeError('One AI thread reported an error before both drones were ready.')

        if not t1.is_alive() or not t2.is_alive():
            raise RuntimeError('One AI thread stopped before both drones were ready.')

        if AI1_READY_FILE.exists() and AI2_READY_FILE.exists():
            return

        time.sleep(0.10)

    raise TimeoutError('Timed out waiting for both AI drones to connect and become ready.')


def main():
    workdir = Path.cwd()
    worker_path = workdir / AI_WORKER_SCRIPT

    print('')
    print('One-process dual AI-deck inspection V10_CORNERS_PARALLEL_SAFE')
    print(f'Working directory: {workdir}')
    print('')

    if not worker_path.exists():
        raise FileNotFoundError(f'Missing companion AI worker script: {worker_path}')

    print('SETUP CHECK:')
    print('  1. This script uses ONE Crazyradio dongle.')
    print('  2. Both AI URIs use radio://0.')
    print(f'     AI 1 URI: {AI1_URI}')
    print(f'     AI 2 URI: {AI2_URI}')
    print('  3. Recommended: AI1 and AI2 use the same channel/datarate but different addresses.')
    print('  4. AI1 starts 75 cm in front of the merged safe-zone center.')
    print('  5. AI2 starts 75 cm behind the merged safe-zone center.')
    print('  6. AI1 and AI2 connect first, wait on the ground, then take off together.')
    print('  7. AI1 stays in the front half, AI2 stays in the back half.')
    print('  8. V10 stable: AI2 starts first, verifies takeoff, uses soft safety, and skips stuck targets.')
    print('  9. Keyboard: L or SPACE = smooth land both AI drones, E = emergency stop/disarm.')
    print('')

    clear_control_files()
    start_keyboard_monitor()

    # Initialize drivers once in this process.
    cflib.crtp.init_drivers()

    # The worker script also calls cflib.crtp.init_drivers().
    # After initializing once here, make further calls harmless no-ops.
    cflib.crtp.init_drivers = lambda *args, **kwargs: None

    common_env = {
        'AI_ENTRY_SPEED': AI_ENTRY_SPEED,
        'AI_SCAN_SPEED': AI_SCAN_SPEED,
        'AI_INSPECT_SPEED': AI_INSPECT_SPEED,
        'AI_RETURN_SPEED': AI_RETURN_SPEED,
        'AI_RECOVERY_SPEED': AI_RECOVERY_SPEED,
        'MISSION_CONTROL_DIR': MISSION_CONTROL_DIR,
        'FRONT_BACK_SPLIT_BUFFER': FRONT_BACK_SPLIT_BUFFER,
        'AI_PATH_EDGE_MARGIN': AI_PATH_EDGE_MARGIN,
        'AI_RUNTIME_EDGE_MARGIN': AI_RUNTIME_EDGE_MARGIN,
        'AI_START_FILE': AI_START_FILE,
        'AI_START_TIMEOUT_SECONDS': AI_READY_TIMEOUT_SECONDS,
        'AI_TARGET_TIMEOUT_SECONDS': os.environ.get('AI_TARGET_TIMEOUT_SECONDS', '22.0'),
        'AI_TARGET_STUCK_SECONDS': os.environ.get('AI_TARGET_STUCK_SECONDS', '8.0'),
        'AI_TARGET_PROGRESS_EPS_M': os.environ.get('AI_TARGET_PROGRESS_EPS_M', '0.07'),
        'AI_RUNTIME_SPLIT_BUFFER': os.environ.get('AI_RUNTIME_SPLIT_BUFFER', '0.20'),
        'AI_HARD_EDGE_MARGIN': os.environ.get('AI_HARD_EDGE_MARGIN', '0.04'),
        'AI_RUNTIME_OBSTACLE_MARGIN': os.environ.get('AI_RUNTIME_OBSTACLE_MARGIN', '0.05'),
        'AI_CORNER_MARGIN': os.environ.get('AI_CORNER_MARGIN', '0.32'),
        'AI_PERIMETER_SPACING': os.environ.get('AI_PERIMETER_SPACING', '0.75'),
        'AI_SPARSE_INTERIOR': os.environ.get('AI_SPARSE_INTERIOR', '1'),
    }

    mod1_env = {
        **common_env,
        'AI_DRONE_ID': 1,
        'AI_URI': AI1_URI,
        'AI1_START_OFFSET_X': AI1_START_OFFSET_X,
        'AI1_START_OFFSET_Y': AI1_START_OFFSET_Y,
        'AI1_TAKEOFF_HEIGHT': AI1_TAKEOFF_HEIGHT,
        'AI_READY_FILE': AI1_READY_FILE,
    }

    mod2_env = {
        **common_env,
        'AI_DRONE_ID': 2,
        'AI_URI': AI2_URI,
        'AI2_START_OFFSET_X': AI2_START_OFFSET_X,
        'AI2_START_OFFSET_Y': AI2_START_OFFSET_Y,
        'AI2_TAKEOFF_HEIGHT': AI2_TAKEOFF_HEIGHT,
        'AI_READY_FILE': AI2_READY_FILE,
    }

    ai1_module = load_worker_module(worker_path, 'ai_worker_front_a1_parallel', mod1_env)
    ai2_module = load_worker_module(worker_path, 'ai_worker_back_a2_parallel', mod2_env)

    errors = []
    t1 = threading.Thread(target=run_module_main, args=(ai1_module, 'AI1/front', errors), daemon=False)
    t2 = threading.Thread(target=run_module_main, args=(ai2_module, 'AI2/back', errors), daemon=False)

    try:
        print('')
        print('Starting AI threads. They will connect first and wait on the ground.')
        print('')

        print('Starting AI2/back connection thread first...')
        t2.start()
        time.sleep(AI2_THREAD_CONNECT_DELAY_SECONDS)

        print('Starting AI1/front connection thread...')
        t1.start()

        print('Waiting until both AI drones are connected and ready...')
        wait_until_both_ai_ready(t1, t2, errors)

        print('')
        print('Both AI drones are ready on the ground.')
        print(f'Opening parallel takeoff gate in {BOTH_READY_SETTLE_SECONDS:.1f}s...')
        time.sleep(BOTH_READY_SETTLE_SECONDS)

        AI_START_FILE.write_text('start both ai drones\n', encoding='utf-8')
        print('Parallel takeoff gate opened. AI1 and AI2 should take off now.')
        print('')

        while t1.is_alive() or t2.is_alive():
            t1.join(timeout=0.5)
            t2.join(timeout=0.5)

    except KeyboardInterrupt:
        print('')
        print('Ctrl+C pressed. Requesting smooth land for both AI drones.')
        request_land()
        while t1.is_alive() or t2.is_alive():
            t1.join(timeout=0.5)
            t2.join(timeout=0.5)
        raise

    except Exception as exc:
        print('')
        print(f'AI parallel launcher error: {type(exc).__name__}: {exc}')
        print('Requesting smooth land for both AI drones.')
        request_land()
        while t1.is_alive() or t2.is_alive():
            t1.join(timeout=0.5)
            t2.join(timeout=0.5)

    print('')
    print('Parallel AI inspection finished.')

    if errors:
        print('')
        print('One or more AI threads reported an error:')
        for label, exc in errors:
            print(f'  {label}: {type(exc).__name__}: {exc}')


if __name__ == '__main__':
    main()
