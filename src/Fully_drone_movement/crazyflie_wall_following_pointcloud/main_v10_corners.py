#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V10 full mission:
  1. One-radio dual Multiranger mapping.
  2. Only if both mappers succeed, start the one-radio two-AI PARALLEL inspection.

Required files in same folder:
  dual_multiranger_single_radio_mapper_v3_safe.py
  ai_deck_dual_front_back_single_radio_v10_parallel_safe.py
  ai_deck_dual_front_back_inspection_v7_parallel_gate.py
  wall_following.py
"""

import subprocess
import sys
import time
from pathlib import Path


DUAL_MAPPER_SCRIPT = 'dual_multiranger_single_radio_mapper_v8_meeting_guard.py'
AI_SINGLE_RADIO_SCRIPT = 'ai_v10_corners.py'

WAIT_BEFORE_AI_TAKEOFF_SECONDS = 8.0


def countdown(seconds, text):
    for remaining in range(int(seconds), 0, -1):
        print(f'{text} in {remaining}...')
        time.sleep(1.0)


def terminate_process(process, name):
    if process is None or process.poll() is not None:
        return

    print(f'Closing {name} process...')
    try:
        process.terminate()
        try:
            process.wait(timeout=8.0)
            print(f'{name} process closed.')
        except subprocess.TimeoutExpired:
            print(f'{name} did not close, killing it.')
            process.kill()
            process.wait(timeout=5.0)
    except Exception as e:
        print(f'Could not terminate {name}: {e}')


def main():
    workdir = Path.cwd()
    mapper_path = workdir / DUAL_MAPPER_SCRIPT
    ai_path = workdir / AI_SINGLE_RADIO_SCRIPT

    if not mapper_path.exists():
        raise FileNotFoundError(f'Missing mapper script: {mapper_path}')
    if not ai_path.exists():
        raise FileNotFoundError(f'Missing AI script: {ai_path}')

    print('')
    print('FULL MISSION V10_CORNERS_AI_PARALLEL_SAFE: one-radio dual Multiranger mapping -> parallel AI inspection')
    print(f'Working directory: {workdir}')
    print('')
    print('SETUP CHECK:')
    print('  1. Use ONE Crazyradio dongle.')
    print('  2. Mappers must both connect before any takeoff.')
    print('  3. Multiranger addresses:')
    print('     M1: radio://0/80/2M/E7E7E7E701')
    print('     M2: radio://0/80/2M/E7E7E7E703')
    print('  4. Lower heights now:')
    print('     M1=0.20 m, M2=0.32 m, AI1=0.30 m, AI2=0.45 m')
    print('  5. AI parallel mode: AI2 connects first, then AI1; both wait on the ground, then launch together. Place AI1 75 cm forward and AI2 75 cm behind the merged center.')
    print('  6. During mapper/AI flight: L = land, E = emergency. For AI V10, L or SPACE lands both AI drones. Mappers start 30 cm apart; meeting guard lands both before they cross/fly under each other. AI drones fly together but stay in separated front/back halves.')
    print('  7. AI addresses:')
    print('     AI1: radio://0/84/2M/E7E7E7E702')
    print('     AI2: radio://0/84/2M/E7E7E7E704')
    print('')
    input('Press Enter to start V8 meeting-guard dual mapper... ')

    mapper_process = None
    ai_process = None

    try:
        mapper_process = subprocess.Popen([sys.executable, str(mapper_path)], cwd=str(workdir))
        mapper_return = mapper_process.wait()

        print('')
        print(f'V8 meeting-guard dual mapper finished with return code {mapper_return}.')
        if mapper_return != 0:
            print('Mapper failed, so AI inspection will not start.')
            return

        countdown(WAIT_BEFORE_AI_TAKEOFF_SECONDS, 'Parallel AI inspection starts')

        ai_process = subprocess.Popen([sys.executable, str(ai_path)], cwd=str(workdir))
        ai_return = ai_process.wait()

        print('')
        print(f'AI inspection finished with return code {ai_return}.')
        print('Full V10 stable parallel-AI mission finished.')

    except KeyboardInterrupt:
        print('')
        print('Ctrl+C pressed. Stopping mission.')
        terminate_process(mapper_process, 'V3 dual mapper')
        terminate_process(ai_process, 'dual AI')
        print('If any drone is still flying, use emergency stop / land immediately.')

    except Exception as e:
        print('')
        print(f'Full mission error: {type(e).__name__}: {e}')
        terminate_process(mapper_process, 'V3 dual mapper')
        terminate_process(ai_process, 'dual AI')


if __name__ == '__main__':
    main()
