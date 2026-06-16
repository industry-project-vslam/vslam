"""Small Crazyflie safety test.

This script arms the drone, takes off slowly to 25 cm, hovers briefly, and lands.
Use it only in an open test area with a Flow deck attached.
"""

from __future__ import annotations

import logging
import os
import select
import sys
import time
from threading import Event, Thread

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper


URI = uri_helper.uri_from_env(default="usb://0")

TARGET_HEIGHT = 0.25
ASCEND_RATE = 0.03
DESCEND_RATE = 0.03
HOVER_TIME = 3.0
COMMAND_PERIOD = 0.1

deck_attached_event = Event()
emergency_event = Event()

logging.basicConfig(level=logging.ERROR)


def emergency_stop(cf: Crazyflie) -> None:
    print("\nEmergency stop: motors off")
    for _ in range(10):
        cf.commander.send_stop_setpoint()
        time.sleep(0.05)


def keyboard_listener() -> None:
    if os.name == "nt":
        import msvcrt

        while not emergency_event.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getwch().lower()
                if key in ("e", "q"):
                    emergency_event.set()
                    break
            time.sleep(0.05)
        return

    import termios
    import tty

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while not emergency_event.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if readable and sys.stdin.read(1).lower() in ("e", "q"):
                emergency_event.set()
                break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def flow_deck_callback(_name: str, value_str: str) -> None:
    if int(value_str):
        print("Flow deck detected")
        deck_attached_event.set()
    else:
        print("Flow deck not detected")


def reset_estimator(cf: Crazyflie) -> None:
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    time.sleep(2.0)


def send_hover(cf: Crazyflie, height: float) -> None:
    cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, height)


def slow_takeoff(cf: Crazyflie) -> bool:
    print("Slow takeoff. Press e or q for emergency stop.")
    height = 0.05

    while height < TARGET_HEIGHT:
        if emergency_event.is_set():
            emergency_stop(cf)
            return False

        send_hover(cf, height)
        print(f"height command: {height:.2f} m")
        height += ASCEND_RATE * COMMAND_PERIOD
        time.sleep(COMMAND_PERIOD)

    return True


def hover(cf: Crazyflie) -> bool:
    print("Hovering. Press e or q for emergency stop.")
    start = time.time()

    while time.time() - start < HOVER_TIME:
        if emergency_event.is_set():
            emergency_stop(cf)
            return False

        send_hover(cf, TARGET_HEIGHT)
        time.sleep(COMMAND_PERIOD)

    return True


def slow_land(cf: Crazyflie) -> bool:
    print("Slow landing. Press e or q for emergency stop.")
    height = TARGET_HEIGHT

    while height > 0.03:
        if emergency_event.is_set():
            emergency_stop(cf)
            return False

        send_hover(cf, height)
        print(f"height command: {height:.2f} m")
        height -= DESCEND_RATE * COMMAND_PERIOD
        time.sleep(COMMAND_PERIOD)

    for _ in range(10):
        send_hover(cf, 0.02)
        time.sleep(COMMAND_PERIOD)

    cf.commander.send_stop_setpoint()
    print("Landed. Motors off.")
    return True


def main() -> None:
    print("Starting safe Crazyflie hover test")
    print(f"Connecting to Crazyflie on {URI}")
    print("Emergency key: e or q")

    cflib.crtp.init_drivers()

    keyboard_thread = Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache")) as scf:
        cf = scf.cf
        cf.param.add_update_callback(group="deck", name="bcFlow2", cb=flow_deck_callback)

        if not deck_attached_event.wait(timeout=5):
            print("No Flow deck detected. Stop.")
            sys.exit(1)

        reset_estimator(cf)
        cf.supervisor.send_arming_request(True)
        time.sleep(1.0)

        try:
            if slow_takeoff(cf) and hover(cf):
                slow_land(cf)
        except KeyboardInterrupt:
            emergency_event.set()
            emergency_stop(cf)
        finally:
            cf.commander.send_stop_setpoint()
            print("Program finished.")


if __name__ == "__main__":
    main()
