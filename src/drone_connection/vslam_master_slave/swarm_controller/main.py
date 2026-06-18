from __future__ import annotations

import argparse
from dataclasses import replace

from .ai_streams import AIStreamManager
from .config import DroneRole, default_half_group_config, default_swarm_config
from .drones import CrazyflieDrone, SimulationDrone
from .formation import FormationModel
from .logs import SwarmLogger
from .ranger import RangerMonitor
from .simulation_stub import run_simulation
from .state_machine import RangerLedStreamingSwarm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-formation Ranger-led streaming swarm MVP")
    parser.add_argument("--simulate", action="store_true", help="Run with simulated drones and fake Ranger readings")
    parser.add_argument("--half-group", action="store_true", help="Use X_FRONT + O1 + O2 only")
    parser.add_argument("--scenario", default="open_space", help="Simulation scenario")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--mission-id", default="fixed_swarm")
    parser.add_argument("--log-root", default="stream_out/fixed_formation_missions")
    parser.add_argument("--x-front-uri", default="radio://0/82/2M/E7E7E7E701")
    parser.add_argument("--x-back-uri", default="")
    parser.add_argument("--o1-uri", default="")
    parser.add_argument("--o2-uri", default="")
    parser.add_argument("--o3-uri", default="")
    parser.add_argument("--o4-uri", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.simulate:
        run_simulation(args.scenario, steps=args.steps, half_group=args.half_group)
        return

    config = default_half_group_config() if args.half_group else default_swarm_config()
    uri_by_id = {
        "X_FRONT": args.x_front_uri,
        "X_BACK": args.x_back_uri,
        "O1": args.o1_uri,
        "O2": args.o2_uri,
        "O3": args.o3_uri,
        "O4": args.o4_uri,
    }
    config.drones = [
        drone
        for drone in config.drones
        if uri_by_id.get(drone.drone_id) or drone.role == DroneRole.FRONT_RANGER
    ]
    for index, drone in enumerate(config.drones):
        config.drones[index] = replace(drone, uri=uri_by_id.get(drone.drone_id, drone.uri))

    formation = FormationModel(config.drones, formation_config=config.formation)
    drones = {drone.drone_id: CrazyflieDrone(drone, rate_hz=config.setpoint_hz) for drone in config.drones}
    ranger_monitor = RangerMonitor(config)
    logger = SwarmLogger(root=args.log_root, mission_id=args.mission_id)
    streams = AIStreamManager()

    swarm = RangerLedStreamingSwarm(config, formation, drones, ranger_monitor, streams, logger)
    try:
        swarm.run(max_steps=args.steps)
    finally:
        logger.close()
        for drone in drones.values():
            close = getattr(drone, "close", None)
            if close is not None:
                close()


def build_simulation_drones() -> dict[str, SimulationDrone]:
    config = default_swarm_config()
    return {drone.drone_id: SimulationDrone(drone) for drone in config.drones}


if __name__ == "__main__":
    main()
