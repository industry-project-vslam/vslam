from __future__ import annotations

import math
from dataclasses import dataclass

from .ai_streams import AIStreamManager
from .config import default_half_group_config, default_swarm_config
from .drones import SimulationDrone
from .formation import FormationModel
from .logs import SwarmLogger
from .ranger import RangerMonitor, RangerReading
from .state_machine import RangerLedStreamingSwarm


@dataclass(frozen=True)
class FakeRoomScenario:
    name: str
    front: RangerReading
    back: RangerReading
    probe_after: float | None = None


def scenarios() -> dict[str, FakeRoomScenario]:
    return {
        "open_space": FakeRoomScenario(
            "open_space",
            front=RangerReading(front=4.0, back=4.0, left=3.0, right=3.0, up=2.5, valid=_valid()),
            back=RangerReading(front=4.0, back=4.0, left=3.0, right=3.0, up=2.5, valid=_valid()),
        ),
        "wall_ahead": FakeRoomScenario(
            "wall_ahead",
            front=RangerReading(front=0.80, back=4.0, left=2.0, right=2.4, up=2.5, valid=_valid()),
            back=RangerReading(front=4.0, back=3.0, left=2.0, right=2.0, up=2.5, valid=_valid()),
            probe_after=0.82,
        ),
        "local_obstacle": FakeRoomScenario(
            "local_obstacle",
            front=RangerReading(front=0.80, back=4.0, left=2.5, right=3.0, up=2.5, valid=_valid()),
            back=RangerReading(front=4.0, back=3.0, left=2.0, right=2.0, up=2.5, valid=_valid()),
            probe_after=4.2,
        ),
        "local_obstacle_left_clear": FakeRoomScenario(
            "local_obstacle_left_clear",
            front=RangerReading(front=0.80, back=4.0, left=3.0, right=2.5, up=2.5, valid=_valid()),
            back=RangerReading(front=4.0, back=3.0, left=2.0, right=2.0, up=2.5, valid=_valid()),
            probe_after=4.2,
        ),
        "front_critical": FakeRoomScenario(
            "front_critical",
            front=RangerReading(front=0.50, back=4.0, left=3.0, right=3.0, up=2.5, valid=_valid()),
            back=RangerReading(front=4.0, back=3.0, left=2.0, right=2.0, up=2.5, valid=_valid()),
        ),
        "front_critical_back_unknown": FakeRoomScenario(
            "front_critical_back_unknown",
            front=RangerReading(
                front=0.50,
                back=math.nan,
                left=3.0,
                right=3.0,
                up=2.5,
                valid={"front": True, "back": False, "left": True, "right": True, "up": True},
            ),
            back=RangerReading(front=4.0, back=3.0, left=2.0, right=2.0, up=2.5, valid=_valid()),
        ),
        "side_blocked": FakeRoomScenario(
            "side_blocked",
            front=RangerReading(front=0.80, back=4.0, left=0.40, right=0.40, up=2.5, valid=_valid()),
            back=RangerReading(front=4.0, back=3.0, left=2.0, right=2.0, up=2.5, valid=_valid()),
        ),
        "ambiguous_wide_obstacle": FakeRoomScenario(
            "ambiguous_wide_obstacle",
            front=RangerReading(front=0.80, back=4.0, left=1.0, right=1.0, up=2.5, valid=_valid()),
            back=RangerReading(front=4.0, back=3.0, left=1.0, right=1.0, up=2.5, valid=_valid()),
            probe_after=2.7,
        ),
        "side_unknown": FakeRoomScenario(
            "side_unknown",
            front=RangerReading(front=4.0, back=4.0, left=math.nan, right=3.0, up=2.5, valid={"front": True, "back": True, "left": False, "right": True, "up": True}),
            back=RangerReading(front=4.0, back=3.0, left=math.nan, right=3.0, up=2.5, valid={"front": True, "back": True, "left": False, "right": True, "up": True}),
        ),
        "turn_reslot_safe": FakeRoomScenario(
            "turn_reslot_safe",
            front=RangerReading(front=3.1, back=4.0, left=2.5, right=3.2, up=2.5, valid=_valid()),
            back=RangerReading(front=4.0, back=3.0, left=3.2, right=2.5, up=2.5, valid=_valid()),
            probe_after=3.15,
        ),
        "turn_reslot_one_side_blocked": FakeRoomScenario(
            "turn_reslot_one_side_blocked",
            front=RangerReading(front=3.1, back=4.0, left=3.2, right=0.6, up=2.5, valid=_valid()),
            back=RangerReading(front=4.0, back=3.0, left=0.6, right=3.2, up=2.5, valid=_valid()),
            probe_after=3.15,
        ),
    }


def build_simulated_swarm(
    scenario_name: str,
    log_root: str = "stream_out/sim_missions",
    half_group: bool = False,
) -> RangerLedStreamingSwarm:
    scenario = scenarios()[scenario_name]
    config = default_half_group_config() if half_group else default_swarm_config()
    enabled = [drone for drone in config.drones if drone.drone_id in {"X_FRONT", "X_BACK", "O1", "O2", "O3", "O4"}]
    config.drones = enabled
    formation = FormationModel(config.drones, formation_config=config.formation)
    drones = {drone.drone_id: SimulationDrone(drone) for drone in config.drones}
    monitor = RangerMonitor(config)
    monitor.update_front(scenario.front)
    monitor.update_back(scenario.back)
    logger = SwarmLogger(root=log_root, mission_id=f"sim_{scenario.name}")
    streams = AIStreamManager()
    return RangerLedStreamingSwarm(config, formation, drones, monitor, streams, logger)


def run_simulation(scenario_name: str = "open_space", steps: int = 3, half_group: bool = False) -> None:
    swarm = build_simulated_swarm(scenario_name, half_group=half_group)
    try:
        swarm.run(max_steps=steps)
        mode = "half-group" if half_group else "full"
        print(f"Simulation complete: {scenario_name}, mode={mode}, final_state={swarm.state.value}, logs={swarm.logger.run_dir}")
    finally:
        swarm.logger.close()


def _valid() -> dict[str, bool]:
    return {"front": True, "back": True, "left": True, "right": True, "up": True}


if __name__ == "__main__":
    run_simulation()
