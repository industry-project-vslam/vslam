from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Frontier:
    frontier_id: str
    timestamp: float
    heading: float
    state: str
    front_initial: float
    probe_direction: str
    front_after: float
    classification: str
    confidence: str
    status: str
    breadcrumb_index: int


@dataclass(frozen=True)
class Breadcrumb:
    seq: int
    timestamp: float
    command: str
    distance: float
    heading: float
    formation_center_estimate: tuple[float, float]
    formation_mode: str
    ranger_snapshot_id: int | None = None


@dataclass
class FrontierManager:
    frontiers: list[Frontier] = field(default_factory=list)
    breadcrumbs: list[Breadcrumb] = field(default_factory=list)

    def add_breadcrumb(
        self,
        seq: int,
        command: str,
        distance: float,
        heading: float,
        formation_center_estimate: tuple[float, float],
        formation_mode: str,
        ranger_snapshot_id: int | None = None,
    ) -> Breadcrumb:
        breadcrumb = Breadcrumb(
            seq=seq,
            timestamp=time.time(),
            command=command,
            distance=distance,
            heading=heading,
            formation_center_estimate=formation_center_estimate,
            formation_mode=formation_mode,
            ranger_snapshot_id=ranger_snapshot_id,
        )
        self.breadcrumbs.append(breadcrumb)
        return breadcrumb

    def save_frontier(
        self,
        heading: float,
        state: str,
        front_initial: float,
        probe_direction: str,
        front_after: float,
        classification: str,
        confidence: str,
        status: str,
    ) -> Frontier:
        frontier = Frontier(
            frontier_id=f"frontier_{len(self.frontiers) + 1:04d}",
            timestamp=time.time(),
            heading=heading,
            state=state,
            front_initial=front_initial,
            probe_direction=probe_direction,
            front_after=front_after,
            classification=classification,
            confidence=confidence,
            status=status,
            breadcrumb_index=max(0, len(self.breadcrumbs) - 1),
        )
        self.frontiers.append(frontier)
        return frontier

    def update_frontier(self, frontier_id: str, status: str) -> Frontier | None:
        for index, frontier in enumerate(self.frontiers):
            if frontier.frontier_id == frontier_id:
                updated = Frontier(**{**asdict(frontier), "status": status})
                self.frontiers[index] = updated
                return updated
        return None

    def list_unresolved(self) -> list[Frontier]:
        resolved = {"OBSTACLE_BYPASSED", "WALL_CONFIRMED"}
        return [frontier for frontier in self.frontiers if frontier.status not in resolved]

    def export_frontiers(self) -> list[dict[str, object]]:
        return [asdict(frontier) for frontier in self.frontiers]
