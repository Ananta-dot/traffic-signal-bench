"""
Traffic simulation engine for a 2x2 intersection grid.

Pure Python, no external dependencies beyond stdlib.
Vehicles arrive stochastically (Poisson), move through intersections
when their lane has green, one vehicle per timestep per lane.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from models import Direction, SignalPhase, IncidentType


def _poisson(rng: random.Random, lam: float) -> int:
    """Sample from Poisson distribution using Knuth's algorithm.
    
    Pure stdlib — no numpy needed.
    """
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p < L:
            return k - 1


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIRECTIONS = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]
STEP_SECONDS = 5  # Each simulation step = 5 real seconds
MAX_QUEUE = 50     # Cap per lane to prevent unbounded growth


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Vehicle:
    id: int
    arrival_step: int
    direction: Direction
    is_left_turn: bool = False
    is_emergency: bool = False
    cleared: bool = False


@dataclass
class Intersection:
    id: str
    row: int
    col: int
    current_phase: SignalPhase = SignalPhase.NS_GREEN
    phase_timer: int = 30  # seconds remaining in current phase
    phase_start_step: int = 0

    # Queues: direction -> list of vehicles
    through_queues: dict[Direction, list[Vehicle]] = field(default_factory=dict)
    left_turn_queues: dict[Direction, list[Vehicle]] = field(default_factory=dict)

    # Pedestrian state: crossing_id -> (count_waiting, max_wait_seconds)
    pedestrian_waiting: dict[str, int] = field(default_factory=dict)
    pedestrian_max_wait: dict[str, int] = field(default_factory=dict)

    # Track last green time per direction for starvation detection
    last_green_step: dict[Direction, int] = field(default_factory=dict)

    # Track phase changes for flicker detection
    phase_change_history: list[int] = field(default_factory=list)

    # Blocked lanes due to incidents
    blocked_lanes: dict[Direction, bool] = field(default_factory=dict)

    def __post_init__(self):
        for d in DIRECTIONS:
            self.through_queues.setdefault(d, [])
            self.left_turn_queues.setdefault(d, [])
            self.last_green_step.setdefault(d, 0)
            self.blocked_lanes.setdefault(d, False)
        self.pedestrian_waiting.setdefault("ns", 0)
        self.pedestrian_waiting.setdefault("ew", 0)
        self.pedestrian_max_wait.setdefault("ns", 0)
        self.pedestrian_max_wait.setdefault("ew", 0)


@dataclass
class Incident:
    incident_type: IncidentType
    intersection_id: str
    direction: Optional[Direction]
    start_step: int
    duration_steps: int  # -1 = until resolved
    description: str
    active: bool = True


# ---------------------------------------------------------------------------
# Simulation Engine
# ---------------------------------------------------------------------------

class TrafficSimulator:
    """Manages a 2x2 grid of intersections with vehicle and pedestrian flow."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.step_count = 0
        self.total_cleared = 0
        self.vehicle_counter = 0
        self.intersections: dict[str, Intersection] = {}
        self.incidents: list[Incident] = []
        self.replay_log: list[dict] = []

        # Arrival rates (vehicles per step per direction) — vary by time of day
        self.base_arrival_rates: dict[Direction, float] = {
            Direction.NORTH: 0.3,
            Direction.SOUTH: 0.3,
            Direction.EAST: 0.25,
            Direction.WEST: 0.25,
        }

        # Build 2x2 grid
        for r in range(2):
            for c in range(2):
                iid = f"int_{r}_{c}"
                self.intersections[iid] = Intersection(id=iid, row=r, col=c)

    def reset(self, seed: Optional[int] = None):
        """Reset simulation to initial state."""
        if seed is not None:
            self.rng = random.Random(seed)
        self.step_count = 0
        self.total_cleared = 0
        self.vehicle_counter = 0
        self.incidents = []
        self.replay_log = []
        for iid in self.intersections:
            self.intersections[iid] = Intersection(
                id=iid,
                row=self.intersections[iid].row,
                col=self.intersections[iid].col,
            )

    def get_time_of_day(self, start_hour: int = 7) -> tuple[str, int]:
        """Return current time as HH:MM string and minutes since midnight."""
        total_seconds = start_hour * 3600 + self.step_count * STEP_SECONDS
        hours = (total_seconds // 3600) % 24
        minutes = (total_seconds % 3600) // 60
        return f"{hours:02d}:{minutes:02d}", hours * 60 + minutes

    def _arrival_rate_multiplier(self, time_minutes: int) -> dict[Direction, float]:
        """Time-of-day traffic pattern multipliers."""
        hour = time_minutes / 60
        multipliers = {}
        for d in DIRECTIONS:
            base = 1.0
            if d in (Direction.NORTH, Direction.SOUTH):
                # Morning rush NS heavy (7-9am), evening lighter
                if 7 <= hour <= 9:
                    base = 1.8
                elif 17 <= hour <= 19:
                    base = 1.2
            else:
                # Evening rush EW heavy (5-7pm), morning lighter
                if 7 <= hour <= 9:
                    base = 1.0
                elif 17 <= hour <= 19:
                    base = 1.8
            multipliers[d] = base
        return multipliers

    def _spawn_vehicles(self, intersection: Intersection, time_minutes: int):
        """Spawn new vehicles based on Poisson arrival rates."""
        multipliers = self._arrival_rate_multiplier(time_minutes)
        for d in DIRECTIONS:
            if intersection.blocked_lanes.get(d, False):
                # Reduced arrivals on blocked lanes (vehicles reroute)
                rate = self.base_arrival_rates[d] * multipliers[d] * 0.3
            else:
                rate = self.base_arrival_rates[d] * multipliers[d]

            # Poisson arrival: number of vehicles this step
            n_arrivals = _poisson(self.rng, rate)
            for _ in range(n_arrivals):
                if len(intersection.through_queues[d]) + len(intersection.left_turn_queues[d]) >= MAX_QUEUE:
                    continue
                self.vehicle_counter += 1
                is_left = self.rng.random() < 0.2  # 20% left turns
                v = Vehicle(
                    id=self.vehicle_counter,
                    arrival_step=self.step_count,
                    direction=d,
                    is_left_turn=is_left,
                )
                if is_left:
                    intersection.left_turn_queues[d].append(v)
                else:
                    intersection.through_queues[d].append(v)

    def _spawn_pedestrians(self, intersection: Intersection):
        """Spawn pedestrians wanting to cross."""
        for crossing in ["ns", "ew"]:
            if self.rng.random() < 0.15:  # 15% chance per step
                intersection.pedestrian_waiting[crossing] += self.rng.randint(1, 3)

    def _green_directions(self, phase: SignalPhase) -> tuple[set[Direction], bool, bool]:
        """Return (directions with green, allows_through, allows_left)."""
        if phase == SignalPhase.NS_GREEN:
            return {Direction.NORTH, Direction.SOUTH}, True, False
        elif phase == SignalPhase.EW_GREEN:
            return {Direction.EAST, Direction.WEST}, True, False
        elif phase == SignalPhase.NS_LEFT_ARROW:
            return {Direction.NORTH, Direction.SOUTH}, False, True
        elif phase == SignalPhase.EW_LEFT_ARROW:
            return {Direction.EAST, Direction.WEST}, False, True
        elif phase == SignalPhase.PEDESTRIAN:
            return set(), False, False  # No vehicles move during ped phase
        elif phase == SignalPhase.ALL_RED:
            return set(), False, False
        return set(), False, False

    def _process_vehicles(self, intersection: Intersection) -> int:
        """Move vehicles through intersection based on current phase. Returns count cleared."""
        green_dirs, allows_through, allows_left = self._green_directions(intersection.current_phase)
        cleared = 0

        for d in green_dirs:
            if intersection.blocked_lanes.get(d, False):
                continue

            # Through traffic: clear up to 3 vehicles per step per direction
            if allows_through:
                for _ in range(min(3, len(intersection.through_queues[d]))):
                    v = intersection.through_queues[d].pop(0)
                    v.cleared = True
                    cleared += 1
                    if v.is_emergency:
                        cleared += 5  # Bonus for clearing emergency vehicles

            # Left turn traffic: clear 1 per step per direction
            if allows_left:
                if intersection.left_turn_queues[d]:
                    v = intersection.left_turn_queues[d].pop(0)
                    v.cleared = True
                    cleared += 1
                # Also allow 1 through vehicle during left arrow phase
                if intersection.through_queues[d]:
                    v = intersection.through_queues[d].pop(0)
                    v.cleared = True
                    cleared += 1

            intersection.last_green_step[d] = self.step_count

        return cleared

    def _process_pedestrians(self, intersection: Intersection):
        """Handle pedestrian crossings during pedestrian phase."""
        if intersection.current_phase == SignalPhase.PEDESTRIAN:
            intersection.pedestrian_waiting["ns"] = 0
            intersection.pedestrian_waiting["ew"] = 0
            intersection.pedestrian_max_wait["ns"] = 0
            intersection.pedestrian_max_wait["ew"] = 0
        elif intersection.current_phase == SignalPhase.NS_GREEN:
            # NS traffic moving, EW pedestrians can't cross — wait grows
            intersection.pedestrian_max_wait["ew"] += STEP_SECONDS
            # NS pedestrians can cross parallel to traffic
            intersection.pedestrian_waiting["ns"] = max(0, intersection.pedestrian_waiting["ns"] - 2)
            if intersection.pedestrian_waiting["ns"] == 0:
                intersection.pedestrian_max_wait["ns"] = 0
        elif intersection.current_phase == SignalPhase.EW_GREEN:
            intersection.pedestrian_max_wait["ns"] += STEP_SECONDS
            intersection.pedestrian_waiting["ew"] = max(0, intersection.pedestrian_waiting["ew"] - 2)
            if intersection.pedestrian_waiting["ew"] == 0:
                intersection.pedestrian_max_wait["ew"] = 0
        else:
            # All other phases: pedestrians just wait
            for crossing in ["ns", "ew"]:
                if intersection.pedestrian_waiting[crossing] > 0:
                    intersection.pedestrian_max_wait[crossing] += STEP_SECONDS

    def _update_phase_timer(self, intersection: Intersection):
        """Decrement phase timer; auto-cycle if expired."""
        intersection.phase_timer -= STEP_SECONDS
        if intersection.phase_timer <= 0:
            # Auto-cycle to next phase
            cycle = [SignalPhase.NS_GREEN, SignalPhase.ALL_RED, SignalPhase.EW_GREEN,
                     SignalPhase.ALL_RED, SignalPhase.NS_GREEN]
            try:
                idx = cycle.index(intersection.current_phase)
                intersection.current_phase = cycle[(idx + 1) % len(cycle)]
            except ValueError:
                intersection.current_phase = SignalPhase.NS_GREEN
            intersection.phase_timer = 5 if intersection.current_phase == SignalPhase.ALL_RED else 30
            intersection.phase_start_step = self.step_count

    def _update_incidents(self):
        """Tick down incident timers, deactivate expired ones."""
        for incident in self.incidents:
            if not incident.active:
                continue
            if incident.duration_steps > 0:
                elapsed = self.step_count - incident.start_step
                if elapsed >= incident.duration_steps:
                    incident.active = False
                    # Unblock lane if accident
                    if incident.incident_type == IncidentType.ACCIDENT and incident.direction:
                        iid = incident.intersection_id
                        if iid in self.intersections:
                            self.intersections[iid].blocked_lanes[incident.direction] = False

    def apply_action(self, intersection_id: str, phase: Optional[SignalPhase] = None,
                     extend_seconds: Optional[int] = None,
                     preempt_direction: Optional[Direction] = None):
        """Apply an agent action to an intersection."""
        if intersection_id not in self.intersections:
            return
        ix = self.intersections[intersection_id]

        if phase is not None and phase != ix.current_phase:
            ix.phase_change_history.append(self.step_count)
            ix.current_phase = phase
            ix.phase_timer = 5 if phase == SignalPhase.ALL_RED else 30
            ix.phase_start_step = self.step_count

        if extend_seconds is not None:
            ix.phase_timer += extend_seconds

        if preempt_direction is not None:
            # Set green for the emergency direction
            if preempt_direction in (Direction.NORTH, Direction.SOUTH):
                ix.current_phase = SignalPhase.NS_GREEN
            else:
                ix.current_phase = SignalPhase.EW_GREEN
            ix.phase_timer = 20
            ix.phase_start_step = self.step_count

    def add_incident(self, incident_type: IncidentType, intersection_id: str,
                     direction: Optional[Direction], duration_steps: int,
                     description: str):
        """Inject an incident into the simulation."""
        incident = Incident(
            incident_type=incident_type,
            intersection_id=intersection_id,
            direction=direction,
            start_step=self.step_count,
            duration_steps=duration_steps,
            description=description,
        )
        self.incidents.append(incident)

        # Apply immediate effects
        if incident_type == IncidentType.ACCIDENT and direction:
            if intersection_id in self.intersections:
                self.intersections[intersection_id].blocked_lanes[direction] = True

    def inject_emergency_vehicle(self, intersection_id: str, direction: Direction):
        """Place an emergency vehicle at the front of a queue."""
        if intersection_id not in self.intersections:
            return
        ix = self.intersections[intersection_id]
        self.vehicle_counter += 1
        ev = Vehicle(
            id=self.vehicle_counter,
            arrival_step=self.step_count,
            direction=direction,
            is_emergency=True,
        )
        ix.through_queues[direction].insert(0, ev)
        self.add_incident(
            IncidentType.EMERGENCY_VEHICLE,
            intersection_id, direction, 30,
            f"Emergency vehicle approaching from {direction.value}"
        )

    def step(self) -> int:
        """Advance simulation by one timestep. Returns total vehicles cleared."""
        self.step_count += 1
        _, time_minutes = self.get_time_of_day()

        step_cleared = 0
        for ix in self.intersections.values():
            self._spawn_vehicles(ix, time_minutes)
            self._spawn_pedestrians(ix)
            cleared = self._process_vehicles(ix)
            step_cleared += cleared
            self._process_pedestrians(ix)
            self._update_phase_timer(ix)

        self._update_incidents()
        self.total_cleared += step_cleared

        # Log for replay
        self.replay_log.append(self._snapshot())

        return step_cleared

    def _snapshot(self) -> dict:
        """Capture current state for replay visualization."""
        time_str, time_min = self.get_time_of_day()
        return {
            "step": self.step_count,
            "time": time_str,
            "intersections": {
                iid: {
                    "phase": ix.current_phase.value,
                    "phase_timer": ix.phase_timer,
                    "queues": {
                        d.value: {
                            "through": len(ix.through_queues[d]),
                            "left": len(ix.left_turn_queues[d]),
                            "has_emergency": any(v.is_emergency for v in ix.through_queues[d]),
                        }
                        for d in DIRECTIONS
                    },
                    "pedestrians": dict(ix.pedestrian_waiting),
                    "ped_max_wait": dict(ix.pedestrian_max_wait),
                    "blocked": {d.value: b for d, b in ix.blocked_lanes.items() if b},
                }
                for iid, ix in self.intersections.items()
            },
            "incidents": [
                {
                    "type": inc.incident_type.value,
                    "intersection": inc.intersection_id,
                    "direction": inc.direction.value if inc.direction else None,
                    "description": inc.description,
                    "active": inc.active,
                }
                for inc in self.incidents if inc.active
            ],
            "total_cleared": self.total_cleared,
            "total_waiting": sum(
                sum(len(q) for q in ix.through_queues.values()) +
                sum(len(q) for q in ix.left_turn_queues.values())
                for ix in self.intersections.values()
            ),
        }

    def get_total_waiting(self) -> int:
        return sum(
            sum(len(q) for q in ix.through_queues.values()) +
            sum(len(q) for q in ix.left_turn_queues.values())
            for ix in self.intersections.values()
        )

    def get_total_pedestrians_waiting(self) -> int:
        return sum(
            sum(ix.pedestrian_waiting.values())
            for ix in self.intersections.values()
        )

    def has_emergency_waiting(self) -> dict[str, list[Direction]]:
        """Return intersection_id -> directions with emergency vehicles queued."""
        result = {}
        for iid, ix in self.intersections.items():
            dirs = []
            for d in DIRECTIONS:
                if any(v.is_emergency for v in ix.through_queues[d]):
                    dirs.append(d)
            if dirs:
                result[iid] = dirs
        return result