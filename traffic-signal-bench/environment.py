"""
TrafficSignalBench — OpenEnv environment.

Wraps the traffic simulator and exposes the standard step() / reset() / state() API.
"""

from __future__ import annotations

from typing import Optional

from models import (
    Action, ActionType, MultiAction, Direction, SignalPhase,
    Observation, IntersectionObservation, LaneQueue,
    IncidentObservation, IncidentType,
    Reward, RewardBreakdown,
    EnvironmentState,
)
from simulator import TrafficSimulator, DIRECTIONS, STEP_SECONDS


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS = {
    "easy": {
        "id": "easy",
        "name": "Single Intersection — Steady Traffic",
        "description": (
            "Control a single intersection with predictable traffic. "
            "Optimize phase timing to minimize average vehicle wait time."
        ),
        "max_steps": 120,        # 10 minutes of sim time
        "use_grid": False,       # Single intersection only
        "incidents": [],
        "difficulty": "easy",
    },
    "medium": {
        "id": "medium",
        "name": "2×2 Grid — Coordinated Signals with Pedestrians",
        "description": (
            "Coordinate four intersections for green wave optimization. "
            "Rush hour pattern shifts from NS-heavy morning to EW-heavy evening. "
            "Pedestrian crossings add timing constraints."
        ),
        "max_steps": 360,        # 30 minutes of sim time
        "use_grid": True,
        "incidents": [],
        "difficulty": "medium",
    },
    "hard": {
        "id": "hard",
        "name": "2×2 Grid — Incidents & Emergency Vehicles",
        "description": (
            "Full 2×2 grid with mid-episode incidents: an accident blocks a lane, "
            "an emergency vehicle needs preemption, and a stadium event causes "
            "a sudden traffic surge. Adapt in real-time."
        ),
        "max_steps": 480,        # 40 minutes of sim time
        "use_grid": True,
        "incidents": [
            {
                "step": 60,
                "type": "accident",
                "intersection": "int_0_1",
                "direction": "east",
                "duration": 120,
                "description": "Accident blocking eastbound lane at intersection (0,1)"
            },
            {
                "step": 150,
                "type": "emergency_vehicle",
                "intersection": "int_1_0",
                "direction": "south",
                "description": "Ambulance approaching from south at intersection (1,0)"
            },
            {
                "step": 240,
                "type": "stadium_event",
                "intersection": "int_1_1",
                "direction": "west",
                "duration": 180,
                "description": "Stadium letting out — surge of westbound traffic at (1,1)"
            },
        ],
        "difficulty": "hard",
    },
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class TrafficSignalEnv:
    """OpenEnv-compliant traffic signal control environment."""

    def __init__(self):
        self.simulator: Optional[TrafficSimulator] = None
        self.task_config: Optional[dict] = None
        self.current_step = 0
        self.max_steps = 0
        self.done = False
        self.cumulative_reward = 0.0
        self._pending_incidents: list[dict] = []
        self._active_intersections: list[str] = []
        self._stadium_surge_active = False
        self._stadium_surge_end = 0
        self._stadium_intersection = ""
        self._stadium_direction = Direction.NORTH

    def reset(self, task_id: str = "easy", seed: int = 42) -> Observation:
        """Reset environment to initial state for given task."""
        if task_id not in TASKS:
            raise ValueError(f"Unknown task: {task_id}. Available: {list(TASKS.keys())}")

        self.task_config = TASKS[task_id]
        self.simulator = TrafficSimulator(seed=seed)
        self.current_step = 0
        self.max_steps = self.task_config["max_steps"]
        self.done = False
        self.cumulative_reward = 0.0
        self._pending_incidents = list(self.task_config["incidents"])
        self._stadium_surge_active = False
        self._stadium_surge_end = 0
        self._stadium_intersection = ""
        self._stadium_direction = Direction.NORTH

        if self.task_config["use_grid"]:
            self._active_intersections = list(self.simulator.intersections.keys())
        else:
            self._active_intersections = ["int_0_0"]

        return self._build_observation()

    def step(self, action: MultiAction) -> tuple[Observation, Reward, bool, dict]:
        """Execute one step: apply actions, advance simulation, compute reward."""
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        # Apply agent actions
        for act in action.actions:
            if act.action_type == ActionType.NOOP:
                continue
            if act.intersection_id and act.intersection_id not in self._active_intersections:
                continue

            if act.action_type == ActionType.SET_PHASE and act.phase:
                self.simulator.apply_action(act.intersection_id, phase=SignalPhase(act.phase))
            elif act.action_type == ActionType.EXTEND_PHASE and act.extend_seconds:
                self.simulator.apply_action(act.intersection_id, extend_seconds=act.extend_seconds)
            elif act.action_type == ActionType.EMERGENCY_PREEMPT and act.preempt_direction:
                self.simulator.apply_action(
                    act.intersection_id,
                    preempt_direction=Direction(act.preempt_direction)
                )

        # Inject scheduled incidents
        self._trigger_incidents()

        # Handle stadium surge (increased arrival rates)
        self._handle_stadium_surge()

        # Advance simulation
        step_cleared = self.simulator.step()
        self.current_step += 1

        # Build observation and reward
        obs = self._build_observation()
        reward = self._compute_reward(step_cleared)
        self.cumulative_reward += reward.total

        # Check termination
        self.done = self.current_step >= self.max_steps
        info = {
            "task_id": self.task_config["id"],
            "cumulative_reward": self.cumulative_reward,
        }

        return obs, reward, self.done, info

    def state(self) -> EnvironmentState:
        """Return full internal state."""
        return EnvironmentState(
            task_id=self.task_config["id"] if self.task_config else "none",
            step_number=self.current_step,
            observation=self._build_observation(),
            cumulative_reward=self.cumulative_reward,
            done=self.done,
            info={"max_steps": self.max_steps},
        )

    def _trigger_incidents(self):
        """Check and trigger any incidents scheduled for this step."""
        remaining = []
        for inc in self._pending_incidents:
            if self.current_step >= inc["step"]:
                if inc["type"] == "accident":
                    self.simulator.add_incident(
                        IncidentType.ACCIDENT,
                        inc["intersection"],
                        Direction(inc["direction"]),
                        inc.get("duration", 60),
                        inc["description"],
                    )
                elif inc["type"] == "emergency_vehicle":
                    self.simulator.inject_emergency_vehicle(
                        inc["intersection"],
                        Direction(inc["direction"]),
                    )
                elif inc["type"] == "stadium_event":
                    self._stadium_surge_active = True
                    self._stadium_surge_end = self.current_step + inc.get("duration", 120)
                    self._stadium_intersection = inc["intersection"]
                    self._stadium_direction = Direction(inc["direction"])
                    self.simulator.add_incident(
                        IncidentType.STADIUM_EVENT,
                        inc["intersection"],
                        Direction(inc["direction"]),
                        inc.get("duration", 120),
                        inc["description"],
                    )
            else:
                remaining.append(inc)
        self._pending_incidents = remaining

    def _handle_stadium_surge(self):
        """Boost arrival rates for stadium event direction."""
        if self._stadium_surge_active:
            if self.current_step >= self._stadium_surge_end:
                self._stadium_surge_active = False
                # Reset rates
                self.simulator.base_arrival_rates[self._stadium_direction] = (
                    {Direction.NORTH: 0.4, Direction.SOUTH: 0.4,
                     Direction.EAST: 0.3, Direction.WEST: 0.3}[self._stadium_direction]
                )
            else:
                # Triple the arrival rate for stadium direction
                self.simulator.base_arrival_rates[self._stadium_direction] = 1.2

    def _build_observation(self) -> Observation:
        """Construct typed observation from simulator state."""
        time_str, time_min = self.simulator.get_time_of_day()

        intersections = []
        for iid in self._active_intersections:
            ix = self.simulator.intersections[iid]
            lane_queues = [
                LaneQueue(
                    direction=d,
                    through_queue=len(ix.through_queues[d]),
                    left_turn_queue=len(ix.left_turn_queues[d]),
                )
                for d in DIRECTIONS
            ]
            intersections.append(IntersectionObservation(
                intersection_id=iid,
                row=ix.row,
                col=ix.col,
                current_phase=ix.current_phase,
                phase_time_remaining=max(0, ix.phase_timer),
                lane_queues=lane_queues,
                pedestrian_waiting=dict(ix.pedestrian_waiting),
                pedestrian_max_wait=dict(ix.pedestrian_max_wait),
            ))

        incidents = [
            IncidentObservation(
                incident_type=inc.incident_type,
                intersection_id=inc.intersection_id,
                direction=inc.direction,
                description=inc.description,
                time_remaining=(
                    max(0, inc.duration_steps - (self.current_step - inc.start_step))
                    if inc.duration_steps > 0 else None
                ),
            )
            for inc in self.simulator.incidents if inc.active
        ]

        return Observation(
            step_number=self.current_step,
            time_of_day=time_str,
            time_of_day_minutes=time_min,
            intersections=intersections,
            incidents=incidents,
            total_vehicles_waiting=self.simulator.get_total_waiting(),
            total_vehicles_cleared=self.simulator.total_cleared,
            total_pedestrians_waiting=self.simulator.get_total_pedestrians_waiting(),
            episode_time_elapsed=self.current_step * STEP_SECONDS,
        )

    def _compute_reward(self, step_cleared: int) -> Reward:
        """Compute shaped reward with multiple components."""
        # --- Throughput reward ---
        throughput_reward = step_cleared * 1.0

        # --- Wait penalty ---
        total_waiting = self.simulator.get_total_waiting()
        wait_penalty = -total_waiting * 0.05

        # --- Pedestrian penalty ---
        ped_penalty = 0.0
        for iid in self._active_intersections:
            ix = self.simulator.intersections[iid]
            for crossing in ["ns", "ew"]:
                max_wait = ix.pedestrian_max_wait.get(crossing, 0)
                if max_wait > 90:
                    ped_penalty -= (max_wait - 90) * 0.1
                elif max_wait > 60:
                    ped_penalty -= (max_wait - 60) * 0.03

        # --- Emergency vehicle penalty ---
        emergency_penalty = 0.0
        emergency_waiting = self.simulator.has_emergency_waiting()
        for iid, dirs in emergency_waiting.items():
            emergency_penalty -= len(dirs) * 5.0  # Heavy penalty per step

        # --- Starvation penalty ---
        starvation_penalty = 0.0
        for iid in self._active_intersections:
            ix = self.simulator.intersections[iid]
            for d in DIRECTIONS:
                steps_since_green = self.current_step - ix.last_green_step.get(d, 0)
                if steps_since_green > 36:  # 180 seconds / 5 = 36 steps
                    starvation_penalty -= 2.0

        # --- Flicker penalty ---
        flicker_penalty = 0.0
        for iid in self._active_intersections:
            ix = self.simulator.intersections[iid]
            recent_changes = [
                t for t in ix.phase_change_history
                if t > self.current_step - 6  # Last 30 seconds
            ]
            if len(recent_changes) > 3:
                flicker_penalty -= (len(recent_changes) - 3) * 1.0

        breakdown = RewardBreakdown(
            throughput_reward=throughput_reward,
            wait_penalty=wait_penalty,
            pedestrian_penalty=ped_penalty,
            emergency_penalty=emergency_penalty,
            starvation_penalty=starvation_penalty,
            flicker_penalty=flicker_penalty,
        )

        total = (throughput_reward + wait_penalty + ped_penalty +
                 emergency_penalty + starvation_penalty + flicker_penalty)

        return Reward(total=round(total, 4), breakdown=breakdown)

    def get_replay_log(self) -> list[dict]:
        """Return replay log for visualization."""
        return self.simulator.replay_log if self.simulator else []