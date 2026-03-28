"""Typed Pydantic models for TrafficSignalBench OpenEnv environment."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Direction(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


class SignalPhase(str, Enum):
    """Pre-defined signal configurations for an intersection."""
    NS_GREEN = "ns_green"           # North-South through traffic green
    EW_GREEN = "ew_green"           # East-West through traffic green
    NS_LEFT_ARROW = "ns_left_arrow" # North-South left turn arrow
    EW_LEFT_ARROW = "ew_left_arrow" # East-West left turn arrow
    ALL_RED = "all_red"             # Safety clearance phase
    PEDESTRIAN = "pedestrian"       # All-walk pedestrian phase


class IncidentType(str, Enum):
    ACCIDENT = "accident"
    EMERGENCY_VEHICLE = "emergency_vehicle"
    STADIUM_EVENT = "stadium_event"


class ActionType(str, Enum):
    SET_PHASE = "set_phase"
    EXTEND_PHASE = "extend_phase"
    EMERGENCY_PREEMPT = "emergency_preempt"
    NOOP = "noop"


# ---------------------------------------------------------------------------
# Observation sub-models
# ---------------------------------------------------------------------------

class LaneQueue(BaseModel):
    """Queue state for a single lane at an intersection approach."""
    direction: Direction
    through_queue: int = Field(ge=0, description="Vehicles waiting to go through")
    left_turn_queue: int = Field(ge=0, description="Vehicles waiting to turn left")


class IntersectionObservation(BaseModel):
    """Observable state of one intersection."""
    intersection_id: str
    row: int
    col: int
    current_phase: SignalPhase
    phase_time_remaining: int = Field(ge=0, description="Seconds left in current phase")
    lane_queues: list[LaneQueue] = Field(description="Queue per approach direction")
    pedestrian_waiting: dict[str, int] = Field(
        default_factory=dict,
        description="Pedestrians waiting per crossing (e.g. 'ns': 3, 'ew': 1)"
    )
    pedestrian_max_wait: dict[str, int] = Field(
        default_factory=dict,
        description="Max wait time (seconds) for pedestrians per crossing"
    )


class IncidentObservation(BaseModel):
    """An active incident visible to the agent."""
    incident_type: IncidentType
    intersection_id: str
    direction: Optional[Direction] = None
    description: str
    time_remaining: Optional[int] = Field(
        None, description="Steps remaining for incident (None = unknown)"
    )


class Observation(BaseModel):
    """Full observation returned by step() and reset()."""
    step_number: int = Field(ge=0)
    time_of_day: str = Field(description="HH:MM format, 24hr")
    time_of_day_minutes: int = Field(description="Minutes since midnight")
    intersections: list[IntersectionObservation]
    incidents: list[IncidentObservation] = Field(default_factory=list)
    total_vehicles_waiting: int = Field(ge=0)
    total_vehicles_cleared: int = Field(ge=0)
    total_pedestrians_waiting: int = Field(ge=0)
    episode_time_elapsed: int = Field(ge=0, description="Seconds elapsed in episode")


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """Action the agent submits each step."""
    action_type: ActionType
    intersection_id: Optional[str] = None
    phase: Optional[SignalPhase] = None
    extend_seconds: Optional[int] = Field(None, ge=5, le=30)
    preempt_direction: Optional[Direction] = None

    class Config:
        use_enum_values = True


class MultiAction(BaseModel):
    """Wrapper for multiple simultaneous actions (one per intersection)."""
    actions: list[Action] = Field(description="One action per intersection, or a single NOOP")


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """Detailed reward components for interpretability."""
    throughput_reward: float = Field(description="Reward for vehicles cleared this step")
    wait_penalty: float = Field(description="Penalty for total vehicles waiting")
    pedestrian_penalty: float = Field(description="Penalty for pedestrian wait violations")
    emergency_penalty: float = Field(description="Penalty for emergency vehicle delay")
    starvation_penalty: float = Field(description="Penalty for any direction starved >180s")
    flicker_penalty: float = Field(description="Penalty for rapid phase changes")


class Reward(BaseModel):
    """Reward returned by step()."""
    total: float = Field(description="Scalar reward signal")
    breakdown: RewardBreakdown


# ---------------------------------------------------------------------------
# State model (full internal state for state() endpoint)
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    """Complete environment state returned by state()."""
    task_id: str
    step_number: int
    observation: Observation
    cumulative_reward: float
    done: bool
    info: dict = Field(default_factory=dict)
