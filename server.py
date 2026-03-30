"""
FastAPI server for TrafficSignalBench OpenEnv environment.

Endpoints:
  POST /reset       — Reset environment for a task
  POST /step        — Submit action, advance simulation
  GET  /state       — Get current state
  GET  /tasks       — List available tasks
  POST /grade       — Grade current trajectory
  GET  /replay      — Get replay log as JSON
  GET  /visualize   — Serve visual debugger HTML page
  GET  /             — Health check / landing page
"""

from __future__ import annotations

import json
from pathlib import Path

from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from environment import TrafficSignalEnv, TASKS
from graders import grade
from models import MultiAction, Action, ActionType, SignalPhase, Direction


app = FastAPI(
    title="TrafficSignalBench",
    description="OpenEnv environment for AI agent traffic signal control",
    version="1.0.0",
)

# Global environment instance
env = TrafficSignalEnv()

# Store trajectory for grading
trajectory: list[dict] = []


def _run_demo_episode():
    """Run a demo episode with heuristic agent so visualizer has data on startup."""
    global trajectory
    from simulator import DIRECTIONS as SIM_DIRS

    obs = env.reset(task_id="hard", seed=42)
    trajectory = []

    while True:
        actions = []
        for ix in obs.intersections:
            ns_q = sum(
                lq.through_queue + lq.left_turn_queue
                for lq in ix.lane_queues if lq.direction in ("north", "south")
            )
            ew_q = sum(
                lq.through_queue + lq.left_turn_queue
                for lq in ix.lane_queues if lq.direction in ("east", "west")
            )
            ns_left = sum(
                lq.left_turn_queue
                for lq in ix.lane_queues if lq.direction in ("north", "south")
            )
            ew_left = sum(
                lq.left_turn_queue
                for lq in ix.lane_queues if lq.direction in ("east", "west")
            )
            total_q = ns_q + ew_q
            phase = ix.current_phase
            timer = ix.phase_time_remaining

            # 1. Emergency preemption — highest priority
            emergency_dir = None
            for inc in obs.incidents:
                if (inc.incident_type == "emergency_vehicle"
                        and inc.intersection_id == ix.intersection_id
                        and inc.direction):
                    emergency_dir = inc.direction
                    break
            if emergency_dir:
                actions.append(Action(
                    action_type=ActionType.EMERGENCY_PREEMPT,
                    intersection_id=ix.intersection_id,
                    preempt_direction=emergency_dir,
                ))
                continue

            # 2. Pedestrian urgency — serve before they hit 90s penalty
            ped_max = max(ix.pedestrian_max_wait.values(), default=0)
            if ped_max > 70:
                actions.append(Action(
                    action_type=ActionType.SET_PHASE,
                    intersection_id=ix.intersection_id,
                    phase=SignalPhase.PEDESTRIAN,
                ))
                continue

            # 3. Proactive phase switching based on queue imbalance
            is_ns_green = phase in ("ns_green", "ns_left_arrow")
            is_ew_green = phase in ("ew_green", "ew_left_arrow")

            # If current green direction has empty queues but other is backed up, switch early
            if is_ns_green and ns_q <= 2 and ew_q > 8 and timer > 10:
                actions.append(Action(
                    action_type=ActionType.SET_PHASE,
                    intersection_id=ix.intersection_id,
                    phase=SignalPhase.EW_GREEN,
                ))
                continue
            if is_ew_green and ew_q <= 2 and ns_q > 8 and timer > 10:
                actions.append(Action(
                    action_type=ActionType.SET_PHASE,
                    intersection_id=ix.intersection_id,
                    phase=SignalPhase.NS_GREEN,
                ))
                continue

            # 4. Handle left turn queues when they build up
            if ns_left > 6 and phase != "ns_left_arrow" and timer <= 10:
                actions.append(Action(
                    action_type=ActionType.SET_PHASE,
                    intersection_id=ix.intersection_id,
                    phase=SignalPhase.NS_LEFT_ARROW,
                ))
                continue
            if ew_left > 6 and phase != "ew_left_arrow" and timer <= 10:
                actions.append(Action(
                    action_type=ActionType.SET_PHASE,
                    intersection_id=ix.intersection_id,
                    phase=SignalPhase.EW_LEFT_ARROW,
                ))
                continue

            # 5. Normal phase expiry — switch to direction with more traffic
            if timer <= 5:
                if ns_q > ew_q * 1.2:
                    target = SignalPhase.NS_GREEN
                elif ew_q > ns_q * 1.2:
                    target = SignalPhase.EW_GREEN
                else:
                    # Alternate
                    target = SignalPhase.EW_GREEN if is_ns_green else SignalPhase.NS_GREEN
                actions.append(Action(
                    action_type=ActionType.SET_PHASE,
                    intersection_id=ix.intersection_id,
                    phase=target,
                ))
            else:
                actions.append(Action(action_type=ActionType.NOOP))

        multi = MultiAction(actions=actions)
        obs, reward, done, info = env.step(multi)
        replay = env.get_replay_log()
        if replay:
            trajectory.append(replay[-1])
        if done:
            break


@app.on_event("startup")
def startup():
    """Pre-populate a demo episode so the visualizer works immediately."""
    print("Running demo episode for visualizer...")
    _run_demo_episode()
    print(f"Demo complete: {len(trajectory)} frames ready.")


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 42


class StepRequest(BaseModel):
    actions: list[Action]


class GradeResponse(BaseModel):
    task_id: str
    score: float
    breakdown: dict[str, float]
    details: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "TrafficSignalBench",
        "description": "OpenEnv environment for traffic signal control",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grade", "/replay", "/visualize"],
    }


@app.get("/tasks")
def list_tasks():
    """List available tasks with descriptions."""
    return {
        tid: {
            "name": t["name"],
            "description": t["description"],
            "difficulty": t["difficulty"],
            "max_steps": t["max_steps"],
        }
        for tid, t in TASKS.items()
    }


@app.post("/reset")
async def reset(req: Optional[ResetRequest] = Body(default=None)):
    """Reset environment for a task. Returns initial observation."""
    global trajectory
    try:
        task_id = req.task_id if req else "easy"
        seed = req.seed if req else 42
        obs = env.reset(task_id=task_id, seed=seed)
        trajectory = []
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    """Submit actions and advance simulation by one step."""
    global trajectory
    try:
        multi = MultiAction(actions=req.actions)
        obs, reward, done, info = env.step(multi)

        # Record trajectory snapshot for grading
        replay = env.get_replay_log()
        if replay:
            trajectory.append(replay[-1])

        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def get_state():
    """Get current environment state."""
    try:
        state = env.state()
        return state.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grade")
def grade_trajectory():
    """Grade the current trajectory."""
    if not trajectory:
        raise HTTPException(status_code=400, detail="No trajectory recorded. Run an episode first.")
    task_id = env.task_config["id"] if env.task_config else "easy"
    result = grade(task_id, trajectory)
    return GradeResponse(
        task_id=result.task_id,
        score=result.score,
        breakdown=result.breakdown,
        details=result.details,
    ).model_dump()


@app.get("/replay")
def get_replay():
    """Return replay log for visualization."""
    return JSONResponse(content=trajectory)


@app.get("/visualize", response_class=HTMLResponse)
def visualize():
    """Serve the visual debugger page."""
    # Try multiple paths to find the visualizer
    candidates = [
        Path(__file__).parent / "static" / "visualizer.html",
        Path.cwd() / "static" / "visualizer.html",
        Path(__file__).parent / "visualizer.html",
        Path.cwd() / "visualizer.html",
    ]
    for p in candidates:
        if p.exists():
            return HTMLResponse(content=p.read_text())

    # Show helpful error
    tried = "\n".join(f"  - {p} (exists: {p.exists()})" for p in candidates)
    return HTMLResponse(
        content=f"<h1>Visualizer not found</h1>"
        f"<p>Looked in:</p><pre>{tried}</pre>"
        f"<p>Current dir: {Path.cwd()}</p>"
        f"<p>Script dir: {Path(__file__).parent}</p>"
        f"<p>Make sure <code>static/visualizer.html</code> exists next to <code>server.py</code></p>"
    )