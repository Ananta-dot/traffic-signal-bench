"""
Baseline inference script for TrafficSignalBench.

Uses the OpenAI API client to run a model against the environment.
Reads credentials from environment variables:
  - API_BASE_URL: The API endpoint for the LLM
  - MODEL_NAME: The model identifier to use
  - HF_TOKEN: Your Hugging Face / API key (optional)

Usage:
  python inference.py                        # Run all tasks with LLM
  python inference.py --task easy             # Run single task
  python inference.py --mock                  # Run with heuristic agent (no API needed)
  python inference.py --server http://...     # Run against deployed server
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

# Try to import openai for LLM-based agent
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from environment import TrafficSignalEnv, TASKS
from graders import grade
from models import Action, ActionType, MultiAction, SignalPhase, Direction


# ---------------------------------------------------------------------------
# Heuristic (mock) agent — no LLM needed
# ---------------------------------------------------------------------------

class HeuristicAgent:
    """Simple fixed-time agent for baseline comparison."""

    def decide(self, observation: dict) -> list[dict]:
        """Return actions based on simple queue-length heuristics."""
        actions = []
        for ix in observation.get("intersections", []):
            iid = ix["intersection_id"]
            phase = ix["current_phase"]
            timer = ix["phase_time_remaining"]

            # Sum queues by direction group
            ns_queue = sum(
                lq["through_queue"] + lq["left_turn_queue"]
                for lq in ix["lane_queues"]
                if lq["direction"] in ("north", "south")
            )
            ew_queue = sum(
                lq["through_queue"] + lq["left_turn_queue"]
                for lq in ix["lane_queues"]
                if lq["direction"] in ("east", "west")
            )

            # Check for emergency vehicles
            has_emergency = any(
                inc["incident_type"] == "emergency_vehicle"
                and inc["intersection_id"] == iid
                for inc in observation.get("incidents", [])
            )

            if has_emergency:
                # Find emergency direction and preempt
                for inc in observation.get("incidents", []):
                    if (inc["incident_type"] == "emergency_vehicle"
                            and inc["intersection_id"] == iid
                            and inc.get("direction")):
                        actions.append({
                            "action_type": "emergency_preempt",
                            "intersection_id": iid,
                            "preempt_direction": inc["direction"],
                        })
                        break
                continue

            # Check pedestrian wait
            ped_max = max(ix.get("pedestrian_max_wait", {}).values(), default=0)
            if ped_max > 80:
                actions.append({
                    "action_type": "set_phase",
                    "intersection_id": iid,
                    "phase": "pedestrian",
                })
                continue

            # Simple demand-responsive: switch to direction with more traffic
            if timer <= 5:
                if ns_queue > ew_queue * 1.3:
                    target = "ns_green"
                elif ew_queue > ns_queue * 1.3:
                    target = "ew_green"
                else:
                    # Alternate
                    target = "ew_green" if phase in ("ns_green", "ns_left_arrow") else "ns_green"

                actions.append({
                    "action_type": "set_phase",
                    "intersection_id": iid,
                    "phase": target,
                })
            else:
                actions.append({"action_type": "noop"})

        return actions if actions else [{"action_type": "noop"}]


# ---------------------------------------------------------------------------
# LLM-based agent
# ---------------------------------------------------------------------------

class LLMAgent:
    """Agent that uses an LLM to decide traffic signal actions."""

    SYSTEM_PROMPT = """You are a traffic signal control agent managing a 2x2 grid of intersections.

Your goal is to minimize vehicle wait times, ensure pedestrian safety, handle incidents,
and maximize throughput.

Available actions per intersection:
- set_phase: Change signal to ns_green, ew_green, ns_left_arrow, ew_left_arrow, all_red, or pedestrian
- extend_phase: Extend current phase by 5-30 seconds
- emergency_preempt: Create green corridor for emergency vehicle (specify direction)
- noop: Do nothing

IMPORTANT rules:
- Don't flicker signals (rapid phase changes are penalized)
- Don't starve any direction for >3 minutes
- Address pedestrian waits >90 seconds (heavy penalty)
- Emergency vehicles should be cleared ASAP (heaviest penalty)
- Coordinate adjacent intersections for green waves when possible

Respond with a JSON array of actions. Example:
[
  {"action_type": "set_phase", "intersection_id": "int_0_0", "phase": "ns_green"},
  {"action_type": "noop"}
]"""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def decide(self, observation: dict) -> list[dict]:
        """Ask LLM to decide actions given current observation."""
        obs_summary = self._format_observation(observation)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": obs_summary},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            content = response.choices[0].message.content.strip()

            # Parse JSON from response
            # Handle markdown code blocks
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            actions = json.loads(content)
            if isinstance(actions, dict):
                actions = [actions]
            return actions

        except Exception as e:
            print(f"  LLM error: {e}, using noop", file=sys.stderr)
            return [{"action_type": "noop"}]

    def _format_observation(self, obs: dict) -> str:
        """Format observation into a concise prompt."""
        lines = [f"Step {obs['step_number']} | Time: {obs['time_of_day']} | "
                 f"Waiting: {obs['total_vehicles_waiting']} | Cleared: {obs['total_vehicles_cleared']}"]

        for ix in obs.get("intersections", []):
            lines.append(f"\n{ix['intersection_id']} — Phase: {ix['current_phase']} "
                         f"(remaining: {ix['phase_time_remaining']}s)")
            for lq in ix["lane_queues"]:
                total = lq["through_queue"] + lq["left_turn_queue"]
                if total > 0:
                    lines.append(f"  {lq['direction']}: {total} vehicles "
                                 f"(through={lq['through_queue']}, left={lq['left_turn_queue']})")
            ped = ix.get("pedestrian_max_wait", {})
            for crossing, wait in ped.items():
                if wait > 30:
                    lines.append(f"  ⚠ Pedestrian {crossing} waiting {wait}s")

        if obs.get("incidents"):
            lines.append("\n⚠ INCIDENTS:")
            for inc in obs["incidents"]:
                lines.append(f"  {inc['incident_type']}: {inc['description']}")

        lines.append("\nDecide actions (JSON array):")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str, agent, seed: int = 42,
                use_server: bool = False, server_url: str = "") -> dict:
    """Run a single episode and return results."""
    print(f"\n{'='*60}")
    print(f"Task: {task_id} | Seed: {seed}")
    print(f"{'='*60}")

    if use_server:
        return _run_episode_server(task_id, agent, seed, server_url)
    else:
        return _run_episode_local(task_id, agent, seed)


def _run_episode_local(task_id: str, agent, seed: int) -> dict:
    """Run episode using local environment."""
    env = TrafficSignalEnv()
    obs = env.reset(task_id=task_id, seed=seed)
    obs_dict = obs.model_dump()

    trajectory = []
    total_reward = 0.0
    step_count = 0

    while True:
        # Agent decides
        action_dicts = agent.decide(obs_dict)

        # Parse actions
        actions = []
        for ad in action_dicts:
            actions.append(Action(**ad))
        multi = MultiAction(actions=actions)

        # Step
        obs, reward, done, info = env.step(multi)
        obs_dict = obs.model_dump()
        total_reward += reward.total
        step_count += 1

        # Record trajectory
        replay = env.get_replay_log()
        if replay:
            trajectory.append(replay[-1])

        # Progress
        if step_count % 30 == 0:
            print(f"  Step {step_count}: waiting={obs.total_vehicles_waiting}, "
                  f"cleared={obs.total_vehicles_cleared}, reward={reward.total:.2f}")

        if done:
            break

    # Grade
    result = grade(task_id, trajectory)
    print(f"\n  Score: {result.score:.4f}")
    print(f"  Breakdown: {result.breakdown}")
    print(f"  {result.details}")
    print(f"  Total reward: {total_reward:.2f}")

    return {
        "task_id": task_id,
        "score": result.score,
        "breakdown": result.breakdown,
        "total_reward": total_reward,
        "steps": step_count,
    }


def _run_episode_server(task_id: str, agent, seed: int, server_url: str) -> dict:
    """Run episode against a deployed server."""
    base = server_url.rstrip("/")

    # Reset
    resp = requests.post(f"{base}/reset", json={"task_id": task_id, "seed": seed})
    resp.raise_for_status()
    obs_dict = resp.json()

    total_reward = 0.0
    step_count = 0

    while True:
        action_dicts = agent.decide(obs_dict)

        resp = requests.post(f"{base}/step", json={"actions": action_dicts})
        resp.raise_for_status()
        result = resp.json()

        obs_dict = result["observation"]
        total_reward += result["reward"]["total"]
        step_count += 1
        done = result["done"]

        if step_count % 30 == 0:
            print(f"  Step {step_count}: waiting={obs_dict['total_vehicles_waiting']}, "
                  f"cleared={obs_dict['total_vehicles_cleared']}")

        if done:
            break

    # Grade
    resp = requests.post(f"{base}/grade")
    grade_result = resp.json()
    print(f"\n  Score: {grade_result['score']:.4f}")
    print(f"  Breakdown: {grade_result['breakdown']}")

    return {
        "task_id": task_id,
        "score": grade_result["score"],
        "breakdown": grade_result["breakdown"],
        "total_reward": total_reward,
        "steps": step_count,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TrafficSignalBench Inference")
    parser.add_argument("--task", type=str, default=None,
                        help="Task ID (easy/medium/hard). Default: run all.")
    parser.add_argument("--mock", action="store_true",
                        help="Use heuristic agent instead of LLM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--server", type=str, default=None,
                        help="Server URL for remote execution")
    args = parser.parse_args()

    # Setup agent
    if args.mock:
        print("Using heuristic (mock) agent")
        agent = HeuristicAgent()
    else:
        api_base = os.environ.get("API_BASE_URL", "")
        model_name = os.environ.get("MODEL_NAME", "")
        if not api_base or not model_name:
            print("ERROR: API_BASE_URL and MODEL_NAME environment variables required.")
            print("Use --mock flag to run with heuristic agent instead.")
            sys.exit(1)

        if not HAS_OPENAI:
            print("ERROR: openai package not installed. Run: pip install openai")
            sys.exit(1)

        client = OpenAI(base_url=api_base, api_key=os.environ.get("HF_TOKEN", "none"))
        agent = LLMAgent(client, model_name)
        print(f"Using LLM agent: {model_name} @ {api_base}")

    # Run tasks
    tasks = [args.task] if args.task else ["easy", "medium", "hard"]
    results = []

    for task_id in tasks:
        result = run_episode(
            task_id, agent, seed=args.seed,
            use_server=bool(args.server),
            server_url=args.server or "",
        )
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['task_id']:8s}: score={r['score']:.4f}  reward={r['total_reward']:.2f}  steps={r['steps']}")

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\n  Average score: {avg_score:.4f}")

    # Write results to file for reproducibility
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to baseline_results.json")


if __name__ == "__main__":
    main()
