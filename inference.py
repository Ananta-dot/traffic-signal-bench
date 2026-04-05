"""
Inference Script for TrafficSignalBench
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from typing import List

from openai import OpenAI

from environment import TrafficSignalEnv, TASKS
from graders import grade
from models import Action, ActionType, MultiAction, SignalPhase, Direction

# ---------------------------------------------------------------------------
# Environment variables (set by hackathon organizers during judging)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TEMPERATURE = 0.1
MAX_TOKENS = 600
SEED = 42

# ---------------------------------------------------------------------------
# System prompt for the LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI traffic signal controller managing intersections in a 2x2 grid.
Each step you receive the current traffic state and must respond with a JSON array of actions.

AVAILABLE ACTIONS (one per intersection):
- {"action_type": "set_phase", "intersection_id": "<id>", "phase": "<phase>"}
  Phases: ns_green, ew_green, ns_left_arrow, ew_left_arrow, all_red, pedestrian
- {"action_type": "extend_phase", "intersection_id": "<id>", "extend_seconds": <5-30>}
- {"action_type": "emergency_preempt", "intersection_id": "<id>", "preempt_direction": "<dir>"}
  Directions: north, south, east, west
- {"action_type": "noop"}

STRATEGY GUIDELINES:
1. EMERGENCY VEHICLES: Highest priority. Immediately preempt to give green to the emergency direction.
2. PEDESTRIANS: If any crossing has waited >70 seconds, switch to pedestrian phase soon.
3. QUEUE BALANCING: Give more green time to the direction with longer queues.
4. DON'T FLICKER: Avoid changing phases more than once every 15-20 seconds.
5. COORDINATION: Try to keep adjacent intersections in compatible phases for green waves.
6. STARVATION: Never leave any direction without green for more than 2.5 minutes.

Respond with ONLY a valid JSON array. No explanations, no markdown. Example:
[{"action_type": "set_phase", "intersection_id": "int_0_0", "phase": "ns_green"}, {"action_type": "noop"}]
""").strip()


# ---------------------------------------------------------------------------
# Observation formatting
# ---------------------------------------------------------------------------

def format_observation(obs: dict) -> str:
    """Format observation into a concise prompt for the LLM."""
    lines = [
        f"Step {obs['step_number']} | Time: {obs['time_of_day']} | "
        f"Total waiting: {obs['total_vehicles_waiting']} | "
        f"Total cleared: {obs['total_vehicles_cleared']}"
    ]

    for ix in obs.get("intersections", []):
        iid = ix["intersection_id"]
        phase = ix["current_phase"]
        timer = ix["phase_time_remaining"]
        lines.append(f"\n[{iid}] Phase: {phase} (remaining: {timer}s)")

        for lq in ix["lane_queues"]:
            total = lq["through_queue"] + lq["left_turn_queue"]
            if total > 0:
                lines.append(
                    f"  {lq['direction']}: {total} vehicles "
                    f"(through={lq['through_queue']}, left={lq['left_turn_queue']})"
                )

        for crossing, wait in ix.get("pedestrian_max_wait", {}).items():
            if wait > 30:
                count = ix.get("pedestrian_waiting", {}).get(crossing, 0)
                lines.append(f"  Pedestrians {crossing.upper()}: {count} waiting, max wait {wait}s")

    if obs.get("incidents"):
        lines.append("\nACTIVE INCIDENTS:")
        for inc in obs["incidents"]:
            dir_str = f" from {inc['direction']}" if inc.get("direction") else ""
            remaining = f" ({inc['time_remaining']} steps left)" if inc.get("time_remaining") else ""
            lines.append(f"  {inc['incident_type']}{dir_str} at {inc['intersection_id']}{remaining}")
            lines.append(f"    {inc['description']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

ACTION_JSON_RE = re.compile(r"\[.*\]", re.DOTALL)


def parse_llm_response(response_text: str) -> list[dict]:
    """Parse the LLM response into a list of action dicts."""
    if not response_text:
        return [{"action_type": "noop"}]

    text = response_text.strip()

    # Strip markdown code fences if present
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                text = part
                break

    # Try direct JSON parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in response
    match = ACTION_JSON_RE.search(text)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    print(f"  Warning: Could not parse LLM response, using noop", file=sys.stderr)
    return [{"action_type": "noop"}]


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str, client: OpenAI, model: str, seed: int = 42) -> dict:
    """Run a single episode using the LLM agent."""
    # START log for this task
    print(f"START task={task_id} seed={seed}")

    env = TrafficSignalEnv()
    obs = env.reset(task_id=task_id, seed=seed)
    obs_dict = obs.model_dump()

    trajectory = []
    total_reward = 0.0
    step_count = 0
    action_history: list[str] = []

    while True:
        # Format observation for LLM
        user_prompt = format_observation(obs_dict)

        # Add recent action history for context
        if action_history:
            recent = action_history[-3:]
            user_prompt += "\n\nYour last actions:\n" + "\n".join(recent)

        user_prompt += "\n\nRespond with a JSON array of actions for this step:"

        # Call LLM
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  LLM error at step {step_count}: {exc}", file=sys.stderr)
            response_text = '[{"action_type": "noop"}]'

        # Parse response into actions
        action_dicts = parse_llm_response(response_text)

        # Convert to typed Action objects
        actions = []
        for ad in action_dicts:
            try:
                actions.append(Action(**ad))
            except Exception:
                actions.append(Action(action_type=ActionType.NOOP))

        if not actions:
            actions = [Action(action_type=ActionType.NOOP)]

        multi = MultiAction(actions=actions)

        # Step environment
        obs, reward, done, info = env.step(multi)
        obs_dict = obs.model_dump()
        total_reward += reward.total
        step_count += 1

        # Record trajectory
        replay = env.get_replay_log()
        if replay:
            trajectory.append(replay[-1])

        # STEP log
        action_summary = ", ".join(
            f"{a.action_type}({a.intersection_id or ''}, {a.phase or a.preempt_direction or ''})"
            for a in actions
        )
        print(f"STEP task={task_id} step={step_count} reward={reward.total:+.2f} waiting={obs.total_vehicles_waiting} cleared={obs.total_vehicles_cleared}")

        action_history.append(f"Step {step_count}: {action_summary} -> reward {reward.total:+.2f}")

        if done:
            break

    # Grade the trajectory
    result = grade(task_id, trajectory)

    # END log for this task
    print(f"END task={task_id} score={result.score:.4f} steps={step_count} reward={total_reward:.2f}")

    return {
        "task_id": task_id,
        "score": result.score,
        "breakdown": result.breakdown,
        "total_reward": total_reward,
        "steps": step_count,
        "details": result.details,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not MODEL_NAME:
        print("ERROR: MODEL_NAME environment variable is required.")
        print("Set: export API_BASE_URL=... MODEL_NAME=... HF_TOKEN=...")
        sys.exit(1)

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is required.")
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    print(f"START inference")
    print(f"  API:   {API_BASE_URL}")
    print(f"  Model: {MODEL_NAME}")

    # Run all 3 tasks
    tasks = ["easy", "medium", "hard"]
    results = []

    for task_id in tasks:
        result = run_episode(task_id, client, MODEL_NAME, seed=SEED)
        results.append(result)

    # Summary
    for r in results:
        print(f"  {r['task_id']:8s}: score={r['score']:.4f}  reward={r['total_reward']:+.2f}  steps={r['steps']}")

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"  Average score: {avg_score:.4f}")

    # Save results for reproducibility
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"END inference avg_score={avg_score:.4f}")


if __name__ == "__main__":
    main()