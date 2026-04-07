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
# Environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = "TrafficSignalBench"

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

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

ACTION_JSON_RE = re.compile(r"\[.*\]", re.DOTALL)

def parse_llm_response(response_text: str) -> list[dict]:
    if not response_text:
        return [{"action_type": "noop"}]

    text = response_text.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                text = part
                break

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass

    match = ACTION_JSON_RE.search(text)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return [{"action_type": "noop"}]


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str, client: OpenAI, model: str, seed: int = 42) -> dict:
    # 1. STRICT START FORMAT
    print(f"[START] task={task_id} env={BENCHMARK} model={model}", flush=True)

    env = TrafficSignalEnv()
    obs = env.reset(task_id=task_id, seed=seed)
    obs_dict = obs.model_dump()

    trajectory = []
    rewards_list = []
    total_reward = 0.0
    step_count = 0
    action_history: list[str] = []

    while True:
        user_prompt = format_observation(obs_dict)

        if action_history:
            recent = action_history[-3:]
            user_prompt += "\n\nYour last actions:\n" + "\n".join(recent)

        user_prompt += "\n\nRespond with a JSON array of actions for this step:"

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
            error_val = "null"
        except Exception as exc:
            print(f"[DEBUG] LLM error: {exc}", file=sys.stderr)
            response_text = '[{"action_type": "noop"}]'
            error_val = str(exc).replace(" ", "_") # No spaces allowed in stdout error trace

        action_dicts = parse_llm_response(response_text)
        
        actions = []
        for ad in action_dicts:
            try:
                actions.append(Action(**ad))
            except Exception:
                actions.append(Action(action_type=ActionType.NOOP))

        if not actions:
            actions = [Action(action_type=ActionType.NOOP)]

        multi = MultiAction(actions=actions)

        obs, reward, done, info = env.step(multi)
        obs_dict = obs.model_dump()
        
        step_reward = float(reward.total)
        total_reward += step_reward
        rewards_list.append(step_reward)
        step_count += 1

        replay = env.get_replay_log()
        if replay:
            trajectory.append(replay[-1])

        # Formatting action string to have no spaces or newlines for stdout
        # Formatting action string safely (handles both Enums and standard strings)
        action_str_parts = []
        for a in actions:
            act_val = a.action_type.value if hasattr(a.action_type, 'value') else str(a.action_type)
            int_id = str(a.intersection_id) if a.intersection_id else "none"
            action_str_parts.append(f"{act_val}:{int_id}")
        
        action_str = "|".join(action_str_parts).replace(" ", "_")
        done_val = str(done).lower()

        # 2. STRICT STEP FORMAT
        print(f"[STEP] step={step_count} action={action_str} reward={step_reward:.2f} done={done_val} error={error_val}", flush=True)

        action_history.append(f"Step {step_count}: {action_str} -> reward {step_reward:+.2f}")

        if done:
            break

    # Grade the trajectory
    result = grade(task_id, trajectory)
    
    # Ensure score is strictly clamped to [0, 1] as requested by validation rules
    clamped_score = max(0.0, min(1.0, float(result.score)))
    success_val = "true" if clamped_score >= 0.1 else "false" # Assuming threshold > 0.1 is success
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)

    # 3. STRICT END FORMAT
    print(f"[END] success={success_val} steps={step_count} score={clamped_score:.3f} rewards={rewards_str}", flush=True)

    return {
        "task_id": task_id,
        "score": clamped_score,
        "total_reward": total_reward,
        "steps": step_count,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not MODEL_NAME:
        print("ERROR: MODEL_NAME environment variable is required.", file=sys.stderr)
        sys.exit(1)

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is required.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    tasks = ["easy", "medium", "hard"]
    results = []

    for task_id in tasks:
        result = run_episode(task_id, client, MODEL_NAME, seed=SEED)
        results.append(result)

    avg_score = sum(r["score"] for r in results) / len(results)
    
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()