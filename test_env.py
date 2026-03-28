"""
Quick test script — validates environment works end-to-end.
Run: python test_env.py
"""

from __future__ import annotations

import json
import sys

from environment import TrafficSignalEnv, TASKS
from models import Action, ActionType, MultiAction, SignalPhase
from graders import grade


def test_task(task_id: str, verbose: bool = False):
    """Run a task with the heuristic agent and validate mechanics."""
    print(f"\n{'='*50}")
    print(f"Testing: {task_id}")
    print(f"{'='*50}")

    env = TrafficSignalEnv()
    obs = env.reset(task_id=task_id, seed=42)

    assert obs.step_number == 0, "Initial step should be 0"
    assert obs.total_vehicles_cleared == 0, "No vehicles cleared initially"
    print(f"  ✓ Reset OK — {len(obs.intersections)} intersections")

    # Run episode with simple alternating strategy
    trajectory = []
    step_count = 0
    total_reward = 0.0

    while True:
        # Smart heuristic: proactive switching based on queue pressure
        actions = []
        for ix in obs.intersections:
            ns_q = sum(lq.through_queue + lq.left_turn_queue
                       for lq in ix.lane_queues if lq.direction in ("north", "south"))
            ew_q = sum(lq.through_queue + lq.left_turn_queue
                       for lq in ix.lane_queues if lq.direction in ("east", "west"))
            phase = ix.current_phase
            timer = ix.phase_time_remaining

            # Pedestrian urgency
            ped_max = max(ix.pedestrian_max_wait.values(), default=0)
            if ped_max > 70:
                actions.append(Action(
                    action_type=ActionType.SET_PHASE,
                    intersection_id=ix.intersection_id,
                    phase=SignalPhase.PEDESTRIAN,
                ))
                continue

            # Proactive switching on imbalance
            is_ns = phase in (SignalPhase.NS_GREEN, SignalPhase.NS_LEFT_ARROW)
            is_ew = phase in (SignalPhase.EW_GREEN, SignalPhase.EW_LEFT_ARROW)
            if is_ns and ns_q <= 2 and ew_q > 8 and timer > 10:
                actions.append(Action(action_type=ActionType.SET_PHASE,
                    intersection_id=ix.intersection_id, phase=SignalPhase.EW_GREEN))
                continue
            if is_ew and ew_q <= 2 and ns_q > 8 and timer > 10:
                actions.append(Action(action_type=ActionType.SET_PHASE,
                    intersection_id=ix.intersection_id, phase=SignalPhase.NS_GREEN))
                continue

            # Normal phase expiry
            if timer <= 5:
                if ns_q > ew_q * 1.2:
                    target = SignalPhase.NS_GREEN
                elif ew_q > ns_q * 1.2:
                    target = SignalPhase.EW_GREEN
                else:
                    target = SignalPhase.EW_GREEN if is_ns else SignalPhase.NS_GREEN
                actions.append(Action(action_type=ActionType.SET_PHASE,
                    intersection_id=ix.intersection_id, phase=target))
            else:
                actions.append(Action(action_type=ActionType.NOOP))

        multi = MultiAction(actions=actions)
        obs, reward, done, info = env.step(multi)
        total_reward += reward.total
        step_count += 1

        replay = env.get_replay_log()
        if replay:
            trajectory.append(replay[-1])

        if verbose and step_count % 60 == 0:
            print(f"  Step {step_count}: waiting={obs.total_vehicles_waiting}, "
                  f"cleared={obs.total_vehicles_cleared}, reward={reward.total:.2f}")
            if obs.incidents:
                for inc in obs.incidents:
                    print(f"    ⚠ {inc.incident_type}: {inc.description}")

        if done:
            break

    print(f"  ✓ Episode complete — {step_count} steps")
    print(f"  ✓ Vehicles cleared: {obs.total_vehicles_cleared}")
    print(f"  ✓ Total reward: {total_reward:.2f}")

    # Test state endpoint
    state = env.state()
    assert state.done is True, "State should show done"
    assert state.task_id == task_id
    print(f"  ✓ state() OK")

    # Test grading
    result = grade(task_id, trajectory)
    assert 0.0 <= result.score <= 1.0, f"Score out of range: {result.score}"
    print(f"  ✓ Graded: {result.score:.4f}")
    print(f"    Breakdown: {result.breakdown}")
    print(f"    {result.details}")

    return result.score


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    scores = {}
    for task_id in ["easy", "medium", "hard"]:
        scores[task_id] = test_task(task_id, verbose=verbose)

    print(f"\n{'='*50}")
    print("ALL TESTS PASSED")
    print(f"{'='*50}")
    for tid, score in scores.items():
        print(f"  {tid:8s}: {score:.4f}")
    print(f"  Average:  {sum(scores.values())/len(scores):.4f}")


if __name__ == "__main__":
    main()