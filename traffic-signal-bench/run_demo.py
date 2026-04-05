"""
Run a demo episode on the server to populate replay data for the visualizer.
Usage: python run_demo.py [--server URL] [--task TASK] [--seed SEED]
"""
import argparse
import requests


def run_demo(base_url: str, task_id: str = "hard", seed: int = 42):
    print(f"Running demo: task={task_id}, seed={seed}, server={base_url}")

    resp = requests.post(f"{base_url}/reset", json={"task_id": task_id, "seed": seed})
    resp.raise_for_status()
    obs = resp.json()
    print(f"  Reset OK — {len(obs['intersections'])} intersections")

    step_count = 0
    while True:
        actions = []
        for ix in obs.get("intersections", []):
            iid = ix["intersection_id"]
            phase = ix["current_phase"]
            timer = ix["phase_time_remaining"]

            ns_q = sum(lq["through_queue"] + lq["left_turn_queue"]
                       for lq in ix["lane_queues"] if lq["direction"] in ("north", "south"))
            ew_q = sum(lq["through_queue"] + lq["left_turn_queue"]
                       for lq in ix["lane_queues"] if lq["direction"] in ("east", "west"))
            ns_left = sum(lq["left_turn_queue"]
                          for lq in ix["lane_queues"] if lq["direction"] in ("north", "south"))
            ew_left = sum(lq["left_turn_queue"]
                          for lq in ix["lane_queues"] if lq["direction"] in ("east", "west"))

            # 1. Emergency
            em_dir = None
            for inc in obs.get("incidents", []):
                if inc["incident_type"] == "emergency_vehicle" and inc["intersection_id"] == iid and inc.get("direction"):
                    em_dir = inc["direction"]
                    break
            if em_dir:
                actions.append({"action_type": "emergency_preempt", "intersection_id": iid, "preempt_direction": em_dir})
                continue

            # 2. Pedestrians
            ped_max = max(ix.get("pedestrian_max_wait", {}).values(), default=0)
            if ped_max > 70:
                actions.append({"action_type": "set_phase", "intersection_id": iid, "phase": "pedestrian"})
                continue

            # 3. Proactive switching on imbalance
            is_ns = phase in ("ns_green", "ns_left_arrow")
            is_ew = phase in ("ew_green", "ew_left_arrow")
            if is_ns and ns_q <= 2 and ew_q > 8 and timer > 10:
                actions.append({"action_type": "set_phase", "intersection_id": iid, "phase": "ew_green"})
                continue
            if is_ew and ew_q <= 2 and ns_q > 8 and timer > 10:
                actions.append({"action_type": "set_phase", "intersection_id": iid, "phase": "ns_green"})
                continue

            # 4. Left turn handling
            if ns_left > 6 and phase != "ns_left_arrow" and timer <= 10:
                actions.append({"action_type": "set_phase", "intersection_id": iid, "phase": "ns_left_arrow"})
                continue
            if ew_left > 6 and phase != "ew_left_arrow" and timer <= 10:
                actions.append({"action_type": "set_phase", "intersection_id": iid, "phase": "ew_left_arrow"})
                continue

            # 5. Normal expiry
            if timer <= 5:
                if ns_q > ew_q * 1.2:
                    target = "ns_green"
                elif ew_q > ns_q * 1.2:
                    target = "ew_green"
                else:
                    target = "ew_green" if is_ns else "ns_green"
                actions.append({"action_type": "set_phase", "intersection_id": iid, "phase": target})
            else:
                actions.append({"action_type": "noop"})

        if not actions:
            actions = [{"action_type": "noop"}]

        resp = requests.post(f"{base_url}/step", json={"actions": actions})
        resp.raise_for_status()
        result = resp.json()
        obs = result["observation"]
        step_count += 1

        if step_count % 60 == 0:
            print(f"  Step {step_count}: waiting={obs['total_vehicles_waiting']}, cleared={obs['total_vehicles_cleared']}")
        if result["done"]:
            break

    resp = requests.post(f"{base_url}/grade")
    grade = resp.json()
    print(f"\n  Score: {grade['score']:.4f}")
    print(f"  Breakdown: {grade['breakdown']}")
    print(f"  Visit {base_url}/visualize to watch the replay")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:7860")
    parser.add_argument("--task", default="hard")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_demo(args.server, args.task, args.seed)