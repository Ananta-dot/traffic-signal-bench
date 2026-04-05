"""
Task graders for TrafficSignalBench.

Each grader scores agent performance on a 0.0–1.0 scale with partial credit.
Scores are deterministic and reproducible given the same trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GradeResult:
    """Grading output with score and breakdown."""
    task_id: str
    score: float  # 0.0 to 1.0
    breakdown: dict[str, float]
    details: str


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def grade_easy(trajectory: list[dict]) -> GradeResult:
    """
    Grade Task 1: Single Intersection, Steady Traffic.

    Metrics:
    - Average vehicles waiting (lower is better)
    - Total throughput (higher is better)
    - No direction starved for >180 seconds

    Baseline (fixed-time): ~0.35 score
    Good agent: 0.6-0.8
    Optimal: 0.9+
    """
    if not trajectory:
        return GradeResult("easy", 0.0, {}, "Empty trajectory")

    total_steps = len(trajectory)
    final = trajectory[-1]

    # --- Throughput score (40%) ---
    total_cleared = final.get("total_cleared", 0)
    # Baseline expectation: ~2 vehicles/step * 120 steps = 240
    throughput_score = _clamp(total_cleared / 300)  # 300 = very good

    # --- Wait time score (40%) ---
    avg_waiting = sum(t.get("total_waiting", 0) for t in trajectory) / total_steps
    # Lower is better. 0 waiting = 1.0, 20+ avg = 0.0
    wait_score = _clamp(1.0 - (avg_waiting / 20))

    # --- Starvation score (20%) ---
    # Check if any snapshot shows a direction with very long queue
    max_queue_seen = 0
    for t in trajectory:
        for iid, ix_data in t.get("intersections", {}).items():
            for d, q in ix_data.get("queues", {}).items():
                max_queue_seen = max(max_queue_seen, q.get("through", 0))
    starvation_score = _clamp(1.0 - (max_queue_seen / 30))

    score = round(throughput_score * 0.4 + wait_score * 0.4 + starvation_score * 0.2, 4)

    return GradeResult(
        task_id="easy",
        score=_clamp(score),
        breakdown={
            "throughput": round(throughput_score, 4),
            "wait_time": round(wait_score, 4),
            "starvation": round(starvation_score, 4),
        },
        details=(
            f"Cleared {total_cleared} vehicles over {total_steps} steps. "
            f"Avg waiting: {avg_waiting:.1f}. Max queue: {max_queue_seen}."
        ),
    )


def grade_medium(trajectory: list[dict]) -> GradeResult:
    """
    Grade Task 2: 2×2 Grid, Coordinated Signals + Pedestrians.

    Metrics:
    - Throughput across all intersections (30%)
    - Average wait time (30%)
    - Pedestrian max wait compliance (25%)
    - Green wave coordination (15%)
    """
    if not trajectory:
        return GradeResult("medium", 0.0, {}, "Empty trajectory")

    total_steps = len(trajectory)
    final = trajectory[-1]

    # --- Throughput (30%) ---
    total_cleared = final.get("total_cleared", 0)
    throughput_score = _clamp(total_cleared / 800)

    # --- Wait time (30%) ---
    avg_waiting = sum(t.get("total_waiting", 0) for t in trajectory) / total_steps
    wait_score = _clamp(1.0 - (avg_waiting / 60))

    # --- Pedestrian compliance (25%) ---
    ped_violations = 0
    for t in trajectory:
        for iid, ix_data in t.get("intersections", {}).items():
            for crossing, max_wait in ix_data.get("ped_max_wait", {}).items():
                if max_wait > 90:
                    ped_violations += 1
    ped_score = _clamp(1.0 - (ped_violations / (total_steps * 4)))

    # --- Coordination (15%) ---
    # Measure: do adjacent intersections have compatible phases?
    coord_matches = 0
    coord_total = 0
    for t in trajectory:
        ixs = t.get("intersections", {})
        # Check horizontal pairs: (0,0)-(0,1) and (1,0)-(1,1)
        for row in range(2):
            left = ixs.get(f"int_{row}_0", {})
            right = ixs.get(f"int_{row}_1", {})
            if left and right:
                coord_total += 1
                if left.get("phase") == right.get("phase"):
                    coord_matches += 1
    coord_score = _clamp(coord_matches / max(coord_total, 1))

    score = round(
        throughput_score * 0.3 + wait_score * 0.3 +
        ped_score * 0.25 + coord_score * 0.15,
        4
    )

    return GradeResult(
        task_id="medium",
        score=_clamp(score),
        breakdown={
            "throughput": round(throughput_score, 4),
            "wait_time": round(wait_score, 4),
            "pedestrian_compliance": round(ped_score, 4),
            "coordination": round(coord_score, 4),
        },
        details=(
            f"Cleared {total_cleared} vehicles. Avg waiting: {avg_waiting:.1f}. "
            f"Ped violations: {ped_violations}. Coordination: {coord_matches}/{coord_total}."
        ),
    )


def grade_hard(trajectory: list[dict]) -> GradeResult:
    """
    Grade Task 3: 2×2 Grid with Incidents & Emergency Vehicles.

    Metrics:
    - Throughput (25%)
    - Wait time management (25%)
    - Pedestrian safety (15%)
    - Emergency response (20%) — how fast was emergency vehicle cleared
    - Incident adaptation (15%) — queue management during/after incidents
    """
    if not trajectory:
        return GradeResult("hard", 0.0, {}, "Empty trajectory")

    total_steps = len(trajectory)
    final = trajectory[-1]

    # --- Throughput (25%) ---
    total_cleared = final.get("total_cleared", 0)
    throughput_score = _clamp(total_cleared / 900)

    # --- Wait time (25%) ---
    avg_waiting = sum(t.get("total_waiting", 0) for t in trajectory) / total_steps
    wait_score = _clamp(1.0 - (avg_waiting / 80))

    # --- Pedestrian safety (15%) ---
    ped_violations = 0
    for t in trajectory:
        for iid, ix_data in t.get("intersections", {}).items():
            for crossing, max_wait in ix_data.get("ped_max_wait", {}).items():
                if max_wait > 90:
                    ped_violations += 1
    ped_score = _clamp(1.0 - (ped_violations / (total_steps * 4)))

    # --- Emergency response (20%) ---
    # Count steps where an emergency vehicle was in queue
    emergency_wait_steps = 0
    for t in trajectory:
        for iid, ix_data in t.get("intersections", {}).items():
            for d, q in ix_data.get("queues", {}).items():
                if q.get("has_emergency", False):
                    emergency_wait_steps += 1
    # Perfect = cleared in 1-2 steps, bad = 20+ steps
    emergency_score = _clamp(1.0 - (emergency_wait_steps / 15))

    # --- Incident adaptation (15%) ---
    # Measure queue buildup during incident period (steps 60-180) vs normal
    pre_incident_wait = []
    during_incident_wait = []
    post_incident_wait = []
    for i, t in enumerate(trajectory):
        w = t.get("total_waiting", 0)
        if i < 60:
            pre_incident_wait.append(w)
        elif i < 180:
            during_incident_wait.append(w)
        else:
            post_incident_wait.append(w)

    avg_pre = sum(pre_incident_wait) / max(len(pre_incident_wait), 1)
    avg_during = sum(during_incident_wait) / max(len(during_incident_wait), 1)
    avg_post = sum(post_incident_wait) / max(len(post_incident_wait), 1)

    # Good adaptation = during/post not much worse than pre
    if avg_pre > 0:
        degradation = (avg_during + avg_post) / (2 * avg_pre)
        adapt_score = _clamp(1.0 - (degradation - 1.0) / 3.0)
    else:
        adapt_score = _clamp(1.0 - avg_during / 40)

    score = round(
        throughput_score * 0.25 + wait_score * 0.25 +
        ped_score * 0.15 + emergency_score * 0.20 +
        adapt_score * 0.15,
        4
    )

    return GradeResult(
        task_id="hard",
        score=_clamp(score),
        breakdown={
            "throughput": round(throughput_score, 4),
            "wait_time": round(wait_score, 4),
            "pedestrian_safety": round(ped_score, 4),
            "emergency_response": round(emergency_score, 4),
            "incident_adaptation": round(adapt_score, 4),
        },
        details=(
            f"Cleared {total_cleared} vehicles. Avg waiting: {avg_waiting:.1f}. "
            f"Emergency wait steps: {emergency_wait_steps}. "
            f"Ped violations: {ped_violations}. "
            f"Queue degradation during incidents: pre={avg_pre:.1f}, "
            f"during={avg_during:.1f}, post={avg_post:.1f}."
        ),
    )


# Registry
GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


def grade(task_id: str, trajectory: list[dict]) -> GradeResult:
    """Grade a trajectory for the given task."""
    if task_id not in GRADERS:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(GRADERS.keys())}")
    return GRADERS[task_id](trajectory)
