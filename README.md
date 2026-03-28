# 🚦 TrafficSignalBench

An OpenEnv environment for AI agent traffic signal control. The agent manages traffic lights at a 2×2 intersection grid, optimizing vehicle throughput, minimizing wait times, ensuring pedestrian safety, and adapting to real-time incidents.

## Motivation

Traffic signal control is a critical real-world optimization problem. Cities worldwide struggle with congestion, and adaptive signal control systems can reduce travel times by 10–25%. This environment provides a realistic simulation for training and evaluating AI agents on this task — bridging the gap between toy RL benchmarks and real urban traffic management.

## Environment Description

The simulation models a **2×2 grid of signalized intersections** with:
- **Vehicles** arriving stochastically (Poisson process) with time-of-day patterns (morning NS rush, evening EW rush)
- **Pedestrians** requesting crossings at each intersection
- **Multiple signal phases** per intersection (NS green, EW green, left arrows, pedestrian, all-red)
- **Incidents** including accidents (lane blockages), emergency vehicles (requiring preemption), and stadium events (traffic surges)

Each simulation step represents **5 real-world seconds**. The agent observes queue lengths, signal states, pedestrian wait times, and active incidents, then decides signal configurations for each intersection.

## Action Space

Each step, the agent submits a `MultiAction` containing one action per intersection:

| Action Type | Parameters | Description |
|---|---|---|
| `set_phase` | `intersection_id`, `phase` | Change signal to: `ns_green`, `ew_green`, `ns_left_arrow`, `ew_left_arrow`, `all_red`, `pedestrian` |
| `extend_phase` | `intersection_id`, `extend_seconds` (5–30) | Extend current phase duration |
| `emergency_preempt` | `intersection_id`, `preempt_direction` | Create green corridor for emergency vehicle |
| `noop` | — | Do nothing |

## Observation Space

Each step returns an `Observation` containing:

| Field | Type | Description |
|---|---|---|
| `step_number` | int | Current simulation step |
| `time_of_day` | str | HH:MM format (24hr) |
| `intersections` | list | Per-intersection: signal phase, timer, lane queues (through + left per direction), pedestrian counts and max wait |
| `incidents` | list | Active incidents with type, location, description, time remaining |
| `total_vehicles_waiting` | int | Sum of all queues |
| `total_vehicles_cleared` | int | Cumulative throughput |

## Tasks

| Task | Difficulty | Steps | Description |
|---|---|---|---|
| `easy` | Easy | 120 | Single intersection, steady traffic. Optimize phase timing. |
| `medium` | Medium | 360 | 2×2 grid with pedestrians. Coordinate green waves across rush hour pattern shifts. |
| `hard` | Hard | 480 | Full grid with incidents: lane-blocking accident, emergency vehicle preemption, and stadium traffic surge. |

### Baseline Scores (Heuristic Agent)

| Task | Score | Notes |
|---|---|---|
| `easy` | ~0.35 | Fixed-time alternating baseline |
| `medium` | ~0.25 | No coordination strategy |
| `hard` | ~0.20 | No incident adaptation |

## Reward Function

The reward provides **dense signal** every step with interpretable components:

| Component | Description |
|---|---|
| **Throughput** (+) | Vehicles cleared this step |
| **Wait penalty** (−) | Proportional to total vehicles queued |
| **Pedestrian penalty** (−) | Escalating penalty for wait > 60s, heavy at > 90s |
| **Emergency penalty** (−) | Heavy per-step penalty while emergency vehicle waits |
| **Starvation penalty** (−) | Penalty if any direction gets no green for > 3 minutes |
| **Flicker penalty** (−) | Penalty for > 3 phase changes in 30 seconds |

## Setup

### Local Development

```bash
# Clone and install
pip install -r requirements.txt

# Validate environment mechanics
python test_env.py

# Run baseline with heuristic agent (no API needed)
python inference.py --mock

# Run with LLM agent
export API_BASE_URL="https://your-api-endpoint"
export MODEL_NAME="your-model-name"
python inference.py
```

### Docker

```bash
docker build -t traffic-signal-bench .
docker run -p 7860:7860 traffic-signal-bench
```

Then visit `http://localhost:7860/visualize` for the visual debugger.

### API Usage

```bash
# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard", "seed": 42}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"actions": [{"action_type": "set_phase", "intersection_id": "int_0_0", "phase": "ns_green"}]}'

# Get current state
curl http://localhost:7860/state

# Grade trajectory
curl -X POST http://localhost:7860/grade

# Get replay data
curl http://localhost:7860/replay
```

### Visual Debugger

After running an episode (via `inference.py` or API), visit `/visualize` for an interactive replay:
- Play/pause, step forward/back, seek with slider
- Real-time queue visualization on the 2×2 grid
- Signal phase indicators (green/red per direction)
- Incident markers and pedestrian wait alerts
- Score breakdown panel
- Keyboard shortcuts: Space (play/pause), ←/→ (step)

## Project Structure

```
traffic-signal-bench/
├── models.py          # Pydantic models (Observation, Action, Reward)
├── simulator.py       # Traffic simulation engine
├── environment.py     # OpenEnv environment (step/reset/state)
├── graders.py         # Task graders (0.0–1.0 scoring)
├── server.py          # FastAPI server
├── inference.py       # Baseline inference script
├── test_env.py        # Environment validation tests
├── openenv.yaml       # OpenEnv specification
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container deployment
├── static/
│   └── visualizer.html  # Visual debugger
└── README.md
```

## License

MIT
