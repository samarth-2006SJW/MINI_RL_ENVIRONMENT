---
title: Warehouse Logistics RL Environment
emoji: 🏭
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - logistics
---

# Warehouse Logistics Exception Handler

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment that simulates a dynamic warehouse control system. An AI agent acts as the **Central Dispatcher**, resolving cascading logistics failures — robot breakdowns, inventory shortages, and shipment delays — across three escalating difficulty levels.

## Quick Start

```python
from environment import WarehouseEnvironment

# Create environment (choose: "easy", "medium", "hard")
env = WarehouseEnvironment(
    map_path="configs/warehouse_map.json",
    scenario_path="configs/scenarios.yaml",
    scenario_name="medium"
)

# Reset and get initial observation
obs = env.reset()
print(f"Active exceptions: {len(obs['active_exceptions'])}")

# Step with an action
action = {"command_type": "REROUTE_ORDER", "target_id": "EX_002"}
obs, reward, done, info = env.step(action)
print(f"Reward: {reward:.4f}  Done: {done}")
print(f"Events: {info['event_log']}")
```

## Server Setup

### Docker (Recommended)

```bash
docker build -t warehouse-rl:latest .
docker run --rm -p 7860:7860 \
  -e HF_TOKEN=your_api_key \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  warehouse-rl:latest
```

### Without Docker

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The interactive Gradio UI starts at `http://localhost:7860`.

## Running Inference (Hackathon Evaluation)

The hackathon validator runs `inference.py` directly and checks its structured stdout.

**Required environment variables:**

| Variable | Description |
|---|---|
| `HF_TOKEN` | Your Hugging Face / LLM provider API key |
| `API_BASE_URL` | LLM endpoint (e.g. `https://api.openai.com/v1`) |
| `MODEL_NAME` | Model identifier (e.g. `gpt-4o-mini`) |

```bash
export HF_TOKEN="your_key"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
python inference.py
```

Expected output format:

```
[START] task=easy env=warehouse-logistics-v1 model=gpt-4o-mini
[STEP] step=1 action=REROUTE_ORDER reward=1.0000 done=true error=null
[END] success=true steps=1 score=0.9900
[START] task=medium ...
...
```

## Action Space

| Field | Type | Description |
|---|---|---|
| `command_type` | str | One of: `MOVE_ROBOT`, `REROUTE_ORDER`, `DISPATCH_MAINTENANCE`, `REQUEST_RESTOCK`, `ASSIGN_WORKER`, `RE_POLL_SENSOR`, `WAIT` |
| `target_id` | str \| None | ID of the robot, exception, or order to act on |
| `parameters` | dict | Optional params (e.g. `{"target_location": [3, 4]}` for `MOVE_ROBOT`, `{"component_name": "component_A"}` for `REQUEST_RESTOCK`) |

## Observation Space (`WarehouseState`)

| Field | Type | Description |
|---|---|---|
| `time_step` | int | Current simulation step |
| `worker_availability` | int | Available human workers |
| `inventory_status` | dict | Component name → stock level |
| `robots` | list | List of `Robot` objects (id, location, status, battery_level) |
| `blocked_paths` | list | List of `BlockedPath` objects (id, location, obstruction_type, severity) |
| `active_exceptions` | list | List of `ExceptionIssue` objects requiring resolution |

## Task Scenarios

| Task | Description | Success Criteria |
|---|---|---|
| `easy` | Single shipment delay → locate component, reroute order | All exceptions resolved, inventory above threshold |
| `medium` | Inventory shortage → verify stock, restock, notify | Easy criteria + all inventory above medium threshold |
| `hard` | Cascading robot failure → dispatch maintenance, clear aisle, reroute | Medium criteria + all robots active, battery above threshold |

## Reward

Reward is strictly bounded in `[0.0, 1.0]` per episode using a **progress-delta** method:

- Each step: `reward = current_progress − previous_progress`
- `progress` = fraction of initial exceptions resolved ∈ [0.0, 1.0]
- All exceptions resolved → total episode reward = `1.0`
- No progress made → total episode reward = `0.0`

## Project Structure

```
warehouse-logistics-v1/
├── README.md               # This file
├── openenv.yaml            # OpenEnv manifest
├── Dockerfile              # Container image definition
├── app.py                  # Gradio UI + FastAPI /reset endpoint
├── inference.py            # Hackathon evaluation script
├── environment.py          # Core RL environment logic
├── models.py               # Pydantic schemas (Action & Observation)
├── tasks.py                # Per-difficulty graders (easy/medium/hard)
├── utils.py                # Config loading, collision detection
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Package metadata
└── configs/
    ├── openenv.yaml        # OpenEnv manifest
    ├── warehouse_map.json  # Grid layout, dimensions, charging stations
    ├── scenarios.yaml      # Per-scenario initial conditions
    ├── easy.yaml           # Task config reference (easy)
    ├── medium.yaml         # Task config reference (medium)
    └── hard.yaml           # Task config reference (hard)
```

## Learn More

- [OpenEnv Documentation](https://github.com/meta-pytorch/OpenEnv)
- [Meta PyTorch OpenEnv Hackathon](https://github.com/meta-pytorch/OpenEnv/blob/main/README.md)
