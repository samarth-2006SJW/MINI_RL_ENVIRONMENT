---
title: Warehouse Logistics RL Environment
emoji: 🏭
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: backend/api/app.py
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - logistics
---

# Warehouse Logistics Exception Handler

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment that simulates a dynamic warehouse control system. An AI agent acts as the **Central Dispatcher**, resolving cascading logistics failures — robot breakdowns, inventory shortages, and shipment delays — across three escalating difficulty levels.

## 🏗️ Project Architecture

Our project follows a modern, modular architecture that cleanly separates the RL logic, API layer, and React frontend.

```bash
MINI_RL_ENVIRONMENT/
├── backend/                  # Python Domain
│   ├── api/                  # FastAPI Application Layer
│   │   └── app.py            # Main Dashboard API (Serves Frontend)
│   ├── core/                 # Environment & Reinforcement Learning Logic
│   │   ├── environment.py    # Main OpenEnv Gym-style Interface
│   │   ├── models.py         # Pydantic state/observation schemas
│   │   ├── tasks.py          # RL Evaluation logic (easy/medium/hard)
│   │   └── utils.py          # Collision math and utility logic
│   ├── configs/              # Scenarios & OpenEnv Configurations
│   └── inference.py          # Hackathon validation baseline script
├── frontend/                 # React UI Domain
│   ├── src/                  # React component sources
│   ├── app/                  # Application routing & hooks
│   ├── package.json          # Node dependencies
│   └── vite.config.ts        # Vite configuration
├── scripts/                  
│   └── validate-submission.sh# The OpenEnv validation script
├── Dockerfile                # Multi-stage optimized Docker build
├── openenv.yaml              # OpenEnv Hackathon Manifest
└── pyproject.toml            # Strict Python metadata
```

---

## 🚀 Getting Started (Development Setup)

### 1. Frontend Setup (React & Vite)
To work on the UI Dashboard or compile the production web package:
```bash
cd frontend
npm install

# Option A: Run Vite Developer Server (Hot Reloading)
npm run dev

# Option B: Build Production Package (For Docker / FastAPI Serving)
npm run build
```

### 2. Backend Setup (FastAPI & RL Engine)
To run the simulation environment locally or test the LLM agent:

```bash
# Create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Start the API & Web Dashboard 
python backend/api/app.py
```
> **Note:** Due to internal `sys.path` injection, you can run `python backend/api/app.py` from the root folder without `PYTHONPATH` errors.

---

## 🐳 Running via Docker (Recommended)

To run the entire experience strictly as it will run on the Hugging Face deployed space, utilize the unified Dockerfile.

```bash
docker build -t warehouse-rl:latest .
docker run --rm -p 7860:7860 \
  -e HF_TOKEN=your_api_key \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  warehouse-rl:latest
```

The interactive React Dashboard will be available at `http://localhost:7860`.

---

## 🏆 Hackathon Evaluation (OpenEnv Validator)

The hackathon uses `backend/inference.py` directly and expects structured `[STEP]` logs to parse task performance.

**Required Environment Variables:**
| Variable | Description |
|---|---|
| `HF_TOKEN` | Your LLM provider API key |
| `API_BASE_URL` | LLM API endpoint (e.g., `https://api.openai.com/v1`) |
| `MODEL_NAME` | Model identifier (e.g., `gpt-4o-mini`) |

```bash
# Test the agent sequentially through all 3 difficulties
export HF_TOKEN="your_key"
export API_BASE_URL="https://api.openai.com/v1"
python backend/inference.py
```

### Final Submission Validation Loop

To double check that your Hugging Face Space passes the strict Meta evaluator, use the provided Bash validation script provided in `scripts/`:

```bash
# Requires Docker & openenv-core to be installed
./scripts/validate-submission.sh https://your-hf-space-url.hf.space .
```

---

## 🧠 Core Interfaces

### Action Space (`LogisticsCommand`)
| Field | Type | Description |
|---|---|---|
| `command_type` | `str` | `MOVE_ROBOT`, `REROUTE_ORDER`, `DISPATCH_MAINTENANCE`, `REQUEST_RESTOCK`, `ASSIGN_WORKER`, `RE_POLL_SENSOR`, `WAIT` |
| `target_id` | `str \| None` | ID of the robot, exception, or order to act on |
| `parameters` | `dict` | Optional params (e.g. `{"target_location": [3, 4]}`) |

### Observation Space (`WarehouseState`)
| Field | Type | Description |
|---|---|---|
| `time_step` | `int` | Current simulation frame |
| `worker_availability` | `int` | Available human workers |
| `inventory_status` | `dict` | Component name → stock level |
| `robots` | `list` | Active system robots (battery & status) |
| `blocked_paths` | `list` | Currently inaccessible zones |
| `active_exceptions` | `list` | Errors in the warehouse requiring intervention |

---

## 📊 Reward Logistics

Reward is strictly bounded in `[0.0, 1.0]` per episode using a **progress-delta** method:
- All exceptions resolved → total episode reward = `1.0`
- Partial resolving maps incrementally towards reward goals!
