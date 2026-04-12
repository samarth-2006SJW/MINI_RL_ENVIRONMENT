---
title: Logistics Exception Handler
emoji: 🏭
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
---

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-10b981)](https://github.com/OpenEnv/spec)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Automated Logistics Exception Handler** is an **intelligent logistics platform** and Reinforcement Learning (RL) environment built for the **Meta AI PyTorch OpenEnv Hackathon**. It empowers foundation models and autonomous agents to manage complex warehouse logistics, resolving cascading failures like robot hardware faults, inventory shortages, and blocked pathways in real-time.

---

## 🧠 What It Does

By integrating an interactive visual dashboard with a dynamic physics-based backend, the platform provides a rigorous testing ground for logistics agents acting as the Central Dispatcher over automated warehouse robotics.

- **Dynamic Exception Engine**: Procedurally generates logistics failures (e.g. `ROBOT_BATTERY_LOW`, `PATH_BLOCKED`, `ORDER_DELAYED`).
- **Chain-of-Thought Ready**: Native integration with large language models generating step-by-step reasoning logs.
- **Interactive UI**: A rich Vite/React dashboard with midnight-indigo glassmorphism design, rendering the environment in real time.
- **OpenEnv Compliance**: Fully compatible with the OpenEnv specification for zero-shot RL evaluation.

---

## 🛠 Tech Stack
The project is architected as a modern **monorepo** for seamless development and deployment.

- **Backend**: Python 3.10+, FastAPI, Uvicorn, Python dataclasses.
- **Frontend**: React, Vite, TailwindCSS (Midnight Indigo Glassmorphism), Lucide Icons.
- **Infrastructure**: Docker multi-stage builds.
- **Agent Integration**: Hugging Face Hub, OpenAI-compatible APIs, `openenv-core`.

---

## 📂 Repository Structure
```text
MINI_RL_ENVIRONMENT/
├── backend/                  # Python Domain (FastAPI server, RL logic)
│   ├── api/                  # FastAPI Application Layer
│   ├── core/                 # Environment & Reinforcement Learning Logic (OpenEnv Spec)
│   ├── configs/              # Configurations & Schema references
│   └── inference.py          # Baseline agent evaluation script (Compliance optimized)
├── frontend/                 # React UI Domain
│   ├── src/                  # React component sources
│   ├── app/                  # Application routing & visuals
│   └── package.json          # Node dependencies
├── server/                   # Production validator shims
├── scripts/                  # Bash scripts for OpenEnv validation
├── Dockerfile                # Docker image optimized for deployment
└── openenv.yaml              # OpenEnv Hackathon Manifest
```

---

### Prerequisites
- **Node.js**: >= 18.0.0
- **Python**: >= 3.10
- **Git**

### Quick Start (Recommended)
1. **Clone the repository**:
   ```bash
   git clone https://github.com/samarth-2006SJW/MINI_RL_ENVIRONMENT.git
   cd MINI_RL_ENVIRONMENT
   ```

2. **Install Dependencies**:
   ```bash
   # Install and build React frontend dependencies
   cd frontend
   npm install
   npm run build
   cd ..
   
   # Setup Python backend virtual environment
   python -m venv .venv
   
   # Windows Activation:
   .venv\Scripts\activate
   # Linux/Mac Activation:
   source .venv/bin/activate
   
   # Install core dependencies
   pip install -r requirements.txt
   ```

3. **Run Dev Environment**:
   ```bash
   python backend/api/app.py
   ```
   > The Interactive Dashboard and API will now run locally on [http://localhost:7860](http://localhost:7860).

---

## 🐳 Running via Docker (Production)

You can run the environment identically to how it corresponds to the Hugging Face Spaces environment execution flow by utilizing the unified Docker build:

```bash
# Build the Docker image
docker build -t warehouse-rl:latest .

# Run the container
docker run --rm -p 7860:7860 \
  -e HF_TOKEN="your_huggingface_token" \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  warehouse-rl:latest
```

---

## 🧠 RL Evaluation (OpenEnv Compliance)

The environment supports three escalating evaluation tasks evaluated natively by the `openenv-core` schema logic:

1. `easy_warehouse`: Base inventory logistics.
2. `medium_warehouse`: Escalated robot failures and path blocking.
3. `hard_warehouse`: Maximize multi-modal exception resolution and throughput.

**Running Baseline Evaluation**:
The `inference.py` script triggers the standard evaluation loop for OpenEnv validation, providing native structured logs out of the box in the `[START]`, `[STEP]`, and `[END]` formats:

```bash
export HF_TOKEN="your_huggingface_token"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
python backend/inference.py
```

---

## 🏆 Project Status
| Check | Status |
|---|---|
| **OpenEnv Validate Tool** | Passing (`[OK] Ready`) ✅ |
| **OpenEnv Spec** | Compliant (v0.1.0) 🟢 |
| **Docker Build Pipeline** | Successful 🐳 |
| **Hugging Face Deployment** | Live 🚀 |

---

## 📜 License
Distributed under the **MIT License**.
