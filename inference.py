import json
import os
import time
from pathlib import Path

import openai

from environment import WarehouseEnvironment

# ── Mandatory OpenEnv Hackathon Variables ────────────────────────────────────
# These are injected by the Meta evaluation backend.
# HF_TOKEN  → your Hugging Face / provider API key
# API_BASE_URL → LLM endpoint (e.g. https://api.openai.com/v1)
# MODEL_NAME   → model identifier (e.g. gpt-4o-mini)
API_KEY      = os.environ["HF_TOKEN"]
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")

# ── Tasks: (task_name, scenario_name) ────────────────────────────────────────
# Exactly 3 tasks are required by the hackathon validator.
TASKS = [
    ("easy",   "easy"),
    ("medium", "medium"),
    ("hard",   "hard"),
]

MAX_STEPS = 15   # Well within the 20-min runtime limit on 2vCPU/8GB


def _safe_action_from_model(client: openai.OpenAI, env: WarehouseEnvironment) -> dict:
    """Call the LLM and parse a valid action JSON dict. Falls back to WAIT on any error."""
    prompt = (
        f"Current Warehouse State: {env._state_to_text()}\n"
        "Output ONLY a raw JSON dictionary with keys: command_type, target_id, parameters.\n"
        "Valid command_type values: MOVE_ROBOT, REROUTE_ORDER, DISPATCH_MAINTENANCE, "
        "REQUEST_RESTOCK, ASSIGN_WORKER, RE_POLL_SENSOR, WAIT"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "").strip()
        # Strip markdown code fences if present
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        action = json.loads(raw)
        if isinstance(action, dict):
            return action
    except Exception:
        pass
    return {"command_type": "WAIT", "target_id": None}


def _run_task(client: openai.OpenAI, root: Path, task_name: str, scenario: str) -> None:
    """Run a single task episode and emit [START] / [STEP] / [END] structured logs."""
    env = WarehouseEnvironment(
        str(root / "configs" / "warehouse_map.json"),
        str(root / "configs" / "scenarios.yaml"),
        scenario,
    )

    # ── [START] line ─────────────────────────────────────────────────────────
    print(f"[START] task={task_name} env=warehouse-logistics-v1 model={MODEL_NAME}", flush=True)

    cumulative_reward = 0.0
    steps = 0
    success = "false"

    try:
        env.reset()
        done = False
        for step in range(1, MAX_STEPS + 1):
            steps = step
            action = _safe_action_from_model(client, env)
            _, reward, done, _ = env.step(action)
            cumulative_reward += float(reward)
            action_name = action.get("command_type", "WAIT")

            # ── [STEP] line ───────────────────────────────────────────────────
            print(
                f"[STEP] step={step} action={action_name} reward={reward:.4f} "
                f"done={str(done).lower()} error=null",
                flush=True,
            )
            if done:
                break
            time.sleep(0.05)

        # Final score is the cumulative reward, already bounded to (0.0, 1.0)
        # by environment.py's progress-delta reward scheme.
        score = round(max(0.01, min(0.99, cumulative_reward)), 4)
        success = "true" if score >= 0.1 else "false"

    except Exception as exc:
        score = 0.01
        success = "false"
        print(f"[STEP] step={steps} action=WAIT reward=0.0000 done=true error={exc!r}", flush=True)

    # ── [END] line ────────────────────────────────────────────────────────────
    print(
        f"[END] success={success} steps={steps} score={score:.4f}",
        flush=True,
    )


def main() -> None:
    root   = Path(__file__).parent
    client = openai.OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_name, scenario in TASKS:
        _run_task(client, root, task_name, scenario)


if __name__ == "__main__":
    main()
