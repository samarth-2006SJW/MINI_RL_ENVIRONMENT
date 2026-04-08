import json
import os
import time
from pathlib import Path

import openai

from environment import WarehouseEnvironment

# Mandatory Variables from Environment
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASKS = [
    ("warehouse-easy", "easy"),
    ("warehouse-medium", "medium"),
    ("warehouse-hard", "hard"),
]
MAX_STEPS = 15
MIN_SCORE = 0.01
MAX_SCORE = 0.99


def _safe_action_from_model(client: openai.OpenAI | None, env: WarehouseEnvironment) -> dict:
    if client is None:
        return {"command_type": "WAIT", "target_id": None}
    prompt = (
        f"Current State: {env._state_to_text()}\n"
        "Output ONLY raw JSON dictionary for Warehouse Action."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw_out = (response.choices[0].message.content or "").strip()
        if "```" in raw_out:
            raw_out = raw_out.split("```")[1].replace("json", "").strip()
        action = json.loads(raw_out)
        if not isinstance(action, dict):
            return {"command_type": "WAIT", "target_id": None}
        return action
    except Exception:
        return {"command_type": "WAIT", "target_id": None}


def _bounded_score(rewards: list[float]) -> float:
    raw_score = sum(rewards) / 5.0 if rewards else 0.0
    return min(MAX_SCORE, max(MIN_SCORE, raw_score))


def _run_task(client: openai.OpenAI | None, root: Path, task_name: str, scenario: str) -> None:
    env = WarehouseEnvironment(
        str(root / "configs" / "warehouse_map.json"),
        str(root / "configs" / "scenarios.yaml"),
        scenario,
    )

    print(f"[START] task={task_name} env=openenv-v1 model={MODEL_NAME}", flush=True)

    rewards: list[float] = []
    steps = 0
    success = "false"
    score = MIN_SCORE

    try:
        env.reset()
        done = False
        for step in range(1, MAX_STEPS + 1):
            steps = step
            action = _safe_action_from_model(client, env)
            _, reward, done, _ = env.step(action)
            rewards.append(float(reward))
            action_name = action.get("command_type", "WAIT")
            print(
                f"[STEP] step={step} action={action_name} reward={reward:.2f} "
                f"done={str(done).lower()} error=null",
                flush=True,
            )
            if done:
                break
            time.sleep(0.05)
        score = _bounded_score(rewards)
        success = "true" if score >= 0.1 else "false"
    except Exception:
        score = MIN_SCORE
        success = "false"

    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={success} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def main() -> None:
    root = Path(__file__).parent
    client = openai.OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None
    for task_name, scenario in TASKS:
        _run_task(client, root, task_name, scenario)


if __name__ == "__main__":
    main()
