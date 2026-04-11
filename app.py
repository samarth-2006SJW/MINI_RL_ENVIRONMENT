"""
FastAPI backend for Warehouse RL Environment.
Serves the React frontend from dist/ and exposes API endpoints
for environment interaction.
"""
import os
import json
import re
import time
from typing import Optional
from pathlib import Path

import openai
import asyncio

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

from environment import WarehouseEnvironment
from models import WarehouseState, RobotStatus

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(title="Warehouse RL - OVERSEER ENGINE")

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------
def _get_llm_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("LLM_BASE_URL", None)
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return openai.OpenAI(**client_kwargs)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
VALID_COMMANDS = {
    "MOVE_ROBOT",
    "REQUEST_RESTOCK",
    "DISPATCH_MAINTENANCE",
    "REROUTE_ORDER",
    "ASSIGN_WORKER",
    "RE_POLL_SENSOR",
    "WAIT",
}

def _build_prompt(state_text: str, recent_history: str, failed_actions: str) -> str:
    return f"""
### ROLE
You are the Autonomous Warehouse Controller. Your mission is to resolve logistics exceptions with ZERO wasted moves.

### CURRENT SYSTEM STATE
{state_text}

### RECENT ACTION HISTORY
{recent_history}

### AVOID THESE FAILED ACTIONS
{failed_actions}

### OPERATIONAL HIERARCHY (Strictly Follow)
1. CRITICAL: If 'Exception Count' > 0, identify the specific ID and issue the resolution command (REROUTE_ORDER, REQUEST_RESTOCK, or DISPATCH_MAINTENANCE).
2. COLLISION AVOIDANCE: Do NOT move a robot to a tile already occupied by another robot or obstacle. Results in -0.1 penalty.
3. MISSION COMPLETE: If 'Exception Count' is 0, your final action MUST be 'WAIT'.
4. REASONING: If 'Recent Action History' shows a 0.0 reward for a command, do NOT repeat it.

### OUTPUT SPECIFICATION
- Output ONLY a raw JSON dictionary. No markdown. No filler.

### JSON SCHEMA
{{"command_type": "MOVE_ROBOT | REQUEST_RESTOCK | DISPATCH_MAINTENANCE | REROUTE_ORDER | ASSIGN_WORKER | WAIT", "target_id": "ID of robot or order", "parameters": {{}}}}

Action:"""


def _extract_json_object(raw_text: str) -> Optional[str]:
    text = (raw_text or "").strip()
    if not text:
        return None
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part.replace("json", "").strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return candidate
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _normalize_action(action: dict) -> dict:
    if not isinstance(action, dict):
        return {"command_type": "WAIT", "target_id": None, "parameters": {}}
    cmd = str(action.get("command_type", "WAIT")).upper().strip()
    if cmd not in VALID_COMMANDS:
        cmd = "WAIT"
    target_id = action.get("target_id", None)
    parameters = action.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}
    return {"command_type": cmd, "target_id": target_id, "parameters": parameters}


def _call_llm_for_action(client: openai.OpenAI, prompt: str) -> tuple[dict, list[str]]:
    errors: list[str] = []
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a warehouse routing expert. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        raw_output = (response.choices[0].message.content or "").strip()
        json_blob = _extract_json_object(raw_output)
        if json_blob is None:
            raise ValueError("No JSON object found in model response.")
        parsed = json.loads(json_blob)
        return _normalize_action(parsed), errors
    except Exception as first_error:
        errors.append(f"Primary parse failed: {first_error}")
        # One controlled retry with strict repair instruction
        try:
            repair_prompt = (
                "Return ONLY valid JSON action object with keys command_type,target_id,parameters. "
                "No markdown, no explanation."
            )
            retry_response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Output only valid JSON."},
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": repair_prompt},
                ],
                temperature=0.0,
            )
            retry_text = (retry_response.choices[0].message.content or "").strip()
            retry_blob = _extract_json_object(retry_text)
            if retry_blob is None:
                raise ValueError("Retry returned no JSON object.")
            retry_parsed = json.loads(retry_blob)
            return _normalize_action(retry_parsed), errors
        except Exception as retry_error:
            errors.append(f"Retry parse failed: {retry_error}")
            return {"command_type": "WAIT", "target_id": None, "parameters": {}}, errors


def _action_reasoning(action: dict, scenario: str, exceptions_before: int) -> str:
    cmd = action.get("command_type", "WAIT")
    target = action.get("target_id", None)
    params = action.get("parameters", {}) or {}
    if cmd == "REQUEST_RESTOCK":
        comp = params.get("component_name", "unknown_component")
        return f"Inventory-first policy ({scenario}): restock `{comp}` to reduce shortage pressure."
    if cmd == "RE_POLL_SENSOR":
        return f"Visibility recovery policy ({scenario}): re-poll robot `{target}` sensor before routing."
    if cmd == "DISPATCH_MAINTENANCE":
        return f"Stability policy ({scenario}): maintenance for robot `{target}` to restore ACTIVE fleet."
    if cmd == "REROUTE_ORDER":
        return f"Exception resolution policy ({scenario}): reroute target `{target}` while exceptions={exceptions_before}."
    if cmd == "ASSIGN_WORKER":
        return f"Congestion policy ({scenario}): assign worker to unblock `{target}`."
    if cmd == "MOVE_ROBOT":
        return f"Mobility policy ({scenario}): move robot `{target}` with collision-aware pathing."
    return f"Conservative policy ({scenario}): WAIT selected due to low-confidence or completed state."


def _heuristic_action(env: WarehouseEnvironment, scenario: str) -> dict:
    state = env.current_state
    if state is None:
        return {"command_type": "WAIT", "target_id": None}

    if scenario == "easy":
        if state.active_exceptions:
            return {"command_type": "REROUTE_ORDER", "target_id": state.active_exceptions[0].id}
        return {"command_type": "WAIT", "target_id": None}

    if scenario == "medium":
        for comp_name, qty in state.inventory_status.items():
            if qty <= 50:
                return {
                    "command_type": "REQUEST_RESTOCK",
                    "target_id": None,
                    "parameters": {"component_name": comp_name},
                }
        if state.active_exceptions:
            return {"command_type": "REROUTE_ORDER", "target_id": state.active_exceptions[0].id}
        return {"command_type": "WAIT", "target_id": None}

    if scenario == "hard":
        for comp_name, qty in state.inventory_status.items():
            if qty <= 50:
                return {
                    "command_type": "REQUEST_RESTOCK",
                    "target_id": None,
                    "parameters": {"component_name": comp_name},
                }
        for robot in state.robots:
            if robot.status == RobotStatus.SENSOR_FAILURE:
                return {"command_type": "RE_POLL_SENSOR", "target_id": robot.id}
        for robot in state.robots:
            if robot.status != RobotStatus.ACTIVE:
                return {"command_type": "DISPATCH_MAINTENANCE", "target_id": robot.id}
        if state.active_exceptions:
            return {"command_type": "REROUTE_ORDER", "target_id": state.active_exceptions[0].id}
        return {"command_type": "WAIT", "target_id": None}

    return {"command_type": "WAIT", "target_id": None}


def _format_state_for_prompt(state: WarehouseState) -> str:
    """Convert warehouse state to a text summary for the LLM prompt."""
    if state is None:
        return "No state available."
    robots_info = []
    for r in state.robots:
        robots_info.append(
            f"  - {r.id}: status={r.status}, battery={r.battery_level:.1f}%, location={r.location}"
        )
    exceptions_info = []
    for e in state.active_exceptions:
        exceptions_info.append(f"  - {e.id}: type={e.type}, severity={e.severity}")
    inventory_info = [f"  - {k}: {v}" for k, v in state.inventory_status.items()]
    return (
        f"Time Step: {state.time_step}\n"
        f"Exception Count: {len(state.active_exceptions)}\n"
        f"Robots:\n" + "\n".join(robots_info) + "\n"
        f"Active Exceptions:\n" + ("\n".join(exceptions_info) if exceptions_info else "  None") + "\n"
        f"Inventory Status:\n" + "\n".join(inventory_info) + "\n"
        f"Blocked Paths: {state.blocked_paths}\n"
    )


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "OVERSEER ENGINE ONLINE"}


@app.post("/reset")
@app.post("/api/reset")
def reset_endpoint():
    return {"status": "ok", "message": "Environment reset triggered."}


class SimulationRequest(BaseModel):
    scenario: str = "medium"
    max_steps: int = 20


@app.get("/api/scenario/{name}")
def get_scenario_grid(name: str):
    root = Path(__file__).parent
    try:
        env = WarehouseEnvironment(
            map_path=str(root / "configs" / "warehouse_map.json"),
            scenario_path=str(root / "configs" / "scenarios.yaml"),
            scenario_name=name,
        )
        state = env.reset()
        grid_state = {
            'grid_size': env.map_config.get('dimensions', [10, 10]),
            'robots': [{"id": r["id"], "location": list(r["location"]) if r.get("location") else None, "status": r.get("status"), "battery": r.get("battery_level", 100)} for r in state.get("robots", [])],
            'obstacles': [list(bp["location"]) for bp in state.get("blocked_paths", [])],
            'stations': [{"location": s} for s in env.map_config.get('charging_stations', [])]
        }
        return {"grid_state": grid_state}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/api/simulate")
def run_simulation_api(req: SimulationRequest):
    """Run a full simulation and return results (non-streaming for simplicity)."""
    client = _get_llm_client()
    root = Path(__file__).parent
    env = WarehouseEnvironment(
        map_path=str(root / "configs" / "warehouse_map.json"),
        scenario_path=str(root / "configs" / "scenarios.yaml"),
        scenario_name=req.scenario,
    )
    state = env.reset()

    total_reward = 0.0
    total_actions = 0
    successful_actions = 0
    failed_action_memory = []
    action_history = []
    plot_data = []
    use_heuristic = client is None

    for step in range(1, req.max_steps + 1):
        state_text = _format_state_for_prompt(env.current_state)
        exceptions_before = len(env.current_state.active_exceptions)

        recent_history_str = "None"
        if action_history:
            clean_history = [re.sub(r'<[^>]+>', '', entry) for entry in action_history[-3:]]
            recent_history_str = json.dumps(clean_history)

        failed_actions_str = "None"
        if failed_action_memory:
            failed_actions_str = json.dumps(list(set(failed_action_memory[-5:])))

        error_logs = []
        if use_heuristic:
            action = _heuristic_action(env, req.scenario)
        else:
            prompt = _build_prompt(state_text, recent_history_str, failed_actions_str)
            action, error_logs = _call_llm_for_action(client, prompt)

        _, reward, terminated, info = env.step(action)

        total_reward += reward
        total_actions += 1
        action_key = f"{action.get('command_type')}:{action.get('target_id')}"
        if reward <= 0.0 and action.get("command_type") != "WAIT":
            failed_action_memory.append(action_key)
        if reward > 0.0:
            successful_actions += 1

        logs = info.get("event_log", ["Agent waited."])
        if error_logs:
            logs = error_logs + logs
        exceptions_after = len(env.current_state.active_exceptions)

        log_entry = f"[T:{step:02d}] OP:{action['command_type']} TGT:{action.get('target_id','N/A')} V:+{reward:.2f} | {'; '.join(logs)}"
        action_history.append(log_entry)
        plot_data.append({"step": step, "total_reward": round(total_reward, 3)})

        if terminated or exceptions_after == 0:
            break

    action_eff = (successful_actions / total_actions) if total_actions > 0 else 0.0
    return {
        "scenario": req.scenario,
        "total_reward": round(total_reward, 3),
        "steps_taken": len(plot_data),
        "action_efficiency": round(action_eff, 3),
        "exceptions_remaining": len(env.current_state.active_exceptions),
        "plot_data": plot_data,
        "action_log": action_history,
    }


@app.post("/api/simulate/stream")
async def run_simulation_stream_api(req: SimulationRequest):
    """Run a full simulation and stream results via Server-Sent Events (SSE)."""
    
    async def event_generator():
        client = _get_llm_client()
        root = Path(__file__).parent
        env = WarehouseEnvironment(
            map_path=str(root / "configs" / "warehouse_map.json"),
            scenario_path=str(root / "configs" / "scenarios.yaml"),
            scenario_name=req.scenario,
        )
        state = env.reset()

        total_reward = 0.0
        total_actions = 0
        successful_actions = 0
        failed_action_memory = []
        action_history = []
        plot_data = []
        use_heuristic = client is None

        for step in range(1, req.max_steps + 1):
            state_text = _format_state_for_prompt(env.current_state)
            exceptions_before = len(env.current_state.active_exceptions)

            recent_history_str = "None"
            if action_history:
                clean_history = [re.sub(r'<[^>]+>', '', entry) for entry in action_history[-3:]]
                recent_history_str = json.dumps(clean_history)

            failed_actions_str = "None"
            if failed_action_memory:
                failed_actions_str = json.dumps(list(set(failed_action_memory[-5:])))

            error_logs = []
            if use_heuristic:
                action = _heuristic_action(env, req.scenario)
            else:
                prompt = _build_prompt(state_text, recent_history_str, failed_actions_str)
                action, error_logs = _call_llm_for_action(client, prompt)

            _, reward, terminated, info = env.step(action)

            total_reward += reward
            total_actions += 1
            action_key = f"{action.get('command_type')}:{action.get('target_id')}"
            if reward <= 0.0 and action.get("command_type") != "WAIT":
                failed_action_memory.append(action_key)
            if reward > 0.0:
                successful_actions += 1

            logs = info.get("event_log", ["Agent waited."])
            if error_logs:
                logs = error_logs + logs
            exceptions_after = len(env.current_state.active_exceptions)

            log_entry = f"[T:{step:02d}] OP:{action['command_type']} TGT:{action.get('target_id','N/A')} V:+{reward:.2f} | {'; '.join(logs)}"
            action_history.append(log_entry)
            plot_data.append({"step": step, "total_reward": round(total_reward, 3)})

            action_eff = (successful_actions / total_actions) if total_actions > 0 else 0.0
            metrics = {"total_reward": round(total_reward, 3), "action_efficiency": round(action_eff, 3)}
            
            # Serialize state for React Native Grid
            grid_state = {
                'grid_size': env.map_config.get('dimensions', [10, 10]),
                'robots': [{"id": r.id, "location": list(r.location) if r.location else None, "status": r.status.value if hasattr(r.status, 'value') else r.status, "battery": r.battery_level} for r in env.current_state.robots],
                'obstacles': [list(bp.location) for bp in env.current_state.blocked_paths],
                'stations': [{"location": s} for s in env.map_config.get('charging_stations', [])]
            }

            update_data = {
                "step": step,
                "scenario": req.scenario,
                "total_reward": round(total_reward, 3),
                "action_efficiency": round(action_eff, 3),
                "exceptions_remaining": exceptions_after,
                "active_exceptions": [{"id": e.id, "type": e.type.value if hasattr(e.type, 'value') else e.type, "severity": e.severity, "location": getattr(e, 'location', None)} for e in env.current_state.active_exceptions],
                "plot_data": plot_data,
                "action_log": action_history,
                "grid_state": grid_state,
                "terminated": terminated or exceptions_after == 0
            }

            yield f"data: {json.dumps(update_data)}\n\n"
            await asyncio.sleep(0.5)

            if update_data["terminated"]:
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")




# ---------------------------------------------------------------------------
# Static Files — serve the React SPA build from dist/
# ---------------------------------------------------------------------------
DIST_DIR = Path(__file__).parent / "dist"

if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve index.html for all non-API routes (SPA client-side routing)."""
        file_path = DIST_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(DIST_DIR / "index.html"), headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        })
else:
    @app.get("/")
    def no_build():
        return JSONResponse(
            {"error": "Frontend not built. Run 'npm run build' first."},
            status_code=503,
        )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
