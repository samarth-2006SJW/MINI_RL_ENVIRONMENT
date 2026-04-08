import os
import json
import time
from typing import Optional
import gradio as gr
import openai
from environment import WarehouseEnvironment
from models import WarehouseState, RobotStatus
from fastapi import FastAPI
import uvicorn

app_api = FastAPI()

@app_api.post("/reset")
def reset_endpoint():
    return {"status": "ok", "message": "Environment reset triggered."}

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

# ---------------------------------------------------------------------------
# Visual Renderer
# ---------------------------------------------------------------------------
def render_warehouse_grid(env: WarehouseEnvironment):
    """Converts the environment state into a visual Markdown grid."""
    dims = env.map_config.get("dimensions", [10, 10])
    width, height = dims[0], dims[1]
    
    # Initialize grid with empty floor
    grid = [["⬜" for _ in range(width)] for _ in range(height)]
    
    # Add charging stations
    for station in env.map_config.get("charging_stations", []):
        grid[station[1]][station[0]] = "🔌"
        
    # Add blockages
    for block in env.current_state.blocked_paths:
        grid[block.location[1]][block.location[0]] = "🚧"
        
    # Add exceptions (Orders/Packages)
    for ex in env.current_state.active_exceptions:
        # If the exception is a shortage, we don't have a location, but if it's a breakdown, the robot is the location.
        pass

    # Add robots
    for robot in env.current_state.robots:
        if robot.location:
            icon = "🤖"
            if robot.status == "MAINTENANCE": icon = "🛠️"
            elif robot.status == "SENSOR_FAILURE": icon = "❓"
            grid[robot.location[1]][robot.location[0]] = icon

    # Build Markdown table
    header = "| " + " | ".join([str(i) for i in range(width)]) + " |"
    divider = "| " + " | ".join(["---" for _ in range(width)]) + " |"
    rows = []
    for y in range(height):
        rows.append(f"| " + " | ".join(grid[y]) + f" |")
        
    return f"### Warehouse Live View (10x10)\n\n" + "\n".join([header, divider] + rows)

# ---------------------------------------------------------------------------
# Simulation Logic
# ---------------------------------------------------------------------------
def run_simulation(scenario, max_steps):
    client = _get_llm_client()
    use_heuristic = client is None

    env = WarehouseEnvironment(
        map_path="configs/warehouse_map.json",
        scenario_path="configs/scenarios.yaml",
        scenario_name=scenario
    )
    env.reset()
    
    action_history = []
    failed_action_memory = []
    total_reward = 0.0
    
    for step in range(1, max_steps + 1):
        state_text = env._state_to_text()
        recent_history_str = "\n".join(action_history[-5:]) if action_history else "No history."
        failed_actions_str = ", ".join(failed_action_memory[-5:]) if failed_action_memory else "None"
        
        # 1. Visualize Grid
        # Initial status for thinking state
        exceptions_before = len(env.current_state.active_exceptions)
        mode_label = "Heuristic Fallback" if use_heuristic else "LLM"
        status_md = f"**Step:** {step} | **Mode:** {mode_label} | **Exceptions:** {exceptions_before} | **Total Reward:** {total_reward:.2f} | ⏳ *Generating Action...*"
        
        yield render_warehouse_grid(env), status_md, "\n".join(action_history) if action_history else "System initialized. Agent is preparing..."
        
        # 3. Get LLM Action
        error_logs = []
        if use_heuristic:
            action = _heuristic_action(env, scenario)
        else:
            prompt = _build_prompt(state_text, recent_history_str, failed_actions_str)
            action, error_logs = _call_llm_for_action(client, prompt)
            
        # 4. Step Environment
        _, reward, terminated, info = env.step(action)
            
        total_reward += reward
        action_key = f"{action.get('command_type')}:{action.get('target_id')}"
        if reward <= 0.0 and action.get("command_type") != "WAIT":
            failed_action_memory.append(action_key)
        
        # Log entry
        logs = info.get("event_log", ["Agent waited."])
        if error_logs:
            logs = error_logs + logs
        log_entry = (
            f"Step {step}: mode={'heuristic' if use_heuristic else 'llm'} "
            f"action={json.dumps(action)} reward={reward:.2f} "
            f"exceptions={len(env.current_state.active_exceptions)} "
            f"notes={'; '.join(logs)}"
        )
        action_history.append(log_entry)
        
        # Update metrics AFTER the step
        exceptions_after = len(env.current_state.active_exceptions)
        final_status = f"**Step:** {step} | **Exceptions:** {exceptions_after} | **Total Reward:** {total_reward:.2f}"
        
        # Yield update
        if terminated or exceptions_after == 0:
            if total_reward > 0 and exceptions_after == 0:
                final_status += " | 🏆 **MISSION COMPLETE!**"
            else:
                final_status += " | 🚨 **MISSION ENDED/FAILED**"
            yield render_warehouse_grid(env), final_status, "\n".join(action_history)
            break
            
        yield render_warehouse_grid(env), final_status, "\n".join(action_history)
        time.sleep(1) # Slow down for visualization

# ---------------------------------------------------------------------------
# Gradio UI Shell
# ---------------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")) as demo:
    gr.Markdown("# 🤖 Autonomous Warehouse RL Controller")
    gr.Markdown("Watch an LLM-based agent resolve warehouse logistics exceptions in real-time.")
    
    with gr.Row():
        with gr.Column(scale=1):
            scenario = gr.Dropdown(choices=["easy", "medium", "hard"], value="medium", label="Select Scenario")
            steps = gr.Slider(minimum=5, maximum=50, value=20, step=1, label="Max Steps")
            start_btn = gr.Button("🚀 Start Simulation", variant="primary")
            
        with gr.Column(scale=2):
            status = gr.Markdown("### Status: Ready")
            grid = gr.Markdown("### Grid will appear here...")
            
    with gr.Row():
        logs = gr.Textbox(label="Agent Reasoning & Event Logs", lines=10, interactive=False)

    start_btn.click(run_simulation, inputs=[scenario, steps], outputs=[grid, status, logs])

if __name__ == "__main__":
    # Mount Gradio onto the FastAPI app
    app = gr.mount_gradio_app(app_api, demo, path="/")
    uvicorn.run(app, host="0.0.0.0", port=7860)

