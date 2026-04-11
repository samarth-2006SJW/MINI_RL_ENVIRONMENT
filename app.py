import os
import json
import time
from typing import Optional
import gradio as gr
import pandas as pd
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


def _kpi_markdown(
    step: int,
    mode_label: str,
    total_reward: float,
    exceptions_before: int,
    exceptions_after: int,
    resolved_count: int,
    successful_actions: int,
    total_actions: int,
) -> str:
    action_eff = (successful_actions / total_actions) if total_actions > 0 else 0.0
    return (
        "### Live KPIs\n"
        f"- Mode: **{mode_label}**\n"
        f"- Step: **{step}**\n"
        f"- Total Reward: **{total_reward:.2f}**\n"
        f"- Exceptions: **{exceptions_before} -> {exceptions_after}**\n"
        f"- Resolved Exceptions: **{resolved_count}**\n"
        f"- Action Efficiency: **{action_eff:.0%}**"
    )


def _scenario_summary_markdown(env: WarehouseEnvironment, scenario: str, max_steps: int) -> str:
    state = env.current_state
    if state is None:
        return "### Scenario Summary\n- Environment not initialized"
    low_stock = [k for k, v in state.inventory_status.items() if v <= 50]
    low_stock_txt = ", ".join(low_stock) if low_stock else "none"
    return (
        "### Scenario Summary\n"
        f"- Scenario: **{scenario}**\n"
        f"- Episode Cap (UI): **{max_steps}** steps\n"
        f"- Initial Exceptions: **{len(state.active_exceptions)}**\n"
        f"- Robots: **{len(state.robots)}**\n"
        f"- Blocked Paths: **{len(state.blocked_paths)}**\n"
        f"- Low-Stock Components (<=50): **{low_stock_txt}**"
    )


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
def render_warehouse_grid(env: WarehouseEnvironment) -> str:
    """Converts the environment state into a highly polished HTML/CSS visual grid."""
    dims = env.map_config.get("dimensions", [10, 10])
    width, height = dims[0], dims[1]
    
    robots_map = {tuple(r.location): r for r in env.current_state.robots if r.location}
    stations_set = {tuple(loc) for loc in env.map_config.get("charging_stations", [])}
    blocked_set = {tuple(b.location) for b in env.current_state.blocked_paths}

    html = []
    html.append(f'''
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
      
      .warehouse-wrapper {{
         font-family: 'Inter', sans-serif;
         background: #0f172a;
         padding: 24px;
         border-radius: 16px;
         border: 1px solid #1e293b;
         box-shadow: 0 20px 40px rgba(0,0,0,0.4);
         width: fit-content;
         margin: 0 auto;
      }}
      .warehouse-grid {{
         display: grid;
         grid-template-columns: repeat({width}, 1fr);
         gap: 6px;
         background: #1e293b; 
         padding: 12px;
         border-radius: 12px;
         border: 1px solid #334155;
      }}
      .grid-cell {{
         width: 48px;
         height: 48px;
         background: #0f172a;
         border-radius: 8px;
         display: flex;
         align-items: center;
         justify-content: center;
         font-size: 24px;
         border: 1px solid #1e293b;
         transition: all 0.2s ease;
         position: relative;
      }}
      .grid-cell:hover {{
         transform: scale(1.05);
         border-color: #475569;
         z-index: 10;
      }}
      .cell-station {{
         background: rgba(56, 189, 248, 0.1);
         border-color: rgba(56, 189, 248, 0.4);
         box-shadow: inset 0 0 15px rgba(56, 189, 248, 0.15);
      }}
      .cell-blocked {{
         background: repeating-linear-gradient(
            45deg,
            #451a03,
            #451a03 10px,
            #78350f 10px,
            #78350f 20px
         );
         border-color: #b45309;
      }}
      .robot {{
         filter: drop-shadow(0 0 6px rgba(255,255,255,0.4));
      }}
      .robot-maintenance {{
         filter: drop-shadow(0 0 10px #ef4444);
         animation: pulse-red 1.5s infinite;
      }}
      .robot-sensor {{
         filter: drop-shadow(0 0 10px #eab308);
         animation: pulse-yellow 2s infinite;
      }}
      .battery-bar {{
         position: absolute;
         bottom: 2px;
         left: 4px;
         right: 4px;
         height: 4px;
         background: rgba(0,0,0,0.5);
         border-radius: 2px;
         overflow: hidden;
         border: 1px solid #000;
      }}
      .battery-fill {{
         height: 100%;
      }}
      @keyframes pulse-red {{
         0% {{ opacity: 1; transform: scale(1); }}
         50% {{ opacity: 0.6; transform: scale(0.9); }}
         100% {{ opacity: 1; transform: scale(1); }}
      }}
      @keyframes pulse-yellow {{
         0% {{ opacity: 1; }}
         50% {{ opacity: 0.5; }}
         100% {{ opacity: 1; }}
      }}
    </style>
    <div class="warehouse-wrapper">
    <div style="color:#e2e8f0; font-size: 14px; font-weight:600; margin-bottom:12px; letter-spacing:1px; display:flex; justify-content:space-between;">
        <span>WAREHOUSE LIVE VIEW ({width}x{height})</span>
    </div>
    <div class="warehouse-grid">
    '''
    )

    for y in range(height):
        for x in range(width):
            loc = (x, y)
            cell_classes = ["grid-cell"]
            icon = ""
            battery_html = ""
            
            if loc in blocked_set:
                cell_classes.append("cell-blocked")
                icon = "🚧"
            elif loc in stations_set:
                cell_classes.append("cell-station")
                icon = "⚡"
            
            r = robots_map.get(loc)
            if r:
                batt_color = "#22c55e" if r.battery_level > 50 else ("#eab308" if r.battery_level > 20 else "#ef4444")
                battery_html = f'''<div class="battery-bar"><div class="battery-fill" style="width: {r.battery_level}%; background: {batt_color};"></div></div>'''
                
                if r.status == "MAINTENANCE":
                    icon = f"<div class='robot-maintenance' title='{r.id}: {r.status}'>🚨</div>"
                elif r.status == "SENSOR_FAILURE":
                    icon = f"<div class='robot-sensor' title='{r.id}: {r.status}'>❓</div>"
                else:
                    icon = f"<div class='robot' title='{r.id}: {r.battery_level:.1f}% Battery'>🤖</div>"

            class_str = " ".join(cell_classes)
            html.append(f"<div class='{class_str}' title='{x},{y}'>{icon}{battery_html}</div>")

    html.append("</div>")

    if env.current_state.active_exceptions:
        html.append(f'''
        <div style="margin-top: 16px; background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 8px; padding: 12px; color: #fca5a5; font-size: 13px; display: flex; align-items:center; gap: 8px;">
            <div style="background:#ef4444; width:8px; height:8px; border-radius:50%; animation: pulse-red 1.5s infinite; flex-shrink:0;"></div>
            <b>{len(env.current_state.active_exceptions)} Active Exception(s)</b> requiring REROUTE, RESTOCK, or MAINTENANCE.
        </div>
        ''')
    elif env.current_state.time_step > 0:
        html.append(f'''
        <div style="margin-top: 16px; background: rgba(34, 197, 94, 0.1); border: 1px solid #22c55e; border-radius: 8px; padding: 12px; color: #86efac; font-size: 13px; display: flex; align-items:center; gap: 8px;">
            <div style="background:#22c55e; width:8px; height:8px; border-radius:50%;"></div>
            <b>All Systems Nominal.</b> Zero Active Exceptions.
        </div>
        ''')

    html.append("</div>")
    return "\n".join(html)
# ---------------------------------------------------------------------------
# Simulation Logic
# ---------------------------------------------------------------------------
def format_hacker_terminal(history):
    lines = "<br>".join(history)
    return f"""<div style="background-color: #0d1117; color: #10b981; font-family: 'Courier New', Courier, monospace; padding: 15px; border-radius: 8px; border: 1px solid #30363d; height: 300px; overflow-y: auto; font-size: 13px;">
{lines if lines else 'Initializing agent connection...'}
<span style="animation: blink 1s step-end infinite;">█</span>
</div>
<style>@keyframes blink {{ 50% {{ opacity: 0; }} }}</style>"""

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
    resolved_count = 0
    successful_actions = 0
    total_actions = 0
    
    plot_data = []

    for step in range(1, max_steps + 1):
        state_text = env._state_to_text()
        recent_history_str = "\n".join(action_history[-5:]) if action_history else "No history."
        failed_actions_str = ", ".join(failed_action_memory[-5:]) if failed_action_memory else "None"
        
        exceptions_before = len(env.current_state.active_exceptions)
        mode_label = "Heuristic Fallback" if use_heuristic else "LLM"
        
        if step == 1:
            plot_data.append({"Step": 0, "Total Reward": 0.0})
            
        kpi_md = f"""<div style='color:#e2e8f0; font-family: sans-serif;'>
        <div style='margin-bottom:8px;'><b>Status:</b> ⏳ Generating Action...</div>
        <div style='margin-bottom:8px;'><b>Step:</b> {step} / {max_steps}</div>
        <div style='margin-bottom:8px;'><b>Mode:</b> {mode_label}</div>
        <div style='margin-bottom:8px;'><b>Reward:</b> <span style='color:#22c55e;'>{total_reward:.2f}</span></div>
        <div style='margin-bottom:8px;'><b>Exceptions:</b> <span style='color:#ef4444;'>{exceptions_before}</span></div>
        </div>"""
        
        yield (
            render_warehouse_grid(env),
            pd.DataFrame(plot_data) if plot_data else pd.DataFrame(columns=["Step", "Total Reward"]),
            format_hacker_terminal(action_history),
            kpi_md
        )
        
        error_logs = []
        if use_heuristic:
            action = _heuristic_action(env, scenario)
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
        if exceptions_after < exceptions_before:
            resolved_count += (exceptions_before - exceptions_after)
            
        log_entry = (
            f"<span style='color:#3b82f6'>[Step {step}]</span> action=<span style='color:#d946ef'>{json.dumps(action)}</span> "
            f"reward=<span style='color:#22c55e'>{reward:.2f}</span> "
            f"exceptions=<span style='color:#ef4444'>{exceptions_after}</span> "
            f"<br>&nbsp;&nbsp;↳ <i>{'; '.join(logs)}</i>"
        )
        action_history.append(log_entry)
        
        plot_data.append({"Step": step, "Total Reward": total_reward})
        
        status_text = "🏆 MISSION COMPLETE!" if (total_reward > 0 and exceptions_after == 0) else ("🚨 MISSION FAILED" if terminated else "✅ Step Completed")
        kpi_md = f"""<div style='color:#e2e8f0; font-family: sans-serif;'>
        <div style='margin-bottom:8px;'><b>Status:</b> {status_text}</div>
        <div style='margin-bottom:8px;'><b>Step:</b> {step} / {max_steps}</div>
        <div style='margin-bottom:8px;'><b>Mode:</b> {mode_label}</div>
        <div style='margin-bottom:8px;'><b>Reward:</b> <span style='color:#22c55e;'>{total_reward:.2f}</span></div>
        <div style='margin-bottom:8px;'><b>Exceptions:</b> <span style='color:#ef4444;'>{exceptions_after}</span></div>
        </div>"""
        
        if terminated or exceptions_after == 0:
            yield (
                render_warehouse_grid(env),
                pd.DataFrame(plot_data),
                format_hacker_terminal(action_history),
                kpi_md
            )
            break
            
        yield (
            render_warehouse_grid(env),
            pd.DataFrame(plot_data),
            format_hacker_terminal(action_history),
            kpi_md
        )
        time.sleep(0.5)

# ---------------------------------------------------------------------------
# Gradio UI Shell
# ---------------------------------------------------------------------------
custom_css = """
body { background-color: #0f172a; }
.gradio-container { background-color: #0f172a !important; }
.sidebar-panel { background: rgba(22, 27, 34, 0.7); backdrop-filter: blur(10px); border-radius: 12px; border: 1px solid #1e293b; padding: 20px; }
.stage-panel { background: rgba(22, 27, 34, 0.7); backdrop-filter: blur(10px); border-radius: 12px; border: 1px solid #1e293b; padding: 20px; }
.terminal-panel { background: transparent; border: none; padding: 0; box-shadow: none; margin-top: 15px;}
"""

with gr.Blocks(theme=gr.themes.Base(), css=custom_css) as demo:
    with gr.Row():
        # LEFT SIDEBAR
        with gr.Column(scale=1, elem_classes=["sidebar-panel"]):
            gr.Markdown("<h2 style='color:#e2e8f0; margin-top:0;'>🤖 Logistics Hub</h2>")
            gr.Markdown("<p style='color:#94a3b8; font-size:13px;'>Simulation Engine & Agent Control</p>")
            gr.HTML("<hr style='border-color: #334155;'>")
            
            scenario = gr.Dropdown(choices=["easy", "medium", "hard"], value="medium", label="Scenario Protocol")
            steps = gr.Slider(minimum=5, maximum=50, value=20, step=1, label="Operation Max Time")
            start_btn = gr.Button("🚀 Run Agent", variant="primary")
            
            gr.HTML("<hr style='border-color: #334155; margin-top:20px; margin-bottom:20px;'>")
            gr.Markdown("<h3 style='color:#e2e8f0;'>Quick Stats</h3>")
            kpi_html = gr.HTML(
                "<div style='color:#94a3b8; font-family: sans-serif; font-size:14px;'>"
                "<div style='margin-bottom:8px;'><b>Status:</b> Ready Mode</div>"
                "<div style='margin-bottom:8px;'><b>Step:</b> 0</div>"
                "<div style='margin-bottom:8px;'><b>Reward:</b> 0.00</div>"
                "</div>"
            )

        # MAIN STAGE
        with gr.Column(scale=3):
            # TOP ROW: Grid & LinePlot
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["stage-panel"]):
                    grid = gr.HTML("<div style='color:#94a3b8; font-family: sans-serif;'>Initializing Core System Layer...</div>")
                
                with gr.Column(scale=1, elem_classes=["stage-panel"]):
                    gr.Markdown("<h3 style='color:#e2e8f0; margin-top:0;'>Simulation Trajectory</h3>")
                    plot = gr.LinePlot(
                        x="Step", 
                        y="Total Reward", 
                        title="Cumulative Reward vs Time",
                        width=400, 
                        height=250, 
                        tooltip=["Step", "Total Reward"],
                        color_discrete_sequence=["#38bdf8"]
                    )
                    
            # BOTTOM ROW: Terminal Log
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["terminal-panel"]):
                    gr.Markdown("<h3 style='color:#e2e8f0; margin-bottom:5px;'>Agent Event Log</h3>")
                    terminal_log = gr.HTML(
                        '<div style="background-color: #0d1117; color: #10b981; font-family: \'Courier New\', Courier, monospace; padding: 15px; border-radius: 8px; border: 1px solid #30363d; height: 300px; overflow-y: auto; font-size: 13px;">'
                        'Awaiting initialization sequence...<span style="animation: blink 1s step-end infinite;">█</span></div>'
                        '<style>@keyframes blink { 50% { opacity: 0; } }</style>'
                    )

    # Wire up the button
    start_btn.click(
        run_simulation, 
        inputs=[scenario, steps], 
        outputs=[grid, plot, terminal_log, kpi_html]
    )

if __name__ == "__main__":
    app = gr.mount_gradio_app(app_api, demo, path="/")
    uvicorn.run(app, host="0.0.0.0", port=7860)
