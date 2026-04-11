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

def render_warehouse_grid(env):
    width = env.grid_width
    height = env.grid_height
    blocked_set = set(env.blocked_cells)
    stations_set = set(env.charging_stations)
    robots_map = {r.location: r for r in env.robots.values()}

    html = []
    html.append(f'''
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
      
      .warehouse-wrapper {{
         background: linear-gradient(145deg, #0b1120 0%, #080b12 100%);
         border-radius: 16px;
         padding: 24px;
         border: 1px solid rgba(16, 185, 129, 0.15);
         box-shadow: 0 0 40px rgba(16, 185, 129, 0.05), inset 0 0 20px rgba(0, 0, 0, 0.5);
         position: relative;
         overflow: hidden;
      }}
      .warehouse-wrapper::before {{
         content: "";
         position: absolute;
         top: 0; left: 0; right: 0; height: 1px;
         background: linear-gradient(90deg, transparent, #10b981, transparent);
         opacity: 0.5;
      }}
      .hud-header {{
         color: #10b981; 
         font-family: 'Outfit', sans-serif; 
         font-size: 14px; 
         font-weight:800; 
         margin-bottom:16px; 
         letter-spacing:2px; 
         display:flex; 
         justify-content:space-between;
         text-transform: uppercase;
         text-shadow: 0 0 10px rgba(16, 185, 129, 0.4);
      }}
      .warehouse-grid {{
         display: grid;
         grid-template-columns: repeat({width}, 1fr);
         gap: 4px;
         background: #0f172a;
         padding: 10px;
         border-radius: 12px;
         border: 1px solid #1e293b;
      }}
      .grid-cell {{
         aspect-ratio: 1;
         background: #151f32;
         border-radius: 6px;
         display: flex;
         align-items: center;
         justify-content: center;
         position: relative;
         border: 1px solid #1e293b;
         transition: all 0.3s ease;
      }}
      .cell-blocked {{
         background: repeating-linear-gradient(45deg, #1e293b, #1e293b 5px, #151f32 5px, #151f32 10px);
         border-color: #334155;
      }}
      .cell-station {{
         background: rgba(14, 165, 233, 0.05);
         border-color: rgba(14, 165, 233, 0.3);
         box-shadow: inset 0 0 10px rgba(14, 165, 233, 0.1);
      }}
      .station-icon {{
         color: #0ea5e9;
         text-shadow: 0 0 8px #0ea5e9;
         font-size: 1.2rem;
         animation: float 2s ease-in-out infinite;
      }}
      
      /* Robot Design */
      .robot-core {{
         width: 60%;
         height: 60%;
         border-radius: 50%;
         position: relative;
         display: flex;
         align-items: center;
         justify-content: center;
         box-shadow: 0 0 15px currentColor;
         z-index: 10;
      }}
      .robot-core::after {{
         content: "";
         position: absolute;
         inset: -4px;
         border-radius: 50%;
         border: 2px solid currentColor;
         border-top-color: transparent;
         animation: spin 3s linear infinite;
      }}
      .robot-normal {{ color: #10b981; background: rgba(16,185,129,0.2); }}
      .robot-warning {{ color: #eab308; background: rgba(234,179,8,0.2); }}
      .robot-critical {{ color: #ef4444; background: rgba(239,68,68,0.2); }}
      .robot-maintenance {{ color: #ef4444; background: rgba(239,68,68,0.2); animation: pulse 1s infinite; }}
      .robot-sensor {{ color: #f59e0b; background: rgba(245,158,11,0.2); animation: pulse 2s infinite; }}
      
      .bot-id {{
         position: absolute;
         top: -10px;
         font-family: 'JetBrains Mono', monospace;
         font-size: 8px;
         color: #cbd5e1;
         background: #0f172a;
         padding: 1px 4px;
         border-radius: 4px;
         border: 1px solid #334155;
      }}
      
      @keyframes spin {{ 100% {{ transform: rotate(360deg); }} }}
      @keyframes float {{ 0%, 100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-3px); }} }}
      @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}
      
      .exception-alert {{
         margin-top: 20px;
         background: linear-gradient(90deg, rgba(239,68,68,0.15) 0%, rgba(239,68,68,0.05) 100%);
         border-left: 4px solid #ef4444;
         border-radius: 4px 8px 8px 4px;
         padding: 16px;
         color: #fca5a5;
         font-family: 'Outfit', sans-serif;
         font-size: 14px;
         display: flex;
         align-items: center;
         gap: 12px;
         backdrop-filter: blur(5px);
      }}
      .nominal-alert {{
         margin-top: 20px;
         background: linear-gradient(90deg, rgba(16,185,129,0.15) 0%, rgba(16,185,129,0.05) 100%);
         border-left: 4px solid #10b981;
         border-radius: 4px 8px 8px 4px;
         padding: 16px;
         color: #6ee7b7;
         font-family: 'Outfit', sans-serif;
         font-size: 14px;
         display: flex;
         align-items: center;
         gap: 12px;
         backdrop-filter: blur(5px);
      }}
    </style>
    <div class="warehouse-wrapper">
    <div class="hud-header">
        <span>🌐 SYSTEM.LIVE_MAP ({width}x{height})</span>
        <span style="color:#64748b">UPLINK: SECURE</span>
    </div>
    <div class="warehouse-grid">
    ''')

    for y in range(height):
        for x in range(width):
            loc = (x, y)
            cell_classes = ["grid-cell"]
            content = ""
            
            if loc in blocked_set:
                cell_classes.append("cell-blocked")
            elif loc in stations_set:
                cell_classes.append("cell-station")
                content = "<div class='station-icon'>⚡</div>"
            
            r = robots_map.get(loc)
            if r:
                color_class = "robot-normal"
                if r.battery_level < 20: color_class = "robot-critical"
                elif r.battery_level < 50: color_class = "robot-warning"
                
                if r.status == "MAINTENANCE":
                    color_class = "robot-maintenance"
                    core_icon = "⚠️"
                elif r.status == "SENSOR_FAILURE":
                    color_class = "robot-sensor"
                    core_icon = "👁️"
                else:
                    core_icon = "•"
                
                content = f'''
                <div class="robot-core {color_class}" title="ID:{r.id} | BAT:{r.battery_level:.1f}% | STAT:{r.status}">
                    <div style="font-size:10px;">{core_icon}</div>
                    <div class="bot-id">{r.id.replace('Robot_', 'R')}</div>
                </div>'''

            class_str = " ".join(cell_classes)
            html.append(f"<div class='{class_str}' title='SRC:{x},{y}'>{content}</div>")

    html.append("</div>")

    if env.current_state.active_exceptions:
        html.append(f'''
        <div class="exception-alert">
            <div style="background:#ef4444; width:10px; height:10px; border-radius:50%; box-shadow: 0 0 10px #ef4444; animation: pulse 1s infinite; flex-shrink:0;"></div>
            <b>DETECTED {len(env.current_state.active_exceptions)} CRITICAL EXCEPTIONS</b>
        </div>
        ''')
    elif env.current_state.time_step > 0:
        html.append(f'''
        <div class="nominal-alert">
            <div style="background:#10b981; width:10px; height:10px; border-radius:50%; box-shadow: 0 0 10px #10b981; flex-shrink:0;"></div>
            <b>ALL SYSTEMS NOMINAL.</b> OVERSEER AI ACTIVE.
        </div>
        ''')

    html.append("</div>")
    return "\n".join(html)

# ---------------------------------------------------------------------------
# Simulation Logic
# ---------------------------------------------------------------------------
def format_hacker_terminal(history):
    lines = "<br>".join(history[-15:]) # Keep only last 15 to prevent huge DOM
    return f'''
    <div style="background: #050914; color: #10b981; font-family: 'JetBrains Mono', Courier, monospace; padding: 20px; border-radius: 12px; border: 1px solid rgba(16,185,129,0.2); height: 350px; overflow-y: auto; font-size: 13px; position:relative; box-shadow: inset 0 0 30px rgba(0,0,0,0.8);">
        <div style="position:absolute; top:0; left:0; right:0; height:100%; background: linear-gradient(transparent 50%, rgba(16,185,129,0.02) 50%); background-size: 100% 4px; pointer-events:none; z-index:1;"></div>
        <div style="position:relative; z-index:2;">
            {lines if lines else 'SYSTEM BOOT... OVERSEER AI WAITING FOR UPLINK...'}
            <span style="animation: blink 1s step-end infinite; color:#38bdf8;">█</span>
        </div>
    </div>
    <style>@keyframes blink {{ 50% {{ opacity: 0; }} }}</style>
    '''

def run_simulation(scenario, max_steps):
    client = _get_llm_client()
    env = WarehouseEnvironment(scenario_key=scenario)
    state = env.reset()
    
    total_reward = 0.0
    total_actions = 0
    successful_actions = 0
    failed_action_memory = []
    
    action_history = []
    plot_data = []
    
    # Extract heuristic flag directly for now (keeping original logic intact)
    use_heuristic = False
    
    # Initialize the plot layout dynamically
    yield (
        render_warehouse_grid(env),
        pd.DataFrame(columns=["Step", "Total Reward"]),
        format_hacker_terminal([]),
        "<div style='color:#64748b; font-family: Outfit, sans-serif;'>Initializing Simulation Engine...</div>"
    )

    for step in range(1, max_steps + 1):
        state_text = _format_state_for_prompt(env.current_state)
        exceptions_before = len(env.current_state.active_exceptions)
        resolved_count = 0
        
        recent_history_str = "None"
        if action_history:
            # We strip html to leave bare text for LLM
            clean_history = [re.sub(r'<[^>]+>', '', idx) for idx in action_history[-3:]]
            recent_history_str = json.dumps(clean_history)
            
        failed_actions_str = "None"
        if failed_action_memory:
            failed_actions_str = json.dumps(list(set(failed_action_memory[-5:])))
            
        mode_label = "<span style='color:#f59e0b;'>Heuristic Engine</span>" if use_heuristic else "<span style='color:#8b5cf6;'>LLM Chain-of-Thought Core</span>"
        
        # Intermediate yield 
        kpi_md = f'''<div style='color:#e2e8f0; font-family: Outfit, sans-serif;'>
        <div style='display:flex; justify-content:space-between; margin-bottom:12px; padding-bottom:8px; border-bottom:1px solid #1e293b;'>
            <b style='color:#94a3b8;'>STATUS</b><span style='color:#38bdf8; font-weight:800;'>COMPUTING...</span>
        </div>
        <div style='display:flex; justify-content:space-between; margin-bottom:12px;'>
            <b style='color:#94a3b8;'>STEP</b><span style='color:#cbd5e1;'>{step} / {max_steps}</span>
        </div>
        <div style='display:flex; justify-content:space-between; margin-bottom:12px;'>
            <b style='color:#94a3b8;'>MODE</b><span>{mode_label}</span>
        </div>
        <div style='display:flex; justify-content:space-between; margin-bottom:12px;'>
            <b style='color:#94a3b8;'>NET REWARD</b><span style='color:#10b981; font-weight:800; text-shadow:0 0 8px rgba(16,185,129,0.5);'>{total_reward:.2f}</span>
        </div>
        </div>'''
        
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
            f"<span style='color:#38bdf8'>[T:{step:02d}]</span> "
            f"<span style='color:#cbd5e1'>OP:</span> <span style='color:#ec4899; font-weight:bold;'>{action['command_type']}</span> "
            f"<span style='color:#cbd5e1'>TGT:</span> <span style='color:#eab308'>{action.get('target_id','N/A')}</span> "
            f"<span style='color:#cbd5e1'>V:</span> <span style='color:#10b981'>+{reward:.2f}</span><br>"
            f"&nbsp;&nbsp;<span style='color:#64748b'>↳ {'; '.join(logs)}</span>"
        )
        action_history.append(log_entry)
        
        plot_data.append({"Step": step, "Total Reward": total_reward})
        
        status_text = "<span style='color:#10b981; font-weight:800; text-shadow:0 0 10px rgba(16,185,129,0.5);'>MISSION COMPLETE</span>" if (total_reward > 0 and exceptions_after == 0) else ("<span style='color:#ef4444; font-weight:800; text-shadow:0 0 10px rgba(239,68,68,0.5);'>MISSION WARNING</span>" if terminated else "<span style='color:#38bdf8;'>IN PROGRESS</span>")
        kpi_md = f'''<div style='color:#e2e8f0; font-family: Outfit, sans-serif;'>
        <div style='display:flex; justify-content:space-between; margin-bottom:12px; padding-bottom:8px; border-bottom:1px solid #1e293b;'>
            <b style='color:#94a3b8;'>STATUS</b><span>{status_text}</span>
        </div>
        <div style='display:flex; justify-content:space-between; margin-bottom:12px;'>
            <b style='color:#94a3b8;'>STEP</b><span style='color:#cbd5e1;'>{step} / {max_steps}</span>
        </div>
        <div style='display:flex; justify-content:space-between; margin-bottom:12px;'>
            <b style='color:#94a3b8;'>MODE</b><span>{mode_label}</span>
        </div>
        <div style='display:flex; justify-content:space-between; margin-bottom:12px;'>
            <b style='color:#94a3b8;'>NET REWARD</b><span style='color:#10b981; font-weight:800; text-shadow:0 0 8px rgba(16,185,129,0.5);'>{total_reward:.2f}</span>
        </div>
        <div style='display:flex; justify-content:space-between; margin-bottom:12px;'>
            <b style='color:#94a3b8;'>EXCEPTIONS</b><span style='color:#ef4444;'>{exceptions_after}</span>
        </div>
        </div>'''
        
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
# Gradio Premium UI Shell
# ---------------------------------------------------------------------------
custom_css = '''
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

body { background-color: #030712 !important; }
.gradio-container { background-color: #030712 !important; max-width: 1400px !important; margin: auto !important; font-family: 'Outfit', sans-serif !important; }

/* Dashboard Wrapper Styles */
.sidebar-panel { 
    background: #0b1120 !important; 
    border-radius: 16px !important; 
    border: 1px solid rgba(56, 189, 248, 0.1) !important; 
    box-shadow: 0 0 30px rgba(0, 0, 0, 0.5) !important;
    padding: 24px !important; 
}
.stage-panel { 
    background: #0b1120 !important; 
    border-radius: 16px !important; 
    border: 1px solid rgba(56, 189, 248, 0.1) !important; 
    padding: 24px !important; 
}
.terminal-panel { 
    background: transparent !important; 
    border: none !important; 
    padding: 0 !important; 
    box-shadow: none !important; 
    margin-top: 20px !important;
}

/* Control Element Overrides */
.gr-button-primary {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
    transition: all 0.3s ease !important;
    font-weight: 800 !important;
    letter-spacing: 1px !important;
    border-radius: 8px !important;
}
.gr-button-primary:hover {
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
    transform: translateY(-2px) !important;
}
.gr-input, .gr-box, .gr-dropdown {
    background-color: #0f172a !important;
    border: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
.gr-input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 1px #38bdf8 !important;
}

/* Typography Overrides */
h1, h2, h3, h4, p, span {
    font-family: 'Outfit', sans-serif !important;
}
'''

# Define Custom Theme using Gradio Theme Builder
premium_theme = gr.themes.Base(
    primary_hue="emerald",
    neutral_hue="slate",
    spacing_size="sm",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
).set(
    body_background_fill="#030712",
    body_text_color="#f8fafc",
    background_fill_primary="#0b1120",
    background_fill_secondary="#030712",
    border_color_primary="rgba(56, 189, 248, 0.1)",
    block_background_fill="#0b1120",
    block_border_width="1px",
    block_border_color="rgba(56, 189, 248, 0.1)",
    block_radius="16px",
    button_primary_background_fill="linear-gradient(135deg, #10b981, #059669)",
    button_primary_background_fill_hover="linear-gradient(135deg, #34d399, #10b981)",
    button_primary_text_color="#ffffff",
    panel_background_fill="#0b1120"
)

with gr.Blocks(theme=premium_theme, css=custom_css, title="EcoNav Premium AI") as demo:
    with gr.Row():
        # LEFT SIDEBAR
        with gr.Column(scale=1, elem_classes=["sidebar-panel"]):
            gr.Markdown("<h2 style='color:#f8fafc; margin-top:0; font-weight:800; letter-spacing:1px;'><span style='color:#10b981;'>OVERSEER</span> ENGINE</h2>")
            gr.Markdown("<p style='color:#94a3b8; font-size:13px; font-family: Outfit;'>Neural Logistics Optimization Dashboard</p>")
            gr.HTML("<div style='height:1px; background:linear-gradient(90deg, #10b981, transparent); margin:20px 0;'></div>")
            
            scenario = gr.Dropdown(choices=["easy", "medium", "hard"], value="medium", label="Simulation Matrix")
            steps = gr.Slider(minimum=5, maximum=50, value=20, step=1, label="Epoch Constraints")
            start_btn = gr.Button("INITIALIZE UPLINK", variant="primary")
            
            gr.HTML("<div style='height:1px; background:linear-gradient(90deg, transparent, #38bdf8, transparent); margin:20px 0;'></div>")
            gr.Markdown("<h3 style='color:#e2e8f0; font-weight:600; letter-spacing:1px;'>TELEMETRY</h3>")
            kpi_html = gr.HTML(
                "<div style='color:#64748b; font-family: Outfit, sans-serif; font-size:14px;'>"
                "<div style='margin-bottom:12px; display:flex; justify-content:space-between;'><b>STATUS</b><span style='color:#10b981;'>SYSTEM READY</span></div>"
                "<div style='margin-bottom:12px; display:flex; justify-content:space-between;'><b>STEP</b><span>0</span></div>"
                "<div style='margin-bottom:12px; display:flex; justify-content:space-between;'><b>REWARD</b><span>0.00</span></div>"
                "</div>"
            )

        # MAIN STAGE
        with gr.Column(scale=3):
            # TOP ROW: Grid & LinePlot
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["stage-panel"]):
                    grid = gr.HTML("<div style='color:#38bdf8; font-family: JetBrains Mono; text-align:center; padding:100px 0; border: 1px dashed #1e293b; border-radius:12px;'>WAITING FOR ENVIRONMENT SYNC...</div>")
                
                with gr.Column(scale=1, elem_classes=["stage-panel"]):
                    gr.Markdown("<h3 style='color:#e2e8f0; margin-top:0; font-weight:800; letter-spacing:1px;'>REWARD TRAJECTORY</h3>")
                    plot = gr.LinePlot(
                        x="Step", 
                        y="Total Reward", 
                        title="Cumulative Return vs Epochs",
                        tooltip=["Step", "Total Reward"],
                    )
                    
            # BOTTOM ROW: Terminal Log
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["terminal-panel"]):
                    gr.Markdown("<h3 style='color:#e2e8f0; margin-bottom:5px; font-weight:800; letter-spacing:1px;'>SYSTEM CONSOLE</h3>")
                    terminal_log = gr.HTML(
                        '<div style="background: #050914; color: #10b981; font-family: \\\'JetBrains Mono\\\', Courier, monospace; padding: 20px; border-radius: 12px; border: 1px solid rgba(16,185,129,0.2); height: 350px; overflow-y: auto; font-size: 13px; position:relative; box-shadow: inset 0 0 30px rgba(0,0,0,0.8);">'
                        '<div style="position:absolute; top:0; left:0; right:0; height:100%; background: linear-gradient(transparent 50%, rgba(16,185,129,0.02) 50%); background-size: 100% 4px; pointer-events:none; z-index:1;"></div>'
                        '<div style="position:relative; z-index:2;">Awaiting user command...<span style="animation: blink 1s step-end infinite; color:#38bdf8;">█</span></div>'
                        '</div><style>@keyframes blink { 50% { opacity: 0; } }</style>'
                    )

    # Wire up the button
    start_btn.click(
        run_simulation, 
        inputs=[scenario, steps], 
        outputs=[grid, plot, terminal_log, kpi_html]
    )

if __name__ == "__main__":
    app_api = FastAPI()
    app = gr.mount_gradio_app(app_api, demo, path="/")
    uvicorn.run(app, host="0.0.0.0", port=7860)
