import os
import json
import time
import gradio as gr
import openai
from environment import WarehouseEnvironment
from models import WarehouseState
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

def _build_prompt(state_text: str, recent_history: str) -> str:
    return f"""
### ROLE
You are the Autonomous Warehouse Controller. Your mission is to resolve logistics exceptions with ZERO wasted moves.

### CURRENT SYSTEM STATE
{state_text}

### RECENT ACTION HISTORY
{recent_history}

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
    if not client:
        yield "ERROR: OPENAI_API_KEY not found in Environment Variables.", "", ""
        return

    env = WarehouseEnvironment(
        map_path="configs/warehouse_map.json",
        scenario_path="configs/scenarios.yaml",
        scenario_name=scenario
    )
    env.reset()
    
    action_history = []
    total_reward = 0.0
    
    for step in range(1, max_steps + 1):
        state_text = env._state_to_text()
        recent_history_str = "\n".join(action_history[-5:]) if action_history else "No history."
        
        # 1. Visualize Grid
        # Initial status for thinking state
        exceptions_before = len(env.current_state.active_exceptions)
        status_md = f"**Step:** {step} | **Exceptions:** {exceptions_before} | **Total Reward:** {total_reward:.2f} | ⏳ *Generating Action...*"
        
        yield render_warehouse_grid(env), status_md, "\n".join(action_history) if action_history else "System initialized. Agent is preparing..."
        
        # 3. Get LLM Action
        prompt = _build_prompt(state_text, recent_history_str)
        error_logs = []
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a warehouse routing expert. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            llm_output = response.choices[0].message.content.strip()
            # Basic cleanup
            if "```" in llm_output:
                llm_output = llm_output.split("```")[1].replace("json", "").strip()
            action = json.loads(llm_output)
        except Exception as e:
            action = {"command_type": "ERROR", "target_id": None}
            error_logs = [f"API/Parse Error: {str(e)}"]
            
        # 4. Step Environment
        if action.get("command_type") == "ERROR":
            reward = 0.0
            terminated = False
            info = {"event_log": error_logs}
        else:
            _, reward, terminated, info = env.step(action)
            
        total_reward += reward
        
        # Log entry
        logs = error_logs if error_logs else info.get("event_log", ["Agent waited."])
        log_entry = f"Step {step}: {action.get('command_type')} -> {', '.join(logs)}"
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

