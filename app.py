import os
import json
import time
import gradio as gr
import openai
from environment import WarehouseEnvironment
from models import WarehouseState

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
        grid_md = render_warehouse_grid(env)
        
        # 2. Status Dashboard
        exceptions = len(env.current_state.active_exceptions)
        status_md = f"**Step:** {step} | **Exceptions:** {exceptions} | **Total Reward:** {total_reward:.2f}"
        
        yield grid_md, status_md, "Agent is thinking..."
        
        # 3. Get LLM Action
        prompt = _build_prompt(state_text, recent_history_str)
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
            action = {"command_type": "WAIT", "target_id": None}
            
        # 4. Step Environment
        _, reward, terminated, info = env.step(action)
        total_reward += reward
        
        # Log entry
        logs = info.get("event_log", ["Agent waited."])
        log_entry = f"Step {step}: {action['command_type']} -> {', '.join(logs)}"
        action_history.append(log_entry)
        
        # Yield update
        yield render_warehouse_grid(env), status_md, "\n".join(action_history)
        
        if terminated or exceptions == 0:
            yield render_warehouse_grid(env), status_md + " | **MISSION COMPLETE!**", "\n".join(action_history)
            break
            
        time.sleep(1) # Slow down for visualization

