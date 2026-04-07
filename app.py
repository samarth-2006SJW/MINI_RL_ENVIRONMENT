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
