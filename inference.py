import os
import sys
import json
import time
import openai
from environment import WarehouseEnvironment

def main():
    """
    Entry point for judges. Sets up the environment and runs an LLM-based agent.
    Security First: Uses os.getenv to avoid hard-coded API keys.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set. Security First: DO NOT hardcode your API key.")
        sys.exit(1)

    # Configurable: Judges use OpenAI/GPT-4o by default.
    # For free testing, set LLM_BASE_URL and LLM_MODEL in your environment.
    base_url = os.getenv("LLM_BASE_URL", None)  # None = default OpenAI
    model_name = os.getenv("LLM_MODEL", "gpt-4o")  # Default: GPT-4o for judges
    
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.OpenAI(**client_kwargs)
    
    # Initialize Environment
    map_path = "configs/warehouse_map.json"
    scenario_path = "configs/scenarios.yaml"
    
    if not os.path.exists(map_path) or not os.path.exists(scenario_path):
        print(f"Warning: missing configuration files ({map_path} or {scenario_path}). Make sure to run this from the project root.")

    # Initialize environment
    env = WarehouseEnvironment(
        map_path=map_path, 
        scenario_path=scenario_path, 
        scenario_name="hard" # testing hard mode
    )
    
    env.reset()
    terminated = False
    action_history = []
    
    print("--- Starting Agent Loop ---")

    while not terminated:
        time.sleep(3)
        state_text = env._state_to_text()
        recent_history = "\n".join(action_history[-5:]) if action_history else "No previous actions taken."
        
        prompt = f"""
### ROLE
You are the Autonomous Warehouse Controller. Your mission is to resolve logistics exceptions with ZERO wasted moves.

### CURRENT SYSTEM STATE
{state_text}

### RECENT ACTION HISTORY
{recent_history}

### OPERATIONAL HIERARCHY (Strictly Follow)
1. CRITICAL: If 'Exception Count' > 0, identify the specific ID (e.g., ORD_101) and issue the resolution command (REROUTE_ORDER, REQUEST_RESTOCK, or DISPATCH_MAINTENANCE).
2. COLLISION AVOIDANCE: Do NOT move a robot to a tile already occupied by another robot or obstacle. This results in a -0.1 penalty.
3. MISSION COMPLETE: If 'Exception Count' is 0, DO NOT move robots. Your final action MUST be 'WAIT'. In warehouse RL, standing still after a job is done is the highest efficiency state.
4. REASONING: If 'Recent Action History' shows a 0.0 reward for a command, do NOT repeat it. Change your parameters or command type.

### OUTPUT SPECIFICATION
- Output ONLY a raw JSON dictionary.
- No markdown code blocks (```json).
- No conversational filler.

### JSON SCHEMA
{{
  "command_type": "MOVE_ROBOT | REQUEST_RESTOCK | DISPATCH_MAINTENANCE | REROUTE_ORDER | ASSIGN_WORKER | WAIT",
  "target_id": "ID of robot or order",
  "parameters": {{ "target_location": [x, y] or other relevant fields }}
}}

Action:"""
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a warehouse routing and exception handling expert. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )

            llm_output = response.choices[0].message.content.strip()
            
            # Clean markdown formatting if present
            if llm_output.startswith("```json"):
                llm_output = llm_output.replace("```json\n", "").replace("```", "").strip()
            elif llm_output.startswith("```"):
                llm_output = llm_output.replace("```\n", "").replace("```", "").strip()

            action = json.loads(llm_output)
            
            # Step the environment
            _, reward, terminated, info = env.step(action)
            if not env.current_state.active_exceptions or action.get("command_type") == "WAIT":
                print(f"\n--- [SUCCESS] MISSION ACCOMPLISHED ---")
                print(f"Final Reward: {env.total_reward} | Status: Goal Reached")
                terminated = True
                break
            event_strs = " | ".join(info.get("event_log", []))
            
            action_summary = f"Attempted Action: {action.get('command_type')} -> Result: {event_strs} | Reward: {reward}"
            action_history.append(action_summary)
            
            # Avoid API 429 limits
            time.sleep(1)
            
        except json.JSONDecodeError:
            print("[ERROR] LLM produced invalid JSON. Applying -0.1 penalty and advancing time...")
            _, _, terminated, _ = env.step({"command_type": "WAIT", "target_id": None})
            env.total_reward -= 0.1
            action_history.append("Error: Produced invalid JSON format. Agent waited.")
            time.sleep(1)
        except Exception as e:
            print(f"[ERROR] Inference issue occurred: {e}")
            # Step environment artificially to prevent infinite loops if API fails repeatedly
            print("Terminating run prematurely due to API exception.")
            break

if __name__ == "__main__":
    main()
