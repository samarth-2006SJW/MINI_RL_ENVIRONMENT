import os
import json
import time
import openai
from pathlib import Path
from environment import WarehouseEnvironment

# Mandatory Variables from Environment
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

def main():
    # 1. Initialize
    client = openai.OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    root = Path(__file__).parent
    env = WarehouseEnvironment(str(root/"configs"/"warehouse_map.json"), str(root/"configs"/"scenarios.yaml"), "hard")
    
    # [START] - Must be first line
    print(f"[START] task=warehouse-logistics env=openenv-v1 model={MODEL_NAME}", flush=True)

    rewards = []
    steps = 0
    success = "false"
    
    try:
        obs = env.reset()
        done = False
        
        for step in range(1, 16): # Max steps for validation
            steps = step
            prompt = f"Current State: {env._state_to_text()}\nOutput ONLY raw JSON dictionary for Warehouse Action."
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            raw_out = response.choices[0].message.content.strip()
            # Clean JSON
            if "```" in raw_out:
                raw_out = raw_out.split("```")[1].replace("json", "").strip()
            
            try:
                action = json.loads(raw_out)
            except:
                action = {"command_type": "WAIT", "target_id": None}
            
            # Step Env
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            
            # [STEP] - Exact Format
            action_name = action.get("command_type", "WAIT")
            print(f"[STEP] step={step} action={action_name} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            
            if done: break
            time.sleep(0.1)

        score = min(max(sum(rewards) / 5.0, 0.0), 1.0)
        success = "true" if score >= 0.1 else "false"
        
    except Exception as e:
        # Graceful end on error
        pass

    # [END] - Exact Format
    r_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(f"[END] success={success} steps={steps} score={score:.2f} rewards={r_str}", flush=True)

if __name__ == "__main__":
    main()