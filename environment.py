import json
import random
import os
from typing import Dict, List, Optional, Tuple, Union

try:
    import openenv
    BaseEnv = getattr(openenv, "BaseEnv", object)
except ImportError:
    class BaseEnv:
        pass

from models import WarehouseState, Robot, BlockedPath, CommandType, ExceptionIssue, LogisticsCommand, RobotStatus
from utils import load_config, check_collision
import tasks

class WarehouseEnvironment(BaseEnv):
    """
    Core logic environment for Reinforcement Learning based Warehouse Logistics Exception Handler.
    Manages state, actions, observations, and rewards focusing on strict serializability 
    and dynamic map loading.
    """

    def __init__(self, map_path: str, scenario_path: str, scenario_name: str = "easy"):
        """
        Initializes the WarehouseEnvironment by loading map and scenario configurations,
        and setting up the initial state.

        Args:
            map_path (str): File path to the JSON configuration containing warehouse layout and limits.
            scenario_path (str): File path to the JSON configuration containing initial exceptions/tasks.
            scenario_name (str): The name of the specific scenario to load from the scenario_path.
        """
        self.map_path = map_path
        self.scenario_path = scenario_path
        self.scenario_name = scenario_name
        
        self.map_config = load_config(self.map_path)
        all_scenarios = load_config(self.scenario_path)
        self.scenario_config = all_scenarios.get(self.scenario_name, {})
        
        self.current_state: Optional[WarehouseState] = None
        self.total_reward = 0.0
        self._initialize_state()

    def _initialize_state(self) -> None:
        """
        Creates the initial WarehouseState using the parsed map and scenario configurations.
        """
        # Parse initial robots setup (as a list in scenario config)
        robots: List[Robot] = []
        for r_data in self.scenario_config.get("robots", []):
            robots.append(Robot(**r_data))
        
        # Parse blocked paths
        blocked_paths: List[BlockedPath] = []
        for path_data in self.scenario_config.get("blocked_paths", []):
            blocked_paths.append(BlockedPath(**path_data))
            
        # Parse initial exceptions
        active_exceptions: List[ExceptionIssue] = []
        for issue_data in self.scenario_config.get("active_exceptions", []):
            active_exceptions.append(ExceptionIssue(**issue_data))
            
        # Load inventory status from scenario configuration
        inventory_status: Dict[str, int] = self.scenario_config.get("inventory_status", {})

        # Load worker availability from scenario
        worker_availability = self.scenario_config.get("worker_availability", 0)

        # Load dynamic configurations
        config_data = self.scenario_config.get("config", {})

        # Populate state
        self.current_state = WarehouseState(
            time_step=0,
            worker_availability=worker_availability,
            inventory_status=inventory_status,
            robots=robots,
            blocked_paths=blocked_paths,
            active_exceptions=active_exceptions,
            config=config_data
        )

    def reset(self) -> Dict:
        """
        Resets the environment to its initial configuration.
        
        Returns:
            Dict: The initial observation of the environment state as a pure dictionary.
        """
        self._initialize_state()
        self.total_reward = 0.0
        print("[START] Mission: Warehouse Logistics initialized.")
        return self.state()

    def state(self) -> Dict:
        """
        Retrieves the current state of the warehouse environment.
        Uses model_dump() for pure dictionary output to support FastAPI seamlessly.
        Masks the location of robots experiencing SENSOR_FAILURE.
        
        Returns:
            Dict: A pure dictionary representation of the current WarehouseState.
        """
        if self.current_state is None:
            raise ValueError("Environment state is not initialized.")
            
        obs = self.current_state.model_dump()
        
        import enum
        def _convert_enums(obj):
            if isinstance(obj, dict):
                return {k: _convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_convert_enums(x) for x in obj]
            elif isinstance(obj, enum.Enum):
                return obj.value
            return obj
            
        obs = _convert_enums(obs)
        
        # Observation Masking: Hide location if SENSOR_FAILURE
        for robot in obs.get("robots", []):
            status = robot.get("status")
            if status == getattr(RobotStatus.SENSOR_FAILURE, 'value', 'SENSOR_FAILURE') or status == 'SENSOR_FAILURE':
                robot["location"] = None
                
        return obs
        
    def step(self, action: Union[Dict, LogisticsCommand]) -> Tuple[Dict, float, bool, Dict]:
        """
        Executes a step in the environment given an action.
        Evaluates reward logic and checks for task completion.
        
        Args:
            action (Union[Dict, LogisticsCommand]): Action mapping or command to execute.
            
        Returns:
            Tuple[Dict, float, bool, Dict]: (observation, reward, terminated, info)
        """
        reward = 0.0
        terminated = False
        info = {"event_log": []}
        
        # Save exact action dict/LogisticsCommand for logging
        action_repr = action.model_dump() if isinstance(action, LogisticsCommand) else action

        try:
            if isinstance(action, dict):
                action_cmd = LogisticsCommand(**action)
            elif isinstance(action, LogisticsCommand):
                action_cmd = action
            else:
                raise ValueError("Action must be a LogisticsCommand or a dictionary.")
        except Exception as e:
            info["event_log"].append(f"Action validation failed: {str(e)}")
            action_cmd = LogisticsCommand(command_type=CommandType.WAIT)
            reward -= 0.1
            
        # Identify the target robot if there is one
        target_robot = None
        if action_cmd.target_id:
            for r in self.current_state.robots:
                if r.id == action_cmd.target_id:
                    target_robot = r
                    break
        
        # Execute action based on type
        if action_cmd.command_type == CommandType.MOVE_ROBOT:
            if target_robot:
                if target_robot.battery_level <= 0.0:
                    reward -= 0.1
                    info["event_log"].append("Movement failed: Battery depleted")
                else:
                    # Decrease battery by 0.5% for every MOVE_ROBOT attempt
                    target_robot.battery_level = max(0.0, target_robot.battery_level - 0.5)
                    
                    if target_robot.battery_level == 0.0:
                        reward -= 0.1
                        info["event_log"].append(f"Robot {target_robot.id} battery is empty.")
                    
                    params = action_cmd.parameters or {}
                    location_list = params.get("target_location")
                    
                    # Check target location is valid
                    if location_list is not None and len(location_list) == 2:
                        target_location = (int(location_list[0]), int(location_list[1]))
                        grid_dims = self.map_config.get("dimensions", [10, 10])
                        grid_size = (int(grid_dims[0]), int(grid_dims[1]))
                        
                        # Validate move with collision check
                        if not check_collision(target_location, grid_size, self.current_state.robots, self.current_state.blocked_paths):
                            target_robot.location = target_location
                            info["event_log"].append(f"Robot {target_robot.id} moved to {target_location}.")
                        else:
                            reward -= 0.1
                            info["event_log"].append(f"Collision detected for robot {target_robot.id}.")

        elif action_cmd.command_type == CommandType.RE_POLL_SENSOR:
            if target_robot and target_robot.status == RobotStatus.SENSOR_FAILURE:
                target_robot.status = RobotStatus.ACTIVE
                reward += 0.2
                info["event_log"].append(f"Robot {target_robot.id} sensor restored.")

        elif action_cmd.command_type == CommandType.REQUEST_RESTOCK:
            params = action_cmd.parameters or {}
            component_name = params.get("component_name")
            
            # Fallback if LLM forgets parameter: pick first item under 100
            if not component_name and self.current_state.inventory_status:
                for item, qty in self.current_state.inventory_status.items():
                    if qty < 100:
                        component_name = item
                        break
                if not component_name:
                    component_name = list(self.current_state.inventory_status.keys())[0]
                    
            if component_name:
                current_qty = self.current_state.inventory_status.get(component_name, 0)
                if current_qty >= 100:
                    reward -= 0.05
                    info["event_log"].append(f"Restock failed: {component_name} already at max capacity.")
                else:
                    new_qty = min(100, current_qty + 20)
                    self.current_state.inventory_status[component_name] = new_qty
                    if new_qty > current_qty:
                        reward += 0.5
                    info["event_log"].append(f"Restocked {component_name}. New qty: {new_qty}.")

        elif action_cmd.command_type in [CommandType.REROUTE_ORDER, CommandType.DISPATCH_MAINTENANCE, CommandType.ASSIGN_WORKER]:
            target_id = action_cmd.target_id
            resolved_any = False
            
            # 1. Physics level resolution
            if action_cmd.command_type == CommandType.DISPATCH_MAINTENANCE:
                for r in self.current_state.robots:
                    if r.id == target_id and r.status != RobotStatus.ACTIVE:
                        r.status = RobotStatus.ACTIVE
                        r.battery_level = 100.0
                        reward += 0.5
                        info["event_log"].append(f"Maintenance restored Robot {r.id}.")
                        resolved_any = True

            if action_cmd.command_type == CommandType.ASSIGN_WORKER:
                for bp in self.current_state.blocked_paths[:]:
                    if bp.id == target_id:
                        self.current_state.blocked_paths.remove(bp)
                        reward += 0.5
                        info["event_log"].append(f"Worker cleared blocked path {bp.id}.")
                        resolved_any = True
            
            # 2. Logical level exception resolution
            for exc in self.current_state.active_exceptions[:]:
                if target_id == exc.id or (exc.affected_orders and target_id in exc.affected_orders) or (target_id in exc.description):
                    self.current_state.active_exceptions.remove(exc)
                    reward += 0.5
                    info["event_log"].append(f"Exception resolved via {action_cmd.command_type.value} on target {target_id}.")
                    resolved_any = True
            
            if not resolved_any:
                reward -= 0.1
                info["event_log"].append(f"Action {action_cmd.command_type.value} failed: Target {target_id} not resolvable or already cleared.")

        # Wildcard Noise: 1% chance for any active robot to encounter SENSOR_FAILURE
        for r in self.current_state.robots:
            if r.status == RobotStatus.ACTIVE:
                if random.random() < 0.01:
                    r.status = RobotStatus.SENSOR_FAILURE

        self.current_state.time_step += 1

        # Check completion
        try:
            if tasks.check_completion(self.current_state, difficulty=self.scenario_name):
                reward += 1.0
                terminated = True
                info["event_log"].append("Task completed successfully!")
            elif self.current_state.time_step >= 50:
                reward -= 1.0
                terminated = True
                info["event_log"].append("Max time steps reached. Terminating early.")
        except Exception as e:
            info["event_log"].append(f"Error checking completion: {str(e)}")

        self.total_reward += reward
        exception_count = len(self.current_state.active_exceptions)
        print(f"[STEP] Action: {action_repr} | Reward: {reward} | Exception Count: {exception_count}.")
        
        if terminated:
            # Simple success logic: positive total reward
            status = "True" if reward > 0 else "False"
            print(f"[END] Total Reward: {self.total_reward} | Success: {status}.")

        return self.state(), reward, terminated, info

    def _state_to_text(self) -> str:
        """
        Helper method that converts the complex WarehouseState dictionary/Pydantic model 
        into a clean, concise English string intended for LLM prompting.
        """
        if not self.current_state:
            return "Environment state is not initialized."
            
        state = self.current_state
        text = [
            f"=== SYSTEM VITAL SIGNS ===",
            f"Time Step: {state.time_step}",
            f"Exception Count: {len(state.active_exceptions)}",
            f"=========================="
        ]
        
        # Robots
        robot_texts = []
        for r in state.robots:
            status_val = getattr(r.status, 'value', r.status)
            robot_texts.append(f"Robot {r.id}: Location {r.location}, Status: {status_val}, Battery: {r.battery_level}%")
        text.append("Robots:\n- " + "\n- ".join(robot_texts))
        
        # Inventory
        if state.inventory_status:
            inv_texts = []
            for item, qty in state.inventory_status.items():
                inv_texts.append(f"{item}: {qty} units")
            text.append("Inventory:\n- " + "\n- ".join(inv_texts))
        else:
            text.append("Inventory: Empty")
            
        # Exceptions
        if state.active_exceptions:
            exc_texts = []
            for e in state.active_exceptions:
                type_val = getattr(e.type, 'value', e.type)
                exc_texts.append(f"Exception {e.id} ({type_val}): {e.description} (Severity {e.severity}, affects {e.affected_orders})")
            text.append("Active Exceptions:\n- " + "\n- ".join(exc_texts))
        else:
            text.append("Active Exceptions: None")
            
        return "\n\n".join(text)
# segment 1 running veryfying script 
#if __name__ == "__main__":
    #env = WarehouseEnvironment("configs/warehouse_map.json", "configs/scenarios.yaml") 
    #print("Initial State Loaded:", env.state())

#segment 1 and segment 2 combined running script 
# if __name__ == "__main__":
#     env = WarehouseEnvironment("configs/warehouse_map.json", "configs/scenarios.yaml") 
#     initial_state = env.state()
    
#     print("--- 🚦 Segment 2 Verification Test ---")
    
#     if initial_state.get("robots"):
#         test_robot = initial_state["robots"][0]
#         r_id = test_robot.get("id")
        
#         print(f"✅ Found Robot: {r_id} at {test_robot.get('location')}")
        
#         # Test Action
#         action = {
#             "command_type": "MOVE_ROBOT",
#             "target_id": r_id,
#             "parameters": {"target_location": [2, 1]} 
#         }
        
#         print(f"⚡ Moving {r_id} to [2, 1]...")
#         next_obs, reward, terminated, info = env.step(action)
        
#         # Result Verification
#         new_robot = next_obs["robots"][0]
#         print(f"📍 Result Location: {new_robot.get('location')}")
#         print(f"🔋 Result Battery: {new_robot.get('battery_level')}%")
#         print(f"⏱️ Time Step: {next_obs.get('time_step')}")
#     else:
#         print("❌ No robots found in config!")    

#segment 3 verification script 
# if __name__ == "__main__":
#     env = WarehouseEnvironment("configs/warehouse_map.json", "configs/scenarios.yaml")
    
#     print("--- 🏆 FINAL SYSTEM INTEGRATION TEST ---")
    
#     # 1. Test REQUEST_RESTOCK (Reward +0.5)
#     print("\n📦 Action: Requesting Restock for component_A...")
#     restock_act = {
#         "command_type": "REQUEST_RESTOCK",
#         "target_id": "R1",
#         "parameters": {"component_name": "component_A"} 
#     }
#     obs, reward, term, info = env.step(restock_act)
#     print(f"Result -> Reward: {reward}, New Qty: {obs['inventory_status']['component_A']}, Info: {info['event_log'][-1]}")

#     # 2. Test RE_POLL_SENSOR (Reward +0.2)
#     # Pehle force fail karte hain test ke liye
#     env.current_state.robots[0].status = RobotStatus.SENSOR_FAILURE
#     print("\n📡 Action: Re-polling Sensor for R1...")
#     poll_act = {"command_type": "RE_POLL_SENSOR", "target_id": "R1"}
#     obs, reward, term, info = env.step(poll_act)
#     print(f"Result -> Reward: {reward}, New Status: {obs['robots'][0]['status']}, Info: {info['event_log'][-1]}")

#     # 3. Test Task Completion (Reward +1.0 + Terminated)
#     # Note: Ye tabhi True hoga agar aapka tasks.py ka logic meet ho raha ho
#     print(f"\n🏁 Mission Status -> Terminated: {term}")
#     if term:
#         print("🎉 SCENARIO CLEARED!")
#     else:
#         print("🛠️ Mission ongoing (Tasks not yet completed).")

#brutal verifier or collision case verifier:-
# if __name__ == "__main__":
    # env = WarehouseEnvironment("configs/warehouse_map.json", "configs/scenarios.yaml")
    
    # print("--- ⚖️ BRUTAL JUDGE: STRESS TESTING ENVIRONMENT ---")
    
    # # TEST 1: Collision Penalty (Moving into a wall or blocked path)
    # # Scenario mein (2,3) par 'fallen_pallet' hai, wahan move karte hain.
    # collision_act = {
    #     "command_type": "MOVE_ROBOT",
    #     "target_id": "R1",
    #     "parameters": {"target_location": [2, 3]} 
    # }
    # _, reward, _, info = env.step(collision_act)
    # print(f"💥 Collision Test -> Reward: {reward} (Expect: -0.1), Event: {info['event_log'][-1]}")

    # # TEST 2: Battery Death Penalty
    # # Battery ko force-drain karte hain zero tak
    # env.current_state.robots[0].battery_level = 0.4 
    # death_act = {
    #     "command_type": "MOVE_ROBOT",
    #     "target_id": "R1",
    #     "parameters": {"target_location": [1, 1]}
    # }
    # _, reward, term, info = env.step(death_act)
    # print(f"💀 Battery Death Test -> Reward: {reward} (Expect: -0.1), Terminated: {term}")

    # # TEST 3: Tasks.py Integration (Easy Mode)
    # # Saari exceptions clear karte hain aur inventory badhate hain
    # env.current_state.active_exceptions = []
    # env.current_state.inventory_status = {"component_A": 100, "component_B": 100}
    
    # # Ek harmless move karte hain state update karne ke liye
    # final_act = {"command_type": "RE_POLL_SENSOR", "target_id": "R1"}
    # _, reward, term, info = env.step(final_act)
    
    # print(f"\n🏆 Final Completion Test -> Terminated: {term} (Expect: True if criteria met)")
    # if term:
    #     print("✅ SUCCESS: The Judge is satisfied with tasks.py logic.")

    # TEST: Reward Hacking Fix
# Pehle inventory ko full kardo
# if __name__ == "__main__":
#     # 1. PEHLE ENV KO INITIALIZE KARO (Yeh missing tha!)
#     env = WarehouseEnvironment("configs/warehouse_map.json", "configs/scenarios.yaml")
    
#     print("--- ⚖️ BRUTAL JUDGE: REWARD HACKING TEST ---")
    
#     # 2. Inventory ko full kardo (Setup state)
#     # Ensure current_state exists
#     if env.current_state:
#         env.current_state.inventory_status["component_A"] = 100
#         print(f"Initial Inventory A: {env.current_state.inventory_status['component_A']}")

#         # 3. AB TEST KARO: Restock command bhejo jab inventory already full ho
#         action = {
#             "command_type": "REQUEST_RESTOCK",
#             "target_id": "R1",
#             "parameters": {"component_name": "component_A"}
#         }
        
#         print("\n📦 Attempting unnecessary restock (Hacking attempt)...")
#         obs, reward, terminated, info = env.step(action)

#         # 4. RESULT DEKHO
#         print(f"💰 Reward received: {reward}")
#         print(f"📝 Event Log: {info['event_log'][-1]}")

#         if reward < 0:
#             print("\n✅ SUCCESS: Reward Hacking blocked! Agent ko penalty mili.")
#         else:
#             print("\n❌ FAILED: Agent is still getting free points! Logic check karo.")
#     else:
#         print("❌ Error: State not initialized properly.")