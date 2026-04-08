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
                        
                           