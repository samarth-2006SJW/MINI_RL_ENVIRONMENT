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