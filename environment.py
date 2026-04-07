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