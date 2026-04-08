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
    def __init__(self, map_path: str, scenario_path: str, scenario_name: str = "easy"):
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
        robots = [Robot(**r_data) for r_data in self.scenario_config.get("robots", [])]
        blocked_paths = [BlockedPath(**path_data) for path_data in self.scenario_config.get("blocked_paths", [])]
        active_exceptions = [ExceptionIssue(**issue_data) for issue_data in self.scenario_config.get("active_exceptions", [])]
        inventory_status = self.scenario_config.get("inventory_status", {})
        worker_availability = self.scenario_config.get("worker_availability", 0)
        config_data = self.scenario_config.get("config", {})

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
        self._initialize_state()
        self.total_reward = 0.0
        return self.state()

    def state(self) -> Dict:
        if self.current_state is None:
            raise ValueError("Environment state is not initialized.")
        obs = self.current_state.model_dump()
        import enum
        def _convert_enums(obj):
            if isinstance(obj, dict): return {k: _convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list): return [_convert_enums(x) for x in obj]
            elif isinstance(obj, enum.Enum): return obj.value
            return obj
        obs = _convert_enums(obs)
        for robot in obs.get("robots", []):
            if robot.get("status") in [getattr(RobotStatus.SENSOR_FAILURE, 'value', 'SENSOR_FAILURE'), 'SENSOR_FAILURE']:
                robot["location"] = None
        return obs
        
    def step(self, action: Union[Dict, LogisticsCommand]) -> Tuple[Dict, float, bool, Dict]:
        reward = 0.0
        terminated = False
        info = {"event_log": []}
        try:
            if isinstance(action, dict): action_cmd = LogisticsCommand(**action)
            elif isinstance(action, LogisticsCommand): action_cmd = action
            else: raise ValueError("Action must be a dictionary.")
        except Exception as e:
            action_cmd = LogisticsCommand(command_type=CommandType.WAIT)
            reward -= 0.1
            
        target_robot = next((r for r in self.current_state.robots if r.id == action_cmd.target_id), None)
        
        if action_cmd.command_type == CommandType.MOVE_ROBOT and target_robot:
            if target_robot.battery_level <= 0.0:
                reward -= 0.1
            else:
                target_robot.battery_level = max(0.0, target_robot.battery_level - 0.5)
                params = action_cmd.parameters or {}
                loc = params.get("target_location")
                if loc and len(loc) == 2:
                    target_loc = (int(loc[0]), int(loc[1]))
                    grid = self.map_config.get("dimensions", [10, 10])
                    if not check_collision(target_loc, (int(grid[0]), int(grid[1])), self.current_state.robots, self.current_state.blocked_paths):
                        target_robot.location = target_loc
                    else:
                        reward -= 0.1

        elif action_cmd.command_type == CommandType.RE_POLL_SENSOR:
            if target_robot and target_robot.status == RobotStatus.SENSOR_FAILURE:
                target_robot.status = RobotStatus.ACTIVE
                reward += 0.2

        elif action_cmd.command_type == CommandType.REQUEST_RESTOCK:
            params = action_cmd.parameters or {}
            comp = params.get("component_name")
            if comp and self.current_state.inventory_status.get(comp, 0) < 100:
                self.current_state.inventory_status[comp] = min(100, self.current_state.inventory_status[comp] + 20)
                reward += 0.5
            else:
                reward -= 0.05

        elif action_cmd.command_type in [CommandType.REROUTE_ORDER, CommandType.DISPATCH_MAINTENANCE, CommandType.ASSIGN_WORKER]:
            target_id = action_cmd.target_id
            resolved = False
            if action_cmd.command_type == CommandType.DISPATCH_MAINTENANCE:
                for r in self.current_state.robots:
                    if r.id == target_id and r.status != RobotStatus.ACTIVE:
                        r.status, r.battery_level = RobotStatus.ACTIVE, 100.0
                        reward, resolved = reward + 0.5, True
            if action_cmd.command_type == CommandType.ASSIGN_WORKER:
                for bp in self.current_state.blocked_paths[:]:
                    if bp.id == target_id:
                        self.current_state.blocked_paths.remove(bp)
                        reward, resolved = reward + 0.5, True
            for exc in self.current_state.active_exceptions[:]:
                if target_id == exc.id or (exc.affected_orders and target_id in exc.affected_orders):
                    self.current_state.active_exceptions.remove(exc)
                    reward, resolved = reward + 0.5, True
            if not resolved: reward -= 0.1

        self.current_state.time_step += 1
        if tasks.check_completion(self.current_state, difficulty=self.scenario_name):
            reward, terminated = reward + 1.0, True
        elif self.current_state.time_step >= 50:
            reward, terminated = reward - 1.0, True

        self.total_reward += reward
        return self.state(), reward, terminated, info

    def _state_to_text(self) -> str:
        if not self.current_state: return "Not initialized."
        s = self.current_state
        return f"Time: {s.time_step}, Exceptions: {len(s.active_exceptions)}, Robots: {[{'id': r.id, 'loc': r.location, 'batt': r.battery_level} for r in s.robots]}"