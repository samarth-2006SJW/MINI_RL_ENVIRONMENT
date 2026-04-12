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

from backend.core.models import WarehouseState, Robot, BlockedPath, CommandType, ExceptionIssue, LogisticsCommand, RobotStatus, ExceptionType
from backend.core.utils import load_config, check_collision
from backend.core import tasks


class WarehouseEnvironment(BaseEnv):
    """
    Warehouse Logistics Exception Handler RL Environment.

    Reward scheme (strictly bounded 0.0 – 1.0):
      Each step reward = current_progress – previous_progress
      where progress ∈ [0.0, 1.0] measures fraction of initial exceptions resolved.
      This guarantees sum(rewards) ∈ [0.0, 1.0] over any episode,
      satisfying the OpenEnv hackathon validator constraint.
    """

    def __init__(self, map_path: str, scenario_path: str, scenario_name: str = "easy"):
        self.map_path      = map_path
        self.scenario_path = scenario_path
        self.scenario_name = scenario_name
        self.map_config    = load_config(self.map_path)

        all_scenarios      = load_config(self.scenario_path)
        self.scenario_config = all_scenarios.get(self.scenario_name, {})

        self.current_state: Optional[WarehouseState] = None
        self.total_reward  = 0.0
        self._initial_exceptions = 0
        self._exceptions_resolved_this_step = 0
        self._initialize_state()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _log_event(self, info: Dict, message: str) -> None:
        info.setdefault("event_log", []).append(message)

    def _maybe_inject_dynamic_exception(self, info: Dict) -> None:
        cfg   = self.current_state.config or {}
        chance = float(cfg.get("new_exception_chance", 0.0))
        max_active = int(cfg.get("max_active_exceptions", 6))
        if chance <= 0.0:
            return
        if len(self.current_state.active_exceptions) >= max_active:
            return
        if random.random() < chance:
            idx     = self.current_state.time_step
            new_exc = ExceptionIssue(
                id=f"EX_DYNAMIC_{idx}",
                type=ExceptionType.DELAY,
                description="Dynamic shipment delay detected in live warehouse traffic.",
                affected_orders=[f"ORD_DYNAMIC_{idx}"],
                severity=3,
            )
            self.current_state.active_exceptions.append(new_exc)
            self._log_event(info, f"Injected dynamic exception {new_exc.id}")

    # ── State Init ───────────────────────────────────────────────────────────

    def _initialize_state(self) -> None:
        robots              = [Robot(**r) for r in self.scenario_config.get("robots", [])]
        blocked_paths       = [BlockedPath(**p) for p in self.scenario_config.get("blocked_paths", [])]
        active_exceptions   = [ExceptionIssue(**e) for e in self.scenario_config.get("active_exceptions", [])]
        inventory_status    = self.scenario_config.get("inventory_status", {})
        worker_availability = self.scenario_config.get("worker_availability", 0)
        config_data         = self.scenario_config.get("config", {})

        self.current_state = WarehouseState(
            time_step=0,
            worker_availability=worker_availability,
            inventory_status=inventory_status,
            robots=robots,
            blocked_paths=blocked_paths,
            active_exceptions=active_exceptions,
            config=config_data,
        )

    # ── OpenEnv API ──────────────────────────────────────────────────────────

    def reset(self) -> Dict:
        self._initialize_state()
        self.total_reward         = 0.0
        self._initial_exceptions  = len(self.current_state.active_exceptions)
        self._exceptions_resolved_this_step = 0
        return self.state()

    def state(self) -> Dict:
        if self.current_state is None:
            raise ValueError("Environment state is not initialized — call reset() first.")
        obs = self.current_state.model_dump()
        import enum
        def _convert_enums(obj):
            if isinstance(obj, dict):   return {k: _convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list): return [_convert_enums(x) for x in obj]
            elif isinstance(obj, enum.Enum): return obj.value
            return obj
        obs = _convert_enums(obs)
        for robot in obs.get("robots", []):
            if robot.get("status") in ["SENSOR_FAILURE"]:
                robot["location"] = None
        return obs

    def step(self, action: Union[Dict, LogisticsCommand]) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one environment step.

        Returns
        -------
        (observation, reward, terminated, info)

        Reward uses a Time-Decaying Pot:
        - Max pot starts at 0.99.
        - Pot shrinks dynamically with each step taken (penalizing inefficiency).
        - When an exception is resolved, it awards (1 / total_exceptions) * current_pot.
        - Guarantees exactly 0.0-0.99 math range.
        """
        terminated = False
        info       = {"event_log": []}
        cfg        = self.current_state.config or {}
        max_time_steps = int(cfg.get("max_time_steps", 50))
        self._exceptions_resolved_this_step = 0

        # ── Parse action ─────────────────────────────────────────────────────
        try:
            if isinstance(action, dict):
                action_cmd = LogisticsCommand(**action)
            elif isinstance(action, LogisticsCommand):
                action_cmd = action
            else:
                raise ValueError("Action must be a dict or LogisticsCommand.")
        except Exception as e:
            action_cmd = LogisticsCommand(command_type=CommandType.WAIT)
            self._log_event(info, f"Invalid action payload; defaulted to WAIT ({e})")

        target_robot = next(
            (r for r in self.current_state.robots if r.id == action_cmd.target_id), None
        )

        # ── Execute action ───────────────────────────────────────────────────

        if action_cmd.command_type == CommandType.MOVE_ROBOT and target_robot:
            if target_robot.battery_level <= 0.0:
                self._log_event(info, f"Robot {target_robot.id} has empty battery; move denied")
            else:
                target_robot.battery_level = max(0.0, target_robot.battery_level - 0.5)
                params   = action_cmd.parameters or {}
                loc      = params.get("target_location")
                if loc and len(loc) == 2:
                    target_loc = (int(loc[0]), int(loc[1]))
                    grid = self.map_config.get("dimensions", [10, 10])
                    if not check_collision(target_loc, (int(grid[0]), int(grid[1])), self.current_state.robots, self.current_state.blocked_paths):
                        target_robot.location = target_loc
                        self._log_event(info, f"Moved robot {target_robot.id} to {target_loc}")
                    else:
                        self._log_event(info, f"Collision risk at {target_loc}; move blocked")
                else:
                    self._log_event(info, "MOVE_ROBOT missing target_location; ignored")

        elif action_cmd.command_type == CommandType.RE_POLL_SENSOR:
            if target_robot and target_robot.status == RobotStatus.SENSOR_FAILURE:
                target_robot.status = RobotStatus.ACTIVE
                self._log_event(info, f"Re-polled sensor and reactivated robot {target_robot.id}")
            else:
                self._log_event(info, "RE_POLL_SENSOR: no matching SENSOR_FAILURE target")

        elif action_cmd.command_type == CommandType.REQUEST_RESTOCK:
            params = action_cmd.parameters or {}
            comp   = params.get("component_name")
            if comp and self.current_state.inventory_status.get(comp, 0) < 100:
                self.current_state.inventory_status[comp] = min(
                    100, self.current_state.inventory_status[comp] + 20
                )
                self._log_event(info, f"Restocked {comp} to {self.current_state.inventory_status[comp]}")
            else:
                self._log_event(info, "REQUEST_RESTOCK: invalid component or already maxed")

        elif action_cmd.command_type in [
            CommandType.REROUTE_ORDER,
            CommandType.DISPATCH_MAINTENANCE,
            CommandType.ASSIGN_WORKER,
        ]:
            target_id = action_cmd.target_id
            resolved  = False

            if action_cmd.command_type == CommandType.DISPATCH_MAINTENANCE:
                for r in self.current_state.robots:
                    if r.id == target_id and r.status != RobotStatus.ACTIVE:
                        r.status, r.battery_level = RobotStatus.ACTIVE, 100.0
                        resolved = True
                        self._log_event(info, f"Maintenance dispatched; robot {r.id} restored")

            if action_cmd.command_type == CommandType.ASSIGN_WORKER:
                for bp in self.current_state.blocked_paths[:]:
                    if bp.id == target_id:
                        self.current_state.blocked_paths.remove(bp)
                        resolved = True
                        self._log_event(info, f"Worker assigned; cleared blockage {bp.id}")

            for exc in self.current_state.active_exceptions[:]:
                if target_id == exc.id or (exc.affected_orders and target_id in exc.affected_orders):
                    self.current_state.active_exceptions.remove(exc)
                    resolved = True
                    self._exceptions_resolved_this_step += 1
                    self._log_event(info, f"Resolved exception {exc.id}")

            if not resolved:
                self._log_event(info, f"No resolution target matched for {action_cmd.command_type}")

        else:
            self._log_event(info, f"Action {action_cmd.command_type} executed (WAIT or no-op)")

        # ── Dynamic difficulty hook (disabled by default) ─────────────────────
        self._maybe_inject_dynamic_exception(info)

        # ── Advance time ─────────────────────────────────────────────────────
        self.current_state.time_step += 1

        # ── Compute Time-Decaying Reward (Strictly 0.0-0.99 mathematically bounded) ──
        reward = 0.0
        if self._exceptions_resolved_this_step > 0 and self._initial_exceptions > 0:
            # Pot shrinks by 0.015 each step, but never goes below 0.20 (so solving is always worth *something*)
            current_pot = max(0.20, 0.99 - (self.current_state.time_step * 0.015))
            fraction_resolved = self._exceptions_resolved_this_step / float(self._initial_exceptions)
            reward = round(fraction_resolved * current_pot, 4)

        # ── Termination conditions ────────────────────────────────────────────
        if tasks.check_completion(self.current_state, difficulty=self.scenario_name):
            terminated = True
            self._log_event(info, f"Task completed at step {self.current_state.time_step}")
        elif self.current_state.time_step >= max_time_steps:
            terminated = True
            self._log_event(info, f"Timeout at step {self.current_state.time_step}")

        self.total_reward += reward
        return self.state(), reward, terminated, info

    # ── Rendering helper (used by app.py and inference.py) ───────────────────

    def _state_to_text(self) -> str:
        if not self.current_state:
            return "Not initialized."
        s = self.current_state
        return (
            f"Time: {s.time_step}, "
            f"Exceptions: {len(s.active_exceptions)}, "
            f"Robots: {[{'id': r.id, 'status': r.status.value, 'batt': r.battery_level} for r in s.robots]}, "
            f"Inventory: {dict(s.inventory_status)}"
        )
