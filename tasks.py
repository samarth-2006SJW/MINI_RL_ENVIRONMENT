from typing import Dict, Any
from models import WarehouseState, RobotStatus

def check_easy(state: WarehouseState) -> bool:
    """
    Evaluates if the state meets the 'Easy' difficulty completion criteria.
    Criteria: No active exceptions and all inventory component levels are above configured threshold.
    """
    try:
        if len(state.active_exceptions) > 0:
            return False
            
        # Check inventory levels
        if not state.inventory_status:
            return False
            
        min_inv = state.config.get('min_inventory', 20) if state.config else 20
        for item, stock in state.inventory_status.items():
            if stock <= min_inv:
                return False
                
        return True
    except (TypeError, AttributeError):
        return False

def check_medium(state: WarehouseState) -> bool:
    """
    Evaluates if the state meets the 'Medium' difficulty completion criteria.
    Criteria: Easy criteria passed + No robots in SENSOR_FAILURE + inventory levels > configured threshold.
    """
    try:
        # Explicit call to lower-tier check
        if not check_easy(state):
            return False
            
        # Check inventory > configured higher threshold for medium
        min_inv_medium = state.config.get('min_inventory_medium', 50) if state.config else 50
        for item, stock in state.inventory_status.items():
            if stock <= min_inv_medium:
                return False
                
        # Check robot statuses gracefully
        if state.robots:
            for robot in state.robots:
                if robot.status == RobotStatus.SENSOR_FAILURE:
                    return False
                    
        return True
    except (TypeError, AttributeError):
        return False        

def check_hard(state: WarehouseState) -> bool:
    """
    Evaluates if the state meets the 'Hard' difficulty completion criteria.
    Criteria: Medium criteria passed + time_step <= configured threshold + all robots battery_level > configured threshold.
    """
    try:
        # Explicit call to lower-tier check
        if not check_medium(state):
            return False
            
        max_time = state.config.get('max_time_steps', 50) if state.config else 50
        if state.time_step > max_time:
            return False
            
        min_batt = state.config.get('min_battery', 20) if state.config else 20
        if state.robots:
            for robot in state.robots:
                if robot.battery_level <= min_batt:
                    return False
                    
        return True
    except (TypeError, AttributeError):
        return False

def check_completion(state: WarehouseState, difficulty: str = "easy") -> bool:
    """
    Acts as the single entry point for checking task completion.
    
    Args:
        state (WarehouseState): The current environment state.
        difficulty (str): The requested difficulty level ("easy", "medium", "hard").
        
    Returns:
        bool: True if criteria for the specified difficulty is met, False otherwise.
    """
    try:
        difficulty = difficulty.lower()
        
        if difficulty == "easy":
            return check_easy(state)
        elif difficulty == "medium":
            return check_medium(state)
        elif difficulty == "hard":
            return check_hard(state)
        else:
            return check_easy(state)
    except Exception:
        return False   

def get_task_progress(state: WarehouseState, total_starting_exceptions: int = 5) -> float:
    """
    Returns a percentage (0.0 to 1.0) of how close the agent is to finishing,
    based on resolved exceptions vs. total starting exceptions.
    
    Args:
        state (WarehouseState): The current state.
        total_starting_exceptions (int): The baseline number of exceptions at start.
                                          Defaults to 5 for general calculations.
        
    Returns:
        float: Completion percentage from 0.0 to 1.0.
    """
    try:
        current_exceptions = len(state.active_exceptions)
        if total_starting_exceptions <= 0:
            total_starting_exceptions = 1  # Avoid division by zero
            
        resolved = max(0, total_starting_exceptions - current_exceptions)
        progress = float(resolved) / float(total_starting_exceptions)
        
        # Cap between 0.0 and 1.0
        return max(0.0, min(1.0, progress))
    except (TypeError, AttributeError):
        # Graceful fallback if state is improperly structured
        return 0.0             