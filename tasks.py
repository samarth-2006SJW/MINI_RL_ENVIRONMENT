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