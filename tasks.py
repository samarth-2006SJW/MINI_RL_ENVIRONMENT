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