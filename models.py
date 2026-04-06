from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple, Union, Any
from enum import Enum


class RobotStatus(str, Enum):
    """Enumeration of possible statuses for a warehouse robot."""
    ACTIVE = "ACTIVE"
    IDLE = "IDLE"
    MAINTENANCE = "MAINTENANCE"
    SENSOR_FAILURE = "SENSOR_FAILURE"


class ObstructionSeverity(str, Enum):
    """Enumeration of severity levels for warehouse obstructions."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class ExceptionType(str, Enum):
    """Enumeration of recognized exception types requiring resolution."""
    DELAY = "DELAY"
    SHORTAGE = "SHORTAGE"
    BREAKDOWN = "BREAKDOWN"
    MISROUTING = "MISROUTING"


class CommandType(str, Enum):
    """Enumeration of valid action commands for the RL agent."""
    MOVE_ROBOT = "MOVE_ROBOT"
    REROUTE_ORDER = "REROUTE_ORDER"
    DISPATCH_MAINTENANCE = "DISPATCH_MAINTENANCE"
    REQUEST_RESTOCK = "REQUEST_RESTOCK"
    ASSIGN_WORKER = "ASSIGN_WORKER"
    RE_POLL_SENSOR = "RE_POLL_SENSOR"
    WAIT = "WAIT"