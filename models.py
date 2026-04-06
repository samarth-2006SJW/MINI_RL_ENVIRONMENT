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

class Robot(BaseModel):
    """
    Represents a single automated transport robot within the warehouse.
    
    Attributes:
        id (str): Unique identifier for the robot.
        location (Optional[Tuple[int, int]]): The (x, y) grid coordinates. None if SENSOR_FAILURE.
        status (RobotStatus): Operating condition of the robot.
        battery_level (float): Remaining battery from 0.0 to 100.0.
        assigned_task (Optional[str]): ID of the current assigned order or task.
    """
    id: str = Field(..., description="Unique identifier for the robot")
    location: Optional[Tuple[int, int]] = Field(
        None, description="The (x, y) grid coordinates. None if SENSOR_FAILURE"
    )
    status: RobotStatus = Field(..., description="Operating condition of the robot")
    battery_level: float = Field(
        ..., description="Remaining battery from 0.0 to 100.0", ge=0.0, le=100.0
    )
    assigned_task: Optional[str] = Field(None, description="ID of the current assigned order or task")


class BlockedPath(BaseModel):
    """
    Represents an obstruction in the warehouse grid.
    
    Attributes:
        id (str): Unique identifier for the blockage.
        location (Tuple[int, int]): The (x, y) coordinates of the blockage.
        obstruction_type (str): Text description of what is blocking the path.
        severity (ObstructionSeverity): The impact level of the obstruction.
    """
    id: str = Field(..., description="Unique identifier for the blockage")
    location: Tuple[int, int] = Field(..., description="The (x, y) coordinates of the blockage")
    obstruction_type: str = Field(..., description="Description of the obstruction type")
    severity: ObstructionSeverity = Field(..., description="Impact level of the obstruction")