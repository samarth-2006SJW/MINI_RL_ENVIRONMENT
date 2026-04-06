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

class ExceptionIssue(BaseModel):
    """
    Represents a logistical failure that requires agent intervention.
    
    Attributes:
        id (str): Unique exception identifier.
        type (ExceptionType): The category of the exception.
        description (str): Human-readable explanation of the issue.
        affected_orders (List[str]): List of order IDs delayed or impacted by this issue.
        severity (int): Scale of 1 to 5 indicating priority/severity.
    """
    id: str = Field(..., description="Unique exception identifier")
    type: ExceptionType = Field(..., description="The category of the exception")
    description: str = Field(..., description="Explanation of the issue")
    affected_orders: List[str] = Field(..., description="Order IDs impacted by this issue")
    severity: int = Field(..., description="Scale of 1 to 5 indicating priority", ge=1, le=5)


class WarehouseState(BaseModel):
    """
    The complete Observation Space representing the current environment state.
    This model is strictly structured to serialize seamlessly to JSON, supporting state persistence.
    
    Attributes:
        time_step (int): Current simulation step.
        worker_availability (int): Count of human workers available for assignment.
        inventory_status (Dict[str, int]): Map of item IDs to their current stock levels.
        robots (List[Robot]): Real-time statuses of all warehouse robots.
        blocked_paths (List[BlockedPath]): Known grid obstructions.
        active_exceptions (List[ExceptionIssue]): Outstanding issues requiring agent action.
    """
    time_step: int = Field(..., description="Current simulation step", ge=0)
    worker_availability: int = Field(..., description="Count of human workers available", ge=0)
    inventory_status: Dict[str, int] = Field(
        default_factory=dict, description="Item IDs mapped to stock levels"
    )
    robots: List[Robot] = Field(default_factory=list, description="All warehouse robots")
    blocked_paths: List[BlockedPath] = Field(default_factory=list, description="Known grid obstructions")
    active_exceptions: List[ExceptionIssue] = Field(
        default_factory=list, description="Outstanding logistical issues"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Dynamic thresholds loaded from scenarios.yaml"
    )

class LogisticsCommand(BaseModel):
    """
    The Action Space representing interventions requested by the RL agent.
    
    Attributes:
        command_type (CommandType): The Enum category of the action.
        target_id (Optional[str]): Unique identifier of the robot, exception, or order.
        parameters (Optional[Dict[str, Union[str, int, float, List[int]]]]): Additional flexible parameters.
    """
    command_type: CommandType = Field(..., description="The action category")
    target_id: Optional[str] = Field(None, description="Identifier of the target entity")
    parameters: Optional[Dict[str, Union[str, int, float, List[int], List[str]]]] = Field(
        default_factory=dict,
        description="Flexible key-value pairs for command parameters (supports mixed types)"
    )
    
class LogisticsCommand(BaseModel):
    """
    The Action Space representing interventions requested by the RL agent.
    
    Attributes:
        command_type (CommandType): The Enum category of the action.
        target_id (Optional[str]): Unique identifier of the robot, exception, or order.
        parameters (Optional[Dict[str, Union[str, int, float, List[int]]]]): Additional flexible parameters.
    """
    command_type: CommandType = Field(..., description="The action category")
    target_id: Optional[str] = Field(None, description="Identifier of the target entity")
    parameters: Optional[Dict[str, Union[str, int, float, List[int], List[str]]]] = Field(
        default_factory=dict,
        description="Flexible key-value pairs for command parameters (supports mixed types)"
    )