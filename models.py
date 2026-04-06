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