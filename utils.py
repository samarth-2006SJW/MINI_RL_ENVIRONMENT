import heapq
import json
import os
from typing import List, Dict, Optional, Tuple, Set

# Local imports from our models contract
from models import Robot, BlockedPath, ExceptionType


def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Manhattan distance heuristic for A* pathfinding."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def find_shortest_path(
    start: Tuple[int, int], 
    end: Tuple[int, int], 
    grid_size: Tuple[int, int], 
    obstructions: List[BlockedPath]
) -> Optional[List[Tuple[int, int]]]:
    """
    Computes optimal path from start to end using A* algorithm.
    
    Args:
        start: Starting (x, y) coordinate.
        end: Goal (x, y) coordinate.
        grid_size: Maximum grid bounds (width, height).
        obstructions: List of BlockedPath instances to avoid.
        
    Returns:
        A list of coordinates representing the path. Returns None if unreachable.
    """