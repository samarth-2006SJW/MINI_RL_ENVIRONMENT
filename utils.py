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
    if start == end:
        return [start]
        
    # Convert obstructions into a set of coordinates for O(1) lookups
    obs_set: Set[Tuple[int, int]] = {obs.location for obs in obstructions}
    
    # Priority queue: (f_score, (x, y))
    frontier = []
    heapq.heappush(frontier, (0, start))
    
    # Store where we came from and cost to get there
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    g_score: Dict[Tuple[int, int], int] = {start: 0}
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while frontier:
        current_f, current = heapq.heappop(frontier)
        
        if current == end:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
            
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Boundary checks
            if not (0 <= neighbor[0] < grid_size[0] and 0 <= neighbor[1] < grid_size[1]):
                continue
                
            # Obstruction check
            if neighbor in obs_set:
                continue
                
            tentative_g = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, end)
                heapq.heappush(frontier, (f_score, neighbor))
                came_from[neighbor] = current
                
    return None  # Unreachable

def check_collision(
    target_coordinate: Tuple[int, int], 
    grid_size: Tuple[int, int], 
    robots: List[Robot], 
    blocked_paths: List[BlockedPath]
) -> bool:
    """
    Checks if a target coordinate results in a collision.
    
    Args:
        target_coordinate: Proposed (x, y) move.
        grid_size: Total dimensions of the warehouse grid.
        robots: List of current active robots.
        blocked_paths: List of current known grid obstructions.
        
    Returns:
        True if the move results in a collision, False otherwise.
    """
    x, y = target_coordinate
    
    # 1. Wall Checks
    if not (0 <= x < grid_size[0] and 0 <= y < grid_size[1]):
        return True
        
    # 2. Blocked Path Checks
    for block in blocked_paths:
        if block.location == target_coordinate:
            return True
            
    # 3. Dynamic Robot Collisions
    # Gracefully ignore robots experiencing SENSOR_FAILURE where location is None
    for r in robots:
        if r.location is not None and r.location == target_coordinate:
            return True
            
    return False

