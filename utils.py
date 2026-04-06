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
