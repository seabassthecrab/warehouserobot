import argparse
import math
import random
import time
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import patches
from PIL import Image

# ------------------------ Environment -------------------------------------

FREE = 0
RACK = 1

@dataclass
class WarehouseMap:
    rows: int
    cols: int
    grid: np.ndarray = field(init=False)

    def __post_init__(self):
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

    def add_racks(self, rack_rows: int = 2, spacing: int = 4, margin: int = 2):
        r = margin
        while r + rack_rows < self.rows - margin:
            for rr in range(r, r + rack_rows):
                for c in range(margin, self.cols - margin):
                    self.grid[rr, c] = RACK
            r += rack_rows + spacing

    def is_free(self, cell: Tuple[int,int]) -> bool:
        r, c = cell
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.grid[r,c] == FREE
        return False

    def in_bounds(self, cell):
        r,c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def neighbors4(self, cell):
        r,c = cell
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr,nc]==FREE:
                yield (nr,nc)

    def is_free_inflated(self, cell: Tuple[int,int], radius: int = 1) -> bool:
        """Check if cell and its radius-neighborhood are free (for footprint planning)."""
        r, c = cell
        for dr in range(-radius, radius+1):
            for dc in range(-radius, radius+1):
                nr, nc = r+dr, c+dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    return False
                if self.grid[nr, nc] != FREE:
                    return False
        return True

    def neighbors4_inflated(self, cell, radius: int = 1):
        """Get neighbors considering inflated footprint."""
        r, c = cell
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if self.is_free_inflated((nr, nc), radius):
                yield (nr, nc)

    def draw(self, ax):
        ax.imshow(self.grid.T, origin='lower', cmap='Greys', interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        return ax

# ------------------------ Planners ----------------------------------------

def heuristic(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar_grid(wmap: WarehouseMap, start: Tuple[int,int], goal: Tuple[int,int], 
               inflated: bool = False, radius: int = 1) -> Optional[List[Tuple[int,int]]]:
    """A* with optional obstacle inflation for robot footprint."""
    import heapq
    
    # Check if start and goal are valid
    if inflated:
        if not wmap.is_free_inflated(start, radius) or not wmap.is_free_inflated(goal, radius):
            return None
    else:
        if not wmap.is_free(start) or not wmap.is_free(goal):
            return None
    
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    gscore = {start: 0}
    closed = set()
    
    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        if current in closed: 
            continue
        closed.add(current)
        
        neighbors = wmap.neighbors4_inflated(current, radius) if inflated else wmap.neighbors4(current)
        for nb in neighbors:
            tentative = g + 1
            if tentative < gscore.get(nb, 1e9):
                gscore[nb] = tentative
                came_from[nb] = current
                heapq.heappush(open_set, (tentative + heuristic(nb, goal), tentative, nb))
    return None

def bresenham_line(a, b):
    (x0,y0) = a; (x1,y1) = b
    x0=int(x0); y0=int(y0); x1=int(x1); y1=int(y1)
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dy <= dx:
        err = dx // 2
        while True:
            points.append((x,y))
            if x == x1: break
            x += sx
            err -= dy
            if err < 0:
                y += sy
                err += dx
    else:
        err = dy // 2
        while True:
            points.append((x,y))
            if y == y1: break
            y += sy
            err -= dx
            if err < 0:
                x += sx
                err += dy
    return points

def collision_free_segment(wmap: WarehouseMap, a, b):
    for p in bresenham_line(a,b):
        if not wmap.is_free(p):
            return False
    return True

def build_prm(wmap: WarehouseMap, nsamples=400, k=8, seed=0, inflated=False, radius=2):
    """Build PRM with optional footprint inflation."""
    rng = random.Random(seed)
    samples = set()
    rows, cols = wmap.rows, wmap.cols
    trials = 0
    
    # Increase trial limit for inflated planning
    max_trials = nsamples * 50 if inflated else nsamples * 20
    
    while len(samples) < nsamples and trials < max_trials:
        trials += 1
        r = rng.randrange(0, rows)
        c = rng.randrange(0, cols)
        if inflated:
            if wmap.is_free_inflated((r,c), radius):
                samples.add((r,c))
        else:
            if wmap.is_free((r,c)):
                samples.add((r,c))
    
    if len(samples) < 50:  # Not enough samples
        print(f"Warning: Only {len(samples)} PRM samples found (target: {nsamples})")
    
    samples = list(samples)
    G = nx.Graph()
    for p in samples:
        G.add_node(p)
    
    for p in samples:
        dists = sorted(((math.hypot(p[0]-q[0], p[1]-q[1]), q) for q in samples if q!=p), key=lambda x:x[0])
        for _,q in dists[:k]:
            if not G.has_edge(p,q) and collision_free_segment(wmap, p, q):
                G.add_edge(p,q, weight=math.hypot(p[0]-q[0], p[1]-q[1]))
    
    print(f"PRM built: {len(samples)} nodes, {G.number_of_edges()} edges")
    return G

def prm_query(prm_graph: nx.Graph, wmap: WarehouseMap, start, goal):
    G = prm_graph.copy()
    if not wmap.is_free(start) or not wmap.is_free(goal):
        return None
    G.add_node(start); G.add_node(goal)
    
    nodes = list(prm_graph.nodes())
    dists_s = sorted(((math.hypot(start[0]-q[0], start[1]-q[1]), q) for q in nodes), key=lambda x:x[0])
    dists_g = sorted(((math.hypot(goal[0]-q[0], goal[1]-q[1]), q) for q in nodes), key=lambda x:x[0])
    
    connected = 0
    for _,q in dists_s[:10]:
        if collision_free_segment(wmap, start, q):
            G.add_edge(start, q, weight=math.hypot(start[0]-q[0], start[1]-q[1]))
            connected+=1
    for _,q in dists_g[:10]:
        if collision_free_segment(wmap, goal, q):
            G.add_edge(goal, q, weight=math.hypot(goal[0]-q[0], goal[1]-q[1]))
            connected+=1
    
    if connected == 0: 
        return None
    try:
        path = nx.shortest_path(G, start, goal, weight='weight')
        return path
    except nx.NetworkXNoPath:
        return None

def rrt_planner(wmap: WarehouseMap, start, goal, max_iters=2000, step=5, 
                goal_sample_rate=0.05, seed=0, inflated=False, radius=2):
    """RRT with optional footprint inflation."""
    rng = random.Random(seed)
    nodes = {start: None}
    pts = [start]
    rows, cols = wmap.rows, wmap.cols
    
    def sample_point():
        if rng.random() < goal_sample_rate:
            return goal
        for _ in range(100):
            r = rng.randrange(0,rows)
            c = rng.randrange(0,cols)
            if inflated:
                if wmap.is_free_inflated((r,c), radius):
                    return (r,c)
            else:
                if wmap.is_free((r,c)):
                    return (r,c)
        return goal
    
    def nearest(p):
        return min(pts, key=lambda q: (q[0]-p[0])**2 + (q[1]-p[1])**2)
    
    for it in range(max_iters):
        q_rand = sample_point()
        q_near = nearest(q_rand)
        vec = (q_rand[0]-q_near[0], q_rand[1]-q_near[1])
        dist = math.hypot(vec[0], vec[1])
        if dist == 0: continue
        
        q_new = (int(q_near[0] + round(step * vec[0] / dist)), 
                 int(q_near[1] + round(step * vec[1] / dist)))
        q_new = (max(0, min(rows-1, q_new[0])), max(0, min(cols-1, q_new[1])))
        
        valid = wmap.is_free_inflated(q_new, radius) if inflated else wmap.is_free(q_new)
        if not valid: 
            continue
        
        if collision_free_segment(wmap, q_near, q_new):
            nodes[q_new] = q_near
            pts.append(q_new)
            
            if math.hypot(q_new[0]-goal[0], q_new[1]-goal[1]) <= step and collision_free_segment(wmap, q_new, goal):
                nodes[goal] = q_new
                path = [goal]
                cur = goal
                visited = {goal}  # Track visited nodes to prevent cycles
                max_path_length = len(nodes) + 10  # Safety limit
                while cur is not None and len(path) < max_path_length:
                    cur = nodes[cur]
                    if cur is not None:
                        if cur in visited:  # Cycle detected!
                            print(f"Warning: Cycle detected in RRT path reconstruction")
                            break
                        path.append(cur)
                    visited.add(cur)
                path.reverse()
                return path
    return None

# ------------------------- ORCA Velocity Obstacles ------------------------

def orca_velocity(robot, others, time_horizon=2.0, max_neighbors=5):
    """
    Simplified ORCA (Optimal Reciprocal Collision Avoidance).
    Returns a preferred velocity adjustment to avoid nearby robots.
    """
    if not others:
        return (0.0, 0.0)
    
    # Get nearby robots
    nearby = []
    for other in others:
        if other.id == robot.id or other.done or other.collided:
            continue
        dist = math.hypot(robot.x - other.x, robot.y - other.y)
        if dist < 5.0:  # Only consider robots within 5 units
            nearby.append((dist, other))
    
    if not nearby:
        return (0.0, 0.0)
    
    # Sort by distance and take closest
    nearby.sort(key=lambda x: x[0])
    nearby = nearby[:max_neighbors]
    
    # Compute avoidance velocity
    avoid_vx, avoid_vy = 0.0, 0.0
    
    for dist, other in nearby:
        if dist < 0.01:
            dist = 0.01
        
        # Relative position
        rel_x = robot.x - other.x
        rel_y = robot.y - other.y
        
        # Relative velocity
        my_vx = robot.v * math.cos(robot.theta)
        my_vy = robot.v * math.sin(robot.theta)
        other_vx = other.v * math.cos(other.theta)
        other_vy = other.v * math.sin(other.theta)
        rel_vx = my_vx - other_vx
        rel_vy = my_vy - other_vy
        
        # Combined radius (safety margin)
        combined_radius = (robot.length + other.length) / 2.0 + 0.5
        
        # If on collision course, add repulsive velocity
        if dist < combined_radius * 2.0:
            # Strength inversely proportional to distance
            strength = (combined_radius * 2.0 - dist) / dist
            avoid_vx += strength * rel_x
            avoid_vy += strength * rel_y
    
    # Normalize and scale
    avoid_mag = math.hypot(avoid_vx, avoid_vy)
    if avoid_mag > 0.01:
        scale = min(robot.max_speed * 0.3, avoid_mag)
        avoid_vx = avoid_vx / avoid_mag * scale
        avoid_vy = avoid_vy / avoid_mag * scale
    
    return (avoid_vx, avoid_vy)

# ------------------------ Multi-Robot Coordination -------------------------

def create_spacetime_reservation(path_cells, robot_id, dt=0.2):
    """
    Create space-time reservations from a path.
    Returns dict: {timestep: set of occupied cells}
    """
    reservations = {}
    time = 0.0
    for i, cell in enumerate(path_cells):
        start_time = int(time / dt)
        end_time = int((time + 3.0) / dt)
        
        for t in range(start_time, end_time + 1):
            if t not in reservations:
                reservations[t] = set()
            reservations[t].add(cell)
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                neighbor = (cell[0] + dr, cell[1] + dc)
                reservations[t].add(neighbor)
        
        time += 1.0
    
    return reservations

def astar_with_reservations(wmap: WarehouseMap, start: Tuple[int,int], goal: Tuple[int,int],
                            reservations: Dict[int, set], inflated: bool = False, 
                            radius: int = 1) -> Optional[List[Tuple[int,int]]]:
    """A* that avoids reserved space-time cells."""
    import heapq
    
    if inflated:
        if not wmap.is_free_inflated(start, radius) or not wmap.is_free_inflated(goal, radius):
            return None
    else:
        if not wmap.is_free(start) or not wmap.is_free(goal):
            return None
    
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, 0, start))
    came_from = {}
    gscore = {(start, 0): 0}
    closed = set()
    
    while open_set:
        _, g, t, current = heapq.heappop(open_set)
        
        if current == goal:
            path = [current]
            node = (current, t)
            while node in came_from:
                node = came_from[node]
                if node[0] is not None:
                    path.append(node[0])
            path.reverse()
            return path
        
        if (current, t) in closed:
            continue
        closed.add((current, t))
        
        neighbors = wmap.neighbors4_inflated(current, radius) if inflated else wmap.neighbors4(current)
        
        for nb in neighbors:
            next_time = t + 1
            if next_time in reservations and nb in reservations[next_time]:
                continue
            
            tentative = g + 1
            node_key = (nb, next_time)
            
            if tentative < gscore.get(node_key, 1e9):
                gscore[node_key] = tentative
                came_from[node_key] = (current, t)
                heapq.heappush(open_set, (tentative + heuristic(nb, goal), tentative, next_time, nb))
        
        if t < 1000:
            wait_node = (current, t + 1)
            if current not in reservations.get(t + 1, set()):
                if g + 1 < gscore.get(wait_node, 1e9):
                    gscore[wait_node] = g + 1
                    came_from[wait_node] = (current, t)
                    heapq.heappush(open_set, (g + 1 + heuristic(current, goal), g + 1, t + 1, current))
    
    return None

def prioritized_planning(wmap: WarehouseMap, robots: List, planner: str = 'astar',
                        prm_graph: Optional[nx.Graph] = None, inflated: bool = True) -> Dict[int, List]:
    """Plan paths for all robots with priorities."""
    sorted_robots = sorted(robots, key=lambda r: r.id)
    
    all_reservations = {}
    planned_paths = {}
    
    for robot in sorted_robots:
        print(f"  Robot {robot.id}: Planning with {len(all_reservations)} timesteps reserved...", end=" ")
        
        if planner == 'astar':
            path = astar_with_reservations(wmap, robot.start_cell, robot.goal_cell,
                                          all_reservations, inflated=inflated, radius=2)
        elif planner == 'prm' and prm_graph is not None:
            path = prm_query(prm_graph, wmap, robot.start_cell, robot.goal_cell)
        elif planner == 'rrt':
            path = rrt_planner(wmap, robot.start_cell, robot.goal_cell, 
                             seed=robot.id, inflated=inflated, radius=2)
        else:
            path = astar_with_reservations(wmap, robot.start_cell, robot.goal_cell,
                                          all_reservations, inflated=inflated, radius=2)
        
        if path is None:
            print("FAILED - trying without reservations...")
            if planner == 'astar':
                path = astar_grid(wmap, robot.start_cell, robot.goal_cell, inflated=inflated, radius=2)
            if path is None and inflated:
                path = astar_grid(wmap, robot.start_cell, robot.goal_cell, inflated=False, radius=1)
        
        planned_paths[robot.id] = path
        
        if path:
            print(f"SUCCESS (length: {len(path)})")
            robot_reservations = create_spacetime_reservation(path, robot.id)
            for t, cells in robot_reservations.items():
                if t not in all_reservations:
                    all_reservations[t] = set()
                all_reservations[t].update(cells)
        else:
            print("FAILED - no path found")
            planned_paths[robot.id] = None
    
    return planned_paths
# ------------------------- Robot dynamics ---------------------------------

@dataclass
class Robot:
    id: int
    start_cell: Tuple[int,int]
    goal_cell: Tuple[int,int]
    length: float = 2.0
    width: float = 0.6
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0
    w: float = 0.0
    max_speed: float = 50.0
    max_accel: float = 10
    max_omega: float = 8
    path_cells: List[Tuple[int,int]] = field(default_factory=list)
    waypoints: List[Tuple[float,float]] = field(default_factory=list)
    waypoint_idx: int = 0
    done: bool = False
    collided: bool = False
    color: Tuple[float,float,float] = (0.2,0.4,1.0)
    enable_orca: bool = False

    
    prev_heading_error: float = 0.0
    ang_kp: float = 4.0   # proportional gain
    ang_kd: float = 0.8   # derivative damping gain

    def initialize_from_cells(self):
        self.x = float(self.start_cell[0]) + 0.5
        self.y = float(self.start_cell[1]) + 0.5
        
        gx = float(self.goal_cell[0]) + 0.5
        gy = float(self.goal_cell[1]) + 0.5
        self.theta = math.atan2(gy - self.y, gx - self.x)
        
        self.waypoints = [(float(r)+0.5, float(c)+0.5) for (r,c) in self.path_cells] if self.path_cells else [(gx,gy)]
        self.waypoint_idx = 0

    def current_goal_point(self):
        if self.waypoint_idx < len(self.waypoints):
            return self.waypoints[self.waypoint_idx]
        return None

    def step_control(self, dt=0.1, other_robots=None):
        wp = self.current_goal_point()
        if wp is None:
            self.v = max(0.0, self.v - self.max_accel * dt)
            if abs(self.v) < 1e-2:
                self.v = 0.0
                self.done = True
            self.w = 0.0
            return

        # vector to waypoint
        dx = wp[0] - self.x
        dy = wp[1] - self.y
        dist = math.hypot(dx, dy)
        desired_theta = math.atan2(dy, dx)

        # ORCA avoidance (damped)
        if self.enable_orca and other_robots:
            avoid_vx, avoid_vy = orca_velocity(self, other_robots)
            avoid_vx *= 0.6
            avoid_vy *= 0.6
            if abs(avoid_vx) < 0.02: avoid_vx = 0.0
            if abs(avoid_vy) < 0.02: avoid_vy = 0.0
            dx += avoid_vx * 2.0
            dy += avoid_vy * 2.0
            desired_theta = math.atan2(dy, dx)

        # heading error normalized
        heading_error = (desired_theta - self.theta + math.pi) % (2*math.pi) - math.pi
        if abs(heading_error) < 0.02:
            heading_error = 0.0

        # PD angular controller
        de = (heading_error - self.prev_heading_error) / dt
        w_cmd = self.ang_kp * heading_error + self.ang_kd * de
        w_cmd = max(-self.max_omega, min(self.max_omega, w_cmd))

        # low-pass filter to smooth w
        alpha = 0.6
        self.w = alpha * self.w + (1 - alpha) * w_cmd
        self.prev_heading_error = heading_error

        # speed profile
        v_cmd = min(self.max_speed * 0.6, 0.6 * dist)
        turn_factor = max(0.3, 1 - abs(heading_error) / (math.pi / 2))
        v_cmd *= turn_factor

        # further slow when turning sharply
        if abs(heading_error) > math.radians(25):
            v_cmd *= 0.1

        # turning radius limit
        min_turn_radius = max(self.length * 0.1, 0.1)
        max_v_for_turn = min_turn_radius * self.max_omega
        if v_cmd > max_v_for_turn:
            v_cmd = max_v_for_turn

        # accel/decel
        if self.v < v_cmd:
            self.v = min(self.v + self.max_accel * dt, v_cmd)
        else:
            self.v = max(self.v - self.max_accel * dt, v_cmd)

        # integrate unicycle dynamics
        self.theta += self.w * dt
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt

        # waypoint progress
        if dist < 0.9 and abs(heading_error) < math.radians(60):
            self.waypoint_idx += 1
            if self.waypoint_idx >= len(self.waypoints):
                self.done = True


# ------------------------ Collision detection ------------------------------

def rectangle_corners(cx, cy, length, width, theta):
    dx = length/2.0
    dy = width/2.0
    corners_local = [ (dx, dy), (dx, -dy), (-dx, -dy), (-dx, dy) ]
    corners = []
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    for lx, ly in corners_local:
        rx = cx + lx * cos_t - ly * sin_t
        ry = cy + lx * sin_t + ly * cos_t
        corners.append((rx, ry))
    return corners

def project_polygon(axis, polygon):
    dots = [axis[0]*p[0] + axis[1]*p[1] for p in polygon]
    return min(dots), max(dots)

def overlap_intervals(a_min, a_max, b_min, b_max):
    return not (a_max < b_min or b_max < a_min)

def polygons_overlap(poly1, poly2):
    axes = []
    for poly in (poly1, poly2):
        n = len(poly)
        for i in range(n):
            x1,y1 = poly[i]
            x2,y2 = poly[(i+1)%n]
            edge = (x2-x1, y2-y1)
            axis = (-edge[1], edge[0])
            norm = math.hypot(axis[0], axis[1])
            if norm == 0: continue
            axes.append((axis[0]/norm, axis[1]/norm))
    
    for axis in axes:
        a_min, a_max = project_polygon(axis, poly1)
        b_min, b_max = project_polygon(axis, poly2)
        if not overlap_intervals(a_min, a_max, b_min, b_max):
            return False
    return True

def robot_collision(robot: Robot, other: Robot):
    poly1 = rectangle_corners(robot.x, robot.y, robot.length, robot.width, robot.theta)
    poly2 = rectangle_corners(other.x, other.y, other.length, other.width, other.theta)
    return polygons_overlap(poly1, poly2)

def robot_map_collision(robot: Robot, wmap: WarehouseMap, margin=0.1):
    """
    Stricter collision detection - checks multiple points along robot body.
    Returns True if robot overlaps with RACK cells.
    """
    corners = rectangle_corners(robot.x, robot.y, robot.length + margin, robot.width + margin, robot.theta)
    
    # Check all corners
    for (cx, cy) in corners:
        r = int(math.floor(cx))
        c = int(math.floor(cy))
        if wmap.in_bounds((r, c)) and wmap.grid[r, c] == RACK:
            return True
    
    # Check center
    center_r = int(math.floor(robot.x))
    center_c = int(math.floor(robot.y))
    if wmap.in_bounds((center_r, center_c)) and wmap.grid[center_r, center_c] == RACK:
        return True
    
    # Check midpoints of robot edges for better coverage
    front_x = robot.x + (robot.length/2) * math.cos(robot.theta)
    front_y = robot.y + (robot.length/2) * math.sin(robot.theta)
    front_r = int(math.floor(front_x))
    front_c = int(math.floor(front_y))
    if wmap.in_bounds((front_r, front_c)) and wmap.grid[front_r, front_c] == RACK:
        return True
    
    return False

# ------------------------ Simulator ---------------------------------------

@dataclass
class Metrics:
    successes: int = 0
    collisions: int = 0
    total_steps: int = 0
    runtimes: List[float] = field(default_factory=list)
    plan_success_rate: float = 0.0
    avg_path_length: float = 0.0

class DynamicMultiRobotSim:
    def __init__(self, wmap: WarehouseMap, robots: List[Robot], planner='astar', 
                 prm_graph: Optional[nx.Graph]=None, seed=0, enable_orca=False,
                 inflated_planning=True, coordinated_planning=True):
        self.wmap = wmap
        self.robots = robots
        self.prm_graph = prm_graph
        self.planner = planner
        self.time = 0.0
        self.dt = 0.2
        self.metrics = Metrics()
        self.enable_orca = enable_orca
        self.inflated_planning = inflated_planning
        self.coordinated_planning = coordinated_planning
        random.seed(seed)
        self.seed = seed
        
        # Set ORCA for all robots
        for r in self.robots:
            r.enable_orca = enable_orca

    def plan_all(self):
        successful_plans = 0
        total_path_length = 0
        
        print("\n=== Path Planning Phase ===")
        
        if self.coordinated_planning and self.planner == 'astar':
            print("Using PRIORITIZED PLANNING (coordinated multi-robot)")
            planned_paths = prioritized_planning(self.wmap, self.robots, self.planner,
                                                self.prm_graph, self.inflated_planning)
            
            for r in self.robots:
                p = planned_paths.get(r.id)
                if p is None or len(p) == 0:
                    print(f"Robot {r.id}: No valid path")
                    r.path_cells = []
                    r.waypoints = []
                else:
                    r.path_cells = p
                    successful_plans += 1
                    total_path_length += len(p)
                
                r.initialize_from_cells()
        else:
            print("Using INDEPENDENT PLANNING (no coordination)")
            for r in self.robots:
                print(f"Robot {r.id}: Planning from {r.start_cell} to {r.goal_cell}...", end=" ")
                
                if self.planner == 'astar':
                    p = astar_grid(self.wmap, r.start_cell, r.goal_cell, 
                                  inflated=self.inflated_planning, radius=2)
                elif self.planner == 'prm' and self.prm_graph is not None:
                    p = prm_query(self.prm_graph, self.wmap, r.start_cell, r.goal_cell)
                elif self.planner == 'rrt':
                    p = rrt_planner(self.wmap, r.start_cell, r.goal_cell, 
                                   seed=r.id, inflated=self.inflated_planning, radius=2)
                else:
                    p = astar_grid(self.wmap, r.start_cell, r.goal_cell)
                
                if p is None:
                    print(f"❌ FAILED (no path found)")
                    if self.inflated_planning:
                        print(f"  Retrying without inflation...")
                        p = astar_grid(self.wmap, r.start_cell, r.goal_cell, inflated=False, radius=0)
                        if p:
                            print(f"  ✓ Found path without inflation (length: {len(p)})")
                    
                if p is None:
                    print(f"  Warning: No valid path exists for robot {r.id}")
                    r.path_cells = []
                    r.waypoints = []
                else:
                    r.path_cells = p
                    successful_plans += 1
                    total_path_length += len(p)
                    print(f"✓ Path found (length: {len(p)} cells)")
                
                r.initialize_from_cells()
        
        self.metrics.plan_success_rate = successful_plans / len(self.robots) if self.robots else 0
        self.metrics.avg_path_length = total_path_length / successful_plans if successful_plans > 0 else 0
        
        print(f"\nPlanning Summary: {successful_plans}/{len(self.robots)} robots have valid paths")
        print("=" * 40)

    def step(self):
        # Update each robot with ORCA
        for r in self.robots:
            if r.done or r.collided:
                continue
            r.step_control(dt=self.dt, other_robots=self.robots if self.enable_orca else None)
        
        # Collision checking
        for i, r in enumerate(self.robots):
            if r.done or r.collided: continue
            
            if robot_map_collision(r, self.wmap):
                r.collided = True
                self.metrics.collisions += 1
                continue
            
            for j in range(i+1, len(self.robots)):
                o = self.robots[j]
                if o.done or o.collided: continue
                if robot_collision(r, o):
                    r.collided = True
                    o.collided = True
                    self.metrics.collisions += 1
        
        # Count successes
        for r in self.robots:
            if r.done and not getattr(r, '_counted', False):
                self.metrics.successes += 1
                setattr(r, '_counted', True)
        
        self.metrics.total_steps += 1
        self.time += self.dt

    def run(self, max_steps=600, visualize=False, outdir='outputs'):
        """Run simulation with adjustable max_steps."""
        frames = []
        start_time = time.time()
        
        for step in range(max_steps):
            if all(r.done or r.collided for r in self.robots):
                break
            self.step()
            if visualize:  # Capture every frame for smoother animation
                frames.append(self.render_frame())
        
        runtime = time.time() - start_time
        self.metrics.runtimes.append(runtime)
        
        # Print status of each robot
        print(f"\nSimulation ended at step {self.metrics.total_steps}/{max_steps}")
        for r in self.robots:
            status = "DONE" if r.done else ("COLLIDED" if r.collided else "INCOMPLETE")
            print(f"  Robot {r.id}: {status} (waypoint {r.waypoint_idx}/{len(r.waypoints)})")
        
        if visualize and frames:
            os.makedirs(outdir, exist_ok=True)
            orca_tag = '_orca' if self.enable_orca else ''
            gif_path = os.path.join(outdir, f'warehouse_{self.planner}_n{len(self.robots)}_s{self.seed}{orca_tag}.gif')
            
            # Save GIF without resizing - preserve original frame dimensions
            imgs = [Image.fromarray(f) for f in frames]
            
            # Verify first frame size
            print(f"Frame dimensions: {imgs[0].size}, Total frames: {len(imgs)}")
            
            # Smoother animation: shorter duration per frame
            frame_duration = int(1000 * self.dt * 0.5)  # 0.5x speed for smoother playback
            
            imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], 
                        duration=frame_duration, loop=0, optimize=False)
            print(f'Saved GIF to {gif_path}')
        
        return frames

    def render_frame(self, cellsize=10, padding=10):
        """Render frame with proper padding to show full warehouse area."""
        rows, cols = self.wmap.rows, self.wmap.cols
        
        # Calculate figure size including padding
        total_width = cols + 2 * padding
        total_height = rows + 2 * padding
        
        # Larger figure with higher DPI for better quality
        fig = plt.figure(figsize=(total_width * 0.35, total_height * 0.35), dpi=100)
        ax = fig.add_subplot(111)
        
        # Remove all margins
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        ax.set_xlim(-padding, cols + padding)
        ax.set_ylim(-padding * 2, rows + padding *2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('white')
        ax.axis('off')  # Turn off axis frame
        
        # Draw racks
        for r in range(rows):
            for c in range(cols):
                if self.wmap.grid[r,c] == RACK:
                    rect = patches.Rectangle((r, c), 1, 1, facecolor=(0.2,0.2,0.2), edgecolor=None)
                    ax.add_patch(rect)
        
        # Draw robot paths
        for robot in self.robots:
            if robot.path_cells and len(robot.path_cells) > 1:
                path_x = [p[0] + 0.5 for p in robot.path_cells]
                path_y = [p[1] + 0.5 for p in robot.path_cells]
                cmap = plt.get_cmap('tab10')
                ax.plot(path_x, path_y, '-', color=cmap(robot.id % 10), 
                       alpha=0.3, linewidth=2, zorder=1)
        
        # Draw waypoints
        for robot in self.robots:
            for wp in robot.waypoints:
                ax.plot(wp[0], wp[1], '.', color=(0.8,0.8,1.0), markersize=2, zorder=2)
        
        # Draw robots
        cmap = plt.get_cmap('tab10')
        for robot in self.robots:
            cx, cy = robot.x, robot.y
            L, W = robot.length, robot.width
            theta_deg = math.degrees(robot.theta)
            
            if robot.collided:
                fcolor = (0.5, 0.5, 0.5)
            else:
                fcolor = cmap(robot.id % 10)
            
            # Create rectangle centered at origin, with length along x-axis
            # The length (L) is the forward direction, width (W) is the side
            rect = patches.Rectangle((-L/2, -W/2), L, W, 
                                     angle=0,  # Start unrotated
                                     facecolor=fcolor, alpha=0.9,
                                     edgecolor='black', linewidth=0.5, zorder=3)
            
            # Apply rotation around center, then translate to robot position
            t = (patches.Affine2D()
                 .rotate(robot.theta)  # Rotate by theta radians
                 .translate(cx, cy)    # Move to robot position
                 + ax.transData)
            rect.set_transform(t)
            ax.add_patch(rect)
            
            # Draw heading arrow at front of robot
            # Arrow points from center towards the front
            arrow_len = L * 0.4  # Slightly longer to be more visible
            arrow_start_x = cx
            arrow_start_y = cy
            arrow_dx = arrow_len * math.cos(robot.theta)
            arrow_dy = arrow_len * math.sin(robot.theta)
            
            ax.arrow(arrow_start_x, arrow_start_y, arrow_dx, arrow_dy,
                    head_width=0.3, head_length=0.25, fc='yellow', ec='black', 
                    linewidth=1.5, zorder=4, length_includes_head=True)
            
            # Start marker
            sx = robot.start_cell[0] + 0.5
            sy = robot.start_cell[1] + 0.5
            ax.plot(sx, sy, 'o', color='blue', markersize=8, markeredgewidth=2, 
                   markeredgecolor='darkblue', zorder=5)
            
            # Goal marker
            gx = robot.goal_cell[0] + 0.5
            gy = robot.goal_cell[1] + 0.5
            ax.plot(gx, gy, 'x', color='green', markersize=10, markeredgewidth=3, zorder=5)
        
        ax.set_aspect('equal', adjustable='box')
        
        fig.canvas.draw()
        
        # Convert to array without any resizing
        buf = fig.canvas.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        img_array = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        img_array = img_array[:, :, :3]  # Drop alpha channel
        
        plt.close(fig)
        return img_array

# --------------------------- Utilities ------------------------------------

def random_free_cell(wmap: WarehouseMap, rng: random.Random, inflated=False, radius=1):
    """Get a random free cell, optionally with footprint clearance."""
    for _ in range(10000):
        r = rng.randrange(0, wmap.rows)
        c = rng.randrange(0, wmap.cols)
        if inflated:
            if wmap.is_free_inflated((r, c), radius):
                return (r, c)
        else:
            if wmap.is_free((r, c)):
                return (r, c)
    
    # Fallback: scan entire map
    for i in range(wmap.rows):
        for j in range(wmap.cols):
            if inflated:
                if wmap.is_free_inflated((i, j), radius):
                    return (i, j)
            else:
                if wmap.is_free((i, j)):
                    return (i, j)
    raise RuntimeError('no free cell')

def make_demo(rows=40, cols=60, num_robots=4, seed=0, inflated=True):
    """Create demo with proper spawn points considering robot footprint."""
    rng = random.Random(seed)
    wmap = WarehouseMap(rows, cols)
    wmap.add_racks(rack_rows=2, spacing=5, margin=2)
    
    robots = []
    used_cells = set()
    min_separation = 3
    
    # Use radius=1.4 for spawning even if inflated planning is enabled
    # This allows spawning in aisles between racks
    spawn_radius = 2
    
    for i in range(num_robots):
        # Find valid start position
        max_attempts = 200
        for attempt in range(max_attempts):
            s = random_free_cell(wmap, rng, inflated=True, radius=spawn_radius)
            
            # Additional check: ensure spawn is not at the extreme edges
            if s[0] < 2 or s[0] > rows - 3 or s[1] < 2 or s[1] > cols - 3:
                continue
            
            # Check minimum separation from other robots
            too_close = False
            for used in used_cells:
                if abs(s[0] - used[0]) + abs(s[1] - used[1]) < min_separation:
                    too_close = True
                    break
            
            if not too_close:
                break
        else:
            print(f"Warning: Could not find well-separated start for robot {i}, using best available")
        
        used_cells.add(s)
        
        # Find valid goal position
        for attempt in range(max_attempts):
            g = random_free_cell(wmap, rng, inflated=True, radius=spawn_radius)
            
            # Ensure goal is also not at extreme edges
            if g[0] < 2 or g[0] > rows - 3 or g[1] < 2 or g[1] > cols - 3:
                continue
            
            # Ensure goal is different from start and not too close to start
            if g != s and abs(g[0] - s[0]) + abs(g[1] - s[1]) > 8:
                # Also check goal isn't too close to other goals
                too_close = False
                for other in robots:
                    if abs(g[0] - other.goal_cell[0]) + abs(g[1] - other.goal_cell[1]) < min_separation:
                        too_close = True
                        break
                if not too_close:
                    break
        
        r = Robot(id=i, start_cell=s, goal_cell=g)
        robots.append(r)
    
    return wmap, robots

# --------------------------- Experiment Runner ----------------------------

def run_experiment(planner='astar', rows=40, cols=60, num_robots=4, num_trials=5, 
                   enable_orca=False, inflated_planning=True, coordinated_planning=True,
                   max_steps=600, visualize_first=False, outdir='outputs'):
    """Run multiple trials and collect statistics."""
    results = {
        'planner': planner,
        'num_robots': num_robots,
        'enable_orca': enable_orca,
        'inflated_planning': inflated_planning,
        'coordinated_planning': coordinated_planning,
        'trials': [],
        'avg_success_rate': 0.0,
        'avg_collision_rate': 0.0,
        'avg_runtime': 0.0,
        'avg_steps': 0.0,
        'avg_plan_success': 0.0,
        'avg_path_length': 0.0
    }
    
    for trial in range(num_trials):
        print(f"  Trial {trial+1}/{num_trials}...", end=' ')
        seed = trial * 100
        wmap, robots = make_demo(rows=rows, cols=cols, num_robots=num_robots, 
                                seed=seed, inflated=inflated_planning)
        
        # Build PRM if needed
        prm = None
        if planner == 'prm':
            prm = build_prm(wmap, nsamples=800, k=8, seed=seed, 
                          inflated=inflated_planning, radius=2)
        
        sim = DynamicMultiRobotSim(wmap, robots, planner=planner, prm_graph=prm, 
                                  seed=seed, enable_orca=enable_orca,
                                  inflated_planning=inflated_planning,
                                  coordinated_planning=coordinated_planning)
        sim.plan_all()
        
        visualize = (trial == 0 and visualize_first)
        sim.run(max_steps=max_steps, visualize=visualize, outdir=outdir)
        
        # Collect metrics
        trial_result = {
            'seed': seed,
            'successes': sim.metrics.successes,
            'collisions': sim.metrics.collisions,
            'steps': sim.metrics.total_steps,
            'runtime': sim.metrics.runtimes[0] if sim.metrics.runtimes else 0.0,
            'plan_success_rate': sim.metrics.plan_success_rate,
            'avg_path_length': sim.metrics.avg_path_length,
            'success_rate': sim.metrics.successes / num_robots,
            'collision_rate': sim.metrics.collisions / num_robots
        }
        results['trials'].append(trial_result)
        print(f"Success: {trial_result['success_rate']:.2f}, Collisions: {trial_result['collision_rate']:.2f}")
    
    # Aggregate results
    results['avg_success_rate'] = np.mean([t['success_rate'] for t in results['trials']])
    results['avg_collision_rate'] = np.mean([t['collision_rate'] for t in results['trials']])
    results['avg_runtime'] = np.mean([t['runtime'] for t in results['trials']])
    results['avg_steps'] = np.mean([t['steps'] for t in results['trials']])
    results['avg_plan_success'] = np.mean([t['plan_success_rate'] for t in results['trials']])
    results['avg_path_length'] = np.mean([t['avg_path_length'] for t in results['trials']])
    
    return results

def run_comparison_experiments(outdir='outputs'):
    """Run comprehensive comparison across planners and configurations."""
    os.makedirs(outdir, exist_ok=True)
    
    configs = [
        # A* variations
        {'planner': 'astar', 'num_robots': 4, 'enable_orca': False, 'inflated_planning': True, 'coordinated_planning': False},
        #{'planner': 'astar', 'num_robots': 4, 'enable_orca': False, 'inflated_planning': True, 'coordinated_planning': True},
        {'planner': 'astar', 'num_robots': 4, 'enable_orca': True, 'inflated_planning': True, 'coordinated_planning': False},
        #{'planner': 'astar', 'num_robots': 4, 'enable_orca': True, 'inflated_planning': True, 'coordinated_planning': True},
        
        # PRM comparison
        {'planner': 'prm', 'num_robots': 4, 'enable_orca': False, 'inflated_planning': True, 'coordinated_planning': False},
        {'planner': 'prm', 'num_robots': 4, 'enable_orca': True, 'inflated_planning': True, 'coordinated_planning': True},
        # RRT comparison
        {'planner': 'rrt', 'num_robots': 4, 'enable_orca': False, 'inflated_planning': True, 'coordinated_planning': False},
        {'planner': 'rrt', 'num_robots': 4, 'enable_orca': True, 'inflated_planning': True, 'coordinated_planning': True},

        
        # Scaling test
        #{'planner': 'astar', 'num_robots': 8, 'enable_orca': False, 'inflated_planning': True, 'coordinated_planning': True},
        #{'planner': 'astar', 'num_robots': 8, 'enable_orca': True, 'inflated_planning': True, 'coordinated_planning': True},
    ]
    
    all_results = []
    
    print("=" * 60)
    print("RUNNING COMPARISON EXPERIMENTS")
    print("=" * 60)
    
    for i, config in enumerate(configs):
        coord_str = "COORD" if config['coordinated_planning'] else "INDEP"
        orca_str = "ORCA" if config['enable_orca'] else "BASIC"
        print(f"\nConfig {i+1}/{len(configs)}: {config['planner'].upper()}, "
              f"robots={config['num_robots']}, {coord_str}, {orca_str}")
        
        results = run_experiment(
            planner=config['planner'],
            num_robots=config['num_robots'],
            num_trials=5,
            enable_orca=config['enable_orca'],
            inflated_planning=config['inflated_planning'],
            coordinated_planning=config.get('coordinated_planning', False),
            visualize_first=(i < 2),  # Visualize first 2 configs
            outdir=outdir
        )
        all_results.append(results)
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison_results(all_results, outdir)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    print(f"{'Config':<50} {'Success':<10} {'Collisions':<12} {'Runtime':<10}")
    print("-" * 80)
    
    for res in all_results:
        config_str = f"{res['planner']}-R{res['num_robots']}"
        if res.get('coordinated_planning'):
            config_str += "-COORD"
        else:
            config_str += "-INDEP"
        if res['enable_orca']:
            config_str += "-ORCA"
        if res['inflated_planning']:
            config_str += "-INF"
        
        print(f"{config_str:<50} {res['avg_success_rate']:<10.2%} "
              f"{res['avg_collision_rate']:<12.2%} {res['avg_runtime']:<10.2f}s")
    
    print("\nResults saved to:", outdir)
    return all_results

def plot_comparison_results(all_results, outdir='outputs'):
    """Generate comparison plots."""
    
    # Extract data for plotting
    labels = []
    success_rates = []
    collision_rates = []
    runtimes = []
    
    for res in all_results:
        label = f"{res['planner']}-R{res['num_robots']}"
        if res['enable_orca']:
            label += "-ORCA"
        labels.append(label)
        success_rates.append(res['avg_success_rate'] * 100)
        collision_rates.append(res['avg_collision_rate'] * 100)
        runtimes.append(res['avg_runtime'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Multi-Robot Warehouse Simulation Comparison', fontsize=16, fontweight='bold')
    
    # Success rate comparison
    ax = axes[0, 0]
    bars = ax.bar(range(len(labels)), success_rates, color='steelblue', alpha=0.8)
    ax.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Task Success Rate', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Target')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Collision rate comparison
    ax = axes[0, 1]
    bars = ax.bar(range(len(labels)), collision_rates, color='tomato', alpha=0.8)
    ax.set_ylabel('Collision Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Robot Collision Rate', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Runtime comparison
    ax = axes[1, 0]
    bars = ax.bar(range(len(labels)), runtimes, color='seagreen', alpha=0.8)
    ax.set_ylabel('Runtime (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('Computation Time', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=8)
    
    # Success vs Collision scatter
    ax = axes[1, 1]
    colors = ['blue' if 'ORCA' not in l else 'orange' for l in labels]
    markers = ['o' if 'astar' in l else ('^' if 'prm' in l else 's') for l in labels]
    
    for i, (sr, cr, label, color, marker) in enumerate(zip(success_rates, collision_rates, labels, colors, markers)):
        ax.scatter(sr, cr, s=150, c=color, marker=marker, alpha=0.7, edgecolors='black', linewidth=1.5)
        ax.annotate(label, (sr, cr), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Collision Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Success vs Collision Trade-off', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add quadrant lines
    ax.axhline(y=np.mean(collision_rates), color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=np.mean(success_rates), color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(outdir, 'comparison_results.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {plot_path}")

# --------------------------- CLI ------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Dynamic Multi-Robot Warehouse Simulator')
    p.add_argument('--mode', choices=['single', 'experiment', 'compare'], default='single',
                   help='Run mode: single simulation, experiment trials, or full comparison')
    p.add_argument('--planner', choices=['astar','prm','rrt'], default='astar',
                   help='Path planning algorithm')
    p.add_argument('--rows', type=int, default=40,
                   help='Warehouse rows')
    p.add_argument('--cols', type=int, default=60,
                   help='Warehouse columns')
    p.add_argument('--num-robots', type=int, default=4,
                   help='Number of robots')
    p.add_argument('--seed', type=int, default=0,
                   help='Random seed')
    p.add_argument('--max-steps', type=int, default=600,
                   help='Maximum simulation steps (increased default for longer runs)')
    p.add_argument('--enable-orca', action='store_true',
                   help='Enable ORCA collision avoidance')
    p.add_argument('--coordinated', action='store_true',
                   help='Enable coordinated planning (robots avoid each other\'s planned paths)')
    p.add_argument('--no-inflation', action='store_true',
                   help='Disable obstacle inflation during planning')
    p.add_argument('--num-trials', type=int, default=5,
                   help='Number of trials for experiment mode')
    p.add_argument('--out', '--outdir', dest='outdir', default='outputs',
                   help='Output directory')
    return p.parse_args()

def main():
    args = parse_args()
    
    if args.mode == 'compare':
        # Run full comparison experiments
        all_results = run_comparison_experiments(outdir=args.outdir)
        
    elif args.mode == 'experiment':
        # Run experiment with specified config
        coord_enabled = getattr(args, 'coordinated', False)
        print(f"Running experiment: {args.planner}, {args.num_robots} robots, {args.num_trials} trials")
        results = run_experiment(
            planner=args.planner,
            rows=args.rows,
            cols=args.cols,
            num_robots=args.num_robots,
            num_trials=args.num_trials,
            enable_orca=args.enable_orca,
            inflated_planning=not args.no_inflation,
            coordinated_planning=coord_enabled,
            max_steps=args.max_steps,
            visualize_first=True,
            outdir=args.outdir
        )
        
        print("\n" + "=" * 60)
        print("EXPERIMENT RESULTS")
        print("=" * 60)
        print(f"Planner: {results['planner']}")
        print(f"Robots: {results['num_robots']}")
        print(f"ORCA: {results['enable_orca']}")
        print(f"Coordinated: {results.get('coordinated_planning', False)}")
        print(f"Inflated Planning: {results['inflated_planning']}")
        print(f"Average Success Rate: {results['avg_success_rate']:.2%}")
        print(f"Average Collision Rate: {results['avg_collision_rate']:.2%}")
        print(f"Average Runtime: {results['avg_runtime']:.2f}s")
        print(f"Average Steps: {results['avg_steps']:.1f}")
        print(f"Average Plan Success: {results['avg_plan_success']:.2%}")
        
    else:
        # Single run mode
        coord_enabled = getattr(args, 'coordinated', False)
        print(f"Running single simulation: {args.planner}, {args.num_robots} robots")
        print(f"Coordinated planning: {coord_enabled}, ORCA: {args.enable_orca}")
        
        wmap, robots = make_demo(rows=args.rows, cols=args.cols, 
                                num_robots=args.num_robots, seed=args.seed,
                                inflated=not args.no_inflation)
        
        prm = None
        if args.planner == 'prm':
            print("Building PRM...")
            prm = build_prm(wmap, nsamples=800, k=8, seed=args.seed, 
                          inflated=not args.no_inflation, radius=1)
        
        sim = DynamicMultiRobotSim(wmap, robots, planner=args.planner, prm_graph=prm, 
                                  seed=args.seed, enable_orca=args.enable_orca,
                                  inflated_planning=not args.no_inflation,
                                  coordinated_planning=coord_enabled)
        
        print("Planning paths...")
        sim.plan_all()
        
        print("Running simulation...")
        frames = sim.run(max_steps=args.max_steps, visualize=True, outdir=args.outdir)
        
        # Save final map
        os.makedirs(args.outdir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        wmap.draw(ax)
        
        for r in robots:
            ax.plot(r.start_cell[0]+0.5, r.start_cell[1]+0.5, 'o', color='blue', markersize=8)
            ax.plot(r.goal_cell[0]+0.5, r.goal_cell[1]+0.5, 'x', color='green', markersize=10, markeredgewidth=2)
            if r.path_cells:
                path_x = [p[0]+0.5 for p in r.path_cells]
                path_y = [p[1]+0.5 for p in r.path_cells]
                ax.plot(path_x, path_y, '-', alpha=0.3, linewidth=1.5)
        
        orca_tag = '_orca' if args.enable_orca else ''
        inf_tag = '_inf' if not args.no_inflation else ''
        plt.title(f'Warehouse {args.planner.upper()}{orca_tag}{inf_tag} - {args.num_robots} robots')
        plt.savefig(os.path.join(args.outdir, f'final_map_{args.planner}{orca_tag}{inf_tag}.png'), 
                   dpi=200, bbox_inches='tight')
        plt.close()
        
        # Print metrics
        print("\n" + "=" * 60)
        print("SIMULATION RESULTS")
        print("=" * 60)
        print(f"Successes: {sim.metrics.successes}/{len(robots)} ({sim.metrics.successes/len(robots):.1%})")
        print(f"Collisions: {sim.metrics.collisions}")
        print(f"Steps: {sim.metrics.total_steps}")
        print(f"Runtime: {sim.metrics.runtimes[0]:.2f}s")
        print(f"Plan Success Rate: {sim.metrics.plan_success_rate:.1%}")
        print(f"Avg Path Length: {sim.metrics.avg_path_length:.1f}")
        print(f"\nOutputs saved to: {args.outdir}")

if __name__ == '__main__':
    main()
