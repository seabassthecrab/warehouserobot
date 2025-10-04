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

def build_prm(wmap: WarehouseMap, nsamples=400, k=8, seed=0, inflated=False, radius=1):
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
                goal_sample_rate=0.05, seed=0, inflated=False, radius=1):
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
                while cur is not None:
                    cur = nodes[cur]
                    if cur is not None: 
                        path.append(cur)
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

# ------------------------- Robot dynamics ---------------------------------

@dataclass
class Robot:
    id: int
    start_cell: Tuple[int,int]
    goal_cell: Tuple[int,int]
    length: float = 2.0
    width: float = 1.0
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0
    w: float = 0.0
    max_speed: float = 1.0
    max_accel: float = 0.5
    max_omega: float = 2.0
    path_cells: List[Tuple[int,int]] = field(default_factory=list)
    waypoints: List[Tuple[float,float]] = field(default_factory=list)
    waypoint_idx: int = 0
    done: bool = False
    collided: bool = False
    color: Tuple[float,float,float] = (0.2,0.4,1.0)
    enable_orca: bool = False

    def initialize_from_cells(self):
        self.x = float(self.start_cell[0]) + 0.5
        self.y = float(self.start_cell[1]) + 0.5
        
        # Initialize heading towards goal for smoother start
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
        """Enhanced controller with optional ORCA collision avoidance."""
        wp = self.current_goal_point()
        if wp is None:
            self.v = max(0.0, self.v - self.max_accel*dt)
            if abs(self.v) < 1e-2:
                self.v = 0.0
                self.done = True
            self.w = 0.0
            return

        # Compute vector to waypoint
        dx = wp[0] - self.x
        dy = wp[1] - self.y
        dist = math.hypot(dx, dy)
        desired_theta = math.atan2(dy, dx)

        # ORCA avoidance if enabled
        if self.enable_orca and other_robots:
            avoid_vx, avoid_vy = orca_velocity(self, other_robots)
            # Blend avoidance into desired direction
            dx += avoid_vx * 2.0
            dy += avoid_vy * 2.0
            desired_theta = math.atan2(dy, dx)

        # Heading error
        heading_error = (desired_theta - self.theta + math.pi) % (2*math.pi) - math.pi

        # Speed profile - more conservative
        v_cmd = min(self.max_speed * 0.6, 0.6 * dist)  # Reduced speed limit
        
        # Slow down MORE for turns
        turn_factor = max(0.15, 1 - abs(heading_error)/(math.pi/3))  # Stricter turn slowdown
        v_cmd *= turn_factor

        # Limit speed for turning radius - more conservative
        min_turn_radius = max(self.length * 0.8, 0.8)  # Larger minimum radius
        max_v_for_turn = min_turn_radius * self.max_omega
        if v_cmd > max_v_for_turn:
            v_cmd = max_v_for_turn

        # Accelerate/decelerate
        if self.v < v_cmd:
            self.v = min(self.v + self.max_accel*dt, v_cmd)
        else:
            self.v = max(self.v - self.max_accel*dt, v_cmd)

        # Angular velocity controller - reduced gain for stability
        self.w = max(-self.max_omega, min(self.max_omega, 2.0 * heading_error))

        # Integrate unicycle dynamics
        self.theta += self.w * dt
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt

        # Check waypoint reached - larger threshold
        if dist < 0.8:  # Increased from 0.6
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

def robot_map_collision(robot: Robot, wmap: WarehouseMap, margin=0.3):
    """Enhanced collision detection with safety margin."""
    corners = rectangle_corners(robot.x, robot.y, robot.length + margin, robot.width + margin, robot.theta)
    for (cx,cy) in corners:
        r = int(math.floor(cx))
        c = int(math.floor(cy))
        if not wmap.in_bounds((r,c)): 
            return True
        if wmap.grid[r,c] == RACK: 
            return True
    
    # Also check center point for safety
    center_r = int(math.floor(robot.x))
    center_c = int(math.floor(robot.y))
    if not wmap.in_bounds((center_r, center_c)):
        return True
    if wmap.grid[center_r, center_c] == RACK:
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
                 inflated_planning=True):
        self.wmap = wmap
        self.robots = robots
        self.prm_graph = prm_graph
        self.planner = planner
        self.time = 0.0
        self.dt = 0.2
        self.metrics = Metrics()
        self.enable_orca = enable_orca
        self.inflated_planning = inflated_planning
        random.seed(seed)
        self.seed = seed
        
        # Set ORCA for all robots
        for r in self.robots:
            r.enable_orca = enable_orca

    def plan_all(self):
        successful_plans = 0
        total_path_length = 0
        
        for r in self.robots:
            if self.planner == 'astar':
                p = astar_grid(self.wmap, r.start_cell, r.goal_cell, 
                              inflated=self.inflated_planning, radius=1)
            elif self.planner == 'prm' and self.prm_graph is not None:
                p = prm_query(self.prm_graph, self.wmap, r.start_cell, r.goal_cell)
            elif self.planner == 'rrt':
                p = rrt_planner(self.wmap, r.start_cell, r.goal_cell, 
                               seed=r.id, inflated=self.inflated_planning, radius=1)
            else:
                p = astar_grid(self.wmap, r.start_cell, r.goal_cell)
            
            if p is None:
                print(f"Warning: Robot {r.id} failed to find path from {r.start_cell} to {r.goal_cell}")
                r.path_cells = []
                r.waypoints = []
            else:
                r.path_cells = p
                successful_plans += 1
                total_path_length += len(p)
            
            r.initialize_from_cells()
        
        self.metrics.plan_success_rate = successful_plans / len(self.robots) if self.robots else 0
        self.metrics.avg_path_length = total_path_length / successful_plans if successful_plans > 0 else 0
        
        print(f"Planning complete: {successful_plans}/{len(self.robots)} robots have valid paths")

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

    def run(self, max_steps=400, visualize=False, outdir='outputs'):
        frames = []
        start_time = time.time()
        
        for step in range(max_steps):
            if all(r.done or r.collided for r in self.robots):
                break
            self.step()
            if visualize and step % 2 == 0:  # Sample every other frame for efficiency
                frames.append(self.render_frame())
        
        runtime = time.time() - start_time
        self.metrics.runtimes.append(runtime)
        
        if visualize and frames:
            os.makedirs(outdir, exist_ok=True)
            orca_tag = '_orca' if self.enable_orca else ''
            gif_path = os.path.join(outdir, f'warehouse_{self.planner}_n{len(self.robots)}_s{self.seed}{orca_tag}.gif')
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=int(1000*self.dt*2), loop=0)
            print(f'Saved GIF to {gif_path}')
        
        return frames

    def render_frame(self, cellsize=8):
        rows, cols = self.wmap.rows, self.wmap.cols
        figw = cols * cellsize / 100.0
        figh = rows * cellsize / 100.0
        
        fig = plt.figure(figsize=(figw, figh), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Draw racks
        for r in range(rows):
            for c in range(cols):
                if self.wmap.grid[r,c] == RACK:
                    rect = patches.Rectangle((r, c), 1, 1, facecolor=(0.3,0.3,0.3), edgecolor=None)
                    ax.add_patch(rect)
        
        # Draw robot paths (FIXED: draw full path, not just remaining waypoints)
        for robot in self.robots:
            if robot.path_cells and len(robot.path_cells) > 1:
                path_x = [p[0] + 0.5 for p in robot.path_cells]
                path_y = [p[1] + 0.5 for p in robot.path_cells]
                cmap = plt.get_cmap('tab10')
                ax.plot(path_x, path_y, '-', color=cmap(robot.id % 10), 
                       alpha=0.3, linewidth=2, zorder=1)
        
        # Draw waypoints lightly
        for robot in self.robots:
            for wp in robot.waypoints:
                ax.plot(wp[0], wp[1], '.', color=(0.8,0.8,1.0), markersize=2, zorder=2)
        
        # Draw robots
        cmap = plt.get_cmap('tab10')
        for robot in self.robots:
            cx, cy = robot.x, robot.y
            L, W = robot.length, robot.width
            angle = math.degrees(robot.theta)
            
            # Color: gray if collided, otherwise colorful
            if robot.collided:
                fcolor = (0.5, 0.5, 0.5)
            else:
                fcolor = cmap(robot.id % 10)
            
            rect = patches.Rectangle((cx - L/2, cy - W/2), L, W, 
                                     angle=angle, facecolor=fcolor, alpha=0.9,
                                     edgecolor='black', linewidth=0.5, zorder=3)
            t = patches.Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData
            rect.set_transform(t)
            ax.add_patch(rect)
            
            # Heading arrow (smaller, properly scaled)
            arrow_len = min(L * 0.3, 0.4)
            ax.arrow(cx, cy, arrow_len*math.cos(robot.theta), arrow_len*math.sin(robot.theta), 
                    head_width=0.15, head_length=0.15, fc='k', ec='k', linewidth=0.8, zorder=4)
            
            # Start marker (blue circle)
            sx = robot.start_cell[0] + 0.5
            sy = robot.start_cell[1] + 0.5
            ax.plot(sx, sy, 'o', color='blue', markersize=6, markeredgewidth=1.5, 
                   markeredgecolor='darkblue', zorder=5)
            
            # Goal marker (green X)
            gx = robot.goal_cell[0] + 0.5
            gy = robot.goal_cell[1] + 0.5
            ax.plot(gx, gy, 'x', color='green', markersize=8, markeredgewidth=2.5, zorder=5)
        
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect('equal')
        fig.canvas.draw()
        
        # Convert to array
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        data = data.reshape((h, w, 4))[:,:,:3]
        plt.close(fig)
        
        img_arr = np.array(Image.fromarray(data).resize((cols*cellsize, rows*cellsize)))
        return img_arr

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
    wmap.add_racks(rack_rows=2, spacing=3, margin=2)
    
    robots = []
    used_cells = set()
    min_separation = 3  # Minimum distance between robot start positions
    
    for i in range(num_robots):
        # Find valid start position
        max_attempts = 100
        for attempt in range(max_attempts):
            s = random_free_cell(wmap, rng, inflated=inflated, radius=1)
            
            # Check minimum separation from other robots
            too_close = False
            for used in used_cells:
                if abs(s[0] - used[0]) + abs(s[1] - used[1]) < min_separation:
                    too_close = True
                    break
            
            if not too_close:
                break
        
        used_cells.add(s)
        
        # Find valid goal position
        for attempt in range(max_attempts):
            g = random_free_cell(wmap, rng, inflated=inflated, radius=1)
            
            # Ensure goal is different from start and not too close to start
            if g != s and abs(g[0] - s[0]) + abs(g[1] - s[1]) > 5:
                break
        
        r = Robot(id=i, start_cell=s, goal_cell=g)
        robots.append(r)
    
    return wmap, robots

# --------------------------- Experiment Runner ----------------------------

def run_experiment(planner='astar', rows=40, cols=60, num_robots=4, num_trials=5, 
                   enable_orca=False, inflated_planning=True, max_steps=400, 
                   visualize_first=False, outdir='outputs'):
    """Run multiple trials and collect statistics."""
    results = {
        'planner': planner,
        'num_robots': num_robots,
        'enable_orca': enable_orca,
        'inflated_planning': inflated_planning,
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
                          inflated=inflated_planning, radius=1)
        
        sim = DynamicMultiRobotSim(wmap, robots, planner=planner, prm_graph=prm, 
                                  seed=seed, enable_orca=enable_orca,
                                  inflated_planning=inflated_planning)
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
        # Basic comparison
        {'planner': 'astar', 'num_robots': 4, 'enable_orca': False, 'inflated_planning': True},
        {'planner': 'prm', 'num_robots': 4, 'enable_orca': False, 'inflated_planning': True},
        {'planner': 'rrt', 'num_robots': 4, 'enable_orca': False, 'inflated_planning': True},
        
        # With ORCA
        {'planner': 'astar', 'num_robots': 4, 'enable_orca': True, 'inflated_planning': True},
        {'planner': 'prm', 'num_robots': 4, 'enable_orca': True, 'inflated_planning': True},
        {'planner': 'rrt', 'num_robots': 4, 'enable_orca': True, 'inflated_planning': True},
        
        # Scaling test
        {'planner': 'astar', 'num_robots': 8, 'enable_orca': False, 'inflated_planning': True},
        {'planner': 'astar', 'num_robots': 8, 'enable_orca': True, 'inflated_planning': True},
    ]
    
    all_results = []
    
    print("=" * 60)
    print("RUNNING COMPARISON EXPERIMENTS")
    print("=" * 60)
    
    for i, config in enumerate(configs):
        print(f"\nConfig {i+1}/{len(configs)}: {config['planner'].upper()}, "
              f"robots={config['num_robots']}, ORCA={config['enable_orca']}, "
              f"inflated={config['inflated_planning']}")
        
        results = run_experiment(
            planner=config['planner'],
            num_robots=config['num_robots'],
            num_trials=5,
            enable_orca=config['enable_orca'],
            inflated_planning=config['inflated_planning'],
            visualize_first=(i < 3),  # Visualize first 3 configs
            outdir=outdir
        )
        all_results.append(results)
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison_results(all_results, outdir)
    
    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY RESULTS")
    print("=" * 60)
    print(f"{'Config':<40} {'Success':<10} {'Collisions':<12} {'Runtime':<10}")
    print("-" * 60)
    
    for res in all_results:
        config_str = f"{res['planner']}-R{res['num_robots']}"
        if res['enable_orca']:
            config_str += "-ORCA"
        if res['inflated_planning']:
            config_str += "-INF"
        
        print(f"{config_str:<40} {res['avg_success_rate']:<10.2%} "
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
    p.add_argument('--max-steps', type=int, default=400,
                   help='Maximum simulation steps')
    p.add_argument('--enable-orca', action='store_true',
                   help='Enable ORCA collision avoidance')
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
        print(f"Running experiment: {args.planner}, {args.num_robots} robots, {args.num_trials} trials")
        results = run_experiment(
            planner=args.planner,
            rows=args.rows,
            cols=args.cols,
            num_robots=args.num_robots,
            num_trials=args.num_trials,
            enable_orca=args.enable_orca,
            inflated_planning=not args.no_inflation,
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
        print(f"Inflated Planning: {results['inflated_planning']}")
        print(f"Average Success Rate: {results['avg_success_rate']:.2%}")
        print(f"Average Collision Rate: {results['avg_collision_rate']:.2%}")
        print(f"Average Runtime: {results['avg_runtime']:.2f}s")
        print(f"Average Steps: {results['avg_steps']:.1f}")
        print(f"Average Plan Success: {results['avg_plan_success']:.2%}")
        
    else:
        # Single run mode
        print(f"Running single simulation: {args.planner}, {args.num_robots} robots")
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
                                  inflated_planning=not args.no_inflation)
        
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
