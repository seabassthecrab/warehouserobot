

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

    def draw(self, ax):
        # draw racks as dark gray blocks
        ax.imshow(self.grid.T, origin='lower', cmap='Greys', interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        return ax

# ------------------------ Planners ----------------------------------------

def heuristic(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar_grid(wmap: WarehouseMap, start: Tuple[int,int], goal: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    import heapq
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    gscore = {start: 0}
    closed = set()
    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        if current in closed: continue
        closed.add(current)
        for nb in wmap.neighbors4(current):
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

def build_prm(wmap: WarehouseMap, nsamples=400, k=8, seed=0):
    rng = random.Random(seed)
    samples = set()
    rows, cols = wmap.rows, wmap.cols
    trials = 0
    while len(samples) < nsamples and trials < nsamples*20:
        trials += 1
        r = rng.randrange(0, rows)
        c = rng.randrange(0, cols)
        if wmap.is_free((r,c)):
            samples.add((r,c))
    samples = list(samples)
    G = nx.Graph()
    for p in samples:
        G.add_node(p)
    for p in samples:
        dists = sorted(((math.hypot(p[0]-q[0], p[1]-q[1]), q) for q in samples if q!=p), key=lambda x:x[0])
        for _,q in dists[:k]:
            if not G.has_edge(p,q) and collision_free_segment(wmap, p, q):
                G.add_edge(p,q, weight=math.hypot(p[0]-q[0], p[1]-q[1]))
    return G

def prm_query(prm_graph: nx.Graph, wmap: WarehouseMap, start, goal):
    G = prm_graph.copy()
    if not wmap.is_free(start) or not wmap.is_free(goal):
        return None
    G.add_node(start); G.add_node(goal)
    # connect to nearest
    nodes = list(prm_graph.nodes())
    dists_s = sorted(((math.hypot(start[0]-q[0], start[1]-q[1]), q) for q in nodes), key=lambda x:x[0])
    dists_g = sorted(((math.hypot(goal[0]-q[0], goal[1]-q[1]), q) for q in nodes), key=lambda x:x[0])
    connected = 0
    for _,q in dists_s[:10]:
        if collision_free_segment(wmap, start, q):
            G.add_edge(start, q, weight=math.hypot(start[0]-q[0], start[1]-q[1])); connected+=1
    for _,q in dists_g[:10]:
        if collision_free_segment(wmap, goal, q):
            G.add_edge(goal, q, weight=math.hypot(goal[0]-q[0], goal[1]-q[1])); connected+=1
    if connected == 0: return None
    try:
        path = nx.shortest_path(G, start, goal, weight='weight')
        return path
    except nx.NetworkXNoPath:
        return None

def rrt_planner(wmap: WarehouseMap, start, goal, max_iters=2000, step=5, goal_sample_rate=0.05, seed=0):
    rng = random.Random(seed)
    nodes = {start: None}
    pts = [start]
    rows, cols = wmap.rows, wmap.cols
    def sample_point():
        if rng.random() < goal_sample_rate:
            return goal
        return (rng.randrange(0,rows), rng.randrange(0,cols))
    def nearest(p):
        return min(pts, key=lambda q: (q[0]-p[0])**2 + (q[1]-p[1])**2)
    for it in range(max_iters):
        q_rand = sample_point()
        q_near = nearest(q_rand)
        vec = (q_rand[0]-q_near[0], q_rand[1]-q_near[1])
        dist = math.hypot(vec[0], vec[1])
        if dist == 0: continue
        q_new = (int(q_near[0] + round(step * vec[0] / dist)), int(q_near[1] + round(step * vec[1] / dist)))
        q_new = (max(0, min(rows-1, q_new[0])), max(0, min(cols-1, q_new[1])))
        if not wmap.is_free(q_new): continue
        if collision_free_segment(wmap, q_near, q_new):
            nodes[q_new] = q_near; pts.append(q_new)
            if math.hypot(q_new[0]-goal[0], q_new[1]-goal[1]) <= step and collision_free_segment(wmap, q_new, goal):
                nodes[goal] = q_new
                path = [goal]; cur = goal
                while cur is not None:
                    cur = nodes[cur]
                    if cur is not None: path.append(cur)
                path.reverse(); return path
    return None

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
    v: float = 0.0   # linear velocity (units per second)
    w: float = 0.0   # angular velocity (rad/s)
    max_speed: float = 1.0
    max_accel: float = 0.5
    max_omega: float = 2.0
    path_cells: List[Tuple[int,int]] = field(default_factory=list)
    waypoints: List[Tuple[float,float]] = field(default_factory=list)
    waypoint_idx: int = 0
    done: bool = False
    collided: bool = False
    color: Tuple[float,float,float] = (0.2,0.4,1.0)

    def initialize_from_cells(self):
        # initialize continuous pose centered in the start cell, with small random theta
        self.x = float(self.start_cell[0]) + 0.5
        self.y = float(self.start_cell[1]) + 0.5
        self.theta = random.uniform(-math.pi, math.pi)
        # goal as center of goal cell
        gx = float(self.goal_cell[0]) + 0.5
        gy = float(self.goal_cell[1]) + 0.5
        self.waypoints = [(float(r)+0.5, float(c)+0.5) for (r,c) in self.path_cells] if self.path_cells else [(gx,gy)]
        self.waypoint_idx = 0

    def current_goal_point(self):
        if self.waypoint_idx < len(self.waypoints):
            return self.waypoints[self.waypoint_idx]
        return None

    
    def step_control(self, dt=0.1):
        """
        Updated controller for safe turning:
        - Slows down during large heading errors
        - Limits speed based on angular velocity (turn radius)
    - Uses proportional controller for heading
        - Handles reaching waypoints
        """
        wp = self.current_goal_point()
        if wp is None:
            # at final goal
            self.v = max(0.0, self.v - self.max_accel*dt)  # slow to stop
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

        # Heading error in [-pi, pi]
        heading_error = (desired_theta - self.theta + math.pi) % (2*math.pi) - math.pi

        # Speed profile: reduce speed when turning or near waypoint
        v_cmd = min(self.max_speed, 0.8 * dist)
    
        # Slow down based on heading error
        turn_factor = max(0.1, 1 - abs(heading_error)/(math.pi/2))
        v_cmd *= turn_factor

        # Limit speed to maintain minimum turning radius
        min_turn_radius = max(self.length/2, 0.5)  # half length or minimum safety
        max_v_for_turn = min_turn_radius * self.max_omega
        if v_cmd > max_v_for_turn:
            v_cmd = max_v_for_turn

        # Accelerate/decelerate toward v_cmd
        if self.v < v_cmd:
            self.v = min(self.v + self.max_accel*dt, v_cmd)
        else:
            self.v = max(self.v - self.max_accel*dt, v_cmd)

        # Angular velocity proportional controller
        self.w = max(-self.max_omega, min(self.max_omega, 2.5 * heading_error))

        # Integrate unicycle dynamics
        self.theta += self.w * dt
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt

        # Check if reached waypoint (within 0.6 units)
        if dist < 0.6:
            self.waypoint_idx += 1
            if self.waypoint_idx >= len(self.waypoints):
                self.done = True


# ------------------------ Collision detection ------------------------------

def rectangle_corners(cx, cy, length, width, theta):
    # rectangle centered at (cx,cy), length along x (forward), width along y
    dx = length/2.0; dy = width/2.0
    corners_local = [ (dx, dy), (dx, -dy), (-dx, -dy), (-dx, dy) ]
    corners = []
    cos_t = math.cos(theta); sin_t = math.sin(theta)
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
    # Separating Axis Theorem for convex polygons (rectangles)
    # axes: normals of each polygon edge (for rectangles, 4 axes)
    axes = []
    for poly in (poly1, poly2):
        n = len(poly)
        for i in range(n):
            x1,y1 = poly[i]; x2,y2 = poly[(i+1)%n]
            edge = (x2-x1, y2-y1)
            # normal axis
            axis = (-edge[1], edge[0])
            # normalize
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

def robot_map_collision(robot: Robot, wmap: WarehouseMap):
    # check corners against rack cells: if any corner lies inside a rack cell, count as collision
    corners = rectangle_corners(robot.x, robot.y, robot.length, robot.width, robot.theta)
    for (cx,cy) in corners:
        # convert to grid cell indices (row, col) using floor
        r = int(math.floor(cx)); c = int(math.floor(cy))
        if not wmap.in_bounds((r,c)): return True
        if wmap.grid[r,c] == RACK: return True
    return False

# ------------------------ Simulator ---------------------------------------

@dataclass
class Metrics:
    successes: int = 0
    collisions: int = 0
    total_steps: int = 0
    runtimes: List[float] = field(default_factory=list)

class DynamicMultiRobotSim:
    def __init__(self, wmap: WarehouseMap, robots: List[Robot], planner='astar', prm_graph: Optional[nx.Graph]=None, seed=0):
        self.wmap = wmap
        self.robots = robots
        self.prm_graph = prm_graph
        self.planner = planner
        self.time = 0.0
        self.dt = 0.2
        self.metrics = Metrics()
        random.seed(seed)
        self.seed = seed

    def plan_all(self):
        for r in self.robots:
            if self.planner == 'astar':
                p = astar_grid(self.wmap, r.start_cell, r.goal_cell)
            elif self.planner == 'prm' and self.prm_graph is not None:
                p = prm_query(self.prm_graph, self.wmap, r.start_cell, r.goal_cell)
            elif self.planner == 'rrt':
                p = rrt_planner(self.wmap, r.start_cell, r.goal_cell, seed=r.id)
            else:
                p = astar_grid(self.wmap, r.start_cell, r.goal_cell)
            if p is None:
                r.path_cells = []
                r.waypoints = []
            else:
                r.path_cells = p
            r.initialize_from_cells()

    def step(self):
        # update each robot's controller and pose
        for r in self.robots:
            if r.done or r.collided:
                continue
            r.step_control(dt=self.dt)
        # collision checking between robots and with map
        for i, r in enumerate(self.robots):
            if r.done or r.collided: continue
            # map collision
            if robot_map_collision(r, self.wmap):
                r.collided = True
                self.metrics.collisions += 1
                continue
            # robot-robot collisions
            for j in range(i+1, len(self.robots)):
                o = self.robots[j]
                if o.done or o.collided: continue
                if robot_collision(r, o):
                    r.collided = True; o.collided = True
                    self.metrics.collisions += 1
        # count successes
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
            if visualize:
                frames.append(self.render_frame())
        runtime = time.time() - start_time
        self.metrics.runtimes.append(runtime)
        # save frames if requested
        if visualize and frames:
            os.makedirs(outdir, exist_ok=True)
            gif_path = os.path.join(outdir, f'dynamic_warehouse_{self.planner}_n{len(self.robots)}_s{self.seed}.gif')
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=int(1000*self.dt), loop=0)
            print('Saved GIF to', gif_path)
        return frames

    def render_frame(self, cellsize=8):
        rows, cols = self.wmap.rows, self.wmap.cols
        figw = cols * cellsize / 100.0; figh = rows * cellsize / 100.0
        # create RGB array
        img = np.ones((rows, cols, 3), dtype=np.uint8)*255
        for r in range(rows):
            for c in range(cols):
                if self.wmap.grid[r,c] == RACK:
                    img[r,c] = np.array([80,80,80], dtype=np.uint8)
        # draw robots footprint onto img using patches via matplotlib then convert to array
        fig = plt.figure(figsize=(figw, figh), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_xlim(0, cols); ax.set_ylim(0, rows)
        ax.set_xticks([]); ax.set_yticks([])
        # draw map as background
        for r in range(rows):
            for c in range(cols):
                if self.wmap.grid[r,c] == RACK:
                    rect = patches.Rectangle((r, c), 1, 1, facecolor=(0.3,0.3,0.3), edgecolor=None)
                    ax.add_patch(rect)
        # draw robot waypoints lightly
        for robot in self.robots:
            for wp in robot.waypoints:
                ax.plot(wp[0], wp[1], '.', color=(0.8,0.8,1.0), markersize=2)
        # draw robots as oriented rectangles
        cmap = plt.get_cmap('tab10')
        for robot in self.robots:
            cx, cy = robot.x, robot.y
            L, W = robot.length, robot.width
            angle = math.degrees(robot.theta)
            rect = patches.Rectangle((cx - L/2, cy - W/2), L, W, angle=angle, facecolor=cmap(robot.id%10), alpha=0.9)
            t = patches.Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData
            rect.set_transform(t)
            ax.add_patch(rect)
            # draw heading arrow
            hx = cx + (L/2) * math.cos(robot.theta)
            hy = cy + (L/2) * math.sin(robot.theta)
            ax.arrow(cx, cy, 0.4*math.cos(robot.theta), 0.4*math.sin(robot.theta), head_width=0.2, head_length=0.2, fc='k', ec='k')
            # draw goal marker
            gx = robot.goal_cell[0] + 0.5; gy = robot.goal_cell[1] + 0.5
            ax.plot(gx, gy, 'x', color='green')
        ax.set_xlim(0, cols); ax.set_ylim(0, rows)
        ax.set_aspect('equal')
        fig.canvas.draw()
        # convert to numpy array
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w,h = fig.canvas.get_width_height()
        data = data.reshape((h, w, 4))  # RGBA
        data = data[:,:,:3]  # drop alpha
        plt.close(fig)
        # downsample to rows*cellsize x cols*cellsize
        img_arr = np.array(Image.fromarray(data).resize((cols*cellsize, rows*cellsize)))
        return img_arr

# --------------------------- Utilities ------------------------------------

def random_free_cell(wmap: WarehouseMap, rng: random.Random):
    for _ in range(10000):
        r = rng.randrange(0, wmap.rows); c = rng.randrange(0, wmap.cols)
        if wmap.is_free((r,c)): return (r,c)
    for i in range(wmap.rows):
        for j in range(wmap.cols):
            if wmap.is_free((i,j)): return (i,j)
    raise RuntimeError('no free cell')

def make_demo(rows=40, cols=60, num_robots=4, seed=0):
    rng = random.Random(seed)
    wmap = WarehouseMap(rows, cols)
    wmap.add_racks(rack_rows=2, spacing=3, margin=2)
    robots = []
    for i in range(num_robots):
        s = random_free_cell(wmap, rng); g = random_free_cell(wmap, rng)
        count = 0
        while g == s and count < 50:
            g = random_free_cell(wmap, rng); count += 1
        r = Robot(id=i, start_cell=s, goal_cell=g)
        robots.append(r)
    return wmap, robots

# --------------------------- CLI ------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--planner', choices=['astar','prm','rrt'], default='astar')
    p.add_argument('--rows', type=int, default=40)
    p.add_argument('--cols', type=int, default=60)
    p.add_argument('--num-robots', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-steps', type=int, default=400)
    p.add_argument('--out', '--outdir', dest='outdir', default='outputs')
    return p.parse_args()

def main():
    args = parse_args()
    rng = random.Random(args.seed)
    wmap, robots = make_demo(rows=args.rows, cols=args.cols, num_robots=args.num_robots, seed=args.seed)
    prm = None
    if args.planner == 'prm':
        prm = build_prm(wmap, nsamples=800, k=8, seed=args.seed)
    sim = DynamicMultiRobotSim(wmap, robots, planner=args.planner, prm_graph=prm, seed=args.seed)
    sim.plan_all()
    frames = sim.run(max_steps=args.max_steps, visualize=True, outdir=args.outdir)
    # save final map figure
    os.makedirs(args.outdir, exist_ok=True)
    plt.figure(figsize=(6,6))
    ax = plt.gca()
    wmap.draw(ax)
    for r in robots:
        ax.plot(r.start_cell[0]+0.5, r.start_cell[1]+0.5, 'o', color='blue')
        ax.plot(r.goal_cell[0]+0.5, r.goal_cell[1]+0.5, 'x', color='green')
    plt.title(f'Dynamic warehouse final - planner={args.planner} robots={args.num_robots}')
    plt.savefig(os.path.join(args.outdir, 'dynamic_warehouse_final_map.png'), dpi=200, bbox_inches='tight')
    plt.close()
    # metrics
    print('Successes:', sim.metrics.successes, 'Collisions:', sim.metrics.collisions, 'Steps:', sim.metrics.total_steps)
    print('GIF/frames saved to', args.outdir)

if __name__ == '__main__':
    main()

