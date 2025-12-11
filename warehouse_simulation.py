"""
Warehouse Multi-Robot Motion Planning Simulation
Implements A*, PRM, RRT* with and without ORCA collision avoidance
Includes differential drive dynamics and multi-robot coordination
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import heapq
from collections import defaultdict
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import json


@dataclass
class PerformanceMetrics:
    """Track performance metrics for each planning method"""
    planner_name: str
    num_robots: int
    planning_time: float
    execution_time: float
    path_length: float
    num_collisions: int
    success_rate: float
    computational_cost: int  # Number of nodes explored
    smoothness: float  # Path smoothness metric


class WarehouseEnvironment:
    """Represents a warehouse environment with obstacles"""

    def __init__(self, width: float, height: float, grid_resolution: float = 0.5):
        self.width = width
        self.height = height
        self.grid_resolution = grid_resolution
        self.obstacles = []
        self.shelves = []

    def add_obstacle(self, x: float, y: float, width: float, height: float):
        """Add rectangular obstacle"""
        self.obstacles.append((x, y, width, height))

    def add_shelf_row(self, start_x: float, start_y: float, num_shelves: int,
                      shelf_width: float, shelf_height: float, spacing: float):
        """Add a row of warehouse shelves"""
        for i in range(num_shelves):
            x = start_x + i * (shelf_width + spacing)
            self.add_obstacle(x, start_y, shelf_width, shelf_height)
            self.shelves.append((x, start_y, shelf_width, shelf_height))

    def create_warehouse_layout(self, layout_type: str = "simple"):
        """Create predefined warehouse layouts"""
        if layout_type == "simple":
            # Simple warehouse with 2 rows of shelves
            self.add_shelf_row(5, 5, 4, 2, 4, 2)
            self.add_shelf_row(5, 15, 4, 2, 4, 2)

        elif layout_type == "complex":
            # Complex warehouse with multiple aisles
            self.add_shelf_row(5, 5, 6, 2, 3, 1.5)
            self.add_shelf_row(5, 12, 6, 2, 3, 1.5)
            self.add_shelf_row(5, 19, 6, 2, 3, 1.5)

        elif layout_type == "dense":
            # Dense warehouse for stress testing
            for row in range(5):
                self.add_shelf_row(3, 3 + row * 5, 8, 1.5, 2, 1)

    def is_collision(self, x: float, y: float, robot_radius: float = 0.3) -> bool:
        """Check if position collides with obstacles"""
        # Check boundaries
        if x < robot_radius or x > self.width - robot_radius:
            return True
        if y < robot_radius or y > self.height - robot_radius:
            return True

        # Check obstacles
        for ox, oy, ow, oh in self.obstacles:
            if (ox - robot_radius <= x <= ox + ow + robot_radius and
                oy - robot_radius <= y <= oy + oh + robot_radius):
                return True
        return False

    def is_line_collision_free(self, x1: float, y1: float, x2: float, y2: float,
                               robot_radius: float = 0.3, num_checks: int = 20) -> bool:
        """Check if line segment is collision-free"""
        for i in range(num_checks + 1):
            t = i / num_checks
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if self.is_collision(x, y, robot_radius):
                return False
        return True


class DifferentialDriveRobot:
    """Differential drive robot with dynamics"""

    def __init__(self, x: float, y: float, theta: float, robot_id: int):
        self.x = x
        self.y = y
        self.theta = theta  # Orientation in radians
        self.robot_id = robot_id

        # Physical parameters
        self.wheel_base = 0.5  # Distance between wheels (m)
        self.wheel_radius = 0.1  # Wheel radius (m)
        self.max_wheel_velocity = 2.0  # rad/s
        self.max_linear_velocity = 1.0  # m/s
        self.max_angular_velocity = 1.0  # rad/s
        self.radius = 0.3  # Robot radius for collision detection

        # Current velocities
        self.v = 0.0  # Linear velocity
        self.omega = 0.0  # Angular velocity

        # Path and control
        self.path = []
        self.current_path_index = 0
        self.reached_goal = False

        # History for visualization
        self.trajectory = [(x, y)]

    def set_velocities(self, v: float, omega: float):
        """Set linear and angular velocities with limits"""
        self.v = np.clip(v, -self.max_linear_velocity, self.max_linear_velocity)
        self.omega = np.clip(omega, -self.max_angular_velocity, self.max_angular_velocity)

    def update(self, dt: float):
        """Update robot state using differential drive kinematics"""
        # Differential drive kinematic model
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.omega * dt

        # Normalize theta to [-pi, pi]
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        # Record trajectory
        self.trajectory.append((self.x, self.y))

    def get_position(self) -> Tuple[float, float]:
        """Get current position"""
        return (self.x, self.y)

    def get_state(self) -> Tuple[float, float, float]:
        """Get full state (x, y, theta)"""
        return (self.x, self.y, self.theta)

    def follow_path(self, dt: float, lookahead_distance: float = 0.5):
        """Simple path following controller"""
        if not self.path or self.current_path_index >= len(self.path):
            self.set_velocities(0, 0)
            self.reached_goal = True
            return

        # Get target point
        target_x, target_y = self.path[self.current_path_index]

        # Calculate distance to target
        dx = target_x - self.x
        dy = target_y - self.y
        distance = np.sqrt(dx**2 + dy**2)

        # Move to next waypoint if close enough
        if distance < 0.3:
            self.current_path_index += 1
            if self.current_path_index >= len(self.path):
                self.set_velocities(0, 0)
                self.reached_goal = True
                return
            target_x, target_y = self.path[self.current_path_index]
            dx = target_x - self.x
            dy = target_y - self.y
            distance = np.sqrt(dx**2 + dy**2)

        # Calculate desired heading
        desired_theta = np.arctan2(dy, dx)

        # Calculate heading error
        theta_error = np.arctan2(np.sin(desired_theta - self.theta),
                                 np.cos(desired_theta - self.theta))

        # Simple proportional controller
        k_linear = 0.5
        k_angular = 2.0

        v = k_linear * distance
        omega = k_angular * theta_error

        self.set_velocities(v, omega)


class AStarPlanner:
    """A* path planning algorithm"""

    def __init__(self, environment: WarehouseEnvironment):
        self.env = environment
        self.grid_resolution = environment.grid_resolution

    def heuristic(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Euclidean distance heuristic"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def get_neighbors(self, pos: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Get valid neighboring positions (8-connected grid)"""
        x, y = pos
        neighbors = []

        # 8-connected grid
        for dx in [-self.grid_resolution, 0, self.grid_resolution]:
            for dy in [-self.grid_resolution, 0, self.grid_resolution]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if not self.env.is_collision(nx, ny):
                    neighbors.append((nx, ny))

        return neighbors

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Tuple[List[Tuple[float, float]], int]:
        """
        Plan path from start to goal using A*
        Returns: (path, nodes_explored)
        """
        # Round to grid
        start = (round(start[0] / self.grid_resolution) * self.grid_resolution,
                round(start[1] / self.grid_resolution) * self.grid_resolution)
        goal = (round(goal[0] / self.grid_resolution) * self.grid_resolution,
               round(goal[1] / self.grid_resolution) * self.grid_resolution)

        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        nodes_explored = 0

        while open_set:
            current = heapq.heappop(open_set)[1]
            nodes_explored += 1

            if self.heuristic(current, goal) < self.grid_resolution:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path, nodes_explored

            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + self.heuristic(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return [], nodes_explored  # No path found


class PRMPlanner:
    """Probabilistic Roadmap (PRM) path planning algorithm"""

    def __init__(self, environment: WarehouseEnvironment, num_samples: int = 800,
                 connection_radius: float = 3.5):
        self.env = environment
        self.num_samples = num_samples
        self.connection_radius = connection_radius
        self.roadmap = defaultdict(list)
        self.samples = []

    def build_roadmap(self):
        """Build PRM roadmap by sampling and connecting"""
        self.samples = []
        self.roadmap = defaultdict(list)

        # Random sampling
        attempts = 0
        max_attempts = self.num_samples * 10

        while len(self.samples) < self.num_samples and attempts < max_attempts:
            x = np.random.uniform(0, self.env.width)
            y = np.random.uniform(0, self.env.height)

            if not self.env.is_collision(x, y):
                self.samples.append((x, y))
            attempts += 1

        # Connect nearby samples
        for i, sample1 in enumerate(self.samples):
            for j, sample2 in enumerate(self.samples):
                if i >= j:
                    continue

                dist = np.sqrt((sample1[0] - sample2[0])**2 + (sample1[1] - sample2[1])**2)

                if dist <= self.connection_radius:
                    if self.env.is_line_collision_free(sample1[0], sample1[1],
                                                      sample2[0], sample2[1]):
                        self.roadmap[i].append(j)
                        self.roadmap[j].append(i)

    def find_nearest_sample(self, pos: Tuple[float, float]) -> int:
        """Find nearest roadmap sample to position"""
        min_dist = float('inf')
        nearest_idx = -1

        for i, sample in enumerate(self.samples):
            dist = np.sqrt((sample[0] - pos[0])**2 + (sample[1] - pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        return nearest_idx

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Tuple[List[Tuple[float, float]], int]:
        """
        Plan path using PRM + A* on roadmap
        Returns: (path, nodes_explored)
        """
        # Build roadmap if not already built
        if not self.samples:
            self.build_roadmap()

        # Add start and goal to roadmap temporarily
        start_idx = len(self.samples)
        goal_idx = len(self.samples) + 1

        temp_samples = self.samples + [start, goal]
        temp_roadmap = dict(self.roadmap)
        temp_roadmap[start_idx] = []
        temp_roadmap[goal_idx] = []

        # Connect start and goal to nearby samples
        for i, sample in enumerate(self.samples):
            # Ensure node exists in temp_roadmap
            if i not in temp_roadmap:
                temp_roadmap[i] = []

            # Connect start
            dist = np.sqrt((sample[0] - start[0])**2 + (sample[1] - start[1])**2)
            if dist <= self.connection_radius:
                if self.env.is_line_collision_free(start[0], start[1], sample[0], sample[1]):
                    temp_roadmap[start_idx].append(i)
                    temp_roadmap[i].append(start_idx)

            # Connect goal
            dist = np.sqrt((sample[0] - goal[0])**2 + (sample[1] - goal[1])**2)
            if dist <= self.connection_radius:
                if self.env.is_line_collision_free(goal[0], goal[1], sample[0], sample[1]):
                    temp_roadmap[goal_idx].append(i)
                    temp_roadmap[i].append(goal_idx)

        # A* on roadmap
        open_set = []
        heapq.heappush(open_set, (0, start_idx))

        came_from = {}
        g_score = {start_idx: 0}

        def heuristic(idx1, idx2):
            p1 = temp_samples[idx1]
            p2 = temp_samples[idx2]
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        f_score = {start_idx: heuristic(start_idx, goal_idx)}
        nodes_explored = 0

        while open_set:
            current = heapq.heappop(open_set)[1]
            nodes_explored += 1

            if current == goal_idx:
                # Reconstruct path
                path_indices = [current]
                while current in came_from:
                    current = came_from[current]
                    path_indices.append(current)
                path_indices.reverse()

                path = [temp_samples[i] for i in path_indices]
                return path, nodes_explored

            for neighbor in temp_roadmap[current]:
                tentative_g = g_score[current] + heuristic(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal_idx)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return [], nodes_explored


class RRTStarPlanner:
    """RRT* path planning algorithm"""

    def __init__(self, environment: WarehouseEnvironment, max_iterations: int = 2000,
                 step_size: float = 0.5, search_radius: float = 1.5):
        self.env = environment
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.search_radius = search_radius

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Tuple[List[Tuple[float, float]], int]:
        """
        Plan path using RRT*
        Returns: (path, nodes_explored)
        """
        # Tree structure: node_id -> (x, y, parent_id, cost)
        tree = {0: (start[0], start[1], None, 0)}

        for iteration in range(self.max_iterations):
            # Sample random point (with goal bias)
            if np.random.random() < 0.1:
                rand_point = goal
            else:
                rand_point = (np.random.uniform(0, self.env.width),
                            np.random.uniform(0, self.env.height))

            # Find nearest node
            nearest_id = min(tree.keys(),
                           key=lambda k: np.sqrt((tree[k][0] - rand_point[0])**2 +
                                               (tree[k][1] - rand_point[1])**2))
            nearest = tree[nearest_id]

            # Steer towards random point
            dist = np.sqrt((rand_point[0] - nearest[0])**2 + (rand_point[1] - nearest[1])**2)
            if dist < 1e-6:
                continue

            theta = np.arctan2(rand_point[1] - nearest[1], rand_point[0] - nearest[0])
            new_x = nearest[0] + min(self.step_size, dist) * np.cos(theta)
            new_y = nearest[1] + min(self.step_size, dist) * np.sin(theta)

            # Check collision
            if self.env.is_collision(new_x, new_y):
                continue
            if not self.env.is_line_collision_free(nearest[0], nearest[1], new_x, new_y):
                continue

            # Find near nodes for rewiring
            near_ids = []
            for node_id, node in tree.items():
                dist = np.sqrt((node[0] - new_x)**2 + (node[1] - new_y)**2)
                if dist < self.search_radius:
                    near_ids.append(node_id)

            # Choose best parent
            best_parent = nearest_id
            best_cost = nearest[3] + np.sqrt((new_x - nearest[0])**2 + (new_y - nearest[1])**2)

            for near_id in near_ids:
                near_node = tree[near_id]
                cost = near_node[3] + np.sqrt((new_x - near_node[0])**2 + (new_y - near_node[1])**2)

                if cost < best_cost:
                    if self.env.is_line_collision_free(near_node[0], near_node[1], new_x, new_y):
                        best_parent = near_id
                        best_cost = cost

            # Add new node
            new_id = len(tree)
            tree[new_id] = (new_x, new_y, best_parent, best_cost)

            # Rewire tree
            for near_id in near_ids:
                near_node = tree[near_id]
                new_cost = best_cost + np.sqrt((new_x - near_node[0])**2 + (new_y - near_node[1])**2)

                if new_cost < near_node[3]:
                    if self.env.is_line_collision_free(new_x, new_y, near_node[0], near_node[1]):
                        tree[near_id] = (near_node[0], near_node[1], new_id, new_cost)

            # Check if goal reached
            if np.sqrt((new_x - goal[0])**2 + (new_y - goal[1])**2) < self.step_size:
                # Connect to goal
                goal_id = len(tree)
                goal_cost = best_cost + np.sqrt((new_x - goal[0])**2 + (new_y - goal[1])**2)
                tree[goal_id] = (goal[0], goal[1], new_id, goal_cost)

                # Reconstruct path
                path = []
                current_id = goal_id
                while current_id is not None:
                    node = tree[current_id]
                    path.append((node[0], node[1]))
                    current_id = node[2]
                path.reverse()

                return path, len(tree)

        return [], len(tree)


class ORCACollisionAvoidance:
    """Optimal Reciprocal Collision Avoidance (ORCA) implementation"""

    def __init__(self, time_horizon: float = 2.0, max_speed: float = 1.0):
        self.time_horizon = time_horizon
        self.max_speed = max_speed

    def compute_velocity(self, robot: DifferentialDriveRobot,
                        other_robots: List[DifferentialDriveRobot],
                        preferred_velocity: Tuple[float, float]) -> Tuple[float, float]:
        """
        Compute ORCA-compliant velocity for a robot
        Returns: (vx, vy) velocity components
        """
        # Start with preferred velocity
        vx_pref, vy_pref = preferred_velocity

        # ORCA constraints (half-planes)
        orca_lines = []

        for other in other_robots:
            if other.robot_id == robot.robot_id:
                continue

            # Relative position and velocity
            rel_pos = np.array([other.x - robot.x, other.y - robot.y])
            rel_vel = np.array([other.v * np.cos(other.theta) - robot.v * np.cos(robot.theta),
                               other.v * np.sin(other.theta) - robot.v * np.sin(robot.theta)])

            dist = np.linalg.norm(rel_pos)
            combined_radius = robot.radius + other.radius

            if dist < 1e-6:
                continue

            # ORCA half-plane
            if dist > combined_radius:
                # No collision
                w = rel_vel - rel_pos / self.time_horizon

                # Compute normal vector
                leg = np.sqrt(dist**2 - combined_radius**2)
                if dist * dist > leg * leg:
                    # Normal pointing away from collision
                    direction = np.array([-rel_pos[1], rel_pos[0]])
                    if np.dot(direction, w) > 0:
                        direction = -direction

                    direction = direction / np.linalg.norm(direction)
                    u = direction * (np.dot(w, direction)) - w
                else:
                    # Already penetrating, push away
                    u = -rel_vel

                orca_lines.append((u / 2, rel_pos / np.linalg.norm(rel_pos)))

        # Find velocity closest to preferred that satisfies ORCA constraints
        # Simplified: project preferred velocity onto safe region
        vx, vy = vx_pref, vy_pref

        # Simple projection approach
        for u, n in orca_lines:
            # Check if current velocity violates constraint
            v_current = np.array([vx, vy])
            if np.dot(v_current - u, n) < 0:
                # Project onto constraint
                v_current = v_current - (np.dot(v_current - u, n)) * n
                vx, vy = v_current[0], v_current[1]

        # Limit to max speed
        speed = np.sqrt(vx**2 + vy**2)
        if speed > self.max_speed:
            vx = vx / speed * self.max_speed
            vy = vy / speed * self.max_speed

        return vx, vy


class MultiRobotSimulation:
    """Main simulation class for multi-robot warehouse system"""

    def __init__(self, environment: WarehouseEnvironment):
        self.env = environment
        self.robots = []
        self.dt = 0.1  # Time step
        self.current_time = 0.0
        self.metrics_history = []

    def add_robot(self, robot: DifferentialDriveRobot):
        """Add robot to simulation"""
        self.robots.append(robot)

    def assign_path(self, robot_id: int, path: List[Tuple[float, float]]):
        """Assign path to robot"""
        for robot in self.robots:
            if robot.robot_id == robot_id:
                robot.path = path
                robot.current_path_index = 0
                robot.reached_goal = False
                break

    def step(self, use_orca: bool = False):
        """Execute one simulation step"""
        if use_orca:
            orca = ORCACollisionAvoidance()

            for robot in self.robots:
                if robot.reached_goal:
                    continue

                # Get preferred velocity from path following
                if robot.path and robot.current_path_index < len(robot.path):
                    target = robot.path[robot.current_path_index]
                    dx = target[0] - robot.x
                    dy = target[1] - robot.y
                    dist = np.sqrt(dx**2 + dy**2)

                    if dist > 0.01:
                        pref_vx = dx / dist * 0.5
                        pref_vy = dy / dist * 0.5
                    else:
                        pref_vx, pref_vy = 0, 0

                    # Compute ORCA velocity
                    vx, vy = orca.compute_velocity(robot, self.robots, (pref_vx, pref_vy))

                    # Convert to differential drive commands
                    speed = np.sqrt(vx**2 + vy**2)
                    if speed > 0.01:
                        desired_theta = np.arctan2(vy, vx)
                        theta_error = np.arctan2(np.sin(desired_theta - robot.theta),
                                                np.cos(desired_theta - robot.theta))

                        robot.set_velocities(speed * 0.8, theta_error * 2.0)
                    else:
                        robot.set_velocities(0, 0)
        else:
            # Normal path following without ORCA
            for robot in self.robots:
                if not robot.reached_goal:
                    robot.follow_path(self.dt)

        # Update all robots
        for robot in self.robots:
            robot.update(self.dt)

        self.current_time += self.dt

    def check_collisions(self) -> int:
        """Check for inter-robot collisions"""
        collision_count = 0
        for i, robot1 in enumerate(self.robots):
            for robot2 in self.robots[i+1:]:
                dist = np.sqrt((robot1.x - robot2.x)**2 + (robot1.y - robot2.y)**2)
                if dist < (robot1.radius + robot2.radius):
                    collision_count += 1
        return collision_count

    def all_robots_reached_goal(self) -> bool:
        """Check if all robots reached their goals"""
        return all(robot.reached_goal for robot in self.robots)

    def reset(self):
        """Reset simulation"""
        self.current_time = 0.0
        for robot in self.robots:
            robot.trajectory = [(robot.x, robot.y)]
            robot.current_path_index = 0
            robot.reached_goal = False


def calculate_path_metrics(path: List[Tuple[float, float]]) -> dict:
    """Calculate metrics for a path"""
    if len(path) < 2:
        return {'length': 0, 'smoothness': 0}

    # Path length
    length = 0
    for i in range(len(path) - 1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        length += np.sqrt(dx**2 + dy**2)

    # Smoothness (average turning angle)
    smoothness = 0
    if len(path) > 2:
        angles = []
        for i in range(1, len(path) - 1):
            v1 = np.array([path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]])
            v2 = np.array([path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]])

            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)

        if angles:
            smoothness = np.mean(angles)

    return {'length': length, 'smoothness': smoothness}


def run_benchmark(planner_name: str, num_robots: int, environment: WarehouseEnvironment,
                 use_orca: bool = False) -> PerformanceMetrics:
    """Run a benchmark scenario"""
    print(f"\nRunning {planner_name} with {num_robots} robots (ORCA: {use_orca})...")

    # Create simulation
    sim = MultiRobotSimulation(environment)

    # Create robots with start/goal positions
    np.random.seed(42 + num_robots)  # Reproducible
    robots = []
    paths = []
    total_nodes_explored = 0
    total_planning_time = 0

    for i in range(num_robots):
        # Random start and goal
        while True:
            start_x = np.random.uniform(2, environment.width - 2)
            start_y = np.random.uniform(2, environment.height - 2)
            if not environment.is_collision(start_x, start_y):
                break

        while True:
            goal_x = np.random.uniform(2, environment.width - 2)
            goal_y = np.random.uniform(2, environment.height - 2)
            dist = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
            if not environment.is_collision(goal_x, goal_y) and dist > 5:
                break

        robot = DifferentialDriveRobot(start_x, start_y, 0, i)
        robots.append(robot)
        sim.add_robot(robot)

        # Plan path
        start_time = time.time()

        if 'A*' in planner_name or 'A-Star' in planner_name:
            planner = AStarPlanner(environment)
            path, nodes = planner.plan((start_x, start_y), (goal_x, goal_y))
        elif 'PRM' in planner_name:
            planner = PRMPlanner(environment, num_samples=800, connection_radius=3.5)
            path, nodes = planner.plan((start_x, start_y), (goal_x, goal_y))
        elif 'RRT' in planner_name:
            planner = RRTStarPlanner(environment, max_iterations=1500)
            path, nodes = planner.plan((start_x, start_y), (goal_x, goal_y))

        planning_time = time.time() - start_time
        total_planning_time += planning_time
        total_nodes_explored += nodes

        if path:
            sim.assign_path(i, path)
            paths.append(path)
        else:
            print(f"  Warning: No path found for robot {i}")
            paths.append([])

    # Run simulation
    max_steps = 5000
    step = 0
    start_exec_time = time.time()

    while step < max_steps and not sim.all_robots_reached_goal():
        sim.step(use_orca=use_orca)
        step += 1

    execution_time = time.time() - start_exec_time

    # Calculate metrics
    total_path_length = 0
    total_smoothness = 0
    success_count = 0

    for robot, path in zip(robots, paths):
        if robot.reached_goal:
            success_count += 1
        if path:
            metrics = calculate_path_metrics(path)
            total_path_length += metrics['length']
            total_smoothness += metrics['smoothness']

    num_collisions = sim.check_collisions()
    success_rate = success_count / num_robots if num_robots > 0 else 0
    avg_path_length = total_path_length / num_robots if num_robots > 0 else 0
    avg_smoothness = total_smoothness / num_robots if num_robots > 0 else 0

    metrics = PerformanceMetrics(
        planner_name=planner_name,
        num_robots=num_robots,
        planning_time=total_planning_time,
        execution_time=execution_time,
        path_length=avg_path_length,
        num_collisions=num_collisions,
        success_rate=success_rate,
        computational_cost=total_nodes_explored,
        smoothness=avg_smoothness
    )

    print(f"  Planning time: {total_planning_time:.3f}s")
    print(f"  Execution time: {execution_time:.3f}s")
    print(f"  Success rate: {success_rate*100:.1f}%")
    print(f"  Collisions: {num_collisions}")
    print(f"  Nodes explored: {total_nodes_explored}")

    return metrics


if __name__ == "__main__":
    print("Warehouse Multi-Robot Motion Planning Simulation")
    print("=" * 60)

    # Create warehouse environment
    env = WarehouseEnvironment(width=30, height=30, grid_resolution=0.5)
    env.create_warehouse_layout("complex")

    print(f"Environment: {env.width}x{env.height}m with {len(env.obstacles)} obstacles")

    # Run benchmarks
    all_metrics = []

    planners = [
        ("A*", False),
        ("PRM", False),
        ("RRT*", False),
        ("A* + ORCA", True),
        ("PRM + ORCA", True),
        ("RRT* + ORCA", True)
    ]

    robot_counts = [1, 2, 4, 8]

    for num_robots in robot_counts:
        print(f"\n{'='*60}")
        print(f"Testing with {num_robots} robots")
        print(f"{'='*60}")

        for planner_name, use_orca in planners:
            metrics = run_benchmark(planner_name, num_robots, env, use_orca)
            all_metrics.append(metrics)

    # Save results
    results = {
        'metrics': [
            {
                'planner': m.planner_name,
                'num_robots': m.num_robots,
                'planning_time': m.planning_time,
                'execution_time': m.execution_time,
                'path_length': m.path_length,
                'collisions': m.num_collisions,
                'success_rate': m.success_rate,
                'nodes_explored': m.computational_cost,
                'smoothness': m.smoothness
            }
            for m in all_metrics
        ]
    }

    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Results saved to benchmark_results.json")
    print(f"{'='*60}")
