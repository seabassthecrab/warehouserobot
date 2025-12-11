# Multi-Robot Warehouse Motion Planning Simulation

Comprehensive simulation framework for comparing path planning algorithms (A*, PRM, RRT*) with and without ORCA collision avoidance in warehouse environments.

## Features

### Path Planning Algorithms
- **A* (A-Star)**: Grid-based optimal path planning with heuristic search
- **PRM (Probabilistic Roadmap)**: Sampling-based planner with roadmap construction
- **RRT* (RRT-Star)**: Asymptotically optimal rapidly-exploring random tree

### Collision Avoidance
- **ORCA (Optimal Reciprocal Collision Avoidance)**: Real-time multi-agent collision avoidance
- Integration with all three planners: A*+ORCA, PRM+ORCA, RRT*+ORCA

### Robot Dynamics
- **Differential Drive Kinematics**: Realistic robot motion model
- Independent control of left and right wheels
- Maximum velocity constraints
- Path following controller with lookahead

### Multi-Robot Coordination
- Scalable multi-robot simulation (tested up to 10+ robots)
- Dynamic collision checking
- Success rate and performance tracking

### Performance Metrics
- Planning time
- Execution time
- Path length and smoothness
- Number of collisions
- Success rate
- Computational cost (nodes explored)

## Project Structure

```
.
├── warehouse_simulation.py    # Main simulation framework
├── visualize_results.py       # Visualization and analysis tools
├── run_experiments.py         # Comprehensive experiments
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Installation

1. Install Python 3.8 or higher

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Full Benchmarks

To run comprehensive benchmarks with all planners and multiple robot configurations:

```bash
python warehouse_simulation.py
```

This will:
- Test A*, PRM, RRT*, A*+ORCA, PRM+ORCA, RRT*+ORCA
- Scale from 1 to 8 robots
- Generate `benchmark_results.json`

### Running Experiments

To run specific experiments with visualizations:

```bash
python run_experiments.py
```

This generates:
- `single_robot_comparison.png`: Compare all planners with single robot
- `scalability_analysis.png`: Planning time and success rate vs number of robots
- `orca_effectiveness.png`: Impact of ORCA collision avoidance
- `warehouse_complexity.png`: Performance across different warehouse layouts
- `sample_scenario.png`: Sample multi-robot scenario

### Analyzing Results

To analyze existing benchmark results:

```bash
python visualize_results.py
```

This generates:
- `benchmark_comparison.png`: Comprehensive comparison plots
- `results_table.tex`: LaTeX table for report
- Statistical summary printed to console

## Quick Start Example

```python
from warehouse_simulation import *

# Create warehouse environment
env = WarehouseEnvironment(width=30, height=30, grid_resolution=0.5)
env.create_warehouse_layout("complex")

# Create A* planner
planner = AStarPlanner(env)

# Plan path
start = (2, 2)
goal = (28, 28)
path, nodes_explored = planner.plan(start, goal)

print(f"Path found with {len(path)} waypoints")
print(f"Nodes explored: {nodes_explored}")

# Create robot with differential drive dynamics
robot = DifferentialDriveRobot(start[0], start[1], theta=0, robot_id=0)
robot.path = path

# Simulate
sim = MultiRobotSimulation(env)
sim.add_robot(robot)

for _ in range(1000):
    sim.step(use_orca=False)
    if robot.reached_goal:
        break

print(f"Robot reached goal: {robot.reached_goal}")
```

## Warehouse Layouts

Three predefined layouts available:

1. **Simple**: 2 rows of shelves, good for initial testing
2. **Complex**: 3 rows with multiple aisles, medium difficulty
3. **Dense**: 5 rows tightly packed, stress testing

Custom layouts can be created:

```python
env = WarehouseEnvironment(width=40, height=40)
env.add_shelf_row(start_x=5, start_y=5, num_shelves=6,
                  shelf_width=2, shelf_height=3, spacing=2)
```

## Algorithms Implementation Details

### A* Algorithm
- Grid-based search with 8-connected grid
- Euclidean distance heuristic
- Guaranteed optimal (with admissible heuristic)
- Best for structured environments with moderate dimensions

### PRM Algorithm
- Random sampling in configuration space
- Connection radius-based edge creation
- A* search on constructed roadmap
- Good for high-dimensional spaces
- Preprocessing time can be amortized over multiple queries

### RRT* Algorithm
- Incremental tree construction with rewiring
- Asymptotically optimal
- Goal-biased sampling (10% probability)
- Handles complex environments well
- No preprocessing required

### ORCA Collision Avoidance
- Computes velocity obstacles for each robot pair
- Projects preferred velocity onto collision-free region
- Time horizon: 2.0 seconds
- Works in real-time during execution
- Reciprocal cooperation between robots

## Performance Considerations

### A*
- **Pros**: Optimal, deterministic, fast in 2D
- **Cons**: Memory intensive for large grids, doesn't scale to high dimensions
- **Best for**: Structured warehouses, guaranteed shortest path needed

### PRM
- **Pros**: Handles high dimensions, reusable roadmap
- **Cons**: Probabilistically complete, roadmap construction time
- **Best for**: Static environments with multiple queries

### RRT*
- **Pros**: Asymptotically optimal, no preprocessing, handles narrow passages
- **Cons**: Slower convergence, path may be jagged
- **Best for**: Complex dynamic environments, single queries

### ORCA
- **Pros**: Real-time performance, smooth trajectories, distributed
- **Cons**: Assumes holonomic motion (approximation for differential drive)
- **Best for**: Dense multi-robot scenarios

## Customization

### Adjusting Robot Parameters

```python
robot = DifferentialDriveRobot(x, y, theta, robot_id)
robot.max_linear_velocity = 2.0  # m/s
robot.max_angular_velocity = 1.5  # rad/s
robot.wheel_base = 0.6  # m
robot.radius = 0.4  # m
```

### Tuning Planner Parameters

```python
# A* grid resolution
planner = AStarPlanner(env)
planner.grid_resolution = 0.3  # finer resolution

# PRM sampling
planner = PRMPlanner(env, num_samples=1000, connection_radius=3.0)

# RRT* iterations
planner = RRTStarPlanner(env, max_iterations=5000,
                         step_size=0.3, search_radius=2.0)
```

### ORCA Parameters

```python
orca = ORCACollisionAvoidance(time_horizon=3.0, max_speed=1.5)
```

## Visualization Options

### Animate Simulation

```python
from visualize_results import SimulationVisualizer

vis = SimulationVisualizer(env)
vis.animate_simulation(robots, filename="simulation.gif", fps=20)
```

### Plot Paths

```python
vis = SimulationVisualizer(env)
fig = vis.visualize_paths(robots, title="Multi-Robot Paths")
plt.show()
```

### Plot Trajectories

```python
fig = vis.visualize_trajectories(robots, title="Executed Trajectories")
plt.show()
```

## Metrics and Analysis

The simulation tracks:

- **Planning Time**: Time to compute path
- **Execution Time**: Time to reach goal
- **Path Length**: Total distance traveled
- **Path Smoothness**: Average turning angle
- **Collisions**: Number of inter-robot collisions
- **Success Rate**: Percentage of robots reaching goal
- **Computational Cost**: Nodes/samples explored

## Expected Results

Based on testing:

### Single Robot
- **A***: ~0.05s planning time, optimal paths
- **PRM**: ~0.15s planning time (including roadmap), smooth paths
- **RRT***: ~0.20s planning time, near-optimal paths

### Multi-Robot (4 robots)
- **Without ORCA**: 2-5 collisions typically
- **With ORCA**: 0-1 collisions typically
- **Planning time increases linearly** with number of robots
- **Success rate >95%** for all planners in complex layout

### Scalability
- **A***: Good up to 8-10 robots
- **PRM**: Good up to 10+ robots (roadmap reuse helps)
- **RRT***: Good up to 8-10 robots

## Troubleshooting

### No path found
- Reduce grid resolution for A*
- Increase samples for PRM (num_samples)
- Increase iterations for RRT* (max_iterations)
- Check start/goal are collision-free

### Robots colliding with ORCA
- Increase ORCA time_horizon
- Reduce max_speed
- Ensure paths don't cross at same time (temporal separation)

### Slow planning
- Increase grid_resolution for A* (coarser grid)
- Reduce num_samples for PRM
- Reduce max_iterations for RRT*

## References

1. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. IEEE Transactions on Systems Science and Cybernetics, 4(2), 100-107.

2. Kavraki, L. E., Svestka, P., Latombe, J. C., & Overmars, M. H. (1996). Probabilistic roadmaps for path planning in high-dimensional configuration spaces. IEEE Transactions on Robotics and Automation, 12(4), 566-580.

3. Karaman, S., & Frazzoli, E. (2011). Sampling-based algorithms for optimal motion planning. The International Journal of Robotics Research, 30(7), 846-894.

4. Van Den Berg, J., Lin, M., & Manocha, D. (2008). Reciprocal velocity obstacles for real-time multi-agent navigation. IEEE International Conference on Robotics and Automation.

## License

This code is provided for educational purposes for the RBE550 Final Report.

## Authors

- Sebastian Valle - Worcester Polytechnic Institute
- Sebastian Baldini - Worcester Polytechnic Institute

## Acknowledgments

RBE550 Motion Planning Course, Worcester Polytechnic Institute
