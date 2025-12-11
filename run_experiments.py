"""
Comprehensive test scenarios and experiments for the report
"""

import numpy as np
import matplotlib.pyplot as plt
from warehouse_simulation import *
from visualize_results import *
import time


def experiment_single_robot_comparison():
    """Compare all planners with a single robot"""
    print("\n" + "="*80)
    print("EXPERIMENT 1: Single Robot Comparison")
    print("="*80)

    env = WarehouseEnvironment(width=30, height=30, grid_resolution=0.5)
    env.create_warehouse_layout("complex")

    start = (2, 2)
    goal = (28, 28)

    results = {}

    # Test each planner
    planners_to_test = [
        ("A*", AStarPlanner),
        ("PRM", PRMPlanner),
        ("RRT*", RRTStarPlanner)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Single Robot Path Planning Comparison', fontsize=16, fontweight='bold')

    for idx, (name, PlannerClass) in enumerate(planners_to_test):
        print(f"\nTesting {name}...")

        if name == "A*":
            planner = PlannerClass(env)
        elif name == "PRM":
            planner = PlannerClass(env, num_samples=800, connection_radius=3.5)
        else:
            planner = PlannerClass(env, max_iterations=2000)

        start_time = time.time()
        path, nodes = planner.plan(start, goal)
        planning_time = time.time() - start_time

        metrics = calculate_path_metrics(path)

        results[name] = {
            'path': path,
            'planning_time': planning_time,
            'nodes_explored': nodes,
            'path_length': metrics['length'],
            'smoothness': metrics['smoothness']
        }

        print(f"  Planning time: {planning_time:.4f}s")
        print(f"  Nodes explored: {nodes}")
        print(f"  Path length: {metrics['length']:.2f}m")
        print(f"  Smoothness: {metrics['smoothness']:.4f}")

        # Visualize
        ax = axes[idx]
        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{name}\nTime: {planning_time:.3f}s, Length: {metrics["length"]:.1f}m')

        # Draw obstacles
        for ox, oy, ow, oh in env.obstacles:
            rect = patches.Rectangle((ox, oy), ow, oh,
                                     linewidth=1, edgecolor='black',
                                     facecolor='gray', alpha=0.7)
            ax.add_patch(rect)

        # Draw path
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path')
            ax.plot(path_x, path_y, 'co', markersize=3)

        # Draw start and goal
        ax.plot(start[0], start[1], 'go', markersize=12, label='Start')
        ax.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')
        ax.legend()

    plt.tight_layout()
    plt.savefig('single_robot_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved: single_robot_comparison.png")

    return results


def experiment_scalability_test():
    """Test scalability with increasing number of robots"""
    print("\n" + "="*80)
    print("EXPERIMENT 2: Scalability Test")
    print("="*80)

    env = WarehouseEnvironment(width=35, height=35, grid_resolution=0.5)
    env.create_warehouse_layout("complex")

    robot_counts = [1, 2, 4, 6, 8, 10]
    results = {
        'A*': {'planning_times': [], 'success_rates': []},
        'PRM': {'planning_times': [], 'success_rates': []},
        'RRT*': {'planning_times': [], 'success_rates': []}
    }

    for num_robots in robot_counts:
        print(f"\n--- Testing with {num_robots} robots ---")

        for planner_name in ['A*', 'PRM', 'RRT*']:
            print(f"  {planner_name}...", end=' ')

            np.random.seed(42 + num_robots)
            total_planning_time = 0
            success_count = 0

            for i in range(num_robots):
                # Generate random start/goal
                while True:
                    start_x = np.random.uniform(2, env.width - 2)
                    start_y = np.random.uniform(2, env.height - 2)
                    if not env.is_collision(start_x, start_y):
                        break

                while True:
                    goal_x = np.random.uniform(2, env.width - 2)
                    goal_y = np.random.uniform(2, env.height - 2)
                    dist = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
                    if not env.is_collision(goal_x, goal_y) and dist > 8:
                        break

                # Plan path
                start_time = time.time()

                if planner_name == 'A*':
                    planner = AStarPlanner(env)
                elif planner_name == 'PRM':
                    planner = PRMPlanner(env, num_samples=800, connection_radius=3.5)
                else:
                    planner = RRTStarPlanner(env, max_iterations=1500)

                path, _ = planner.plan((start_x, start_y), (goal_x, goal_y))
                planning_time = time.time() - start_time

                total_planning_time += planning_time
                if path:
                    success_count += 1

            avg_planning_time = total_planning_time / num_robots
            success_rate = success_count / num_robots

            results[planner_name]['planning_times'].append(avg_planning_time)
            results[planner_name]['success_rates'].append(success_rate * 100)

            print(f"Avg time: {avg_planning_time:.4f}s, Success: {success_rate*100:.1f}%")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Scalability Analysis', fontsize=16, fontweight='bold')

    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']

    # Planning time
    for idx, planner_name in enumerate(['A*', 'PRM', 'RRT*']):
        ax1.plot(robot_counts, results[planner_name]['planning_times'],
                marker=markers[idx], color=colors[idx], linewidth=2,
                markersize=8, label=planner_name)

    ax1.set_xlabel('Number of Robots', fontsize=12)
    ax1.set_ylabel('Average Planning Time (s)', fontsize=12)
    ax1.set_title('Planning Time vs Number of Robots')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Success rate
    for idx, planner_name in enumerate(['A*', 'PRM', 'RRT*']):
        ax2.plot(robot_counts, results[planner_name]['success_rates'],
                marker=markers[idx], color=colors[idx], linewidth=2,
                markersize=8, label=planner_name)

    ax2.set_xlabel('Number of Robots', fontsize=12)
    ax2.set_ylabel('Success Rate (%)', fontsize=12)
    ax2.set_title('Success Rate vs Number of Robots')
    ax2.set_ylim([0, 105])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('scalability_analysis.png', dpi=300, bbox_inches='tight')
    print("\nSaved: scalability_analysis.png")

    return results


def experiment_orca_effectiveness():
    """Compare planners with and without ORCA"""
    print("\n" + "="*80)
    print("EXPERIMENT 3: ORCA Collision Avoidance Effectiveness")
    print("="*80)

    env = WarehouseEnvironment(width=30, height=30, grid_resolution=0.5)
    env.create_warehouse_layout("simple")

    num_robots = 6
    np.random.seed(123)

    # Setup robots
    robots_data = []
    for i in range(num_robots):
        while True:
            start_x = np.random.uniform(3, env.width - 3)
            start_y = np.random.uniform(3, env.height - 3)
            if not env.is_collision(start_x, start_y):
                break

        while True:
            goal_x = np.random.uniform(3, env.width - 3)
            goal_y = np.random.uniform(3, env.height - 3)
            dist = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
            if not env.is_collision(goal_x, goal_y) and dist > 10:
                break

        robots_data.append({
            'start': (start_x, start_y),
            'goal': (goal_x, goal_y)
        })

    results = {}

    for planner_name in ['A*', 'PRM', 'RRT*']:
        for use_orca in [False, True]:
            label = f"{planner_name}" + (" + ORCA" if use_orca else "")
            print(f"\nTesting {label}...")

            # Create simulation
            sim = MultiRobotSimulation(env)

            # Create robots and plan paths
            for i, rd in enumerate(robots_data):
                robot = DifferentialDriveRobot(rd['start'][0], rd['start'][1], 0, i)
                sim.add_robot(robot)

                # Plan path
                if planner_name == 'A*':
                    planner = AStarPlanner(env)
                elif planner_name == 'PRM':
                    planner = PRMPlanner(env, num_samples=800, connection_radius=3.5)
                else:
                    planner = RRTStarPlanner(env, max_iterations=1500)

                path, _ = planner.plan(rd['start'], rd['goal'])
                if path:
                    sim.assign_path(i, path)

            # Run simulation
            max_steps = 4000
            step = 0
            start_time = time.time()

            while step < max_steps and not sim.all_robots_reached_goal():
                sim.step(use_orca=use_orca)
                step += 1

            execution_time = time.time() - start_time

            # Count collisions
            num_collisions = sim.check_collisions()
            success_count = sum(1 for r in sim.robots if r.reached_goal)

            results[label] = {
                'execution_time': execution_time,
                'collisions': num_collisions,
                'success_rate': success_count / num_robots * 100,
                'steps': step
            }

            print(f"  Execution time: {execution_time:.2f}s")
            print(f"  Collisions: {num_collisions}")
            print(f"  Success rate: {success_count/num_robots*100:.1f}%")
            print(f"  Steps: {step}")

    # Create comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('ORCA Effectiveness Comparison (6 Robots)', fontsize=16, fontweight='bold')

    labels = list(results.keys())
    collisions = [results[l]['collisions'] for l in labels]
    success_rates = [results[l]['success_rate'] for l in labels]
    exec_times = [results[l]['execution_time'] for l in labels]

    # Collisions
    ax = axes[0]
    colors_bar = ['lightcoral', 'lightgreen', 'lightcoral', 'lightgreen', 'lightcoral', 'lightgreen']
    ax.bar(range(len(labels)), collisions, color=colors_bar)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Number of Collisions')
    ax.set_title('Collision Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Success rate
    ax = axes[1]
    ax.bar(range(len(labels)), success_rates, color=colors_bar)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate Comparison')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')

    # Execution time
    ax = axes[2]
    ax.bar(range(len(labels)), exec_times, color=colors_bar)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Execution Time Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('orca_effectiveness.png', dpi=300, bbox_inches='tight')
    print("\nSaved: orca_effectiveness.png")

    return results


def experiment_warehouse_complexity():
    """Test different warehouse complexities"""
    print("\n" + "="*80)
    print("EXPERIMENT 4: Warehouse Complexity Impact")
    print("="*80)

    layouts = ['simple', 'complex', 'dense']
    num_robots = 4

    results = {layout: {} for layout in layouts}

    for layout in layouts:
        print(f"\n--- Testing {layout} layout ---")

        env = WarehouseEnvironment(width=30, height=30, grid_resolution=0.5)
        env.create_warehouse_layout(layout)

        print(f"  Obstacles: {len(env.obstacles)}")

        for planner_name in ['A*', 'PRM', 'RRT*']:
            print(f"  {planner_name}...", end=' ')

            np.random.seed(42)
            total_time = 0
            success_count = 0
            total_length = 0

            for i in range(num_robots):
                # Generate positions
                while True:
                    start_x = np.random.uniform(2, env.width - 2)
                    start_y = np.random.uniform(2, env.height - 2)
                    if not env.is_collision(start_x, start_y):
                        break

                while True:
                    goal_x = np.random.uniform(2, env.width - 2)
                    goal_y = np.random.uniform(2, env.height - 2)
                    dist = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
                    if not env.is_collision(goal_x, goal_y) and dist > 8:
                        break

                # Plan
                start_time = time.time()

                if planner_name == 'A*':
                    planner = AStarPlanner(env)
                elif planner_name == 'PRM':
                    planner = PRMPlanner(env, num_samples=800, connection_radius=3.5)
                else:
                    planner = RRTStarPlanner(env, max_iterations=1500)

                path, _ = planner.plan((start_x, start_y), (goal_x, goal_y))
                planning_time = time.time() - start_time

                total_time += planning_time
                if path:
                    success_count += 1
                    metrics = calculate_path_metrics(path)
                    total_length += metrics['length']

            avg_time = total_time / num_robots
            success_rate = success_count / num_robots * 100
            avg_length = total_length / success_count if success_count > 0 else 0

            results[layout][planner_name] = {
                'time': avg_time,
                'success': success_rate,
                'length': avg_length
            }

            print(f"Time: {avg_time:.4f}s, Success: {success_rate:.1f}%, Length: {avg_length:.2f}m")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Impact of Warehouse Complexity', fontsize=16, fontweight='bold')

    x = np.arange(len(layouts))
    width = 0.25

    # Planning time
    ax = axes[0]
    astar_times = [results[l]['A*']['time'] for l in layouts]
    prm_times = [results[l]['PRM']['time'] for l in layouts]
    rrt_times = [results[l]['RRT*']['time'] for l in layouts]
    ax.bar(x - width, astar_times, width, label='A*', color='blue', alpha=0.7)
    ax.bar(x, prm_times, width, label='PRM', color='green', alpha=0.7)
    ax.bar(x + width, rrt_times, width, label='RRT*', color='red', alpha=0.7)
    ax.set_xlabel('Warehouse Layout')
    ax.set_ylabel('Avg Planning Time (s)')
    ax.set_title('Planning Time by Complexity')
    ax.set_xticks(x)
    ax.set_xticklabels(layouts)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Success rate
    ax = axes[1]
    astar_success = [results[l]['A*']['success'] for l in layouts]
    prm_success = [results[l]['PRM']['success'] for l in layouts]
    rrt_success = [results[l]['RRT*']['success'] for l in layouts]
    ax.bar(x - width, astar_success, width, label='A*', color='blue', alpha=0.7)
    ax.bar(x, prm_success, width, label='PRM', color='green', alpha=0.7)
    ax.bar(x + width, rrt_success, width, label='RRT*', color='red', alpha=0.7)
    ax.set_xlabel('Warehouse Layout')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate by Complexity')
    ax.set_xticks(x)
    ax.set_xticklabels(layouts)
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Path length
    ax = axes[2]
    astar_length = [results[l]['A*']['length'] for l in layouts]
    prm_length = [results[l]['PRM']['length'] for l in layouts]
    rrt_length = [results[l]['RRT*']['length'] for l in layouts]
    ax.bar(x - width, astar_length, width, label='A*', color='blue', alpha=0.7)
    ax.bar(x, prm_length, width, label='PRM', color='green', alpha=0.7)
    ax.bar(x + width, rrt_length, width, label='RRT*', color='red', alpha=0.7)
    ax.set_xlabel('Warehouse Layout')
    ax.set_ylabel('Avg Path Length (m)')
    ax.set_title('Path Length by Complexity')
    ax.set_xticks(x)
    ax.set_xticklabels(layouts)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('warehouse_complexity.png', dpi=300, bbox_inches='tight')
    print("\nSaved: warehouse_complexity.png")

    return results


def create_sample_visualization():
    """Create a sample visualization for the report"""
    print("\n" + "="*80)
    print("Creating Sample Visualization for Report")
    print("="*80)

    env = WarehouseEnvironment(width=30, height=30, grid_resolution=0.5)
    env.create_warehouse_layout("complex")

    # Create a multi-robot scenario
    sim = MultiRobotSimulation(env)

    robot_configs = [
        ((3, 3), (27, 27)),
        ((27, 3), (3, 27)),
        ((3, 15), (27, 15)),
        ((15, 3), (15, 27))
    ]

    for i, (start, goal) in enumerate(robot_configs):
        robot = DifferentialDriveRobot(start[0], start[1], 0, i)

        # Use A* for planning
        planner = AStarPlanner(env)
        path, _ = planner.plan(start, goal)

        if path:
            robot.path = path
            sim.add_robot(robot)

    # Create visualization
    vis = SimulationVisualizer(env)
    fig = vis.visualize_paths(sim.robots, title="Multi-Robot Warehouse Path Planning (A*)")
    plt.savefig('sample_scenario.png', dpi=300, bbox_inches='tight')
    print("Saved: sample_scenario.png")

    return fig


if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPREHENSIVE EXPERIMENTAL ANALYSIS")
    print("Multi-Robot Warehouse Motion Planning")
    print("="*80)

    # Run all experiments
    exp1_results = experiment_single_robot_comparison()
    exp2_results = experiment_scalability_test()
    exp3_results = experiment_orca_effectiveness()
    exp4_results = experiment_warehouse_complexity()
    create_sample_visualization()

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print("\nGenerated files:")
    print("  1. single_robot_comparison.png")
    print("  2. scalability_analysis.png")
    print("  3. orca_effectiveness.png")
    print("  4. warehouse_complexity.png")
    print("  5. sample_scenario.png")
    print("\nThese figures are ready to be included in your LaTeX report!")
    print("="*80)
