"""
Visualization and analysis tools for warehouse simulation results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import json
from warehouse_simulation import *


class SimulationVisualizer:
    """Visualize simulation results"""

    def __init__(self, environment: WarehouseEnvironment):
        self.env = environment
        self.fig = None
        self.ax = None

    def setup_plot(self):
        """Setup matplotlib figure"""
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_xlim(0, self.env.width)
        self.ax.set_ylim(0, self.env.height)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')

        # Draw obstacles
        for ox, oy, ow, oh in self.env.obstacles:
            rect = patches.Rectangle((ox, oy), ow, oh,
                                     linewidth=1, edgecolor='black',
                                     facecolor='gray', alpha=0.7)
            self.ax.add_patch(rect)

    def visualize_paths(self, robots: List[DifferentialDriveRobot], title: str = "Paths"):
        """Visualize planned paths"""
        self.setup_plot()
        self.ax.set_title(title)

        colors = plt.cm.tab10(np.linspace(0, 1, len(robots)))

        for i, robot in enumerate(robots):
            # Draw path
            if robot.path:
                path_x = [p[0] for p in robot.path]
                path_y = [p[1] for p in robot.path]
                self.ax.plot(path_x, path_y, 'o-', color=colors[i],
                           alpha=0.5, label=f'Robot {robot.robot_id} path',
                           markersize=3)

            # Draw start position
            circle = plt.Circle((robot.path[0][0] if robot.path else robot.x,
                                robot.path[0][1] if robot.path else robot.y),
                               robot.radius, color=colors[i], alpha=0.3)
            self.ax.add_patch(circle)
            self.ax.plot(robot.path[0][0] if robot.path else robot.x,
                        robot.path[0][1] if robot.path else robot.y,
                        'o', color=colors[i], markersize=10, label=f'Robot {i} start')

            # Draw goal position
            if robot.path:
                self.ax.plot(robot.path[-1][0], robot.path[-1][1],
                           '*', color=colors[i], markersize=15, label=f'Robot {i} goal')

        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return self.fig

    def visualize_trajectories(self, robots: List[DifferentialDriveRobot], title: str = "Trajectories"):
        """Visualize actual trajectories"""
        self.setup_plot()
        self.ax.set_title(title)

        colors = plt.cm.tab10(np.linspace(0, 1, len(robots)))

        for i, robot in enumerate(robots):
            if robot.trajectory:
                traj_x = [p[0] for p in robot.trajectory]
                traj_y = [p[1] for p in robot.trajectory]
                self.ax.plot(traj_x, traj_y, '-', color=colors[i],
                           alpha=0.6, linewidth=2, label=f'Robot {i} trajectory')

                # Start and end markers
                self.ax.plot(traj_x[0], traj_y[0], 'o', color=colors[i], markersize=10)
                self.ax.plot(traj_x[-1], traj_y[-1], '*', color=colors[i], markersize=15)

        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return self.fig

    def animate_simulation(self, robots: List[DifferentialDriveRobot],
                          filename: str = "simulation.gif", fps: int = 10):
        """Create animation of simulation"""
        self.setup_plot()

        colors = plt.cm.tab10(np.linspace(0, 1, len(robots)))

        # Find max trajectory length
        max_len = max(len(robot.trajectory) for robot in robots)

        robot_patches = []
        trajectory_lines = []

        for i, robot in enumerate(robots):
            # Create robot patch
            circle = plt.Circle((robot.trajectory[0][0], robot.trajectory[0][1]),
                               robot.radius, color=colors[i], alpha=0.7)
            self.ax.add_patch(circle)
            robot_patches.append(circle)

            # Create trajectory line
            line, = self.ax.plot([], [], '-', color=colors[i], alpha=0.3, linewidth=1)
            trajectory_lines.append(line)

        def init():
            return robot_patches + trajectory_lines

        def animate(frame):
            for i, robot in enumerate(robots):
                if frame < len(robot.trajectory):
                    x, y = robot.trajectory[frame]
                    robot_patches[i].center = (x, y)

                    # Update trajectory
                    traj_x = [p[0] for p in robot.trajectory[:frame+1]]
                    traj_y = [p[1] for p in robot.trajectory[:frame+1]]
                    trajectory_lines[i].set_data(traj_x, traj_y)

            return robot_patches + trajectory_lines

        anim = FuncAnimation(self.fig, animate, init_func=init,
                           frames=max_len, interval=1000/fps, blit=True)

        anim.save(filename, writer=PillowWriter(fps=fps))
        print(f"Animation saved to {filename}")

        return anim


def plot_benchmark_results(results_file: str = 'benchmark_results.json'):
    """Plot benchmark results from JSON file"""
    with open(results_file, 'r') as f:
        data = json.load(f)

    metrics = data['metrics']

    # Organize data by planner
    planners = {}
    for m in metrics:
        planner = m['planner']
        if planner not in planners:
            planners[planner] = {
                'num_robots': [],
                'planning_time': [],
                'execution_time': [],
                'path_length': [],
                'collisions': [],
                'success_rate': [],
                'nodes_explored': [],
                'smoothness': []
            }

        planners[planner]['num_robots'].append(m['num_robots'])
        planners[planner]['planning_time'].append(m['planning_time'])
        planners[planner]['execution_time'].append(m['execution_time'])
        planners[planner]['path_length'].append(m['path_length'])
        planners[planner]['collisions'].append(m['collisions'])
        planners[planner]['success_rate'].append(m['success_rate'] * 100)
        planners[planner]['nodes_explored'].append(m['nodes_explored'])
        planners[planner]['smoothness'].append(m['smoothness'])

    # Create comprehensive plots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Multi-Robot Path Planning Performance Comparison', fontsize=16, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(planners)))
    markers = ['o', 's', '^', 'D', 'v', 'p']

    # Plot 1: Planning Time vs Number of Robots
    ax = axes[0, 0]
    for i, (planner, data) in enumerate(planners.items()):
        ax.plot(data['num_robots'], data['planning_time'],
               marker=markers[i % len(markers)], color=colors[i],
               label=planner, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Robots')
    ax.set_ylabel('Planning Time (s)')
    ax.set_title('Planning Time vs Number of Robots')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Execution Time vs Number of Robots
    ax = axes[0, 1]
    for i, (planner, data) in enumerate(planners.items()):
        ax.plot(data['num_robots'], data['execution_time'],
               marker=markers[i % len(markers)], color=colors[i],
               label=planner, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Robots')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Execution Time vs Number of Robots')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Success Rate vs Number of Robots
    ax = axes[1, 0]
    for i, (planner, data) in enumerate(planners.items()):
        ax.plot(data['num_robots'], data['success_rate'],
               marker=markers[i % len(markers)], color=colors[i],
               label=planner, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Robots')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate vs Number of Robots')
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Number of Collisions vs Number of Robots
    ax = axes[1, 1]
    for i, (planner, data) in enumerate(planners.items()):
        ax.plot(data['num_robots'], data['collisions'],
               marker=markers[i % len(markers)], color=colors[i],
               label=planner, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Robots')
    ax.set_ylabel('Number of Collisions')
    ax.set_title('Collisions vs Number of Robots')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Computational Cost (Nodes Explored)
    ax = axes[2, 0]
    for i, (planner, data) in enumerate(planners.items()):
        ax.plot(data['num_robots'], data['nodes_explored'],
               marker=markers[i % len(markers)], color=colors[i],
               label=planner, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Robots')
    ax.set_ylabel('Nodes Explored')
    ax.set_title('Computational Cost vs Number of Robots')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Average Path Length
    ax = axes[2, 1]
    for i, (planner, data) in enumerate(planners.items()):
        ax.plot(data['num_robots'], data['path_length'],
               marker=markers[i % len(markers)], color=colors[i],
               label=planner, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Robots')
    ax.set_ylabel('Average Path Length (m)')
    ax.set_title('Path Length vs Number of Robots')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
    print("Benchmark comparison saved to benchmark_comparison.png")

    return fig


def create_comparison_table(results_file: str = 'benchmark_results.json'):
    """Create LaTeX table for report"""
    with open(results_file, 'r') as f:
        data = json.load(f)

    metrics = data['metrics']

    # Create LaTeX table
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{Performance Comparison of Path Planning Algorithms}\n"
    latex += "\\label{tab:performance}\n"
    latex += "\\begin{tabular}{|l|c|c|c|c|c|c|}\n"
    latex += "\\hline\n"
    latex += "\\textbf{Planner} & \\textbf{Robots} & \\textbf{Plan (s)} & \\textbf{Exec (s)} & "
    latex += "\\textbf{Success \\%} & \\textbf{Collisions} & \\textbf{Path (m)} \\\\\n"
    latex += "\\hline\n"

    for m in metrics:
        latex += f"{m['planner']} & {m['num_robots']} & "
        latex += f"{m['planning_time']:.3f} & {m['execution_time']:.2f} & "
        latex += f"{m['success_rate']*100:.1f} & {m['collisions']} & "
        latex += f"{m['path_length']:.2f} \\\\\n"

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    with open('results_table.tex', 'w') as f:
        f.write(latex)

    print("LaTeX table saved to results_table.tex")
    return latex


def generate_statistical_summary(results_file: str = 'benchmark_results.json'):
    """Generate statistical summary"""
    with open(results_file, 'r') as f:
        data = json.load(f)

    metrics = data['metrics']

    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)

    # Group by planner
    planners = {}
    for m in metrics:
        planner = m['planner']
        if planner not in planners:
            planners[planner] = []
        planners[planner].append(m)

    for planner, data in planners.items():
        print(f"\n{planner}:")
        print("-" * 40)

        planning_times = [d['planning_time'] for d in data]
        execution_times = [d['execution_time'] for d in data]
        success_rates = [d['success_rate'] for d in data]
        collisions = [d['collisions'] for d in data]

        print(f"  Planning Time:   {np.mean(planning_times):.3f} ± {np.std(planning_times):.3f} s")
        print(f"  Execution Time:  {np.mean(execution_times):.2f} ± {np.std(execution_times):.2f} s")
        print(f"  Success Rate:    {np.mean(success_rates)*100:.1f}% ± {np.std(success_rates)*100:.1f}%")
        print(f"  Avg Collisions:  {np.mean(collisions):.2f} ± {np.std(collisions):.2f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    print("Visualization and Analysis Tools")
    print("="*60)

    # Check if results exist
    try:
        # Plot benchmark results
        print("\nGenerating benchmark comparison plots...")
        plot_benchmark_results()

        # Generate statistical summary
        generate_statistical_summary()

        # Create LaTeX table
        print("\nGenerating LaTeX table...")
        create_comparison_table()

        print("\n" + "="*60)
        print("Analysis complete!")
        print("Generated files:")
        print("  - benchmark_comparison.png")
        print("  - results_table.tex")

    except FileNotFoundError:
        print("\nError: benchmark_results.json not found.")
        print("Please run warehouse_simulation.py first to generate results.")
