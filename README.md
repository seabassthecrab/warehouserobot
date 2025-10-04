how to run the file

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy matplotlib networkx pillow scipy

python warehouse_sim_enhanced.py --planner astar --num-robots 4
python warehouse_sim_enhanced.py --planner prm --num-robots 4

# Single run with visualization
python warehouse_sim_enhanced.py --mode single --planner astar --num-robots 6 --enable-orca

# Run experiment (5 trials)
python warehouse_sim_enhanced.py --mode experiment --planner prm --num-robots 4 --num-trials 5

# Full comparison (all planners, with/without ORCA)
python warehouse_sim_enhanced.py --mode compare
