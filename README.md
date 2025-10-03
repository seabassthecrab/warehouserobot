how to run the file

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy matplotlib networkx pillow scipy

python warehouse_sim_dynamic.py --planner astar --num-robots 4 --seed 42 --out outputs --max-steps 600
