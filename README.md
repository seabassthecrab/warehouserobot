# Warehouse Robotics Simulator

how to run the file

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

warehouse_sim --planner astar --num-robots 4
warehouse_sim --planner prm --num-robots 4
```

## Single run with visualization

```bash
warehouse_sim --mode single --planner astar --num-robots 6 --enable-orca
```

## Run experiment (5 trials)

```bash
warehouse_sim --mode experiment --planner prm --num-robots 4 --num-trials 5
```

## Full comparison (all planners, with/without ORCA)

```bash
warehouse_sim --mode compare
```
