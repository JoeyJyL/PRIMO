import gurobipy as gp
from gurobipy import GRB
import math

def solve_charging_station_deployment():
    # --- 1. Data Input ---
    R_max = 5.0

    candidates = {
        1: (2, 2),  # CS1
        2: (2, 8),  # CS2
        3: (8, 2),  # CS3
        4: (8, 8)   # CS4
    }

    targets = {
        1: (1, 1),  # T1
        2: (1, 9),  # T2
        3: (9, 1),  # T3
        4: (9, 9),  # T4
        5: (5, 5),  # T5 (Center)
        6: (2, 5)   # T6 (Left Edge)
    }

    cand_ids = list(candidates.keys())
    target_ids = list(targets.keys())

    def get_dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # Coverage matrix a[i, j]
    a = {}
    for i in target_ids:
        for j in cand_ids:
            a[i, j] = 1 if get_dist(targets[i], candidates[j]) <= R_max else 0

    # Valid assignment pairs
    valid_pairs = [(i, j) for i in target_ids for j in cand_ids if a[i, j] == 1]

    # --- 2. Model ---
    model = gp.Model("UAV_Station_Assignment")

    # Variables
    x = model.addVars(cand_ids, vtype=GRB.BINARY, name="x")
    # Assignment variables (continuous — integral by construction)
    y = model.addVars(valid_pairs, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="y")

    # Objective: Minimize number of stations
    model.setObjective(gp.quicksum(x[j] for j in cand_ids), GRB.MINIMIZE)

    # Constraints

    # 1. Assignment: each target assigned to exactly one station
    for i in target_ids:
        model.addConstr(
            gp.quicksum(y[i, j] for j in cand_ids if a[i, j] == 1) == 1,
            name=f"Assign_{i}"
        )

    # 2. Linking: can only assign to an open station
    for i, j in valid_pairs:
        model.addConstr(y[i, j] <= x[j], name=f"Link_{i}_{j}")

    # --- 3. Solve ---
    model.optimize()

    # --- 4. Output ---
    if model.status == GRB.OPTIMAL:
        open_stations = [j for j in cand_ids if x[j].X > 0.5]

        coverage_map = {}
        for i in target_ids:
            for j in cand_ids:
                if a[i, j] == 1 and y[i, j].X > 0.5:
                    coverage_map[i] = [j]

        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "open_stations": open_stations,
            "coverage_details": coverage_map
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_charging_station_deployment())
