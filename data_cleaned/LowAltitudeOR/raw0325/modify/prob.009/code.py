
import gurobipy as gp
from gurobipy import GRB
import math

def solve_charging_station_deployment():
    # --- 1. Data Input ---
    R_max = 5.0
    budget = 3  # max number of stations

    candidates = {
        1: (2, 2), # CS1
        2: (2, 8), # CS2
        3: (8, 2), # CS3
        4: (8, 8)  # CS4
    }

    targets = {
        1: (1, 1), # T1
        2: (1, 9), # T2
        3: (9, 1), # T3
        4: (9, 9), # T4
        5: (5, 5), # T5 (Center)
        6: (2, 5)  # T6 (Left Edge)
    }

    cand_ids = list(candidates.keys())
    target_ids = list(targets.keys())

    # Distance Helper
    def get_dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # Pre-calculate Coverage Matrix a[i, j]
    a = {}
    for i in target_ids:
        for j in cand_ids:
            dist = get_dist(targets[i], candidates[j])
            if dist <= R_max:
                a[i, j] = 1
            else:
                a[i, j] = 0

    # --- 2. Model ---
    model = gp.Model("UAV_Maximal_Coverage")

    # Variables
    x = model.addVars(cand_ids, vtype=GRB.BINARY, name="x")
    w = model.addVars(target_ids, vtype=GRB.BINARY, name="w")

    # Objective: Maximize number of covered targets
    model.setObjective(gp.quicksum(w[i] for i in target_ids), GRB.MAXIMIZE)

    # Constraints

    # 1. Coverage logic: target i covered only if at least one covering station is open
    for i in target_ids:
        model.addConstr(w[i] <= gp.quicksum(a[i, j] * x[j] for j in cand_ids), name=f"CoverLogic_{i}")

    # 2. Station budget
    model.addConstr(gp.quicksum(x[j] for j in cand_ids) <= budget, name="Budget")

    # --- 3. Solve ---
    model.optimize()

    # --- 4. Output ---
    if model.status == GRB.OPTIMAL:
        open_stations = [j for j in cand_ids if x[j].X > 0.5]
        covered = [i for i in target_ids if w[i].X > 0.5]
        uncovered = [i for i in target_ids if w[i].X < 0.5]

        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "open_stations": open_stations,
            "covered_targets": covered,
            "uncovered_targets": uncovered
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_charging_station_deployment())
