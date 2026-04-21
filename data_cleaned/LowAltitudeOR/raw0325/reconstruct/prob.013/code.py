import gurobipy as gp
from gurobipy import GRB
import math

def solve_3df_mom():
    # --- 1. Data Input ---
    candidates = {
        1: {'loc': (2, 2), 'cost': 100, 'noise': 5},
        2: {'loc': (2, 8), 'cost': 120, 'noise': 8},
        3: {'loc': (8, 2), 'cost': 110, 'noise': 4},
        4: {'loc': (8, 8), 'cost': 130, 'noise': 6},
        5: {'loc': (5, 5), 'cost': 90,  'noise': 3}
    }

    zones = {
        1: {'loc': (1, 1), 'dem': 50},
        2: {'loc': (1, 9), 'dem': 40},
        3: {'loc': (9, 1), 'dem': 60},
        4: {'loc': (9, 9), 'dem': 45}
    }

    P_max = 3
    R_max = 4.0
    W_cov = 10.0
    W_cost = 1.0
    W_noise = 2.0

    cand_ids = list(candidates.keys())
    zone_ids = list(zones.keys())

    def get_dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # Coverage matrix
    a = {}
    for i in zone_ids:
        for j in cand_ids:
            a[i, j] = 1 if get_dist(zones[i]['loc'], candidates[j]['loc']) <= R_max else 0

    # Composite penalty coefficient
    penalty = {j: W_cost * candidates[j]['cost'] + W_noise * candidates[j]['noise']
               for j in cand_ids}

    # --- 2. Model ---
    model = gp.Model("CityFly_OR")

    x = model.addVars(cand_ids, vtype=GRB.BINARY, name="Open")
    y = model.addVars(zone_ids, vtype=GRB.BINARY, name="Covered")

    # Objective with merged penalty
    term_cov = gp.quicksum(zones[i]['dem'] * y[i] for i in zone_ids)
    term_pen = gp.quicksum(penalty[j] * x[j] for j in cand_ids)
    model.setObjective(W_cov * term_cov - term_pen, GRB.MAXIMIZE)

    # Constraints

    # 1. Max Vertiports
    model.addConstr(gp.quicksum(x[j] for j in cand_ids) <= P_max, name="MaxP")

    # 2. Coverage via OR general constraint (replaces aggregate inequality)
    for i in zone_ids:
        covering_vars = [x[j] for j in cand_ids if a[i, j] == 1]
        if covering_vars:
            model.addGenConstrOr(y[i], covering_vars, name=f"CoverOR_{i}")
        else:
            model.addConstr(y[i] == 0, name=f"NoCover_{i}")

    # --- 3. Solve ---
    model.optimize()

    # --- 4. Output ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "open_vertiports": [j for j in cand_ids if x[j].X > 0.5],
            "covered_zones": [i for i in zone_ids if y[i].X > 0.5]
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_3df_mom())
