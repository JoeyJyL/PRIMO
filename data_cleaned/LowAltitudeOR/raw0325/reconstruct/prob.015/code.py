import gurobipy as gp
from gurobipy import GRB
import math

def solve_queuing_location():
    # --- 1. Data Input ---
    candidates = {
        1: {'loc': (5, 5), 'fixed': 2000},  # V1 Center
        2: {'loc': (2, 5), 'fixed': 1000},   # V2 West
        3: {'loc': (8, 5), 'fixed': 1000}    # V3 East
    }

    zones = {
        1: {'loc': (2, 8), 'dem': 100},
        2: {'loc': (5, 8), 'dem': 120},
        3: {'loc': (8, 8), 'dem': 100},
        4: {'loc': (5, 2), 'dem': 80}
    }

    alpha_congestion = 0.05
    access_rate = 1.0

    cand_ids = list(candidates.keys())
    zone_ids = list(zones.keys())
    num_zones = len(zone_ids)

    def get_dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    c_access = {}
    for i in zone_ids:
        for j in cand_ids:
            c_access[i, j] = get_dist(zones[i]['loc'], candidates[j]['loc']) * access_rate

    # --- 2. Model ---
    model = gp.Model("UAM_Queuing_VarElim")

    # Variables (L_j eliminated)
    y = model.addVars(cand_ids, vtype=GRB.BINARY, name="Open")
    x = model.addVars(zone_ids, cand_ids, vtype=GRB.BINARY, name="Assign")

    # Objective
    # 1. Fixed Cost
    obj_fixed = gp.quicksum(candidates[j]['fixed'] * y[j] for j in cand_ids)

    # 2. Access Cost
    obj_access = gp.quicksum(
        zones[i]['dem'] * c_access[i, j] * x[i, j]
        for i in zone_ids for j in cand_ids)

    # 3. Congestion Cost: alpha * (sum_i D_i x_{ij})^2 — L_j substituted directly
    congestion_exprs = []
    for j in cand_ids:
        load_expr = gp.quicksum(zones[i]['dem'] * x[i, j] for i in zone_ids)
        congestion_exprs.append(alpha_congestion * load_expr * load_expr)
    obj_congest = gp.quicksum(congestion_exprs)

    model.setObjective(obj_fixed + obj_access + obj_congest, GRB.MINIMIZE)

    # Constraints

    # 1. Single Assignment
    for i in zone_ids:
        model.addConstr(
            gp.quicksum(x[i, j] for j in cand_ids) == 1,
            name=f"Assign_{i}")

    # 2. Aggregated Capacity Linking (replaces individual x_{ij} <= y_j)
    for j in cand_ids:
        model.addConstr(
            gp.quicksum(x[i, j] for i in zone_ids) <= num_zones * y[j],
            name=f"AggCap_{j}")

    # --- 3. Solve ---
    model.optimize()

    # --- 4. Output ---
    if model.status == GRB.OPTIMAL:
        sol_load = {}
        for j in cand_ids:
            sol_load[j] = sum(zones[i]['dem'] * x[i, j].X for i in zone_ids)
        return {
            "status": "optimal",
            "obj": round(model.ObjVal, 2),
            "open_vertiports": [j for j in cand_ids if y[j].X > 0.5],
            "loads": sol_load,
            "congestion_costs": {j: alpha_congestion * (sol_load[j]**2) for j in cand_ids}
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_queuing_location())
