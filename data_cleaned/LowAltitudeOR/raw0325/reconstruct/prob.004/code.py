import gurobipy as gp
from gurobipy import GRB
import math

def solve_uav_fas_routing():
    # --- 1. Data Input ---
    L_range = 22.0
    FAS_cost = 100.0
    Travel_cost_per_km = 1.0

    nodes = {
        1: (0, 0),   # FAS_1
        2: (20, 0),  # FAS_2
        3: (10, 10), # FAS_3
        4: (5, 0),   # T_A
        5: (10, 0),  # T_B
        6: (15, 0)   # T_C
    }
    
    fas_ids = [1, 2, 3]
    target_ids = [4, 5, 6]
    all_ids = list(nodes.keys())

    def get_dist(i, j):
        p1, p2 = nodes[i], nodes[j]
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    dist = {(i, j): get_dist(i, j) for i in all_ids for j in all_ids if i != j}

    # --- 2. Model ---
    model = gp.Model("FAS_UAV_Indicator")

    # Variables
    y = model.addVars(fas_ids, vtype=GRB.BINARY, name="OpenFAS")
    x = model.addVars(dist.keys(), vtype=GRB.BINARY, name="Flow")
    u = model.addVars(target_ids, vtype=GRB.CONTINUOUS, lb=0, ub=L_range, name="AccumDist")

    # Objective
    cost_fas = gp.quicksum(FAS_cost * y[k] for k in fas_ids)
    cost_travel = gp.quicksum(dist[i, j] * Travel_cost_per_km * x[i, j] for i, j in dist.keys())
    model.setObjective(cost_fas + cost_travel, GRB.MINIMIZE)

    # Constraints

    # 1. Target Visitation
    for t in target_ids:
        model.addConstr(gp.quicksum(x[i, t] for i in all_ids if (i, t) in dist) == 1, name=f"In_{t}")
        model.addConstr(gp.quicksum(x[t, j] for j in all_ids if (t, j) in dist) == 1, name=f"Out_{t}")

    # 2. Start/End at FAS
    model.addConstr(
        gp.quicksum(x[k, j] for k in fas_ids for j in target_ids if (k, j) in dist) == 1,
        name="Start_at_FAS"
    )
    model.addConstr(
        gp.quicksum(x[i, k] for i in target_ids for k in fas_ids if (i, k) in dist) == 1,
        name="End_at_FAS"
    )

    # 3. FAS Opening Logic
    for k in fas_ids:
        model.addConstr(
            gp.quicksum(x[k, j] for j in target_ids if (k, j) in dist) <= y[k],
            name=f"Open_Start_{k}"
        )
        model.addConstr(
            gp.quicksum(x[i, k] for i in target_ids if (i, k) in dist) <= y[k],
            name=f"Open_End_{k}"
        )

    # 4. Distance Propagation with Indicator Constraints (replacing Big-M)
    # FAS to target initialization
    for k in fas_ids:
        for t in target_ids:
            if (k, t) in dist:
                model.addGenConstrIndicator(
                    x[k, t], True,
                    u[t] >= dist[k, t],
                    name=f"Ind_Init_{k}_{t}"
                )

    # Target to target propagation
    for i in target_ids:
        for j in target_ids:
            if i != j and (i, j) in dist:
                model.addGenConstrIndicator(
                    x[i, j], True,
                    u[j] >= u[i] + dist[i, j],
                    name=f"Ind_MTZ_{i}_{j}"
                )

    # 5. Return range check with indicator
    for t in target_ids:
        for k in fas_ids:
            if (t, k) in dist:
                model.addGenConstrIndicator(
                    x[t, k], True,
                    u[t] + dist[t, k] <= L_range,
                    name=f"Ind_Return_{t}_{k}"
                )

    # Solve
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": round(model.ObjVal, 2),
            "open_fas": [k for k in fas_ids if y[k].X > 0.5]
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_uav_fas_routing())
