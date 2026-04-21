import gurobipy as gp
from gurobipy import GRB
import math

def solve_uav_fas_routing():
    # --- 1. Data Input ---
    # Parameters
    L_range = 22.0
    FAS_cost = 100.0
    Travel_cost_per_km = 1.0

    # Nodes
    # FAS: 1, 2, 3. Targets: 4, 5, 6
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
    model = gp.Model("FAS_UAV_Routing")

    # Variables
    y = model.addVars(fas_ids, vtype=GRB.BINARY, name="OpenFAS")
    x = model.addVars(dist.keys(), vtype=GRB.BINARY, name="Flow")
    u = model.addVars(target_ids, vtype=GRB.CONTINUOUS, name="AccumDist")

    # Objective
    cost_fas = gp.quicksum(FAS_cost * y[k] for k in fas_ids)
    cost_travel = gp.quicksum(dist[i, j] * Travel_cost_per_km * x[i, j] for i, j in dist.keys())
    model.setObjective(cost_fas + cost_travel, GRB.MINIMIZE)

    # Constraints

    # 1. Target Visitation (Must enter and leave each target once)
    for t in target_ids:
        model.addConstr(gp.quicksum(x[i, t] for i in all_ids if (i,t) in dist) == 1, name=f"In_{t}")
        model.addConstr(gp.quicksum(x[t, j] for j in all_ids if (t,j) in dist) == 1, name=f"Out_{t}")

    # 2. Start/End Constraints (single drone)
    model.addConstr(gp.quicksum(x[k, j] for k in fas_ids for j in target_ids if (k,j) in dist) == 1, name="Start_at_FAS")
    model.addConstr(gp.quicksum(x[i, k] for i in target_ids for k in fas_ids if (i,k) in dist) == 1, name="End_at_FAS")

    # 3. FAS Opening Logic
    for k in fas_ids:
        model.addConstr(gp.quicksum(x[k, j] for j in target_ids if (k,j) in dist) <= y[k], name=f"Open_Start_{k}")
        model.addConstr(gp.quicksum(x[i, k] for i in target_ids if (i,k) in dist) <= y[k], name=f"Open_End_{k}")

    # 4. Range & Connectivity (MTZ)
    M = 100
    for k in fas_ids:
        for t in target_ids:
            if (k, t) in dist:
                model.addConstr(u[t] >= dist[k, t] - M*(1 - x[k, t]), name=f"Init_{k}_{t}")

    for i in target_ids:
        for j in target_ids:
            if i != j and (i, j) in dist:
                model.addConstr(u[j] >= u[i] + dist[i, j] - M*(1 - x[i, j]), name=f"MTZ_{i}_{j}")

    # Max Range Check at Targets
    for t in target_ids:
        model.addConstr(u[t] <= L_range, name=f"RangeCheck_{t}")
    
    # Check return to FAS range
    for t in target_ids:
        for k in fas_ids:
            if (t, k) in dist:
                model.addConstr(u[t] + dist[t, k] <= L_range + M*(1 - x[t, k]), name=f"FinalRange_{t}_{k}")

    # 5. Airspace Coordination Constraint
    # If both FAS_1 and FAS_2 are operational, FAS_3 must also be activated
    model.addConstr(y[1] + y[2] <= 1 + y[3], name="Airspace_Coordination")

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
