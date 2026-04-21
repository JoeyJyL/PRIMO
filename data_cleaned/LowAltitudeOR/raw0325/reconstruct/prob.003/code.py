import gurobipy as gp
from gurobipy import GRB
import math

def solve_mdlrp_monitoring():
    # --- 1. Data Input ---
    L_range = 30.0
    
    # Nodes: 1,2 (Depots), 3,4,5,6 (Targets)
    coords = {
        1: (0, 0),   # Depot A
        2: (10, 0),  # Depot B
        3: (0, 5),   # T1
        4: (2, 5),   # T2
        5: (8, 5),   # T3
        6: (10, 5)   # T4
    }
    
    depots = {
        1: {'cost': 100},
        2: {'cost': 120}
    }
    
    targets = [3, 4, 5, 6]
    all_nodes = list(coords.keys())
    n_targets = len(targets)
    
    def get_dist(i, j):
        p1, p2 = coords[i], coords[j]
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    dist = {(i, j): get_dist(i, j) for i in all_nodes for j in all_nodes if i != j}

    M_big = L_range + 100  # Big-M for distance propagation

    # --- 2. Model ---
    model = gp.Model("MDLRP-Flow")

    # Variables
    y = model.addVars(depots.keys(), vtype=GRB.BINARY, name="OpenDepot")
    x = model.addVars(dist.keys(), vtype=GRB.BINARY, name="Route")
    f = model.addVars(dist.keys(), vtype=GRB.CONTINUOUS, lb=0, name="Flow")
    d = model.addVars(all_nodes, vtype=GRB.CONTINUOUS, lb=0, name="CumDist")

    # Objective: Fixed Cost + Routing Cost
    obj_fixed = gp.quicksum(depots[k]['cost'] * y[k] for k in depots)
    obj_route = gp.quicksum(dist[i, j] * x[i, j] for i, j in dist)
    model.setObjective(obj_fixed + obj_route, GRB.MINIMIZE)

    # Constraints

    # 1. Visit every target exactly once
    for t in targets:
        model.addConstr(gp.quicksum(x[i, t] for i in all_nodes if i != t) == 1, name=f"In_{t}")
        model.addConstr(gp.quicksum(x[t, j] for j in all_nodes if j != t) == 1, name=f"Out_{t}")

    # 2. Depot Flow & Activation
    for k in depots:
        model.addConstr(
            gp.quicksum(x[k, j] for j in targets) == gp.quicksum(x[j, k] for j in targets),
            name=f"DepotFlow_{k}"
        )
        model.addConstr(
            gp.quicksum(x[k, j] for j in targets) <= n_targets * y[k],
            name=f"Activate_{k}"
        )

    # 3. Single-commodity flow subtour elimination
    # Flow capacity: flow only on active arcs
    for (i, j) in dist:
        model.addConstr(f[i, j] <= n_targets * x[i, j], name=f"FlowCap_{i}_{j}")

    # Flow conservation at targets: each target consumes 1 unit
    for t in targets:
        model.addConstr(
            gp.quicksum(f[i, t] for i in all_nodes if i != t) -
            gp.quicksum(f[t, j] for j in all_nodes if j != t) == 1,
            name=f"FlowCons_{t}"
        )

    # Flow source at depots: only open depots can source flow
    for k in depots:
        model.addConstr(
            gp.quicksum(f[k, j] for j in all_nodes if j != k) <= n_targets * y[k],
            name=f"FlowSource_{k}"
        )

    # 4. Distance tracking for range constraint (separate from subtour elimination)
    for k in depots:
        model.addConstr(d[k] == 0, name=f"DepotDist_{k}")

    for i in all_nodes:
        for j in targets:
            if i != j:
                model.addConstr(
                    d[j] >= d[i] + dist[i, j] - M_big * (1 - x[i, j]),
                    name=f"DistProp_{i}_{j}"
                )

    # 5. Range constraint at targets
    for t in targets:
        model.addConstr(d[t] <= L_range, name=f"Range_{t}")

    # 6. Return range: if returning from target to depot
    for t in targets:
        for k in depots:
            model.addConstr(
                d[t] + dist[t, k] <= L_range + M_big * (1 - x[t, k]),
                name=f"ReturnRange_{t}_{k}"
            )

    # 7. Prevent Depot-to-Depot
    model.addConstr(
        gp.quicksum(x[k1, k2] for k1 in depots for k2 in depots if k1 != k2) == 0,
        name="NoD2D"
    )

    # --- Solve ---
    model.optimize()

    # --- Output ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": round(model.ObjVal, 2),
            "open_depots": [k for k in depots if y[k].X > 0.5]
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_mdlrp_monitoring())
