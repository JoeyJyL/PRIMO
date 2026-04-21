import gurobipy as gp
from gurobipy import GRB
import math

def solve_mdlrp_monitoring():
    # --- 1. Data Input ---
    L_range = 30.0
    
    # Nodes: 1,2 (Depots), 3,4,5,6 (Targets)
    # Coordinates
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
    
    def get_dist(i, j):
        p1, p2 = coords[i], coords[j]
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    dist = {(i, j): get_dist(i, j) for i in all_nodes for j in all_nodes if i != j}

    # --- 2. Model ---
    model = gp.Model("MDLRP-UM")

    # Variables
    y = model.addVars(depots.keys(), vtype=GRB.BINARY, name="OpenDepot")
    x = model.addVars(dist.keys(), vtype=GRB.BINARY, name="Route")
    u = model.addVars(all_nodes, vtype=GRB.CONTINUOUS, name="CumDist") # Cumulative distance

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
    # If we leave a depot, we must enter it (conservation) AND it must be open
    for k in depots:
        # Outflow = Inflow
        model.addConstr(gp.quicksum(x[k, j] for j in targets) == gp.quicksum(x[j, k] for j in targets), name=f"Flow_{k}")
        # Activation (Big M)
        model.addConstr(gp.quicksum(x[k, j] for j in targets) <= len(targets) * y[k], name=f"Activate_{k}")

    # 3. Subtour Elimination & Range (MTZ variant for LRP)
    # u[i] is the distance traveled SO FAR when arriving at i.
    # Initialize at Depots: u[k] = 0
    for k in depots:
        model.addConstr(u[k] == 0)

    # Propagation
    M = L_range + 100
    for i in all_nodes:
        for j in all_nodes:
            if i != j and j not in depots: # Don't constrain u[depot] here, it's 0
                model.addConstr(u[j] >= u[i] + dist[i, j] - M * (1 - x[i, j]), name=f"MTZ_{i}_{j}")
    
    # 4. Range Constraint Check (Return to Depot)
    # If we go from i to depot k, the total distance u[i] + dist[i,k] must be <= L
    for i in targets:
        for k in depots:
            model.addConstr(u[i] + dist[i, k] <= L_range + M * (1 - x[i, k]), name=f"RangeReturn_{i}_{k}")
            
    # Also limit u[i] itself <= L (implicit but good to have)
    for i in targets:
        model.addConstr(u[i] <= L_range)
        
    # Prevent Depot-to-Depot
    model.addConstr(gp.quicksum(x[k1, k2] for k1 in depots for k2 in depots if k1!=k2) == 0)

    # --- Solve ---
    model.optimize()

    # --- Output ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "open_depots": [k for k in depots if y[k].X > 0.5]
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_mdlrp_monitoring())