
import gurobipy as gp
from gurobipy import GRB
import math

def solve_dsrflp():
    # --- 1. Data Input ---
    # Sets
    facilities = [1, 2] # A, B
    sites = [1, 2, 3]   # Zone 1, 2, 3
    drones = [1, 2]     # D1, D2
    scenarios = ['S1', 'S2']
    
    # Probabilities
    probs = {'S1': 0.5, 'S2': 0.5}
    
    # Demands D[i][xi]
    demands = {
        'S1': {1: 10, 2: 10, 3: 10},
        'S2': {1: 20, 2: 25, 3: 20}
    }
    
    # Parameters
    L_max = 15.0  # Payload per drone
    B_max = 50.0  # Range km
    W_cap = 40.0  # Facility Inv Cap (reduced due to cold-chain storage limits)
    
    # Costs
    c_open = {1: 500, 2: 600}
    c_inv = 2.0
    c_mana = 10.0
    c_deli = 1.0
    c_uns = 100.0
    c_unu = 5.0
    
    # Coordinates
    loc_f = {1: (0, 0), 2: (10, 0)}
    loc_s = {1: (0, 5), 2: (5, 5), 3: (10, 5)}
    
    # Distance (round trip)
    def get_dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * 2

    dist_ki = {}
    for k in facilities:
        for i in sites:
            dist_ki[k, i] = get_dist(loc_f[k], loc_s[i])

    # --- 2. Model ---
    model = gp.Model("DSRFLP")

    # --- Variables ---
    # Stage 1
    x = model.addVars(facilities, vtype=GRB.BINARY, name="x") # Open
    y = model.addVars(drones, facilities, vtype=GRB.BINARY, name="y") # Assign drone
    z = model.addVars(sites, drones, facilities, vtype=GRB.BINARY, name="z") # Visit
    v = model.addVars(facilities, vtype=GRB.CONTINUOUS, name="v") # Inventory

    # Stage 2 (Recourse)
    w = model.addVars(sites, facilities, scenarios, vtype=GRB.CONTINUOUS, name="w") # Delivered total
    q = model.addVars(sites, drones, facilities, scenarios, vtype=GRB.CONTINUOUS, name="q") # Delivered by drone
    u = model.addVars(sites, scenarios, vtype=GRB.CONTINUOUS, name="u") # Unsatisfied
    r = model.addVars(facilities, scenarios, vtype=GRB.CONTINUOUS, name="r") # Unused

    # --- Objective ---
    # Stage 1 Cost
    obj_stage1 = gp.quicksum(c_open[k] * x[k] for k in facilities) + \
                 gp.quicksum(c_inv * v[k] for k in facilities) + \
                 gp.quicksum(c_mana * z[i, d, k] for i in sites for d in drones for k in facilities)

    # Stage 2 Cost (Expected)
    obj_stage2 = 0
    for xi in scenarios:
        cost_xi = gp.quicksum(c_deli * q[i, d, k, xi] for i in sites for d in drones for k in facilities) + \
                  gp.quicksum(c_uns * u[i, xi] for i in sites) + \
                  gp.quicksum(c_unu * r[k, xi] for k in facilities)
        obj_stage2 += probs[xi] * cost_xi

    model.setObjective(obj_stage1 + obj_stage2, GRB.MINIMIZE)

    # --- Constraints ---
    
    # 1. Topology & Assignment
    for d in drones:
        model.addConstr(gp.quicksum(y[d, k] for k in facilities) <= 1, name=f"DroneAssign_{d}")
        
    for k in facilities:
        for d in drones:
            model.addConstr(y[d, k] <= x[k], name=f"OpenLink_{d}_{k}")
            for i in sites:
                model.addConstr(z[i, d, k] <= y[d, k], name=f"VisitLink_{i}_{d}_{k}")

    # 2. Coverage (Must visit)
    for i in sites:
        model.addConstr(gp.quicksum(z[i, d, k] for d in drones for k in facilities) >= 1, name=f"Cover_{i}")

    # 3. Battery Range
    for k in facilities:
        for d in drones:
            model.addConstr(gp.quicksum(dist_ki[k, i] * z[i, d, k] for i in sites) <= B_max * y[d, k], name=f"Battery_{d}_{k}")

    # 4. Inventory Capacity
    for k in facilities:
        model.addConstr(v[k] <= W_cap * x[k], name=f"InvCap_{k}")

    # 5. Stage 2 Balance (per Scenario)
    for xi in scenarios:
        # Demand Balance
        for i in sites:
            model.addConstr(gp.quicksum(w[i, k, xi] for k in facilities) + u[i, xi] == demands[xi][i], name=f"DemBal_{i}_{xi}")
        
        # Supply Balance
        for k in facilities:
            model.addConstr(gp.quicksum(w[i, k, xi] for i in sites) + r[k, xi] == v[k], name=f"SupBal_{k}_{xi}")
            
        # Drone Flow Link
        for i in sites:
            for k in facilities:
                model.addConstr(w[i, k, xi] == gp.quicksum(q[i, d, k, xi] for d in drones), name=f"FlowLink_{i}_{k}_{xi}")
        
        # Payload Limit
        for i in sites:
            for k in facilities:
                for d in drones:
                    model.addConstr(q[i, d, k, xi] <= L_max * z[i, d, k], name=f"Payload_{i}_{d}_{k}_{xi}")

    # --- Solve ---
    model.optimize()

    # --- Output ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "facilities_open": [k for k in facilities if x[k].X > 0.5],
            "inventory": {k: v[k].X for k in facilities if x[k].X > 0.5},
            "drone_assignments": [(d, k) for d in drones for k in facilities if y[d, k].X > 0.5]
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_dsrflp())
