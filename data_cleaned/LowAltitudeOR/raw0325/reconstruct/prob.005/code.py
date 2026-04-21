import gurobipy as gp
from gurobipy import GRB
import math

def solve_dsrflp():
    # --- 1. Data Input ---
    facilities = [1, 2]
    sites = [1, 2, 3]
    drones = [1, 2]
    scenarios = ['S1', 'S2']
    
    probs = {'S1': 0.5, 'S2': 0.5}
    
    demands = {
        'S1': {1: 10, 2: 10, 3: 10},
        'S2': {1: 20, 2: 25, 3: 20}
    }
    
    L_max = 15.0
    B_max = 50.0
    W_cap = 100.0
    
    c_open = {1: 500, 2: 600}
    c_inv = 2.0
    c_mana = 10.0
    c_deli = 1.0
    c_uns = 100.0
    c_unu = 5.0
    
    loc_f = {1: (0, 0), 2: (10, 0)}
    loc_s = {1: (0, 5), 2: (5, 5), 3: (10, 5)}
    
    n_sites = len(sites)  # Big-M for aggregate linking
    
    def get_dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * 2

    dist_ki = {}
    for k in facilities:
        for i in sites:
            dist_ki[k, i] = get_dist(loc_f[k], loc_s[i])

    # --- 2. Model ---
    model = gp.Model("DSRFLP_Reformulated")

    # --- Variables ---
    # Stage 1
    x = model.addVars(facilities, vtype=GRB.BINARY, name="x")
    y = model.addVars(drones, facilities, vtype=GRB.BINARY, name="y")
    z = model.addVars(sites, drones, facilities, vtype=GRB.BINARY, name="z")
    v = model.addVars(facilities, vtype=GRB.CONTINUOUS, name="v")

    # Stage 2 (w eliminated by substitution — only q, u, r remain)
    q = model.addVars(sites, drones, facilities, scenarios, vtype=GRB.CONTINUOUS, name="q")
    u = model.addVars(sites, scenarios, vtype=GRB.CONTINUOUS, name="u")
    r = model.addVars(facilities, scenarios, vtype=GRB.CONTINUOUS, name="r")

    # --- Objective ---
    obj_stage1 = gp.quicksum(c_open[k] * x[k] for k in facilities) + \
                 gp.quicksum(c_inv * v[k] for k in facilities) + \
                 gp.quicksum(c_mana * z[i, d, k] for i in sites for d in drones for k in facilities)

    obj_stage2 = 0
    for xi in scenarios:
        cost_xi = gp.quicksum(c_deli * q[i, d, k, xi] for i in sites for d in drones for k in facilities) + \
                  gp.quicksum(c_uns * u[i, xi] for i in sites) + \
                  gp.quicksum(c_unu * r[k, xi] for k in facilities)
        obj_stage2 += probs[xi] * cost_xi

    model.setObjective(obj_stage1 + obj_stage2, GRB.MINIMIZE)

    # --- Constraints ---
    
    # 1. Drone assignment
    for d in drones:
        model.addConstr(gp.quicksum(y[d, k] for k in facilities) <= 1, name=f"DroneAssign_{d}")
        
    for k in facilities:
        for d in drones:
            model.addConstr(y[d, k] <= x[k], name=f"OpenLink_{d}_{k}")

    # 2. Aggregate Big-M linking (replaces individual z[i,d,k] <= y[d,k])
    for d in drones:
        for k in facilities:
            model.addConstr(
                gp.quicksum(z[i, d, k] for i in sites) <= n_sites * y[d, k],
                name=f"AggVisitLink_{d}_{k}"
            )

    # 3. Coverage
    for i in sites:
        model.addConstr(gp.quicksum(z[i, d, k] for d in drones for k in facilities) >= 1, name=f"Cover_{i}")

    # 4. Battery Range
    for k in facilities:
        for d in drones:
            model.addConstr(
                gp.quicksum(dist_ki[k, i] * z[i, d, k] for i in sites) <= B_max * y[d, k],
                name=f"Battery_{d}_{k}"
            )

    # 5. Inventory Capacity
    for k in facilities:
        model.addConstr(v[k] <= W_cap * x[k], name=f"InvCap_{k}")

    # 6. Stage 2 Balance (with w eliminated by substitution)
    for xi in scenarios:
        # Demand Balance: sum_{k,d} q_{idk} + u_i = Demand_i (w eliminated)
        for i in sites:
            model.addConstr(
                gp.quicksum(q[i, d, k, xi] for d in drones for k in facilities) + u[i, xi] == demands[xi][i],
                name=f"DemBal_{i}_{xi}"
            )
        
        # Supply Balance: sum_{i,d} q_{idk} + r_k = v_k (w eliminated)
        for k in facilities:
            model.addConstr(
                gp.quicksum(q[i, d, k, xi] for i in sites for d in drones) + r[k, xi] == v[k],
                name=f"SupBal_{k}_{xi}"
            )
        
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
