import gurobipy as gp
from gurobipy import GRB
import math

def solve_double_layer_network():
    # --- 1. Data Input ---
    # Nodes
    supply = (0, 0)
    
    hubs = {
        1: {'loc': (5, 5),  'fixed': 500},
        2: {'loc': (5, -5), 'fixed': 500},
        3: {'loc': (10, 0), 'fixed': 600}
    }
    
    customers = {
        1: {'loc': (6, 6),  'dem': 10},
        2: {'loc': (6, 4),  'dem': 10},
        3: {'loc': (6, -6), 'dem': 10},
        4: {'loc': (11, 1), 'dem': 10}
    }
    
    c_trans = 1.0
    alpha_congestion = 2.0
    
    hub_ids = list(hubs.keys())
    cust_ids = list(customers.keys())

    # Distance Helper
    def get_dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # --- 2. Model ---
    model = gp.Model("DoubleLayerUAV")
    
    # Variables
    y = model.addVars(hub_ids, vtype=GRB.BINARY, name="OpenHub")
    x = model.addVars(hub_ids, cust_ids, vtype=GRB.BINARY, name="Assign")
    load = model.addVars(hub_ids, vtype=GRB.CONTINUOUS, name="HubLoad")
    
    # Objective Terms
    
    # 1. Fixed Cost
    obj_fixed = gp.quicksum(hubs[j]['fixed'] * y[j] for j in hub_ids)
    
    # 2. Transport Cost: S -> H -> C
    obj_trans = 0
    for j in hub_ids:
        d_sh = get_dist(supply, hubs[j]['loc'])
        for k in cust_ids:
            d_hc = get_dist(hubs[j]['loc'], customers[k]['loc'])
            total_dist = d_sh + d_hc
            obj_trans += customers[k]['dem'] * total_dist * c_trans * x[j, k]
            
    # 3. Congestion Cost (Quadratic)
    obj_cong = gp.quicksum(alpha_congestion * load[j] * load[j] for j in hub_ids)
            
    model.setObjective(obj_fixed + obj_trans + obj_cong, GRB.MINIMIZE)
    
    # Constraints
    
    # 1. Assignment
    for k in cust_ids:
        model.addConstr(gp.quicksum(x[j, k] for j in hub_ids) == 1, name=f"Assign_{k}")
        for j in hub_ids:
            model.addConstr(x[j, k] <= y[j], name=f"OpenReq_{j}_{k}")
            
    # 2. Load Definition
    for j in hub_ids:
        model.addConstr(load[j] == gp.quicksum(customers[k]['dem'] * x[j, k] for k in cust_ids), name=f"LoadDef_{j}")
    
    # 3. Minimum Hub Requirement (operational redundancy)
    model.addConstr(gp.quicksum(y[j] for j in hub_ids) >= 3, name="MinHubs")
        
    # --- 3. Solve ---
    model.optimize()
    
    # --- 4. Output ---
    if model.status == GRB.OPTIMAL:
        assignments = {}
        for k in cust_ids:
            for j in hub_ids:
                if x[j, k].X > 0.5:
                    assignments[k] = j
                    
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "open_hubs": [j for j in hub_ids if y[j].X > 0.5],
            "assignments": assignments,
            "hub_loads": {j: load[j].X for j in hub_ids if y[j].X > 0.5}
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_double_layer_network())
