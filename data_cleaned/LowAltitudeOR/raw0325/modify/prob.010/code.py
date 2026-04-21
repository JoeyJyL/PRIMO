import gurobipy as gp
from gurobipy import GRB
import math

def solve_robust_relief_lrp():
    # --- 1. Data Input (Peak Scenario) ---
    truck_cost = 2.0
    drone_cost = 0.2
    drone_range = 5.0
    drone_cap = 10.0
    
    depots = {
        1: {'loc': (0, 0), 'fixed': 500},
        2: {'loc': (20, 0), 'fixed': 600}
    }
    
    villages = {
        3: {'loc': (2, 2),   'dem': 5},  # V1
        4: {'loc': (18, 2),  'dem': 8},  # V2
        5: {'loc': (10, 10), 'dem': 50}, # V3 (Heavy)
        6: {'loc': (5, 5),   'dem': 15}  # V4 (Heavy)
    }
    
    # Combined Nodes
    all_nodes = {**{k: v['loc'] for k,v in depots.items()}, 
                 **{k: v['loc'] for k,v in villages.items()}}
    node_ids = list(all_nodes.keys())
    
    def get_dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    dist = {(i,j): get_dist(all_nodes[i], all_nodes[j]) for i in node_ids for j in node_ids}

    # --- 2. Model ---
    model = gp.Model("RobustReliefLRP")

    # Variables
    y = model.addVars(depots.keys(), vtype=GRB.BINARY, name="OpenDepot")
    z = model.addVars(villages.keys(), depots.keys(), vtype=GRB.BINARY, name="DroneServe")
    v_truck = model.addVars(villages.keys(), vtype=GRB.BINARY, name="TruckServe")
    x = model.addVars(node_ids, node_ids, vtype=GRB.BINARY, name="TruckRoute")
    u = model.addVars(node_ids, vtype=GRB.CONTINUOUS, name="MTZ")

    # --- Objective ---
    # Fixed
    obj_fixed = gp.quicksum(depots[k]['fixed'] * y[k] for k in depots)
    
    # Drone (Round Trip)
    obj_drone = 0
    for i in villages:
        for k in depots:
            d = dist[i, k]
            # Valid only if range and capacity ok
            if d <= drone_range and villages[i]['dem'] <= drone_cap:
                obj_drone += 2 * d * drone_cost * z[i, k]
            else:
                # Force z to 0 if invalid
                model.addConstr(z[i, k] == 0)

    # Truck
    obj_truck = gp.quicksum(dist[i, j] * truck_cost * x[i, j] for i in node_ids for j in node_ids)

    model.setObjective(obj_fixed + obj_drone + obj_truck, GRB.MINIMIZE)

    # --- Constraints ---

    # 1. Coverage
    for i in villages:
        model.addConstr(gp.quicksum(z[i, k] for k in depots) + v_truck[i] == 1, name=f"Cover_{i}")

    # 2. Drone Validity
    for i in villages:
        for k in depots:
            model.addConstr(z[i, k] <= y[k], name=f"DroneDepotOpen_{i}_{k}")

    # 3. Truck emission constraint: at most 2 villages by truck
    model.addConstr(gp.quicksum(v_truck[i] for i in villages) <= 2, name="TruckEmission")

    # 4. Truck Routing Validity
    is_active = model.addVars(node_ids, vtype=GRB.BINARY, name="ActiveNode")
    
    for k in depots:
        model.addConstr(is_active[k] == y[k])
    for i in villages:
        model.addConstr(is_active[i] == v_truck[i])
        
    # Flow Conservation
    for i in node_ids:
        model.addConstr(gp.quicksum(x[i, j] for j in node_ids if i!=j) == is_active[i])
        model.addConstr(gp.quicksum(x[j, i] for j in node_ids if i!=j) == is_active[i])
        
    # Subtour Elimination (MTZ)
    M = len(node_ids)
    for i in node_ids:
        if i == 1: continue
        for j in node_ids:
            if j == 1: continue
            if i != j:
                model.addConstr(u[j] >= u[i] + 1 - M*(1 - x[i, j]))

    # 5. At least one depot open
    model.addConstr(gp.quicksum(y[k] for k in depots) >= 1)

    # --- Solve ---
    model.optimize()

    # --- Output ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": round(model.ObjVal, 2),
            "open_depots": [k for k in depots if y[k].X > 0.5],
            "truck_villages": [i for i in villages if v_truck[i].X > 0.5],
            "drone_villages": [i for i in villages if sum(z[i, k].X for k in depots) > 0.5]
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_robust_relief_lrp())
