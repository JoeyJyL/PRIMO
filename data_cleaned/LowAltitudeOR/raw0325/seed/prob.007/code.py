
import gurobipy as gp
from gurobipy import GRB
import math

def solve_drone_fleet_planning():
    # --- 1. Data Input ---
    # Costs
    truck_cost_km = 1.0
    drone_cost_km = 0.1
    drone_range = 10.0 # radius
    
    # Nodes
    cdc = (0, 0)
    mds = {
        1: {'loc': (10, 10), 'fixed': 100}, # MD1
        2: {'loc': (20, 0),  'fixed': 250}  # MD2
    }
    customers = {
        3: (12, 12), # C1
        4: (8, 8),   # C2
        5: (22, 2),  # C3
        6: (0, 15)   # C4
    }
    
    # Combined Node Set for Truck Routing
    # ID mapping: 0=CDC, 1,2=MDs, 3,4,5,6=Custs
    all_nodes = {0: cdc}
    for k, v in mds.items(): all_nodes[k] = v['loc']
    for k, v in customers.items(): all_nodes[k] = v
    
    node_ids = list(all_nodes.keys())
    
    # Distances
    def get_dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
    dist = {(i,j): get_dist(all_nodes[i], all_nodes[j]) for i in node_ids for j in node_ids}

    # --- 2. Model ---
    model = gp.Model("DroneFleetPlanning")

    # Variables
    y = model.addVars(mds.keys(), vtype=GRB.BINARY, name="OpenMD")
    z = model.addVars(customers.keys(), mds.keys(), vtype=GRB.BINARY, name="AssignDrone")
    v = model.addVars(customers.keys(), vtype=GRB.BINARY, name="AssignTruck")
    
    # Truck Routing Edges
    x = model.addVars(node_ids, node_ids, vtype=GRB.BINARY, name="TruckRoute")
    u_mtz = model.addVars(node_ids, vtype=GRB.CONTINUOUS, name="MTZ")

    # --- Objective ---
    # 1. Fixed Cost
    obj_fixed = gp.quicksum(mds[m]['fixed'] * y[m] for m in mds)
    
    # 2. Drone Variable Cost (Round Trip)
    obj_drone = 0
    for i in customers:
        for m in mds:
            d_im = get_dist(customers[i], mds[m]['loc'])
            # Only allow if within range (soft enforce here via cost or constraints)
            obj_drone += 2 * d_im * drone_cost_km * z[i, m]

    # 3. Truck Variable Cost
    obj_truck = gp.quicksum(dist[i, j] * truck_cost_km * x[i, j] for i in node_ids for j in node_ids)

    model.setObjective(obj_fixed + obj_drone + obj_truck, GRB.MINIMIZE)

    # --- Constraints ---

    # 1. Customer Assignment
    for i in customers:
        model.addConstr(gp.quicksum(z[i, m] for m in mds) + v[i] == 1, name=f"Assign_{i}")

    # 2. Drone Range & MD Logic
    for i in customers:
        for m in mds:
            d_im = get_dist(customers[i], mds[m]['loc'])
            if d_im > drone_range:
                model.addConstr(z[i, m] == 0) # Range limit
            else:
                model.addConstr(z[i, m] <= y[m]) # Must be open

    # 3. Truck Routing Validity
    # Each node i must be visited IF it is active in truck route
    # Active nodes: CDC (always), Open MDs, Truck-Customers
    
    # Helper: is_visited[i]
    is_visited = model.addVars(node_ids, vtype=GRB.BINARY, name="IsVisited")
    
    model.addConstr(is_visited[0] == 1) # CDC always visited
    for m in mds:
        model.addConstr(is_visited[m] == y[m]) # Visit MD if open
    for i in customers:
        model.addConstr(is_visited[i] == v[i]) # Visit Cust if truck-assigned

    # Degree constraints based on is_visited
    for i in node_ids:
        model.addConstr(gp.quicksum(x[i, j] for j in node_ids if i!=j) == is_visited[i])
        model.addConstr(gp.quicksum(x[j, i] for j in node_ids if i!=j) == is_visited[i])

    # 4. Subtour Elimination (MTZ)
    M = len(node_ids)
    for i in node_ids:
        if i == 0: continue
        for j in node_ids:
            if j == 0: continue
            model.addConstr(u_mtz[i] - u_mtz[j] + M * x[i, j] <= M - 1)

    # --- Solve ---
    model.optimize()

    # --- Output ---
    if model.status == GRB.OPTIMAL:
        open_mds = [m for m in mds if y[m].X > 0.5]
        drone_custs = [i for i in customers if sum(z[i, m].X for m in mds) > 0.5]
        truck_custs = [i for i in customers if v[i].X > 0.5]
        
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "open_mds": open_mds,
            "truck_route_customers": truck_custs,
            "drone_served_customers": drone_custs
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_drone_fleet_planning())
