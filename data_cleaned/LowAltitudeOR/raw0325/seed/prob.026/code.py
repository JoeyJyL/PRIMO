import gurobipy as gp
from gurobipy import GRB
import math
import json

def solve_drone_latency_location_routing():
    """
    Solve the Drone Latency Location Routing Problem.
    Minimize total customer waiting time (latency) under worst-case flight time uncertainty.
    """
    
    # --- 1. Data Input ---
    
    # Fulfillment Centers (FCs): ID -> (x, y)
    fcs = {
        'FC1': (50, 50),  # Central Hub
        'FC2': (20, 10)   # South Depot
    }
    
    # Customers: ID -> (x, y, demand in kg)
    customers = {
        'C1': (30, 40, 0.5),
        'C2': (60, 30, 0.4),
        'C3': (45, 70, 0.6),
        'C4': (70, 60, 0.5)
    }
    
    # Parameters
    k = 2          # Number of drones
    Q = 2.0        # Drone payload capacity (kg)
    B = 500        # Battery capacity (Wh)
    DS = 2         # Maximum number of FCs to open
    phi = 2        # Max drones per FC
    W = 1.5        # Drone frame weight (kg)
    M_bat = 0.5    # Battery weight (kg)
    k_prime = 0.05 # Energy coefficient
    drone_speed = 40  # units/min
    rho = 1.0      # Robustness parameter
    uncertainty_ratio = 0.2  # 20% flight time uncertainty
    
    n = len(customers)
    m = len(fcs)
    
    # Sets
    C = list(customers.keys())
    D = list(fcs.keys())
    
    # --- 2. Pre-processing: Calculate flight times ---
    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def get_coords(node):
        if node in fcs:
            return fcs[node]
        else:
            return (customers[node][0], customers[node][1])
    
    # Nominal and robust flight times
    t_bar = {}
    t_robust = {}
    for i in D + C:
        for j in D + C:
            if i != j:
                dist = euclidean_distance(get_coords(i), get_coords(j))
                t_bar[i, j] = dist / drone_speed * 60  # in minutes
                t_robust[i, j] = t_bar[i, j] * (1 + rho * uncertainty_ratio)
    
    # Customer demands
    d = {c: customers[c][2] for c in C}
    
    # Big-M parameter
    M_big = 10000
    
    # --- 3. Optimization Model ---
    model = gp.Model("Drone_Latency_Location_Routing")
    model.setParam('TimeLimit', 300)
    
    # Decision Variables
    
    # z[s]: 1 if FC s is opened
    z = model.addVars(D, vtype=GRB.BINARY, name="z")
    
    # x[i,j]: 1 if arc (i,j) is used in some route
    x = model.addVars([(i, j) for i in D + C for j in D + C if i != j], 
                      vtype=GRB.BINARY, name="x")
    
    # a[i]: arrival position of customer i in its route (1 = first customer visited)
    a = model.addVars(C, vtype=GRB.INTEGER, lb=1, ub=n, name="a")
    
    # w[i]: waiting time of customer i
    w = model.addVars(C, vtype=GRB.CONTINUOUS, lb=0, name="w")
    
    # u[i,j]: flow on arc (i,j) - for subtour elimination and route assignment
    u = model.addVars([(i, j) for i in D + C for j in C if i != j],
                      vtype=GRB.CONTINUOUS, lb=0, ub=n, name="u")
    
    # load[i,j]: cargo load on arc (i,j)
    load = model.addVars([(i, j) for i in D + C for j in D + C if i != j],
                         vtype=GRB.CONTINUOUS, lb=0, ub=Q, name="load")
    
    # --- 4. Objective Function: Minimize Total Latency ---
    model.setObjective(gp.quicksum(w[i] for i in C), GRB.MINIMIZE)
    
    # --- 5. Constraints ---
    
    # Each customer visited exactly once
    for j in C:
        model.addConstr(
            gp.quicksum(x[i, j] for i in D + C if i != j) == 1,
            name=f"visit_{j}"
        )
    
    # Flow conservation at customers
    for i in C:
        model.addConstr(
            gp.quicksum(x[j, i] for j in D + C if j != i) == 
            gp.quicksum(x[i, j] for j in D + C if j != i),
            name=f"flow_{i}"
        )
    
    # Exactly k routes start from FCs
    model.addConstr(
        gp.quicksum(x[s, j] for s in D for j in C) == k,
        name="routes_start"
    )
    
    # Exactly k routes return to FCs
    model.addConstr(
        gp.quicksum(x[i, s] for i in C for s in D) == k,
        name="routes_end"
    )
    
    # FC opening constraints
    for s in D:
        for j in C:
            model.addConstr(x[s, j] <= z[s], name=f"fc_open_{s}_{j}")
    
    # Return only to opened FCs
    for i in C:
        for s in D:
            model.addConstr(x[i, s] <= z[s], name=f"fc_return_{i}_{s}")
    
    # Max FCs constraint
    model.addConstr(gp.quicksum(z[s] for s in D) <= DS, name="max_fcs")
    
    # FC capacity constraint
    for s in D:
        model.addConstr(
            gp.quicksum(x[s, j] for j in C) <= phi,
            name=f"fc_cap_{s}"
        )
    
    # Subtour elimination using MTZ-style constraints
    for i in C:
        for j in C:
            if i != j:
                model.addConstr(
                    a[j] >= a[i] + 1 - n * (1 - x[i, j]),
                    name=f"mtz_{i}_{j}"
                )
    
    # Position reset when coming from FC
    for s in D:
        for j in C:
            model.addConstr(
                a[j] <= 1 + n * (1 - x[s, j]),
                name=f"pos_reset_{s}_{j}"
            )
    
    # Waiting time calculation
    for j in C:
        # Time from FC to first customer
        model.addConstr(
            w[j] >= gp.quicksum(t_robust[s, j] * x[s, j] for s in D) +
                    gp.quicksum(t_robust[i, j] * x[i, j] for i in C if i != j),
            name=f"wait_base_{j}"
        )
        
        # Add cumulative time from previous customers
        for i in C:
            if i != j:
                model.addConstr(
                    w[j] >= w[i] + t_robust[i, j] - M_big * (1 - x[i, j]),
                    name=f"wait_cum_{i}_{j}"
                )
    
    # Payload capacity constraints
    for s in D:
        for j in C:
            model.addConstr(load[s, j] <= Q * x[s, j], name=f"load_cap_fc_{s}_{j}")
            model.addConstr(load[s, j] >= d[j] * x[s, j], name=f"load_min_fc_{s}_{j}")
    
    for i in C:
        for j in C:
            if i != j:
                model.addConstr(load[i, j] <= Q * x[i, j], name=f"load_cap_{i}_{j}")
    
    for i in C:
        for s in D:
            model.addConstr(load[i, s] == 0, name=f"load_return_{i}_{s}")
    
    # Load flow conservation
    for i in C:
        model.addConstr(
            gp.quicksum(load[s, i] for s in D) + 
            gp.quicksum(load[j, i] for j in C if j != i) -
            gp.quicksum(load[i, j] for j in C if j != i) -
            gp.quicksum(load[i, s] for s in D) == d[i],
            name=f"load_flow_{i}"
        )
    
    # --- 6. Solve ---
    model.optimize()
    
    # --- 7. Output ---
    if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
        # Extract solution
        opened_fcs = [s for s in D if z[s].X > 0.5]
        
        # Reconstruct routes
        routes = []
        used_arcs = [(i, j) for i in D + C for j in D + C if i != j and x[i, j].X > 0.5]
        
        for s in D:
            for j in C:
                if x[s, j].X > 0.5:
                    route = [s, j]
                    current = j
                    while True:
                        next_node = None
                        for (a_i, a_j) in used_arcs:
                            if a_i == current and a_j not in route:
                                next_node = a_j
                                break
                        if next_node is None or next_node in D:
                            if next_node is not None:
                                route.append(next_node)
                            break
                        route.append(next_node)
                        current = next_node
                    routes.append(route)
        
        # Calculate total latency
        total_latency = sum(w[i].X for i in C)
        
        result = {
            "status": "optimal" if model.status == GRB.OPTIMAL else "feasible",
            "obj": round(total_latency, 2),
            "opened_fcs": opened_fcs,
            "routes": routes
        }
        return result
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    result = solve_drone_latency_location_routing()
    print(json.dumps(result, indent=4))