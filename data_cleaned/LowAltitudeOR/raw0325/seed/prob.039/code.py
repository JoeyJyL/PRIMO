# -*- coding: utf-8 -*-
"""
Drone Delivery Routing Problem - Minimum Cost Implementation
Based on the paper: "Vehicle Routing Problems for Drone Delivery" (Dorling et al., 2017)
"""

import gurobipy as gp
from gurobipy import GRB
import math
import json

def solve_drone_delivery():
    """
    Solve the drone delivery routing problem to minimize total cost
    (drone purchase cost + battery energy cost) while meeting delivery time constraints.
    """
    
    # ========================================
    # 1. PARAMETERS
    # ========================================
    
    # Drone specifications
    F = 500.0          # Drone cost ($)
    Q = 3.0            # Maximum carrying capacity (kg)
    v = 6.0            # Flight speed (m/s)
    tau = 60.0         # Service time per location (seconds)
    W = 1.5            # Frame weight (kg) - not included in capacity
    
    # Energy model parameters (linear approximation)
    alpha = 0.217      # Power per kg (kW/kg)
    beta = 0.185       # Base power (kW)
    
    # Battery specifications
    xi = 650.0         # Energy density (kJ/kg)
    epsilon = 0.1      # Energy cost ($/kJ)
    
    # Operational constraints
    T = 600.0          # Delivery time limit (seconds)
    M_drones = 5       # Maximum number of drones
    
    # Locations: 0 = depot, 1-5 = customers
    locations = {
        0: (0, 0),       # Depot
        1: (100, 200),   # Customer 1
        2: (300, 100),   # Customer 2
        3: (200, 300),   # Customer 3
        4: (400, 250),   # Customer 4
        5: (150, 400)    # Customer 5
    }
    
    # Customer demands (kg)
    demands = {
        1: 0.8,
        2: 1.2,
        3: 0.6,
        4: 1.0,
        5: 0.9
    }
    
    N = list(locations.keys())          # All locations
    N0 = [i for i in N if i != 0]       # Customer locations only
    
    # Calculate distance matrix
    def euclidean_distance(loc1, loc2):
        return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    dist = {}
    for i in N:
        for j in N:
            if i != j:
                dist[i, j] = euclidean_distance(locations[i], locations[j])
    
    # Maximum number of routes (upper bound)
    max_routes = len(N0)  # At most one route per customer
    R = list(range(max_routes))
    
    # Big M for constraints
    K = 10000.0
    
    print("=" * 60)
    print("DRONE DELIVERY ROUTING PROBLEM - COST MINIMIZATION")
    print("=" * 60)
    print(f"\nProblem Parameters:")
    print(f"  Customers: {len(N0)}")
    print(f"  Drone cost: ${F}")
    print(f"  Max capacity: {Q} kg")
    print(f"  Time limit: {T} seconds ({T/60:.1f} minutes)")
    print(f"  Max drones: {M_drones}")
    
    # ========================================
    # 2. CREATE MODEL
    # ========================================
    
    model = gp.Model("DroneDelivery_MinCost")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 300)  # 5 minute time limit
    
    # ========================================
    # 3. DECISION VARIABLES
    # ========================================
    
    # x[i,j,r] = 1 if route r goes from location i to j
    x = model.addVars(N, N, R, vtype=GRB.BINARY, name="x")
    
    # u[r] = 1 if route r is used
    u = model.addVars(R, vtype=GRB.BINARY, name="u")
    
    # y[i,r] = payload weight leaving location i on route r
    y = model.addVars(N, R, vtype=GRB.CONTINUOUS, lb=0, name="y")
    
    # q[r] = battery weight for route r
    q = model.addVars(R, vtype=GRB.CONTINUOUS, lb=0, name="q")
    
    # z[r] = energy consumed on route r
    z = model.addVars(R, vtype=GRB.CONTINUOUS, lb=0, name="z")
    
    # t[i,r] = time when location i is visited on route r
    t = model.addVars(N, R, vtype=GRB.CONTINUOUS, lb=0, name="t")
    
    # a[r] = arrival time at depot after completing route r
    a = model.addVars(R, vtype=GRB.CONTINUOUS, lb=0, name="a")
    
    # n = number of drones to purchase
    n = model.addVar(vtype=GRB.INTEGER, lb=1, ub=M_drones, name="n")
    
    # ========================================
    # 4. OBJECTIVE FUNCTION
    # ========================================
    
    # Minimize: drone cost + energy cost
    model.setObjective(
        F * n + epsilon * gp.quicksum(z[r] for r in R),
        GRB.MINIMIZE
    )
    
    # ========================================
    # 5. CONSTRAINTS
    # ========================================
    
    # 5.1 Each customer visited exactly once
    for i in N0:
        model.addConstr(
            gp.quicksum(x[i, j, r] for j in N if j != i for r in R) == 1,
            name=f"visit_{i}"
        )
    
    # 5.2 Flow conservation
    for i in N:
        for r in R:
            model.addConstr(
                gp.quicksum(x[i, j, r] for j in N if j != i) ==
                gp.quicksum(x[j, i, r] for j in N if j != i),
                name=f"flow_{i}_{r}"
            )
    
    # 5.3 Route starts at depot
    for r in R:
        model.addConstr(
            gp.quicksum(x[0, j, r] for j in N0) == u[r],
            name=f"start_{r}"
        )
    
    # 5.4 Route ends at depot
    for r in R:
        model.addConstr(
            gp.quicksum(x[i, 0, r] for i in N0) == u[r],
            name=f"end_{r}"
        )
    
    # 5.5 Payload weight tracking
    for r in R:
        # Payload when leaving depot equals sum of demands on route
        model.addConstr(
            y[0, r] == gp.quicksum(demands[j] * x[0, j, r] for j in N0),
            name=f"payload_depot_{r}"
        )
        
        # Payload decreases by demand at each customer
        for i in N0:
            for j in N0:
                if i != j:
                    model.addConstr(
                        y[i, r] >= y[j, r] + demands[j] * x[i, j, r] - K * (1 - x[i, j, r]),
                        name=f"payload_{i}_{j}_{r}"
                    )
    
    # 5.6 Capacity constraint
    for r in R:
        for i in N:
            model.addConstr(
                q[r] + y[i, r] <= Q * u[r],
                name=f"capacity_{i}_{r}"
            )
    
    # 5.7 Energy consumption (simplified linear model)
    for r in R:
        # Energy = sum over all edges of: power * time
        # Power = alpha * (battery + payload) + beta
        # Time = distance/speed + service_time
        energy_expr = gp.LinExpr()
        for i in N:
            for j in N:
                if i != j:
                    travel_time = dist[i, j] / v + tau
                    # Use average payload on edge (approximation)
                    avg_payload = (y[i, r] + y[j, r]) / 2 if j != 0 else y[i, r] / 2
                    power = alpha * (q[r] + avg_payload) + beta
                    energy_expr += power * travel_time * x[i, j, r]
        
        model.addConstr(z[r] == energy_expr, name=f"energy_{r}")
    
    # 5.8 Battery provides enough energy
    for r in R:
        model.addConstr(
            xi * q[r] >= z[r],
            name=f"battery_{r}"
        )
    
    # 5.9 Time tracking
    for r in R:
        for i in N:
            for j in N0:
                if i != j:
                    travel_time = dist[i, j] / v + tau
                    model.addConstr(
                        t[j, r] >= t[i, r] + travel_time - K * (1 - x[i, j, r]),
                        name=f"time_{i}_{j}_{r}"
                    )
    
    # 5.10 Arrival time at depot
    for r in R:
        for i in N0:
            travel_time = dist[i, 0] / v + tau
            model.addConstr(
                a[r] >= t[i, r] + travel_time - K * (1 - x[i, 0, r]),
                name=f"arrival_{i}_{r}"
            )
    
    # 5.11 Time limit constraint
    for r in R:
        model.addConstr(a[r] <= T + K * (1 - u[r]), name=f"timelimit_{r}")
    
    # 5.12 Drone count constraint (simplified - assumes sequential execution)
    # This is a simplification; proper implementation would need scheduling
    model.addConstr(
        n >= gp.quicksum(u[r] for r in R),
        name="drone_count"
    )
    
    # 5.13 Route ordering (symmetry breaking)
    for r in range(len(R) - 1):
        model.addConstr(u[r] >= u[r + 1], name=f"order_{r}")
    
    # ========================================
    # 6. SOLVE
    # ========================================
    
    print("\nSolving optimization model...")
    model.optimize()
    
    # ========================================
    # 7. EXTRACT AND DISPLAY RESULTS
    # ========================================
    
    result = {"status": "unknown", "obj": None}
    
    if model.status == GRB.OPTIMAL:
        result["status"] = "optimal"
        result["obj"] = round(model.ObjVal, 2)
        
        print("\n" + "=" * 60)
        print("OPTIMAL SOLUTION FOUND")
        print("=" * 60)
        print(f"\nTotal Cost: ${model.ObjVal:.2f}")
        
        num_drones = int(n.X + 0.5)
        drone_cost = F * num_drones
        energy_cost = sum(z[r].X for r in R) * epsilon
        
        print(f"  Drone Cost: ${drone_cost:.2f} ({num_drones} drones)")
        print(f"  Energy Cost: ${energy_cost:.2f}")
        
        # Extract routes
        active_routes = [r for r in R if u[r].X > 0.5]
        print(f"\nNumber of Routes: {len(active_routes)}")
        
        for r in active_routes:
            route = [0]  # Start at depot
            current = 0
            visited = set([0])
            
            # Build route by following edges
            while True:
                next_loc = None
                for j in N:
                    if j not in visited and x[current, j, r].X > 0.5:
                        next_loc = j
                        break
                
                if next_loc is None or next_loc == 0:
                    break
                
                route.append(next_loc)
                visited.add(next_loc)
                current = next_loc
            
            route.append(0)  # Return to depot
            
            # Calculate route statistics
            route_distance = sum(dist[route[i], route[i+1]] for i in range(len(route)-1))
            route_time = a[r].X
            route_payload = sum(demands[i] for i in route if i != 0)
            route_battery = q[r].X
            route_energy = z[r].X
            
            print(f"\n  Route {r+1}: {' -> '.join(map(str, route))}")
            print(f"    Customers: {[i for i in route if i != 0]}")
            print(f"    Total payload: {route_payload:.2f} kg")
            print(f"    Battery weight: {route_battery:.3f} kg")
            print(f"    Total weight: {route_payload + route_battery:.3f} kg (capacity: {Q} kg)")
            print(f"    Distance: {route_distance:.1f} m")
            print(f"    Time: {route_time:.1f} s ({route_time/60:.2f} min)")
            print(f"    Energy: {route_energy:.2f} kJ (${route_energy * epsilon:.2f})")
        
    elif model.status == GRB.INFEASIBLE:
        result["status"] = "infeasible"
        print("\n" + "=" * 60)
        print("MODEL IS INFEASIBLE")
        print("=" * 60)
        print("The problem has no feasible solution with the given constraints.")
        
    elif model.status == GRB.TIME_LIMIT:
        result["status"] = "time_limit"
        if model.SolCount > 0:
            result["obj"] = round(model.ObjVal, 2)
            print("\n" + "=" * 60)
            print("TIME LIMIT REACHED - BEST SOLUTION FOUND")
            print("=" * 60)
            print(f"\nBest Cost Found: ${model.ObjVal:.2f}")
        else:
            print("\n" + "=" * 60)
            print("TIME LIMIT REACHED - NO SOLUTION FOUND")
            print("=" * 60)
    else:
        result["status"] = f"status_{model.status}"
        print(f"\nSolver Status: {model.status}")
    
    return result


if __name__ == "__main__":
    result = solve_drone_delivery()
    
    # Save result to JSON
    print("\n" + "=" * 50)
    print(f"Result: {result}")

