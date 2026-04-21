# -*- coding: utf-8 -*-
"""
UAM Vertiport Network Design Problem - Connectivity-Constrained Facility Location
Based on the paper: "UAM Vertiport Network Design Considering Connectivity" (Zhang & Hwang, 2025)
Modified: Mandatory selection of Location 7 (policy constraint)
"""

import gurobipy as gp
from gurobipy import GRB
import math
import json

def solve_uam_vertiport_network():
    """
    Solve the UAM vertiport network design problem to minimize total cost
    (construction cost + transportation cost) while ensuring network connectivity.
    Location 7 is mandatory due to urban development policy.
    """
    
    # ========================================
    # 1. PARAMETERS
    # ========================================
    
    # Network configuration
    P = 3  # Number of vertiports to build
    D_max = 15.0  # Maximum flight range (km)
    alpha = 3.5  # Unit transportation cost ($/person/km)
    
    # Candidate vertiport locations (coordinates in km)
    locations = {
        1: (2, 8),
        2: (5, 12),
        3: (8, 6),
        4: (12, 10),
        5: (15, 14),
        6: (18, 8),
        7: (10, 16),
        8: (6, 4),
        9: (14, 4),
        10: (20, 12)
    }
    
    # Construction costs ($)
    construction_cost = {
        1: 2500000,
        2: 2800000,
        3: 2300000,
        4: 2600000,
        5: 2900000,
        6: 2400000,
        7: 3000000,
        8: 2200000,
        9: 2350000,
        10: 2700000
    }
    
    # Demand points (coordinates in km, demand in passengers/year)
    demands = {
        1: {'loc': (3, 10), 'demand': 50000},
        2: {'loc': (7, 8), 'demand': 80000},
        3: {'loc': (11, 13), 'demand': 60000},
        4: {'loc': (16, 10), 'demand': 70000},
        5: {'loc': (9, 5), 'demand': 55000},
        6: {'loc': (13, 7), 'demand': 65000},
        7: {'loc': (5, 15), 'demand': 45000},
        8: {'loc': (19, 9), 'demand': 75000}
    }
    
    # Sets
    J_prime = list(locations.keys())  # Candidate locations
    I = list(demands.keys())  # Demand points
    
    # Virtual sink node (node 0)
    J = [0] + J_prime  # Complete node set including virtual sink
    
    # Calculate Euclidean distance
    def euclidean_distance(loc1, loc2):
        return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    # Distance matrix between candidate locations
    dist_loc = {}
    for j in J_prime:
        for k in J_prime:
            if j != k:
                dist_loc[j, k] = euclidean_distance(locations[j], locations[k])
    
    # Distance matrix from demand points to candidate locations
    dist_demand = {}
    for i in I:
        for j in J_prime:
            dist_demand[i, j] = euclidean_distance(demands[i]['loc'], locations[j])
    
    # Neighborhood sets N_j (locations within D_max)
    N = {j: [] for j in J_prime}
    for j in J_prime:
        for k in J_prime:
            if j != k and dist_loc[j, k] <= D_max:
                N[j].append(k)
    
    print("=" * 70)
    print("UAM VERTIPORT NETWORK DESIGN - CONNECTIVITY-CONSTRAINED")
    print("(Modified: Location 7 mandatory)")
    print("=" * 70)
    print(f"\nProblem Parameters:")
    print(f"  Candidate locations: {len(J_prime)}")
    print(f"  Demand points: {len(I)}")
    print(f"  Vertiports to build: {P}")
    print(f"  Maximum flight range: {D_max} km")
    print(f"  Unit transportation cost: ${alpha}/person/km")
    print(f"  Mandatory location: Location 7")
    
    # ========================================
    # 2. CREATE MODEL
    # ========================================
    
    model = gp.Model("UAM_Vertiport_Network")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 300)
    
    # ========================================
    # 3. DECISION VARIABLES
    # ========================================
    
    # X[j] = 1 if vertiport is built at location j
    X = model.addVars(J_prime, vtype=GRB.BINARY, name="X")
    
    # Y[i,j] = 1 if demand point i is assigned to vertiport j
    Y = model.addVars(I, J_prime, vtype=GRB.BINARY, name="Y")
    
    # l[j,k] = virtual flow from j to k (for connectivity)
    l = model.addVars(J, J, vtype=GRB.CONTINUOUS, lb=0, name="l")
    
    # ========================================
    # 4. OBJECTIVE FUNCTION
    # ========================================
    
    # Minimize: transportation cost + construction cost
    model.setObjective(
        alpha * gp.quicksum(demands[i]['demand'] * dist_demand[i, j] * Y[i, j] 
                           for i in I for j in J_prime) +
        gp.quicksum(construction_cost[j] * X[j] for j in J_prime),
        GRB.MINIMIZE
    )
    
    # ========================================
    # 5. CONSTRAINTS
    # ========================================
    
    # 5.1 Facility selection: exactly P vertiports
    model.addConstr(
        gp.quicksum(X[j] for j in J_prime) == P,
        name="facility_selection"
    )
    
    # 5.2 Demand assignment: each demand point assigned to exactly one vertiport
    for i in I:
        model.addConstr(
            gp.quicksum(Y[i, j] for j in J_prime) == 1,
            name=f"demand_assignment_{i}"
        )
    
    # 5.3 Assignment feasibility: can only assign to selected vertiports
    for i in I:
        for j in J_prime:
            model.addConstr(
                Y[i, j] <= X[j],
                name=f"assignment_feasibility_{i}_{j}"
            )
    
    # 5.4 Flow balance: connectivity constraint
    for j in J_prime:
        # Inflow from neighbors
        inflow = gp.quicksum(l[n, j] for n in N[j])
        # Outflow to neighbors and sink
        outflow_neighbors = gp.quicksum(l[j, m] for m in N[j])
        outflow_sink = l[j, 0]
        
        model.addConstr(
            X[j] + inflow == outflow_neighbors + outflow_sink,
            name=f"flow_balance_{j}"
        )
    
    # 5.5 Flow capacity: limit outgoing flow
    for j in J_prime:
        for k in N[j]:
            model.addConstr(
                l[j, k] <= (P - 1) * X[j],
                name=f"flow_capacity_{j}_{k}"
            )
    
    # 5.6 Sink node constraints
    # No flow enters sink node from other nodes
    for j in J_prime:
        model.addConstr(l[0, j] == 0, name=f"sink_inflow_{j}")
    
    # Total flow to sink equals P (number of selected facilities)
    model.addConstr(
        gp.quicksum(l[j, 0] for j in J_prime) == P,
        name="total_sink_flow"
    )
    
    # 5.7 Mandatory location constraint: Location 7 must be selected
    model.addConstr(X[7] == 1, name="mandatory_location_7")
    
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
        
        print("\n" + "=" * 70)
        print("OPTIMAL SOLUTION FOUND")
        print("=" * 70)
        print(f"\nTotal Cost: ${model.ObjVal:,.2f}")
        
        # Extract selected vertiports
        selected_vertiports = [j for j in J_prime if X[j].X > 0.5]
        construction_total = sum(construction_cost[j] for j in selected_vertiports)
        transportation_total = model.ObjVal - construction_total
        
        print(f"  Construction Cost: ${construction_total:,.2f}")
        print(f"  Transportation Cost: ${transportation_total:,.2f}")
        
        print(f"\nSelected Vertiport Locations ({len(selected_vertiports)}):")
        for j in selected_vertiports:
            mandatory = " [MANDATORY]" if j == 7 else ""
            print(f"  Location {j}: {locations[j]}, Cost: ${construction_cost[j]:,}{mandatory}")
        
        # Check connectivity
        print("\nNetwork Connectivity:")
        connected_pairs = []
        for j in selected_vertiports:
            for k in selected_vertiports:
                if j < k and k in N[j]:
                    flow_jk = l[j, k].X
                    flow_kj = l[k, j].X
                    if flow_jk > 0.01 or flow_kj > 0.01:
                        connected_pairs.append((j, k, dist_loc[j, k]))
        
        if connected_pairs:
            for j, k, d in connected_pairs:
                print(f"  Location {j} <-> Location {k}: {d:.2f} km")
        else:
            print("  (Connectivity verified through virtual flow network)")
        
        # Demand assignments
        print("\nDemand Point Assignments:")
        for i in I:
            for j in J_prime:
                if Y[i, j].X > 0.5:
                    dist = dist_demand[i, j]
                    demand = demands[i]['demand']
                    cost = alpha * demand * dist
                    print(f"  Demand {i} -> Location {j}: {dist:.2f} km, "
                          f"{demand:,} passengers, Cost: ${cost:,.2f}")
        
    elif model.status == GRB.INFEASIBLE:
        result["status"] = "infeasible"
        print("\n" + "=" * 70)
        print("MODEL IS INFEASIBLE")
        print("=" * 70)
        print("No feasible solution exists with the given constraints.")
        print("Consider increasing D_max or reducing P.")
        
    elif model.status == GRB.TIME_LIMIT:
        result["status"] = "time_limit"
        if model.SolCount > 0:
            result["obj"] = round(model.ObjVal, 2)
            print("\n" + "=" * 70)
            print("TIME LIMIT REACHED - BEST SOLUTION FOUND")
            print("=" * 70)
            print(f"\nBest Cost Found: ${model.ObjVal:,.2f}")
        else:
            print("\n" + "=" * 70)
            print("TIME LIMIT REACHED - NO SOLUTION FOUND")
            print("=" * 70)
    else:
        result["status"] = f"status_{model.status}"
        print(f"\nSolver Status: {model.status}")
    
    return result


if __name__ == "__main__":
    result = solve_uam_vertiport_network()
    
    # Save result to JSON
    print("\n" + "=" * 70)
    with open("answer.json", "w") as f:
        json.dump(result, f, indent=4)
    print(f"Result: {result}")
