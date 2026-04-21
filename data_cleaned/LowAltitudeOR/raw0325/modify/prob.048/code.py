import gurobipy as gp
from gurobipy import GRB
import math

def solve_vertiport_location():
    """
    Multi-objective vertiport location optimization for middle-mile package delivery.
    Minimizes: (1) unserved demand, (2) safety risks, (3) noise nuisance
    """
    
    # ==================== Data Initialization ====================
    
    # Warehouse location
    warehouse = {1: (5, 5)}
    
    # Candidate vertiport zones (zone_id: (x, y, safety_score, noise_score, infrastructure_type))
    vertiport_zones = {
        1: {'loc': (15, 15), 'safety': 0.3, 'noise': 0.2, 'infra': 'medium'},
        3: {'loc': (20, 25), 'safety': 0.2, 'noise': 0.4, 'infra': 'small'},
        4: {'loc': (30, 20), 'safety': 0.6, 'noise': 0.3, 'infra': 'medium'},
        5: {'loc': (10, 25), 'safety': 0.4, 'noise': 0.5, 'infra': 'medium'}
    }
    
    # Customer demand zones (customer_id: (x, y, demand_weight))
    customers = {
        1: {'loc': (18, 18), 'demand': 50},
        2: {'loc': (28, 12), 'demand': 80},
        3: {'loc': (22, 28), 'demand': 60},
        4: {'loc': (32, 22), 'demand': 70},
        5: {'loc': (12, 28), 'demand': 40},
        6: {'loc': (20, 15), 'demand': 90}
    }
    
    # Vertiport types and capacities
    vertiport_types = {
        'small': 288,
        'medium': 1152,
        'large': 4608
    }
    
    # Parameters
    p = 3  # Number of vertiports to select
    f_p = 30  # Maximum drone range (km)
    f_d = 10  # Maximum last-mile distance (km)
    d_min = 2  # Minimum safety distance between vertiports (km)
    d_max = 12  # Maximum safety distance between vertiports (km) — extended from 10
    
    # Sets
    K = list(vertiport_zones.keys())
    W = list(warehouse.keys())
    C = list(customers.keys())
    T = list(vertiport_types.keys())
    
    # Helper function: Euclidean distance
    def dist(loc1, loc2):
        return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    # Precompute coverage matrix v_wck
    v = {}
    for w in W:
        for c in C:
            for k in K:
                w_loc = warehouse[w]
                k_loc = vertiport_zones[k]['loc']
                c_loc = customers[c]['loc']
                
                # Check if vertiport k can serve customer c from warehouse w
                if dist(w_loc, k_loc) <= f_p and dist(k_loc, c_loc) <= f_d:
                    v[w, c, k] = 1
                else:
                    v[w, c, k] = 0
    
    # Precompute distance matrix between vertiport zones
    dist_matrix = {}
    for k1 in K:
        for k2 in K:
            if k1 != k2:
                dist_matrix[k1, k2] = dist(vertiport_zones[k1]['loc'], vertiport_zones[k2]['loc'])
    
    # Filter vertiport types based on infrastructure availability
    available_types = {}
    for k in K:
        infra = vertiport_zones[k]['infra']
        if infra == 'small':
            available_types[k] = ['small']
        elif infra == 'medium':
            available_types[k] = ['small', 'medium']
        else:  # large
            available_types[k] = ['small', 'medium', 'large']
    
    print("=" * 60)
    print("Multi-Objective Vertiport Location Optimization")
    print("=" * 60)
    print(f"\nProblem Parameters:")
    print(f"  Number of vertiports to select: {p}")
    print(f"  Drone range: {f_p} km")
    print(f"  Last-mile distance: {f_d} km")
    print(f"  Safety distance range: [{d_min}, {d_max}] km")
    print(f"\nCandidate zones: {len(K)}")
    print(f"Customer zones: {len(C)}")
    print(f"Total demand: {sum(customers[c]['demand'] for c in C)} packages")
    
    # ==================== Model Building ====================
    
    model = gp.Model("VertiportLocation")
    model.setParam('OutputFlag', 0)
    
    # Decision variables
    x = model.addVars(K, T, vtype=GRB.BINARY, name="x")  # x[k,t]: vertiport type t at zone k
    y = model.addVars(C, vtype=GRB.BINARY, name="y")  # y[c]: customer c is covered
    r = model.addVars(W, C, K, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="r")  # r[w,c,k]: allocation fraction
    
    # Objective 1: Minimize unserved demand (primary objective)
    unserved_demand = gp.quicksum(customers[c]['demand'] * (1 - y[c]) for c in C)
    
    # Objective 2: Minimize safety risk
    safety_risk = gp.quicksum(vertiport_zones[k]['safety'] * x[k, t] for k in K for t in available_types[k])
    
    # Objective 3: Minimize noise nuisance
    noise_nuisance = gp.quicksum(vertiport_zones[k]['noise'] * x[k, t] for k in K for t in available_types[k])
    
    # Multi-objective: weighted sum (prioritize demand coverage)
    model.setObjective(
        1000 * unserved_demand + 10 * safety_risk + 10 * noise_nuisance,
        GRB.MINIMIZE
    )
    
    # ==================== Constraints ====================
    
    # Constraint 1: Select exactly p vertiports
    model.addConstr(
        gp.quicksum(x[k, t] for k in K for t in available_types[k]) == p,
        name="budget"
    )
    
    # Constraint 2: At most one type per zone
    for k in K:
        model.addConstr(
            gp.quicksum(x[k, t] for t in available_types[k]) <= 1,
            name=f"single_type_{k}"
        )
    
    # Constraint 3: Coverage definition
    for w in W:
        for c in C:
            model.addConstr(
                y[c] <= gp.quicksum(x[k, t] * v[w, c, k] for k in K for t in available_types[k]),
                name=f"coverage_{w}_{c}"
            )
    
    # Constraint 4: Demand allocation
    for w in W:
        for c in C:
            for k in K:
                model.addConstr(
                    r[w, c, k] <= gp.quicksum(x[k, t] * v[w, c, k] for t in available_types[k]),
                    name=f"allocation_{w}_{c}_{k}"
                )
    
    # Constraint 5: Coverage requirement
    for w in W:
        for c in C:
            model.addConstr(
                gp.quicksum(r[w, c, k] for k in K) == y[c],
                name=f"coverage_req_{w}_{c}"
            )
    
    # Constraint 6: Capacity constraint
    for k in K:
        model.addConstr(
            gp.quicksum(customers[c]['demand'] * r[w, c, k] for w in W for c in C) <=
            gp.quicksum(vertiport_types[t] * x[k, t] for t in available_types[k]),
            name=f"capacity_{k}"
        )
    
    # Constraint 7: Minimum safety distance
    for k1 in K:
        for k2 in K:
            if k1 < k2 and dist_matrix[k1, k2] < d_min:
                for t1 in available_types[k1]:
                    for t2 in available_types[k2]:
                        model.addConstr(
                            x[k1, t1] + x[k2, t2] <= 1,
                            name=f"min_dist_{k1}_{k2}_{t1}_{t2}"
                        )
    
    # Constraint 8: Maximum safety distance (each vertiport needs at least one neighbor within d_max)
    for k in K:
        neighbors = [k2 for k2 in K if k != k2 and dist_matrix[k, k2] <= d_max]
        if neighbors:
            model.addConstr(
                gp.quicksum(x[k2, t2] for k2 in neighbors for t2 in available_types[k2]) >=
                gp.quicksum(x[k, t] for t in available_types[k]),
                name=f"max_dist_{k}"
            )
    
    # ==================== Solve ====================
    
    print("\n" + "=" * 60)
    print("Solving optimization model...")
    print("=" * 60)
    
    model.optimize()
    
    # ==================== Results ====================
    
    print("\n" + "=" * 60)
    print("Optimization Results")
    print("=" * 60)
    
    if model.status == GRB.OPTIMAL:
        print(f"\nStatus: OPTIMAL")
        print(f"Objective Value: {model.ObjVal:.2f}")
        
        # Extract solution
        selected_vertiports = []
        for k in K:
            for t in available_types[k]:
                if x[k, t].X > 0.5:
                    selected_vertiports.append((k, t))
        
        print(f"\nSelected Vertiports ({len(selected_vertiports)}):")
        for k, t in selected_vertiports:
            loc = vertiport_zones[k]['loc']
            safety = vertiport_zones[k]['safety']
            noise = vertiport_zones[k]['noise']
            capacity = vertiport_types[t]
            print(f"  Zone {k} ({t}): Location {loc}, Safety={safety:.2f}, Noise={noise:.2f}, Capacity={capacity}")
        
        # Calculate objective components
        total_demand = sum(customers[c]['demand'] for c in C)
        served_demand = sum(customers[c]['demand'] * y[c].X for c in C)
        unserved = total_demand - served_demand
        
        total_safety = sum(vertiport_zones[k]['safety'] * x[k, t].X 
                          for k in K for t in available_types[k])
        total_noise = sum(vertiport_zones[k]['noise'] * x[k, t].X 
                         for k in K for t in available_types[k])
        
        print(f"\nObjective Components:")
        print(f"  Unserved Demand: {unserved:.0f} packages ({100*unserved/total_demand:.1f}%)")
        print(f"  Served Demand: {served_demand:.0f} packages ({100*served_demand/total_demand:.1f}%)")
        print(f"  Total Safety Risk: {total_safety:.3f}")
        print(f"  Total Noise Nuisance: {total_noise:.3f}")
        
        print(f"\nCustomer Coverage:")
        for c in C:
            status = "Covered" if y[c].X > 0.5 else "Not Covered"
            demand = customers[c]['demand']
            loc = customers[c]['loc']
            print(f"  Customer {c} (Demand={demand}, Loc={loc}): {status}")
            
            # Show which vertiport serves this customer
            if y[c].X > 0.5:
                for k in K:
                    if sum(r[w, c, k].X for w in W) > 0.01:
                        allocation = sum(r[w, c, k].X for w in W) * 100
                        print(f"    -> Served by Zone {k} ({allocation:.0f}%)")
        
        return {
            "status": "optimal",
            "obj": round(model.ObjVal,2),
            "served_demand": served_demand,
            "total_demand": total_demand,
            "safety_risk": total_safety,
            "noise_nuisance": total_noise,
            "selected_vertiports": selected_vertiports
        }
    
    else:
        print(f"\nStatus: {model.status}")
        if model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS...")
            model.computeIIS()
            print("Infeasible constraints:")
            for c in model.getConstrs():
                if c.IISConstr:
                    print(f"  {c.ConstrName}")
        
        return {"status": "infeasible"}


if __name__ == "__main__":
    result = solve_vertiport_location()
    print("\n" + "=" * 60)
    print(f"Final Result: {result}")
    print("=" * 60)
