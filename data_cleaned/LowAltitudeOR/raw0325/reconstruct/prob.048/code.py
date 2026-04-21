import gurobipy as gp
from gurobipy import GRB
import math

def solve_vertiport_location():
    """
    Multi-objective vertiport location optimization for middle-mile package delivery.
    Binary exclusive assignment formulation (replaces continuous fractional allocation).
    """
    
    # ==================== Data Initialization ====================
    
    # Warehouse location
    warehouse = {1: (5, 5)}
    
    # Candidate vertiport zones
    vertiport_zones = {
        1: {'loc': (15, 15), 'safety': 0.3, 'noise': 0.2, 'infra': 'medium'},
        3: {'loc': (20, 25), 'safety': 0.2, 'noise': 0.4, 'infra': 'small'},
        4: {'loc': (30, 20), 'safety': 0.6, 'noise': 0.3, 'infra': 'medium'},
        5: {'loc': (10, 25), 'safety': 0.4, 'noise': 0.5, 'infra': 'medium'}
    }
    
    # Customer demand zones
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
    p = 3
    f_p = 30
    f_d = 10
    d_min = 2
    d_max = 10
    
    # Sets
    K = list(vertiport_zones.keys())
    W = list(warehouse.keys())
    C = list(customers.keys())
    T = list(vertiport_types.keys())
    
    def dist(loc1, loc2):
        return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    # Precompute coverage matrix
    v = {}
    for w in W:
        for c in C:
            for k in K:
                w_loc = warehouse[w]
                k_loc = vertiport_zones[k]['loc']
                c_loc = customers[c]['loc']
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
    
    # Filter vertiport types based on infrastructure
    available_types = {}
    for k in K:
        infra = vertiport_zones[k]['infra']
        if infra == 'small':
            available_types[k] = ['small']
        elif infra == 'medium':
            available_types[k] = ['small', 'medium']
        else:
            available_types[k] = ['small', 'medium', 'large']
    
    print("=" * 60)
    print("Vertiport Location (Binary Exclusive Assignment Formulation)")
    print("=" * 60)
    print(f"\nProblem Parameters:")
    print(f"  Number of vertiports to select: {p}")
    print(f"  Drone range: {f_p} km, Last-mile: {f_d} km")
    print(f"  Safety distance range: [{d_min}, {d_max}] km")
    print(f"  Candidate zones: {len(K)}, Customer zones: {len(C)}")
    print(f"  Total demand: {sum(customers[c]['demand'] for c in C)} packages")
    
    # ==================== Model Building ====================
    
    model = gp.Model("VertiportLocation_BinaryAssignment")
    model.setParam('OutputFlag', 0)
    
    # Decision variables
    x = model.addVars(K, T, vtype=GRB.BINARY, name="x")
    y = model.addVars(C, vtype=GRB.BINARY, name="y")
    # KEY CHANGE: binary exclusive assignment instead of continuous fractional allocation
    z = model.addVars(C, K, vtype=GRB.BINARY, name="z")
    
    # Objective: weighted sum
    unserved_demand = gp.quicksum(customers[c]['demand'] * (1 - y[c]) for c in C)
    safety_risk = gp.quicksum(vertiport_zones[k]['safety'] * x[k, t] for k in K for t in available_types[k])
    noise_nuisance = gp.quicksum(vertiport_zones[k]['noise'] * x[k, t] for k in K for t in available_types[k])
    
    model.setObjective(
        1000 * unserved_demand + 10 * safety_risk + 10 * noise_nuisance,
        GRB.MINIMIZE
    )
    
    # ==================== Constraints ====================
    
    # C1: Select exactly p vertiports
    model.addConstr(
        gp.quicksum(x[k, t] for k in K for t in available_types[k]) == p,
        name="budget"
    )
    
    # C2: At most one type per zone
    for k in K:
        model.addConstr(
            gp.quicksum(x[k, t] for t in available_types[k]) <= 1,
            name=f"single_type_{k}"
        )
    
    # C3: Coverage definition
    for w in W:
        for c in C:
            model.addConstr(
                y[c] <= gp.quicksum(x[k, t] * v[w, c, k] for k in K for t in available_types[k]),
                name=f"coverage_{w}_{c}"
            )
    
    # C4: Binary assignment — can only assign to reachable selected vertiport
    for w in W:
        for c in C:
            for k in K:
                model.addConstr(
                    z[c, k] <= gp.quicksum(x[k, t] * v[w, c, k] for t in available_types[k]),
                    name=f"assign_{w}_{c}_{k}"
                )
    
    # C5: Exclusive assignment — each covered customer assigned to exactly one vertiport
    for c in C:
        model.addConstr(
            gp.quicksum(z[c, k] for k in K) == y[c],
            name=f"exclusive_{c}"
        )
    
    # C6: Capacity constraint (binary assignment)
    for k in K:
        model.addConstr(
            gp.quicksum(customers[c]['demand'] * z[c, k] for c in C) <=
            gp.quicksum(vertiport_types[t] * x[k, t] for t in available_types[k]),
            name=f"capacity_{k}"
        )
    
    # C7: Minimum safety distance
    for k1 in K:
        for k2 in K:
            if k1 < k2 and dist_matrix[k1, k2] < d_min:
                for t1 in available_types[k1]:
                    for t2 in available_types[k2]:
                        model.addConstr(
                            x[k1, t1] + x[k2, t2] <= 1,
                            name=f"min_dist_{k1}_{k2}_{t1}_{t2}"
                        )
    
    # C8: Maximum safety distance
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
        
        print(f"\nCustomer Coverage (Binary Exclusive Assignment):")
        for c in C:
            status = "Covered" if y[c].X > 0.5 else "Not Covered"
            demand = customers[c]['demand']
            loc = customers[c]['loc']
            print(f"  Customer {c} (Demand={demand}, Loc={loc}): {status}")
            if y[c].X > 0.5:
                for k in K:
                    if z[c, k].X > 0.5:
                        print(f"    -> Exclusively assigned to Zone {k}")
        
        return {
            "status": "optimal",
            "obj": round(model.ObjVal, 2),
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
            for c in model.getConstrs():
                if c.IISConstr:
                    print(f"  {c.ConstrName}")
        return {"status": "infeasible"}


if __name__ == "__main__":
    result = solve_vertiport_location()
    print("\n" + "=" * 60)
    print(f"Final Result: {result}")
    print("=" * 60)
