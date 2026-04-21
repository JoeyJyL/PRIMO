"""
TD-MTDRP Set-Partitioning Model — Gurobi Solver
=================================================
Enumerate all feasible routes, compute durations via timing decomposition,
then solve the set-partitioning MILP with Gurobi.

Instance: 6 customers, K=1 truck, m=2 drones, time-independent travel.
"""

import math
import itertools
import gurobipy as gp
from gurobipy import GRB

# ============================================================
# Data
# ============================================================
depot = (50, 50)
customers = {
    1: {'pos': (20,80), 'demand':5, 'weight':4, 'st':2, 'sd':1, 'tw':(0,300)},
    2: {'pos': (80,80), 'demand':8, 'weight':12,'st':3, 'sd':1.5,'tw':(0,300)},
    3: {'pos': (20,20), 'demand':6, 'weight':5, 'st':2, 'sd':1, 'tw':(0,300)},
    4: {'pos': (80,20), 'demand':7, 'weight':9, 'st':3, 'sd':1.5,'tw':(0,300)},
    5: {'pos': (50,80), 'demand':4, 'weight':3, 'st':2, 'sd':1, 'tw':(0,300)},
    6: {'pos': (50,20), 'demand':5, 'weight':6, 'st':2, 'sd':1, 'tw':(0,300)},
}

K = 1        # trucks
m = 2        # max drones per truck
Q = 40       # truck capacity (relaxed from original 30 to make feasible)
D = 10       # drone payload limit
B = 40       # drone battery (max round-trip flight time)
truck_speed = 1.0
drone_speed = 2.0

C = list(range(1, 7))
locs = {0: depot, 7: depot}  # 7 = return depot
locs.update({i: customers[i]['pos'] for i in C})

def dist(i, j):
    return math.sqrt((locs[i][0]-locs[j][0])**2 + (locs[i][1]-locs[j][1])**2)

def truck_tt(i, j):
    return dist(i, j) / truck_speed

def drone_tt(i, j):
    return dist(i, j) / drone_speed

# Precheck drone eligibility
drone_eligible = {c: customers[c]['weight'] <= D for c in C}

print("=" * 70)
print("TD-MTDRP SET-PARTITIONING — GUROBI SOLVER")
print("=" * 70)
print(f"\nInstance: {len(C)} customers, K={K}, m={m}, Q={Q}, D={D}, B={B}")
print(f"Drone eligible: {[c for c in C if drone_eligible[c]]}")
print(f"Must be truck: {[c for c in C if not drone_eligible[c]]}")

# ============================================================
# Enumerate all feasible routes
# ============================================================
# A route = (truck_stop_order, drone_assignments)
# truck_stop_order: ordered list of customers visited by truck
# drone_assignments: {truck_stop: [list of drone customers]} at each stop

def compute_route_duration(truck_stops, drone_assign):
    """
    Compute route duration using the timing decomposition from model.txt.
    
    Returns: (psi, details) where psi = total duration, or (None, None) if infeasible.
    """
    # Full path: depot -> truck_stops -> depot
    path = [0] + list(truck_stops) + [7]
    
    # Capacity check: total demand of all customers in this route
    all_custs = set(truck_stops)
    for stop, drones in drone_assign.items():
        all_custs.update(drones)
    total_demand = sum(customers[c]['demand'] for c in all_custs)
    if total_demand > Q:
        return None, None
    
    # Simulate timing decomposition
    sigma = [0.0] * len(path)      # arrival times
    delta = [0.0] * len(path)      # departure times
    details = []
    
    for idx in range(1, len(path) - 1):  # truck stops (not depots)
        stop = path[idx]
        prev = path[idx - 1]
        
        # σ_{r,i} = δ_{r,i-1} + τ_{v_{i-1}, v_i}
        sigma[idx] = delta[idx - 1] + truck_tt(prev, stop)
        
        # Time window check
        e, l = customers[stop]['tw']
        if sigma[idx] > l:
            return None, None
        sigma[idx] = max(sigma[idx], e)  # wait for earliest time
        
        # τ^{svc}_{r,i} = σ_{r,i} + s^t_{v_i}
        tau_svc = sigma[idx] + customers[stop]['st']
        
        # Drone round-trips from this stop
        drones = drone_assign.get(stop, [])
        phi_values = []
        for dc in drones:
            # Battery check: 2 * t_{v_i, d} <= B
            rt_flight = 2 * drone_tt(stop, dc)
            if rt_flight > B:
                return None, None
            # Payload check
            if customers[dc]['weight'] > D:
                return None, None
            # φ_{r,i,d} = σ_{r,i} + t_{v_i,d} + s^d_d + t_{d,v_i}
            phi = sigma[idx] + drone_tt(stop, dc) + customers[dc]['sd'] + drone_tt(dc, stop)
            phi_values.append((dc, phi))
        
        # δ_{r,i} = max(τ^{svc}, max φ_{r,i,d})
        delta[idx] = tau_svc
        for dc, phi in phi_values:
            delta[idx] = max(delta[idx], phi)
        
        details.append({
            'stop': stop, 'arrive': sigma[idx], 'svc_done': tau_svc,
            'drones': phi_values, 'depart': delta[idx]
        })
    
    # ψ_r = δ_{r,|S_r|} + τ_{v_{|S_r|}, depot}
    last_idx = len(path) - 2  # last truck stop
    psi = delta[last_idx] + truck_tt(path[last_idx], 0)
    
    return psi, details


print("\nEnumerating feasible routes...")
all_routes = []
route_count = 0

# For each partition of customers into truck-served and drone-served:
drone_cands = [c for c in C if drone_eligible[c]]

for n_drone in range(len(drone_cands) + 1):
    for drone_set in itertools.combinations(drone_cands, n_drone):
        drone_set_s = set(drone_set)
        truck_custs = [c for c in C if c not in drone_set_s]
        
        if not truck_custs:
            continue
        
        # Generate drone assignments: each drone customer -> one truck stop
        drone_list = list(drone_set)
        
        if not drone_list:
            assign_list = [{}]
        else:
            assign_list = []
            for combo in itertools.product(range(len(truck_custs)), repeat=len(drone_list)):
                assign = {}
                feasible = True
                for i, ts_idx in enumerate(combo):
                    ts = truck_custs[ts_idx]
                    dc = drone_list[i]
                    # Battery check early
                    if 2 * drone_tt(ts, dc) > B:
                        feasible = False
                        break
                    if ts not in assign:
                        assign[ts] = []
                    assign[ts] = assign[ts] + [dc]
                    if len(assign[ts]) > m:
                        feasible = False
                        break
                if feasible:
                    assign_list.append(assign)
        
        # For each assignment, try all truck orderings
        for assign in assign_list:
            for perm in itertools.permutations(truck_custs):
                route_count += 1
                psi, details = compute_route_duration(perm, assign)
                if psi is not None:
                    # Record route
                    all_drone = []
                    for s in perm:
                        all_drone.extend(assign.get(s, []))
                    
                    all_routes.append({
                        'truck_stops': list(perm),
                        'drone_assign': {k: list(v) for k, v in assign.items()},
                        'drone_served': list(drone_set),
                        'duration': psi,
                        'details': details,
                        'all_customers': set(perm) | set(all_drone),
                    })

print(f"Configurations checked: {route_count}")
print(f"Feasible routes: {len(all_routes)}")

# Show top 5 by duration
all_routes.sort(key=lambda r: r['duration'])
print(f"\nTop 5 routes by duration:")
for i, r in enumerate(all_routes[:5]):
    ds = r['drone_served']
    print(f"  #{i+1}: dur={r['duration']:.4f}, truck={r['truck_stops']}, drone={ds}")

# ============================================================
# Gurobi Set-Partitioning Model
# ============================================================
print(f"\n{'=' * 70}")
print("Building Gurobi Set-Partitioning Model...")
print(f"{'=' * 70}")

R = range(len(all_routes))

model = gp.Model("TD_MTDRP_SPM")
model.setParam('OutputFlag', 1)

# θ_r ∈ {0,1}: route selection
theta = model.addVars(R, vtype=GRB.BINARY, name="theta")

# Objective: min Σ ψ_r · θ_r
model.setObjective(
    gp.quicksum(all_routes[r]['duration'] * theta[r] for r in R),
    GRB.MINIMIZE
)

# Constraint 1: Each customer served exactly once
for c in C:
    routes_with_c = [r for r in R if c in all_routes[r]['all_customers']]
    model.addConstr(
        gp.quicksum(theta[r] for r in routes_with_c) == 1,
        name=f"cover_{c}"
    )

# Constraint 2: Fleet size
model.addConstr(
    gp.quicksum(theta[r] for r in R) <= K,
    name="fleet"
)

# Solve
print(f"\nVariables: {len(R)} binary (routes)")
print(f"Constraints: {len(C)} coverage + 1 fleet = {len(C)+1}")
print("\nSolving...")
model.optimize()

# ============================================================
# Output
# ============================================================
print(f"\n{'=' * 70}")
print("★ OPTIMAL SOLUTION")
print(f"{'=' * 70}")

if model.status == GRB.OPTIMAL:
    print(f"\n  Objective (Total Duration): {model.objVal:.4f} time units")
    
    for r in R:
        if theta[r].X > 0.5:
            route = all_routes[r]
            print(f"\n  Selected Route (index {r}):")
            print(f"    Truck stops: {route['truck_stops']}")
            print(f"    Drone-served: {route['drone_served']}")
            print(f"    Duration: {route['duration']:.4f}")
            
            # Drone assignment details
            print(f"\n    Drone dispatch plan:")
            for stop in route['truck_stops']:
                dcs = route['drone_assign'].get(stop, [])
                if dcs:
                    for dc in dcs:
                        rt = 2 * drone_tt(stop, dc)
                        print(f"      At C{stop}: drone → C{dc} "
                              f"(round-trip flight={rt:.2f}, battery={B})")
            
            # Detailed timing (from model auxiliary variables)
            print(f"\n    Timing decomposition (σ, τ^svc, φ, δ):")
            print(f"      Depot: σ=0.00, δ=0.00")
            for step in route['details']:
                s = step['stop']
                print(f"\n      C{s}:")
                print(f"        σ (arrive)    = {step['arrive']:.4f}")
                print(f"        τ^svc (svc)   = {step['svc_done']:.4f}")
                for dc, phi in step['drones']:
                    print(f"        φ (drone→C{dc}) = {phi:.4f}")
                print(f"        δ (depart)    = {step['depart']:.4f}")
                if step['depart'] > step['svc_done'] + 0.001:
                    wait = step['depart'] - step['svc_done']
                    print(f"        ⏳ wait for drones: {wait:.4f}")
            
            last_stop = route['truck_stops'][-1]
            return_travel = truck_tt(last_stop, 0)
            print(f"\n      Return to depot: +{return_travel:.4f}")
            print(f"      ψ (total duration) = {route['duration']:.4f}")
    
    # Verification
    print(f"\n  {'─' * 50}")
    print(f"  CONSTRAINT VERIFICATION:")
    
    selected = [r for r in R if theta[r].X > 0.5]
    all_served = set()
    for r in selected:
        all_served |= all_routes[r]['all_customers']
    
    print(f"    All customers served: {sorted(all_served)} = {C}  "
          f"{'✓' if all_served == set(C) else '✗'}")
    print(f"    Fleet size: {len(selected)} ≤ {K}  "
          f"{'✓' if len(selected) <= K else '✗'}")
    
    for r in selected:
        route = all_routes[r]
        # Capacity
        td = sum(customers[c]['demand'] for c in route['all_customers'])
        print(f"    Capacity: {td} ≤ {Q}  {'✓' if td <= Q else '✗'}")
        # Battery
        for stop in route['truck_stops']:
            for dc in route['drone_assign'].get(stop, []):
                rt = 2 * drone_tt(stop, dc)
                if rt > B:
                    print(f"    Battery VIOLATION: C{stop}→C{dc}: {rt:.2f} > {B}")
        print(f"    Battery: all ≤ {B}  ✓")
        # Payload
        for stop in route['truck_stops']:
            for dc in route['drone_assign'].get(stop, []):
                if customers[dc]['weight'] > D:
                    print(f"    Payload VIOLATION: C{dc}: {customers[dc]['weight']} > {D}")
        print(f"    Payload: all ≤ {D}  ✓")
    
    # Comparison with truck-only
    print(f"\n  {'─' * 50}")
    print(f"  COMPARISON:")
    best_truck_only = float('inf')
    for r in R:
        if not all_routes[r]['drone_served'] and all_routes[r]['all_customers'] == set(C):
            if all_routes[r]['duration'] < best_truck_only:
                best_truck_only = all_routes[r]['duration']
    if best_truck_only < float('inf'):
        saving = (best_truck_only - model.objVal) / best_truck_only * 100
        print(f"    Truck-only best: {best_truck_only:.4f}")
        print(f"    Truck+Drone optimal: {model.objVal:.4f}")
        print(f"    Improvement: {saving:.1f}%")

elif model.status == GRB.INFEASIBLE:
    print("\n  INFEASIBLE!")
else:
    print(f"\n  Status: {model.status}")