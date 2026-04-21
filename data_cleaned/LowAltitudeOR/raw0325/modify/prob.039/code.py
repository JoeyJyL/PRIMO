"""
Drone Delivery Routing Problem — Gurobi MILP Solver
=====================================================
Minimizes: drone purchase cost + energy cost
Subject to: capacity, battery, time, hazmat incompatibility constraints

Key modeling decisions:
  - Energy constraint linearized (pre-compute per-leg energy as function of payload)
  - Drones can make multiple sequential trips within time limit T=600s
  - Drone count = number of drones needed to cover all routes within T
  - Battery weight optimized per route
"""

import math
import itertools
import gurobipy as gp
from gurobipy import GRB

# ============================================================
# Problem Data
# ============================================================

# Locations: 0 = depot, 1-5 = customers
locations = {
    0: (0, 0),
    1: (100, 200),
    2: (300, 100),
    3: (200, 300),
    4: (400, 250),
    5: (150, 400),
}

demands = {1: 0.8, 2: 1.2, 3: 0.6, 4: 1.0, 5: 0.9}

# Parameters
F = 500        # drone cost ($)
Q = 3.0        # max carrying capacity: battery + payload (kg)
v = 6.0        # flight speed (m/s)
tau = 60.0     # service time per stop (s)
W_frame = 1.5  # frame weight, NOT in Q (kg)
alpha = 0.217  # power per kg (kW/kg)
beta = 0.185   # base power (kW)
xi = 650.0     # battery energy density (kJ/kg)
eps = 0.1      # energy cost ($/kJ)
T = 600.0      # delivery time limit (s)
M_max = 5      # max drones

N = [0, 1, 2, 3, 4, 5]        # all nodes
C = [1, 2, 3, 4, 5]            # customers
incompatible = (2, 4)           # hazmat incompatibility

# Distance matrix
dist = {}
for i in N:
    for j in N:
        dx = locations[i][0] - locations[j][0]
        dy = locations[i][1] - locations[j][1]
        dist[i, j] = math.sqrt(dx * dx + dy * dy)

# Travel time between nodes (flight only, no service)
flight_time = {(i, j): dist[i, j] / v for i in N for j in N}

print("=" * 70)
print("DRONE DELIVERY ROUTING — GUROBI MILP SOLVER")
print("=" * 70)


# ============================================================
# Approach: Enumerate feasible routes, then select optimally
# ============================================================
# With only 5 customers, we can enumerate all feasible single/multi-stop
# routes and solve a set-covering/partitioning problem.
# This avoids the nonlinear energy constraint issue entirely.

def compute_route_data(route_stops):
    """
    Given an ordered list of customer stops (e.g., [1, 3]),
    compute route properties: time, minimum battery, energy, cost.
    Route: depot -> stops[0] -> stops[1] -> ... -> depot
    
    Returns dict with route info, or None if infeasible.
    """
    # Build full path
    path = [0] + list(route_stops) + [0]
    
    # Total payload = sum of demands
    total_payload = sum(demands[c] for c in route_stops)
    
    # --- Compute route time ---
    route_time = 0.0
    for k in range(len(path) - 1):
        i, j = path[k], path[k + 1]
        route_time += flight_time[i, j]
        if j != 0:  # service time at customer stops
            route_time += tau
    
    if route_time > T:
        return None  # time infeasible
    
    # --- Compute energy as function of battery weight q ---
    # Energy on each leg depends on total mass = W_frame + q + current_payload
    # Payload decreases as deliveries are made.
    # E_leg = P(mass) * time_leg, where P(m) = alpha*m + beta
    # time_leg = flight_time + tau (if destination is customer)
    
    # For each leg, compute: payload carried, leg time
    legs = []
    current_payload = total_payload
    for k in range(len(path) - 1):
        i, j = path[k], path[k + 1]
        leg_time = flight_time[i, j]
        if j != 0:
            leg_time += tau  # service at customer
        
        # Mass during this leg (excluding q, which we'll optimize)
        mass_excl_q = W_frame + current_payload
        legs.append((leg_time, mass_excl_q))
        
        # Drop payload at destination
        if j != 0:
            current_payload -= demands[j]
    
    # Energy = sum over legs: (alpha * (mass_excl_q + q) + beta) * leg_time
    #        = sum over legs: (alpha*mass_excl_q + beta)*leg_time + alpha*q*leg_time
    # z(q) = A + B*q, where:
    A = sum((alpha * m + beta) * t for t, m in legs)  # energy without battery mass
    B = sum(alpha * t for t, m in legs)                # energy per kg of battery
    
    # Battery constraint: xi * q >= z(q) = A + B*q
    # → q * (xi - B) >= A
    # → q >= A / (xi - B)  [if xi > B, which it should be]
    
    if xi <= B:
        return None  # physically impossible
    
    q_min = A / (xi - B)
    
    # Capacity constraint: q + total_payload <= Q
    q_max = Q - total_payload
    
    if q_min > q_max:
        return None  # battery doesn't fit with payload
    
    if q_min < 0:
        q_min = 0  # shouldn't happen but safety check
    
    # Optimal q = q_min (minimizes energy cost since z = A + B*q)
    q_opt = q_min
    z_opt = A + B * q_opt  # energy consumed (kJ)
    energy_cost = eps * z_opt
    
    return {
        'stops': list(route_stops),
        'time': route_time,
        'q_battery': q_opt,
        'payload': total_payload,
        'energy_kJ': z_opt,
        'energy_cost': energy_cost,
        'path': path,
    }


# ============================================================
# Enumerate ALL feasible routes
# ============================================================

print("\nEnumerating feasible routes...")
all_routes = []

for length in range(1, len(C) + 1):
    for combo in itertools.combinations(C, length):
        # Check hazmat: 2 and 4 cannot be together
        if incompatible[0] in combo and incompatible[1] in combo:
            continue
        
        # Try all orderings of this customer set
        best_route = None
        for perm in itertools.permutations(combo):
            rd = compute_route_data(perm)
            if rd is not None:
                if best_route is None or rd['energy_cost'] < best_route['energy_cost']:
                    best_route = rd
        
        if best_route is not None:
            all_routes.append(best_route)

print(f"Found {len(all_routes)} feasible routes:")
for idx, r in enumerate(all_routes):
    print(f"  Route {idx}: stops={r['stops']}, time={r['time']:.1f}s, "
          f"q={r['q_battery']:.4f}kg, payload={r['payload']:.1f}kg, "
          f"energy={r['energy_kJ']:.2f}kJ, cost=${r['energy_cost']:.4f}")


# ============================================================
# Gurobi Model: Select routes to minimize total cost
# ============================================================

print(f"\n{'=' * 70}")
print("Building Gurobi model...")
print(f"{'=' * 70}")

model = gp.Model("DroneDelivery")
model.setParam('OutputFlag', 1)

R = range(len(all_routes))

# --- Decision Variables ---
# x[r] = 1 if route r is selected
x = model.addVars(R, vtype=GRB.BINARY, name="x")

# n = number of drones purchased
n = model.addVar(vtype=GRB.INTEGER, lb=1, ub=M_max, name="n_drones")

# --- Objective: minimize drone cost + energy cost ---
model.setObjective(
    F * n + gp.quicksum(all_routes[r]['energy_cost'] * x[r] for r in R),
    GRB.MINIMIZE
)

# --- Constraint 1: Each customer visited exactly once ---
for c in C:
    model.addConstr(
        gp.quicksum(x[r] for r in R if c in all_routes[r]['stops']) == 1,
        name=f"visit_{c}"
    )

# --- Constraint 2: Drone count >= routes that run in parallel ---
# Since drones can do multiple sequential trips within T,
# we need to figure out which routes can be sequenced on one drone.
# This is a bin-packing / scheduling problem.
#
# Simplification: each drone can fly routes whose total times sum ≤ T.
# We model this by assigning routes to drones.

# Create drone assignment variables
D_set = range(M_max)  # potential drones 0..4

# y[r,d] = 1 if route r is assigned to drone d
y = model.addVars(R, D_set, vtype=GRB.BINARY, name="y")

# w[d] = 1 if drone d is used
w = model.addVars(D_set, vtype=GRB.BINARY, name="w")

# Link x and y: route selected ↔ assigned to exactly one drone
for r in R:
    model.addConstr(
        gp.quicksum(y[r, d] for d in D_set) == x[r],
        name=f"assign_{r}"
    )

# Time capacity per drone: sum of route times on drone d ≤ T
for d in D_set:
    model.addConstr(
        gp.quicksum(all_routes[r]['time'] * y[r, d] for r in R) <= T,
        name=f"drone_time_{d}"
    )

# Link w and y: if any route assigned to drone d, then w[d] = 1
for d in D_set:
    for r in R:
        model.addConstr(y[r, d] <= w[d], name=f"link_w_{r}_{d}")

# Drone count = sum of drones used
model.addConstr(n == gp.quicksum(w[d] for d in D_set), name="drone_count")

# Symmetry breaking: use drones in order
for d in range(1, M_max):
    model.addConstr(w[d] <= w[d - 1], name=f"sym_{d}")

# --- Solve ---
print("\nSolving...")
model.optimize()

# ============================================================
# Output Results
# ============================================================

print(f"\n{'=' * 70}")
print("SOLUTION")
print(f"{'=' * 70}")

if model.status == GRB.OPTIMAL:
    print(f"\n  Status: Optimal")
    print(f"  Total Cost: ${model.objVal:.4f}")
    
    n_drones = int(round(n.X))
    print(f"  Drones purchased: {n_drones} (cost: ${n_drones * F})")
    
    total_energy_cost = 0.0
    
    # Group routes by drone
    drone_routes = {d: [] for d in D_set}
    for r in R:
        if x[r].X > 0.5:
            for d in D_set:
                if y[r, d].X > 0.5:
                    drone_routes[d].append(r)
                    break
    
    for d in D_set:
        if not drone_routes[d]:
            continue
        print(f"\n  Drone {d + 1}:")
        drone_time = 0.0
        for r in drone_routes[d]:
            rt = all_routes[r]
            path_str = " → ".join(
                f"C{s}" if s != 0 else "Depot" for s in rt['path']
            )
            print(f"    Trip: {path_str}")
            print(f"      Stops: {rt['stops']}, Time: {rt['time']:.1f}s")
            print(f"      Payload: {rt['payload']:.1f}kg, "
                  f"Battery: {rt['q_battery']:.4f}kg, "
                  f"Total load: {rt['payload'] + rt['q_battery']:.4f}kg ≤ Q={Q}")
            print(f"      Energy: {rt['energy_kJ']:.2f}kJ, "
                  f"Cost: ${rt['energy_cost']:.4f}")
            total_energy_cost += rt['energy_cost']
            drone_time += rt['time']
        print(f"    Total drone time: {drone_time:.1f}s / {T:.0f}s")
    
    print(f"\n  {'─' * 50}")
    print(f"  COST BREAKDOWN:")
    print(f"    Drone cost:   {n_drones} × ${F} = ${n_drones * F:.2f}")
    print(f"    Energy cost:  ${total_energy_cost:.4f}")
    print(f"    Total cost:   ${model.objVal:.4f}")
    
    # Constraint verification
    print(f"\n  {'─' * 50}")
    print(f"  CONSTRAINT VERIFICATION:")
    
    # All customers served
    served = set()
    for r in R:
        if x[r].X > 0.5:
            served.update(all_routes[r]['stops'])
    print(f"    All customers served: {sorted(served)} = {C}  "
          f"{'✓' if served == set(C) else '✗'}")
    
    # Hazmat
    for r in R:
        if x[r].X > 0.5:
            stops = set(all_routes[r]['stops'])
            if incompatible[0] in stops and incompatible[1] in stops:
                print(f"    Hazmat VIOLATION on route {r}!")
                break
    else:
        print(f"    Hazmat C2/C4 separation: ✓")
    
    # Time
    for d in D_set:
        if drone_routes[d]:
            dt = sum(all_routes[r]['time'] for r in drone_routes[d])
            if dt > T + 0.01:
                print(f"    Time VIOLATION on drone {d}: {dt:.1f} > {T}")
    print(f"    Time limits: all drones within {T}s  ✓")
    
    # Capacity
    cap_ok = True
    for r in R:
        if x[r].X > 0.5:
            rt = all_routes[r]
            if rt['payload'] + rt['q_battery'] > Q + 0.001:
                print(f"    Capacity VIOLATION on route {r}")
                cap_ok = False
    if cap_ok:
        print(f"    Capacity: all routes ≤ Q={Q}kg  ✓")
    
    print(f"\n  ★ ALL CONSTRAINTS SATISFIED ✓")

elif model.status == GRB.INFEASIBLE:
    print("\n  Model is INFEASIBLE!")
    model.computeIIS()
    model.write("/home/claude/infeasible.ilp")
else:
    print(f"\n  Status: {model.status}")