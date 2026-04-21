"""
Flying Sidekick Traveling Salesman Problem (FSTSP) - Gurobi MILP Implementation
Based on Murray and Chu (2015)

Minimizes makespan (total delivery time) for a truck-drone tandem delivery system.

Modification: Dependency constraint — if Customer 4 is drone-served, Customer 1 must also be drone-served.
"""

import gurobipy as gp
from gurobipy import GRB
import math
import itertools

# ============================================================================
# 1. DATA SETUP
# ============================================================================

# Node coordinates (km)
coords = {
    0: (0.0, 0.0),   # Starting depot
    1: (2.0, 5.0),   # Customer 1 (drone OK)
    2: (5.0, 2.0),   # Customer 2 (drone OK)
    3: (7.0, 6.0),   # Customer 3 (drone OK)
    4: (1.0, 8.0),   # Customer 4 (drone OK)
    5: (9.0, 3.0),   # Customer 5 (truck only)
    6: (0.0, 0.0),   # Ending depot (same location as 0)
}

# Sets
C = [1, 2, 3, 4, 5]          # All customers
C_prime = [1, 2, 3, 4]        # Drone-eligible customers
N_0 = [0, 1, 2, 3, 4, 5]     # Departure nodes (depot + customers)
N_plus = [1, 2, 3, 4, 5, 6]  # Arrival nodes (customers + ending depot)
N = [0, 1, 2, 3, 4, 5, 6]    # All nodes
c = 5                          # Number of customers

# Parameters
truck_speed = 40.23   # km/h
drone_speed = 80.47   # km/h
e = 0.5               # Drone endurance (hours)
s_L = 1.0 / 60.0      # Launch time (hours)
s_R = 1.0 / 60.0      # Recovery time (hours)

# Compute Manhattan distance (truck)
def manhattan_dist(i, j):
    return abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])

# Compute Euclidean distance (drone)
def euclidean_dist(i, j):
    return math.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)

# Truck travel times s_ij (hours)
s = {}
for i in N:
    for j in N:
        if i != j:
            s[i, j] = manhattan_dist(i, j) / truck_speed

# Drone travel times sigma_ij (hours)
sigma = {}
for i in N:
    for j in N:
        if i != j:
            sigma[i, j] = euclidean_dist(i, j) / drone_speed

# Big-M constant (upper bound on total time)
M = sum(s[i, j] for i, j in s) / 2 + c * (s_L + s_R) + 5

# ============================================================================
# 2. BUILD SET P: VALID DRONE SORTIES (i, j, k)
# ============================================================================
# i = launch node (in N_0), j = drone customer (in C'), k = retrieval node (in N_+)
# Constraints: j != i, k != j, k != i, sigma_ij + sigma_jk <= e

P = []
for i in N_0:
    for j in C_prime:
        if j == i:
            continue
        for k in N_plus:
            if k == j or k == i:
                continue
            if sigma[i, j] + sigma[j, k] <= e:
                P.append((i, j, k))

print(f"Number of valid drone sorties |P| = {len(P)}")

# Pre-compute lookup sets for efficiency
P_set = set(P)
P_from_i = {i: [(i2, j, k) for (i2, j, k) in P if i2 == i] for i in N_0}
P_serving_j = {j: [(i, j2, k) for (i, j2, k) in P if j2 == j] for j in C_prime}
P_to_k = {k: [(i, j, k2) for (i, j, k2) in P if k2 == k] for k in N_plus}

# ============================================================================
# 3. MODEL CREATION
# ============================================================================

model = gp.Model("FSTSP")
model.Params.TimeLimit = 300
model.Params.MIPGap = 1e-6

# ============================================================================
# 4. DECISION VARIABLES
# ============================================================================

# x[i,j] = 1 if truck travels from node i to node j
x = {}
for i in N_0:
    for j in N_plus:
        if i != j:
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

# y[i,j,k] = 1 if drone launches from i, serves j, retrieves at k
y = {}
for (i, j, k) in P:
    y[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_{k}")

# t[j] = truck arrival time at node j
t = {}
for j in N:
    t[j] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"t_{j}")

# t_prime[j] = drone arrival time at node j
t_prime = {}
for j in N:
    t_prime[j] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"tp_{j}")

# u[i] = position of node i in truck's route (subtour elimination)
u = {}
for i in N_plus:
    u[i] = model.addVar(lb=1, ub=c + 2, vtype=GRB.CONTINUOUS, name=f"u_{i}")

# p[i,j] = 1 if customer i is visited before customer j in truck route
p = {}
for i in C:
    for j in C:
        if i != j:
            p[i, j] = model.addVar(vtype=GRB.BINARY, name=f"p_{i}_{j}")

model.update()

# ============================================================================
# 5. OBJECTIVE: Minimize makespan
# ============================================================================

model.setObjective(t[c + 1], GRB.MINIMIZE)

# ============================================================================
# 6. CONSTRAINTS
# ============================================================================

# (1) Each customer served exactly once (by truck or drone)
for j in C:
    truck_visits = gp.quicksum(x[i, j] for i in N_0 if i != j and (i, j) in x)
    drone_visits = gp.quicksum(y[i2, j, k] for (i2, j2, k) in P if j2 == j)
    model.addConstr(truck_visits + drone_visits == 1, name=f"serve_{j}")

# (2) Truck departs from starting depot exactly once
model.addConstr(
    gp.quicksum(x[0, j] for j in N_plus if (0, j) in x) == 1,
    name="depart_depot"
)

# (3) Truck arrives at ending depot exactly once
model.addConstr(
    gp.quicksum(x[i, c + 1] for i in N_0 if (i, c + 1) in x) == 1,
    name="arrive_depot"
)

# (4) Subtour elimination (MTZ)
for i in C:
    for j in N_plus:
        if i != j and (i, j) in x:
            model.addConstr(
                u[i] - u[j] + 1 <= (c + 2) * (1 - x[i, j]),
                name=f"mtz_{i}_{j}"
            )

# (5) Flow conservation at customer nodes
for j in C:
    inflow = gp.quicksum(x[i, j] for i in N_0 if i != j and (i, j) in x)
    outflow = gp.quicksum(x[j, k] for k in N_plus if k != j and (j, k) in x)
    model.addConstr(inflow == outflow, name=f"flow_{j}")

# (6) Drone launches from any node at most once
for i in N_0:
    if P_from_i[i]:
        model.addConstr(
            gp.quicksum(y[i2, j, k] for (i2, j, k) in P_from_i[i]) <= 1,
            name=f"launch_{i}"
        )

# (7) Drone retrieves at any node at most once
for k in N_plus:
    if P_to_k[k]:
        model.addConstr(
            gp.quicksum(y[i, j, k2] for (i, j, k2) in P_to_k[k]) <= 1,
            name=f"retrieve_{k}"
        )

# (8) Truck must visit launch/retrieval nodes (customer nodes)
for (i, j, k) in P:
    if i in C and k in C:
        lhs = gp.quicksum(x[h, i] for h in N_0 if h != i and (h, i) in x)
        rhs = gp.quicksum(x[l, k] for l in N_0 if l != k and (l, k) in x)
        model.addConstr(2 * y[i, j, k] <= lhs + rhs, name=f"truck_visit_{i}_{j}_{k}")
    elif i in C and k == c + 1:
        lhs = gp.quicksum(x[h, i] for h in N_0 if h != i and (h, i) in x)
        model.addConstr(y[i, j, k] <= lhs, name=f"truck_visit_{i}_{j}_{k}")

# (9) Drone launch from depot: truck must visit retrieval node
for (i, j, k) in P:
    if i == 0:
        rhs = gp.quicksum(x[h, k] for h in N_0 if h != k and (h, k) in x)
        model.addConstr(y[0, j, k] <= rhs, name=f"depot_launch_{j}_{k}")

# (10) Ordering: if drone launches from i, retrieves at k, then u[k] >= u[i] + 1
for i in C:
    for k in N_plus:
        if k != i:
            sorties_ik = [(i, j, k) for j in C_prime if (i, j, k) in P_set]
            if sorties_ik:
                model.addConstr(
                    u[k] - u[i] >= 1 - (c + 2) * (1 - gp.quicksum(y[ss] for ss in sorties_ik)),
                    name=f"order_{i}_{k}"
                )

# (11) Time synchronization at launch node
for i in C:
    sum_launch = gp.quicksum(y[i2, j, k] for (i2, j, k) in P_from_i.get(i, []))
    if P_from_i.get(i, []):
        model.addConstr(t_prime[i] >= t[i] - M * (1 - sum_launch), name=f"sync_launch_lb_{i}")
        model.addConstr(t_prime[i] <= t[i] + M * (1 - sum_launch), name=f"sync_launch_ub_{i}")

# (12) Time synchronization at retrieval node
for k in N_plus:
    sum_retrieve = gp.quicksum(y[i, j, k2] for (i, j, k2) in P_to_k.get(k, []))
    if P_to_k.get(k, []):
        model.addConstr(t_prime[k] >= t[k] - M * (1 - sum_retrieve), name=f"sync_retr_lb_{k}")
        model.addConstr(t_prime[k] <= t[k] + M * (1 - sum_retrieve), name=f"sync_retr_ub_{k}")

# (13a) Truck travel time: t[k] >= t[h] + s[h,k] + service times - M(1 - x[h,k])
for h in N_0:
    for k in N_plus:
        if h != k and (h, k) in x:
            launch_at_k = gp.quicksum(y[k, j2, m] for (k2, j2, m) in P_from_i.get(k, []))
            retrieve_at_k = gp.quicksum(y[i2, j2, k] for (i2, j2, k2) in P_to_k.get(k, []))
            # Truck must wait for its own travel
            model.addConstr(
                t[k] >= t[h] + s[h, k] + s_L * launch_at_k + s_R * retrieve_at_k - M * (1 - x[h, k]),
                name=f"truck_time_{h}_{k}"
            )

# (13b) Truck must also wait for drone at retrieval nodes:
for h in N_0:
    for k in N_plus:
        if h != k and (h, k) in x:
            launch_at_k = gp.quicksum(y[k, j2, m] for (k2, j2, m) in P_from_i.get(k, []))
            retrieve_at_k = gp.quicksum(y[i2, j2, k] for (i2, j2, k2) in P_to_k.get(k, []))
            model.addConstr(
                t[k] >= t_prime[h] + s[h, k] + s_L * launch_at_k + s_R * retrieve_at_k - M * (1 - x[h, k]),
                name=f"truck_time_drone_{h}_{k}"
            )

# (14) Drone travel time: launch node -> customer
for j in C_prime:
    for i in N_0:
        if i != j:
            sorties_ij = [(i, j, k) for k in N_plus if (i, j, k) in P_set]
            if sorties_ij:
                model.addConstr(
                    t_prime[j] >= t_prime[i] + sigma[i, j] - M * (1 - gp.quicksum(y[ss] for ss in sorties_ij)),
                    name=f"drone_to_cust_{i}_{j}"
                )

# (15) Drone travel time: customer -> retrieval node
for j in C_prime:
    for k in N_plus:
        if k != j:
            sorties_jk = [(i, j, k) for i in N_0 if (i, j, k) in P_set]
            if sorties_jk:
                model.addConstr(
                    t_prime[k] >= t_prime[j] + sigma[j, k] + s_R - M * (1 - gp.quicksum(y[ss] for ss in sorties_jk)),
                    name=f"drone_to_retr_{j}_{k}"
                )

# (16) Drone endurance per sortie
for (i, j, k) in P:
    model.addConstr(
        t_prime[k] - (t_prime[j] - sigma[i, j]) <= e + M * (1 - y[i, j, k]),
        name=f"endurance_{i}_{j}_{k}"
    )

# (17) Consecutive sortie ordering
for i in N_0:
    for k in N_plus:
        sorties_ik = [(i, j, k) for j in C_prime if (i, j, k) in P_set]
        if not sorties_ik:
            continue
        for l in C:
            if l == i or l == k:
                continue
            sorties_from_l = P_from_i.get(l, [])
            if not sorties_from_l:
                continue
            if i in C and (i, l) in p:
                model.addConstr(
                    t_prime[l] >= t_prime[k] - M * (
                        3 - gp.quicksum(y[ss] for ss in sorties_ik)
                        - gp.quicksum(y[ss] for ss in sorties_from_l)
                        - p[i, l]
                    ),
                    name=f"consec_{i}_{k}_{l}"
                )

# (18) Ordering variable constraints
for i in C:
    for j in C:
        if i != j:
            model.addConstr(u[i] - u[j] >= 1 - (c + 2) * p[i, j], name=f"ord_lb_{i}_{j}")
            model.addConstr(u[i] - u[j] <= 1 + (c + 2) * (1 - p[i, j]), name=f"ord_ub_{i}_{j}")
for i in C:
    for j in C:
        if i < j:
            model.addConstr(p[i, j] + p[j, i] == 1, name=f"ord_sym_{i}_{j}")

# (19) Initial conditions
model.addConstr(t[0] == 0, name="init_t0")
model.addConstr(t_prime[0] == 0, name="init_tp0")

# (20) Dependency: if Customer 3 is drone-served, Customer 2 must also be drone-served
model.addConstr(
    gp.quicksum(y[i, j, k] for (i, j, k) in P if j == 3) <=
    gp.quicksum(y[i, j, k] for (i, j, k) in P if j == 2),
    name="depend_3_on_2"
)

# ============================================================================
# 7. SOLVE
# ============================================================================

model.optimize()

# ============================================================================
# 8. RESULTS
# ============================================================================

result = {"status": "unknown", "obj": None}

if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
    result["status"] = "optimal" if model.status == GRB.OPTIMAL else "time_limit"
    result["obj"] = round(model.ObjVal * 60, 2)

    print("\n" + "=" * 70)
    print("FSTSP SOLUTION RESULTS")
    print("=" * 70)

    if model.status == GRB.OPTIMAL:
        print(f"Status: OPTIMAL")
    else:
        print(f"Status: TIME_LIMIT (best feasible solution)")

    print(f"Makespan: {model.ObjVal:.4f} hours ({model.ObjVal * 60:.2f} minutes)")

    # Extract truck route
    print("\n--- TRUCK ROUTE ---")
    truck_edges = [(i, j) for (i, j) in x if x[i, j].X > 0.5]
    route = [0]
    current = 0
    visited = set()
    while current != c + 1:
        found = False
        for (i, j) in truck_edges:
            if i == current and j not in visited:
                route.append(j)
                visited.add(j)
                current = j
                found = True
                break
        if not found:
            break
    print(f"Route: {' -> '.join(str(n) for n in route)}")

    print("\nTruck arrival times:")
    for node in route:
        print(f"  Node {node}: t = {t[node].X:.4f} h ({t[node].X * 60:.2f} min)")

    # Extract drone sorties
    print("\n--- DRONE SORTIES ---")
    drone_sorties = [(i, j, k) for (i, j, k) in P if y[i, j, k].X > 0.5]
    if drone_sorties:
        for idx, (i, j, k) in enumerate(drone_sorties):
            flight_time = sigma[i, j] + sigma[j, k]
            print(f"  Sortie {idx + 1}: Launch@Node {i} -> Serve Customer {j} -> Retrieve@Node {k}")
            print(f"    Drone flight time: {flight_time:.4f} h ({flight_time * 60:.2f} min)")
            print(f"    t'[{i}] = {t_prime[i].X:.4f} h, t'[{j}] = {t_prime[j].X:.4f} h, t'[{k}] = {t_prime[k].X:.4f} h")
    else:
        print("  No drone sorties (all customers served by truck)")

    # Summary
    print("\n--- DELIVERY ASSIGNMENT ---")
    truck_served = []
    drone_served = []
    for j in C:
        is_drone = any(y[i, j2, k].X > 0.5 for (i, j2, k) in P if j2 == j)
        if is_drone:
            drone_served.append(j)
        else:
            truck_served.append(j)
    print(f"  Truck-served customers: {truck_served}")
    print(f"  Drone-served customers: {drone_served}")

    # Truck-only time on this route
    truck_travel = sum(s[route[i], route[i+1]] for i in range(len(route) - 1))
    print(f"\n  Truck travel time (route only): {truck_travel:.4f} h ({truck_travel * 60:.2f} min)")
    print(f"  Drone-assisted makespan:        {model.ObjVal:.4f} h ({model.ObjVal * 60:.2f} min)")

    print(f"\n  Result: {result}")

elif model.status == GRB.INFEASIBLE:
    print("\nModel is INFEASIBLE")
    model.computeIIS()
    model.write("fstsp_iis.ilp")
    print("IIS written to fstsp_iis.ilp")
else:
    print(f"\nOptimization ended with status {model.status}")
