"""
Flying Sidekick Traveling Salesman Problem (FSTSP)
Murray & Chu (2015) - Original Formulation
5 customers, customer 5 is truck-only (heavy parcel)
Solved using Gurobi MIP.
"""

import gurobipy as gp
from gurobipy import GRB
import math

# ─────────────────────────────────────────────
# 1. PROBLEM DATA
# ─────────────────────────────────────────────

# Node coordinates (km): 0=start depot, 1-5=customers, 6=end depot
coords = {
    0: (0.0, 0.0),   # Start depot
    1: (2.0, 5.0),   # Customer 1 (drone-eligible)
    2: (5.0, 2.0),   # Customer 2 (drone-eligible)
    3: (7.0, 6.0),   # Customer 3 (drone-eligible)
    4: (1.0, 8.0),   # Customer 4 (drone-eligible)
    5: (9.0, 3.0),   # Customer 5 (truck-only, heavy parcel)
    6: (0.0, 0.0),   # End depot
}

# Sets
customers  = [1, 2, 3, 4, 5]      # C  — all customers
drone_elig = [1, 2, 3, 4]         # C' — drone-eligible only (customer 5 excluded)
N0         = [0, 1, 2, 3, 4, 5]   # departure nodes (depot + customers)
Nplus      = [1, 2, 3, 4, 5, 6]   # arrival nodes   (customers + end depot)
# Drone retrieval must be a customer node the truck visits — NOT the end depot
drone_ret  = [1, 2, 3, 4, 5]      # valid retrieval nodes for drone (customer nodes only)
c          = len(customers)        # 5

# Speed & timing
truck_speed = 40.2336   # km/h (25 mph)
drone_speed = 80.4672   # km/h (50 mph)
e           = 0.5       # drone endurance (hours)
s_L         = 1 / 60    # launch time (hours)
s_R         = 1 / 60    # recovery time (hours)
M           = 1000      # big-M

# ─────────────────────────────────────────────
# 2. TRAVEL TIME MATRICES
# ─────────────────────────────────────────────

def manhattan(a, b):
    return abs(coords[a][0] - coords[b][0]) + abs(coords[a][1] - coords[b][1])

def euclidean(a, b):
    return math.sqrt((coords[a][0]-coords[b][0])**2 + (coords[a][1]-coords[b][1])**2)

# Truck travel times (Manhattan)
s = {(i, j): manhattan(i, j) / truck_speed
     for i in N0 for j in Nplus if i != j}

# Drone travel times (Euclidean) — between all node pairs
all_nodes = list(coords.keys())
sigma = {(i, j): euclidean(i, j) / drone_speed
         for i in all_nodes for j in all_nodes if i != j}

# ─────────────────────────────────────────────
# 3. VALID DRONE SORTIES  P = {(i, j, k)}
#    i = launch node  ∈ N0
#    j = drone customer ∈ C'  (NOT customer 5)
#    k = retrieval node ∈ drone_ret  (customer node, NOT end depot)
#    Constraints: i≠j, j≠k, i≠k, σ_ij + σ_jk ≤ e
# ─────────────────────────────────────────────

sorties = [
    (i, j, k)
    for i in N0
    for j in drone_elig          # only drone-eligible customers
    for k in drone_ret           # retrieval at a customer node (truck must be there)
    if i != j and j != k and i != k
    and sigma[i, j] + sigma[j, k] <= e
]

print(f"Valid drone sorties: {len(sorties)}")
for (i, j, k) in sorties:
    flight_min = (sigma[i, j] + sigma[j, k]) * 60
    print(f"  Launch:{i} → Drone→{j} → Retrieve:{k}  (flight={flight_min:.1f} min)")

# ─────────────────────────────────────────────
# 4. GUROBI MODEL
# ─────────────────────────────────────────────

model = gp.Model("FSTSP_5cust")
model.setParam("OutputFlag", 1)
model.setParam("TimeLimit", 300)

# ── Decision Variables ──────────────────────

# x[i,j] = 1 if truck travels from i to j
x = model.addVars(
    [(i, j) for i in N0 for j in Nplus if i != j],
    vtype=GRB.BINARY, name="x"
)

# y[i,j,k] = 1 if drone: launch at i, serve j, retrieve at k
y = model.addVars(sorties, vtype=GRB.BINARY, name="y")

# t[j]  = truck arrival time at node j
t = model.addVars(all_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="t")

# td[j] = drone arrival time at node j
td = model.addVars(all_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="td")

# p[i,j] = 1 if customer i visited before j in truck route
p = model.addVars(
    [(i, j) for i in customers for j in customers if i != j],
    vtype=GRB.BINARY, name="p"
)

# u[i] = MTZ position of node i in truck route
u = model.addVars(customers, lb=1, ub=c + 2, vtype=GRB.CONTINUOUS, name="u")

# ── Objective: minimize makespan (truck return to end depot, node 6) ──
model.setObjective(t[6], GRB.MINIMIZE)

# ─────────────────────────────────────────────
# 5. CONSTRAINTS
# ─────────────────────────────────────────────

# (C1) Each customer served exactly once — by truck OR drone
for j in customers:
    truck_in   = gp.quicksum(x[i, j] for i in N0 if i != j and (i, j) in x)
    drone_serv = gp.quicksum(y[i, j, k] for (ii, jj, kk) in sorties
                             if jj == j for i in [ii] for k in [kk])
    model.addConstr(truck_in + drone_serv == 1, name=f"serve_{j}")

# (C2) Truck departs start depot exactly once
model.addConstr(gp.quicksum(x[0, j] for j in Nplus) == 1, name="depot_depart")

# (C3) Truck returns to end depot exactly once
model.addConstr(gp.quicksum(x[i, 6] for i in N0) == 1, name="depot_return")

# (C4) Flow conservation at each customer node
for j in customers:
    in_flow  = gp.quicksum(x[i, j] for i in N0 if i != j and (i, j) in x)
    out_flow = gp.quicksum(x[j, k] for k in Nplus if k != j and (j, k) in x)
    model.addConstr(in_flow == out_flow, name=f"flow_{j}")

# (C5) MTZ subtour elimination
for i in customers:
    for j in customers:
        if i != j and (i, j) in x:
            model.addConstr(
                u[i] - u[j] + 1 <= (c + 2) * (1 - x[i, j]),
                name=f"mtz_{i}_{j}"
            )

# (C6) Drone launches from any node at most once
for i in N0:
    launches = [(ii, jj, kk) for (ii, jj, kk) in sorties if ii == i]
    if launches:
        model.addConstr(
            gp.quicksum(y[ii, jj, kk] for (ii, jj, kk) in launches) <= 1,
            name=f"drone_launch_{i}"
        )

# (C7) Drone retrieval at any node at most once
for k in drone_ret:
    retrievals = [(ii, jj, kk) for (ii, jj, kk) in sorties if kk == k]
    if retrievals:
        model.addConstr(
            gp.quicksum(y[ii, jj, kk] for (ii, jj, kk) in retrievals) <= 1,
            name=f"drone_retrieve_{k}"
        )

# (C8) Truck must visit retrieval node k for ALL sorties (including depot launches)
#      Truck must also visit launch node i for customer launches
for (i, j, k) in sorties:
    # Retrieval node k must be on the truck route
    in_k = gp.quicksum(x[l, k] for l in N0 if l != k and (l, k) in x)
    model.addConstr(y[i, j, k] <= in_k, name=f"truck_ret_{i}_{j}_{k}")
    # Launch node i must be on the truck route (depot i=0 is always visited)
    if i in customers:
        in_i = gp.quicksum(x[h, i] for h in N0 if h != i and (h, i) in x)
        model.addConstr(y[i, j, k] <= in_i, name=f"truck_launch_{i}_{j}_{k}")

# (C9) Ordering: if drone launches at customer i, retrieves at customer k
#      → truck must visit i before k
for (i, j, k) in sorties:
    if i in customers and k in customers:
        model.addConstr(
            u[k] - u[i] >= 1 - (c + 2) * (1 - y[i, j, k]),
            name=f"order_{i}_{j}_{k}"
        )

# (C10) Time sync at launch node: drone departs with truck
for i in customers:
    active = [(ii, jj, kk) for (ii, jj, kk) in sorties if ii == i]
    if active:
        z = gp.quicksum(y[ii, jj, kk] for (ii, jj, kk) in active)
        model.addConstr(td[i] >= t[i] - M * (1 - z), name=f"tsync_launch_lo_{i}")
        model.addConstr(td[i] <= t[i] + M * (1 - z), name=f"tsync_launch_hi_{i}")

# (C11) Time sync at retrieval node: truck and drone meet
for k in drone_ret:
    active = [(ii, jj, kk) for (ii, jj, kk) in sorties if kk == k]
    if active:
        z = gp.quicksum(y[ii, jj, kk] for (ii, jj, kk) in active)
        model.addConstr(td[k] >= t[k] - M * (1 - z), name=f"tsync_ret_lo_{k}")
        model.addConstr(td[k] <= t[k] + M * (1 - z), name=f"tsync_ret_hi_{k}")

# (C12) Truck travel time propagation (with launch/recovery overhead at each node)
for h in N0:
    for k in Nplus:
        if h != k and (h, k) in x:
            launch_at_k = (gp.quicksum(y[k, jj, ll] for (ii, jj, ll) in sorties if ii == k)
                           if k in customers else gp.LinExpr(0))
            ret_at_k    = gp.quicksum(y[ii, jj, k] for (ii, jj, kk) in sorties if kk == k)
            model.addConstr(
                t[k] >= t[h] + s[h, k]
                       + s_L * launch_at_k
                       + s_R * ret_at_k
                       - M * (1 - x[h, k]),
                name=f"truck_time_{h}_{k}"
            )

# (C13) Drone travel time: launch node i → customer j
for (i, j, k) in sorties:
    z_ij = gp.quicksum(y[i, j, kk] for (ii, jj, kk) in sorties if ii == i and jj == j)
    model.addConstr(
        td[j] >= td[i] + sigma[i, j] - M * (1 - y[i, j, k]),
        name=f"drone_leg1_{i}_{j}_{k}"
    )

# (C14) Drone travel time: customer j → retrieval node k  (+ recovery time)
for (i, j, k) in sorties:
    model.addConstr(
        td[k] >= td[j] + sigma[j, k] + s_R - M * (1 - y[i, j, k]),
        name=f"drone_leg2_{i}_{j}_{k}"
    )

# (C15) Drone endurance: total flight time per sortie ≤ e
for (i, j, k) in sorties:
    model.addConstr(
        sigma[i, j] + sigma[j, k] <= e + M * (1 - y[i, j, k]),
        name=f"endurance_{i}_{j}_{k}"
    )

# (C16) Non-overlapping sorties — drone can only fly one sortie at a time.
#
#   For any two active sorties A=(i,j,k) and B=(l,m,n), the drone must finish
#   the earlier-launching sortie before it starts the later one.
#
#   The truck route determines launch order:
#     - depot (i=0) always launches before any customer node l
#     - for two customer launches i and l: p[i,l]=1 means i before l
#
#   When sortie A launches before sortie B:
#     td[l] >= td[k] - M*(2 - y[i,j,k] - y[l,m,n])   (depot-first case)
#     td[l] >= td[k] - M*(3 - y[i,j,k] - y[l,m,n] - p[i,l])  (customer-first case)

for (i, j, k) in sorties:
    for (l, m, n) in sorties:
        if (i, j, k) == (l, m, n):
            continue
        if l not in customers:
            continue  # launch node l must be a customer (depot only launches first)

        if i == 0:
            # Depot launch is always first — enforce A finishes before B launches
            model.addConstr(
                td[l] >= td[k] - M * (2 - y[i, j, k] - y[l, m, n]),
                name=f"consec_{i}_{j}_{k}_{l}_{m}_{n}"
            )
        elif i in customers and l in customers and i != l:
            # Use p[i,l]: if truck visits i before l, sortie A precedes B
            model.addConstr(
                td[l] >= td[k] - M * (3 - y[i, j, k] - y[l, m, n] - p[i, l]),
                name=f"consec_{i}_{j}_{k}_{l}_{m}_{n}"
            )


# (C17) Ordering variable consistency with u positions
for i in customers:
    for j in customers:
        if i != j:
            model.addConstr(u[i] - u[j] >= 1 - (c + 2) * p[i, j],  name=f"pij_lo_{i}_{j}")
            model.addConstr(u[i] - u[j] <= 1 + (c + 2) * (1 - p[i, j]), name=f"pij_hi_{i}_{j}")

# Symmetry: exactly one of p[i,j], p[j,i] = 1
for i in customers:
    for j in customers:
        if i < j:
            model.addConstr(p[i, j] + p[j, i] == 1, name=f"psym_{i}_{j}")

# (C18) Initial conditions
model.addConstr(t[0]  == 0, name="t0")
model.addConstr(td[0] == 0, name="td0")

# Position bounds (depot is position 1, customers from 2 onwards)
for j in customers:
    model.addConstr(u[j] >= 2, name=f"u_lb_{j}")

# ─────────────────────────────────────────────
# 6. SOLVE
# ─────────────────────────────────────────────

model.optimize()

result = {"status": "unknown", "obj": None}
# ─────────────────────────────────────────────
# 7. RESULTS
# ─────────────────────────────────────────────

if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
    result["status"] = "optimal" if model.status == GRB.OPTIMAL else "time_limit"
    result["obj"] = round(model.ObjBound*60,2)
    print("\n" + "="*65)
    print("  FSTSP OPTIMAL SOLUTION  (5 customers, customer 5 truck-only)")
    print("="*65)
    print(f"  Makespan (total delivery time): {t[6].X * 60:.2f} minutes\n")

    # Reconstruct truck route
    truck_edges = [(i, j) for (i, j) in x if x[i, j].X > 0.5]
    route = [0]
    while route[-1] != 6:
        nxt = [j for (i, j) in truck_edges if i == route[-1]]
        if not nxt:
            break
        route.append(nxt[0])

    print("  TRUCK ROUTE:")
    for nd in route:
        arrow = " → " if nd != route[-1] else ""
        print(f"    Node {nd}  (t={t[nd].X*60:.2f} min){arrow}", end="")
    print()

    # Drone sorties
    active_sorties = [(i, j, k) for (i, j, k) in sorties if y[i, j, k].X > 0.5]
    print("\n  DRONE SORTIES:")
    if active_sorties:
        for (i, j, k) in active_sorties:
            flight_min = (sigma[i, j] + sigma[j, k]) * 60
            print(f"    Launch node {i} (t={td[i].X*60:.2f} min) "
                  f"→ Deliver to node {j} (t={td[j].X*60:.2f} min) "
                  f"→ Retrieve at node {k} (t={td[k].X*60:.2f} min)  "
                  f"[flight={flight_min:.1f} min]")
    else:
        print("    No drone sorties used — all customers served by truck.")

    print("\n  CUSTOMER SERVICE SUMMARY:")
    for cust in customers:
        by_truck = any(x[i, cust].X > 0.5 for i in N0 if (i, cust) in x)
        by_drone = any(y[i, cust, k].X > 0.5 for (i, jj, k) in sorties if jj == cust)
        if by_truck:
            print(f"    Customer {cust}: TRUCK   (arrival t={t[cust].X*60:.2f} min)"
                  + ("  [truck-only, heavy parcel]" if cust == 5 else ""))
        elif by_drone:
            print(f"    Customer {cust}: DRONE   (delivery t={td[cust].X*60:.2f} min)")
        else:
            print(f"    Customer {cust}: ⚠ NOT SERVED — check model feasibility")

    print(f"\n status: {result}  |  Obj: {model.ObjBound*60:.2f} min")
    print("="*65)

else:
    print(f"No feasible solution found. Gurobi status: {model.Status}")