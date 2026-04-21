"""
Flying Sidekick Traveling Salesman Problem (FSTSP) — Flow-Based Reformulation
5 customers, customer 5 is truck-only (heavy parcel).
Uses single-commodity flow (SCF) subtour elimination instead of MTZ.
Uses time-based ordering variable coupling instead of position-based coupling.
Based on Murray & Chu (2015).
"""

import gurobipy as gp
from gurobipy import GRB
import math
import json

# ─────────────────────────────────────────────
# 1. PROBLEM DATA
# ─────────────────────────────────────────────

coords = {
    0: (0.0, 0.0),   # Start depot
    1: (2.0, 5.0),   # Customer 1 (drone-eligible)
    2: (5.0, 2.0),   # Customer 2 (drone-eligible)
    3: (7.0, 6.0),   # Customer 3 (drone-eligible)
    4: (1.0, 8.0),   # Customer 4 (drone-eligible)
    5: (9.0, 3.0),   # Customer 5 (truck-only, heavy parcel)
    6: (0.0, 0.0),   # End depot
}

customers  = [1, 2, 3, 4, 5]
drone_elig = [1, 2, 3, 4]
N0         = [0, 1, 2, 3, 4, 5]
Nplus      = [1, 2, 3, 4, 5, 6]
drone_ret  = [1, 2, 3, 4, 5]
c          = len(customers)        # 5
all_nodes  = list(coords.keys())

truck_speed = 40.2336   # km/h
drone_speed = 80.4672   # km/h
e           = 0.5       # drone endurance (hours)
s_L         = 1 / 60    # launch time (hours)
s_R         = 1 / 60    # recovery time (hours)
M           = 1000
EPS         = 0.001     # small epsilon for time-based ordering

# ─────────────────────────────────────────────
# 2. TRAVEL TIME MATRICES
# ─────────────────────────────────────────────

def manhattan(a, b):
    return abs(coords[a][0] - coords[b][0]) + abs(coords[a][1] - coords[b][1])

def euclidean(a, b):
    return math.sqrt((coords[a][0]-coords[b][0])**2 + (coords[a][1]-coords[b][1])**2)

s = {(i, j): manhattan(i, j) / truck_speed for i in N0 for j in Nplus if i != j}
sigma = {(i, j): euclidean(i, j) / drone_speed for i in all_nodes for j in all_nodes if i != j}

# ─────────────────────────────────────────────
# 3. VALID DRONE SORTIES
# ─────────────────────────────────────────────

sorties = [
    (i, j, k)
    for i in N0
    for j in drone_elig
    for k in drone_ret
    if i != j and j != k and i != k
    and sigma[i, j] + sigma[j, k] <= e
]

print(f"Valid drone sorties: {len(sorties)}")

# ─────────────────────────────────────────────
# 4. GUROBI MODEL
# ─────────────────────────────────────────────

model = gp.Model("FSTSP_FlowBased_5cust")
model.setParam("OutputFlag", 1)
model.setParam("TimeLimit", 300)

# ── Decision Variables ──────────────────────

x = model.addVars(
    [(i, j) for i in N0 for j in Nplus if i != j],
    vtype=GRB.BINARY, name="x"
)

y = model.addVars(sorties, vtype=GRB.BINARY, name="y")

t = model.addVars(all_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="t")

td = model.addVars(all_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="td")

# p[i,j] = 1 if customer i visited before j in truck route (retained for sortie ordering)
p = model.addVars(
    [(i, j) for i in customers for j in customers if i != j],
    vtype=GRB.BINARY, name="p"
)

# f[i,j] = single-commodity flow on truck arc (i,j) — replaces MTZ u variables
truck_arcs = [(i, j) for i in N0 for j in Nplus if i != j]
f = model.addVars(truck_arcs, lb=0.0, ub=c, vtype=GRB.CONTINUOUS, name="f")

# ── Objective ────────────────────────────────
model.setObjective(t[6], GRB.MINIMIZE)

# ─────────────────────────────────────────────
# 5. CONSTRAINTS
# ─────────────────────────────────────────────

# (C1) Each customer served exactly once
for j in customers:
    truck_in = gp.quicksum(x[i, j] for i in N0 if i != j and (i, j) in x)
    drone_serv = gp.quicksum(y[ii, jj, kk] for (ii, jj, kk) in sorties if jj == j)
    model.addConstr(truck_in + drone_serv == 1, name=f"serve_{j}")

# (C2) Truck departs depot
model.addConstr(gp.quicksum(x[0, j] for j in Nplus) == 1, name="depot_depart")

# (C3) Truck returns to end depot
model.addConstr(gp.quicksum(x[i, 6] for i in N0) == 1, name="depot_return")

# (C4) Flow conservation
for j in customers:
    in_flow = gp.quicksum(x[i, j] for i in N0 if i != j and (i, j) in x)
    out_flow = gp.quicksum(x[j, k] for k in Nplus if k != j and (j, k) in x)
    model.addConstr(in_flow == out_flow, name=f"truck_flow_{j}")

# (C5) Single-Commodity Flow Subtour Elimination (replaces MTZ)
truck_visit = {j: gp.quicksum(x[i, j] for i in N0 if i != j and (i, j) in x) for j in customers}
model.addConstr(
    gp.quicksum(f[0, j] for j in Nplus if (0, j) in f) ==
    gp.quicksum(truck_visit[j] for j in customers),
    name="scf_depot_out"
)
for j in customers:
    model.addConstr(
        gp.quicksum(f[i, j] for i in N0 if i != j and (i, j) in f) -
        gp.quicksum(f[j, k] for k in Nplus if k != j and (j, k) in f) ==
        truck_visit[j],
        name=f"scf_conserve_{j}"
    )
for (i, j) in truck_arcs:
    model.addConstr(f[i, j] <= c * x[i, j], name=f"scf_cap_{i}_{j}")

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

# (C8) Truck must visit launch and retrieval nodes
for (i, j, k) in sorties:
    in_k = gp.quicksum(x[l, k] for l in N0 if l != k and (l, k) in x)
    model.addConstr(y[i, j, k] <= in_k, name=f"truck_ret_{i}_{j}_{k}")
    if i in customers:
        in_i = gp.quicksum(x[h, i] for h in N0 if h != i and (h, i) in x)
        model.addConstr(y[i, j, k] <= in_i, name=f"truck_launch_{i}_{j}_{k}")

# (C9) Time-based drone sortie ordering (replaces u-based ordering)
for (i, j, k) in sorties:
    if i in customers and k in customers:
        model.addConstr(
            t[k] >= t[i] + EPS - M * (1 - y[i, j, k]),
            name=f"time_order_{i}_{j}_{k}"
        )

# (C10) Time sync at launch node
for i in customers:
    active = [(ii, jj, kk) for (ii, jj, kk) in sorties if ii == i]
    if active:
        z = gp.quicksum(y[ii, jj, kk] for (ii, jj, kk) in active)
        model.addConstr(td[i] >= t[i] - M * (1 - z), name=f"tsync_launch_lo_{i}")
        model.addConstr(td[i] <= t[i] + M * (1 - z), name=f"tsync_launch_hi_{i}")

# (C11) Time sync at retrieval node
for k in drone_ret:
    active = [(ii, jj, kk) for (ii, jj, kk) in sorties if kk == k]
    if active:
        z = gp.quicksum(y[ii, jj, kk] for (ii, jj, kk) in active)
        model.addConstr(td[k] >= t[k] - M * (1 - z), name=f"tsync_ret_lo_{k}")
        model.addConstr(td[k] <= t[k] + M * (1 - z), name=f"tsync_ret_hi_{k}")

# (C12) Truck travel time propagation
for h in N0:
    for k in Nplus:
        if h != k and (h, k) in x:
            launch_at_k = (gp.quicksum(y[k, jj, ll] for (ii, jj, ll) in sorties if ii == k)
                           if k in customers else gp.LinExpr(0))
            ret_at_k = gp.quicksum(y[ii, jj, k] for (ii, jj, kk) in sorties if kk == k)
            model.addConstr(
                t[k] >= t[h] + s[h, k]
                       + s_L * launch_at_k
                       + s_R * ret_at_k
                       - M * (1 - x[h, k]),
                name=f"truck_time_{h}_{k}"
            )

# (C13) Drone travel: launch → customer
for (i, j, k) in sorties:
    model.addConstr(
        td[j] >= td[i] + sigma[i, j] - M * (1 - y[i, j, k]),
        name=f"drone_leg1_{i}_{j}_{k}"
    )

# (C14) Drone travel: customer → retrieval
for (i, j, k) in sorties:
    model.addConstr(
        td[k] >= td[j] + sigma[j, k] + s_R - M * (1 - y[i, j, k]),
        name=f"drone_leg2_{i}_{j}_{k}"
    )

# (C15) Drone endurance
for (i, j, k) in sorties:
    model.addConstr(
        sigma[i, j] + sigma[j, k] <= e + M * (1 - y[i, j, k]),
        name=f"endurance_{i}_{j}_{k}"
    )

# (C16) Non-overlapping sorties (consecutive sortie ordering)
for (i, j, k) in sorties:
    for (l, m, n) in sorties:
        if (i, j, k) == (l, m, n):
            continue
        if l not in customers:
            continue
        if i == 0:
            model.addConstr(
                td[l] >= td[k] - M * (2 - y[i, j, k] - y[l, m, n]),
                name=f"consec_{i}_{j}_{k}_{l}_{m}_{n}"
            )
        elif i in customers and l in customers and i != l:
            model.addConstr(
                td[l] >= td[k] - M * (3 - y[i, j, k] - y[l, m, n] - p[i, l]),
                name=f"consec_{i}_{j}_{k}_{l}_{m}_{n}"
            )

# (C17) Time-based ordering variable coupling (replaces MTZ position-based coupling)
for i in customers:
    for j in customers:
        if i != j:
            model.addConstr(
                t[j] >= t[i] + EPS - M * (1 - p[i, j]),
                name=f"ptime_{i}_{j}"
            )

# Ordering symmetry
for i in customers:
    for j in customers:
        if i < j:
            model.addConstr(p[i, j] + p[j, i] == 1, name=f"psym_{i}_{j}")

# (C18) Initial conditions
model.addConstr(t[0]  == 0, name="t0")
model.addConstr(td[0] == 0, name="td0")

# ─────────────────────────────────────────────
# 6. SOLVE
# ─────────────────────────────────────────────

model.optimize()

# ─────────────────────────────────────────────
# 7. RESULTS
# ─────────────────────────────────────────────

result = {"status": "unknown", "obj": None}

if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
    obj_minutes = round(model.ObjVal * 60, 2)
    result["status"] = "optimal" if model.Status == GRB.OPTIMAL else "time_limit"
    result["obj"] = obj_minutes

    print(f"\n{'='*65}")
    print(f"  FSTSP OPTIMAL SOLUTION (Flow-Based, 5 customers)")
    print(f"{'='*65}")
    print(f"  Makespan (total delivery time): {obj_minutes:.2f} minutes\n")

    # Truck route
    truck_edges = [(i, j) for (i, j) in x if x[i, j].X > 0.5]
    route = [0]
    while route[-1] != 6:
        nxt = [j for (i, j) in truck_edges if i == route[-1]]
        if not nxt:
            break
        route.append(nxt[0])

    print("  TRUCK ROUTE:")
    for nd in route:
        arrow = " -> " if nd != route[-1] else ""
        print(f"    Node {nd}  (t={t[nd].X*60:.2f} min){arrow}", end="")
    print()

    # Drone sorties
    active_sorties = [(i, j, k) for (i, j, k) in sorties if y[i, j, k].X > 0.5]
    print("\n  DRONE SORTIES:")
    if active_sorties:
        for (i, j, k) in active_sorties:
            flight_min = (sigma[i, j] + sigma[j, k]) * 60
            print(f"    Launch:{i} -> Deliver:{j} -> Retrieve:{k}  [flight={flight_min:.1f} min]")
    else:
        print("    No drone sorties used.")

    # Flow values
    print("\n  FLOW VALUES (SCF subtour elimination):")
    for (i, j) in truck_arcs:
        if f[i, j].X > 0.01:
            print(f"    f[{i},{j}] = {f[i,j].X:.2f}")

    print(f"\n  CUSTOMER SERVICE SUMMARY:")
    for cust in customers:
        by_truck = any(x[i, cust].X > 0.5 for i in N0 if (i, cust) in x)
        by_drone = any(y[i, cust, k].X > 0.5 for (i, jj, k) in sorties if jj == cust)
        if by_truck:
            print(f"    Customer {cust}: TRUCK   (arrival t={t[cust].X*60:.2f} min)"
                  + ("  [truck-only]" if cust == 5 else ""))
        elif by_drone:
            print(f"    Customer {cust}: DRONE   (delivery t={td[cust].X*60:.2f} min)")

    print(f"\n  Result: {json.dumps(result)}")
    print(f"{'='*65}")
else:
    print(f"No feasible solution found. Gurobi status: {model.Status}")

print(json.dumps(result))
