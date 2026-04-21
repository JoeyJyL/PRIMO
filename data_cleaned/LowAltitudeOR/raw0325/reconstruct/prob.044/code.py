"""
Flying Sidekick Traveling Salesman Problem (FSTSP) — Flow-Based Reformulation
Uses single-commodity flow (SCF) subtour elimination instead of MTZ.
Uses time-based drone sortie ordering instead of position-based ordering.
Based on Murray & Chu (2015), extended from Liu, Li & Khojandi (2022).
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
    1: (3.0, 4.0),   # Customer 1
    2: (6.0, 1.0),   # Customer 2
    3: (8.0, 5.0),   # Customer 3
    4: (2.0, 7.0),   # Customer 4
    5: (0.0, 0.0),   # End depot
}

all_nodes  = list(coords.keys())          # {0,1,2,3,4,5}
customers  = [1, 2, 3, 4]                 # C
drone_elig = [1, 2, 3, 4]                 # C'
N0         = [0, 1, 2, 3, 4]             # departure nodes
Nplus      = [1, 2, 3, 4, 5]             # arrival nodes
drone_ret  = [1, 2, 3, 4]                # valid retrieval nodes for drone
c          = len(customers)               # 4

truck_speed  = 40.2336   # km/h
drone_speed  = 80.4672   # km/h
e            = 0.5       # drone endurance (hours)
s_L          = 1 / 60    # launch time (hours)
s_R          = 1 / 60    # recovery time (hours)
M            = 1000      # big-M constant
EPS          = 0.001     # small epsilon for time-based ordering (hours)

# ─────────────────────────────────────────────
# 2. DISTANCE / TRAVEL TIME MATRICES
# ─────────────────────────────────────────────

def manhattan(a, b):
    return abs(coords[a][0] - coords[b][0]) + abs(coords[a][1] - coords[b][1])

def euclidean(a, b):
    return math.sqrt((coords[a][0]-coords[b][0])**2 + (coords[a][1]-coords[b][1])**2)

# Truck travel time (Manhattan / truck_speed)
s = {(i, j): manhattan(i, j) / truck_speed for i in N0 for j in Nplus if i != j}
# Drone travel time (Euclidean / drone_speed)
sigma = {(i, j): euclidean(i, j) / drone_speed for i in N0 for j in all_nodes if i != j}

# Valid drone sorties: (i, j, k) where i=launch, j=customer, k=retrieval
sorties = [
    (i, j, k)
    for i in N0
    for j in drone_elig
    for k in drone_ret
    if i != j and j != k and i != k
    and sigma.get((i, j), 1e9) + sigma.get((j, k), 1e9) <= e
]

print(f"Number of valid drone sorties: {len(sorties)}")

# ─────────────────────────────────────────────
# 3. GUROBI MODEL
# ─────────────────────────────────────────────

model = gp.Model("FSTSP_FlowBased")
model.setParam("OutputFlag", 1)
model.setParam("TimeLimit", 300)

# ── Decision Variables ──────────────────────

# x[i,j] = 1 if truck travels directly from i to j
x = model.addVars(
    [(i, j) for i in N0 for j in Nplus if i != j],
    vtype=GRB.BINARY, name="x"
)

# y[i,j,k] = 1 if drone launches at i, serves customer j, retrieved at k
y = model.addVars(sorties, vtype=GRB.BINARY, name="y")

# t[j] = truck arrival time at node j
t = model.addVars(all_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="t")

# td[j] = drone arrival time at node j
td = model.addVars(all_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="td")

# f[i,j] = single-commodity flow on truck arc (i,j) — replaces MTZ u variables
truck_arcs = [(i, j) for i in N0 for j in Nplus if i != j]
f = model.addVars(truck_arcs, lb=0.0, ub=c, vtype=GRB.CONTINUOUS, name="f")

# ── Objective ────────────────────────────────
model.setObjective(t[5], GRB.MINIMIZE)

# ── Constraints ──────────────────────────────

# (C1) Each customer served exactly once (truck OR drone)
for j in customers:
    drone_serves_j = gp.quicksum(y[ii, jj, kk] for (ii, jj, kk) in sorties if jj == j)
    truck_serves_j = gp.quicksum(x[i, j] for i in N0 if i != j)
    model.addConstr(truck_serves_j + drone_serves_j == 1, name=f"serve_{j}")

# (C2) Truck departs depot exactly once
model.addConstr(gp.quicksum(x[0, j] for j in Nplus) == 1, name="depot_depart")

# (C3) Truck returns to end depot exactly once
model.addConstr(gp.quicksum(x[i, 5] for i in N0) == 1, name="depot_return")

# (C4) Flow conservation for truck at each customer node
for j in customers:
    model.addConstr(
        gp.quicksum(x[i, j] for i in N0 if i != j) ==
        gp.quicksum(x[j, k] for k in Nplus if k != j),
        name=f"truck_flow_{j}"
    )

# (C5) Single-Commodity Flow Subtour Elimination (replaces MTZ)
# Depot outflow = number of truck-visited customers
truck_visit = {j: gp.quicksum(x[i, j] for i in N0 if i != j) for j in customers}
model.addConstr(
    gp.quicksum(f[0, j] for j in Nplus if (0, j) in f) ==
    gp.quicksum(truck_visit[j] for j in customers),
    name="scf_depot_out"
)
# Each truck-visited customer absorbs one unit of flow
for j in customers:
    model.addConstr(
        gp.quicksum(f[i, j] for i in N0 if i != j and (i, j) in f) -
        gp.quicksum(f[j, k] for k in Nplus if k != j and (j, k) in f) ==
        truck_visit[j],
        name=f"scf_conserve_{j}"
    )
# Flow capacity linking
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
for k in Nplus:
    retrievals = [(ii, jj, kk) for (ii, jj, kk) in sorties if kk == k]
    if retrievals:
        model.addConstr(
            gp.quicksum(y[ii, jj, kk] for (ii, jj, kk) in retrievals) <= 1,
            name=f"drone_retrieve_{k}"
        )

# (C8) Truck must visit BOTH launch node and retrieval node
for (i, j, k) in sorties:
    in_k = gp.quicksum(x[l, k] for l in N0 if l != k and (l, k) in x)
    model.addConstr(y[i, j, k] <= in_k, name=f"truck_visit_ret_{i}_{j}_{k}")
    if i in customers:
        in_i = gp.quicksum(x[h, i] for h in N0 if h != i)
        model.addConstr(y[i, j, k] <= in_i, name=f"truck_visit_launch_{i}_{j}_{k}")

# (C9) Time-based drone sortie ordering (replaces MTZ u-based ordering)
for (i, j, k) in sorties:
    if i in customers and k in customers:
        model.addConstr(
            t[k] >= t[i] + EPS - M * (1 - y[i, j, k]),
            name=f"time_order_{i}_{j}_{k}"
        )

# (C10) Time sync at launch node: drone departs when truck arrives
for i in customers:
    active_launches = [ijk for ijk in sorties if ijk[0] == i]
    if active_launches:
        launch_sum = gp.quicksum(y[ii, jj, kk] for (ii, jj, kk) in active_launches)
        model.addConstr(td[i] >= t[i] - M * (1 - launch_sum), name=f"tsync_launch_lo_{i}")
        model.addConstr(td[i] <= t[i] + M * (1 - launch_sum), name=f"tsync_launch_hi_{i}")

# (C11) Time sync at retrieval node
for k in Nplus:
    active_ret = [ijk for ijk in sorties if ijk[2] == k]
    if active_ret:
        ret_sum = gp.quicksum(y[ii, jj, kk] for (ii, jj, kk) in active_ret)
        model.addConstr(td[k] >= t[k] - M * (1 - ret_sum), name=f"tsync_ret_lo_{k}")
        model.addConstr(td[k] <= t[k] + M * (1 - ret_sum), name=f"tsync_ret_hi_{k}")

# (C12) Truck travel time propagation
for h in N0:
    for k in Nplus:
        if h != k and (h, k) in x:
            launch_at_k = gp.quicksum(
                y[k, jj, ll] for (ii, jj, ll) in sorties if ii == k
            ) if k in N0 else gp.LinExpr(0)
            ret_at_k = gp.quicksum(
                y[ii, jj, k] for (ii, jj, kk) in sorties if kk == k
            )
            model.addConstr(
                t[k] >= t[h] + s[h, k]
                       + s_L * (launch_at_k if k in customers else gp.LinExpr(0))
                       + s_R * ret_at_k
                       - M * (1 - x[h, k]),
                name=f"truck_time_{h}_{k}"
            )

# (C13) Drone travel: launch node → customer j
for (i, j, k) in sorties:
    model.addConstr(
        td[j] >= td[i] + sigma[i, j] - M * (1 - y[i, j, k]),
        name=f"drone_to_cust_{i}_{j}_{k}"
    )

# (C14) Drone travel: customer j → retrieval node k (include s_R)
for (i, j, k) in sorties:
    model.addConstr(
        td[k] >= td[j] + sigma[j, k] + s_R - M * (1 - y[i, j, k]),
        name=f"drone_to_ret_{i}_{j}_{k}"
    )

# (C15) Drone endurance per sortie
for (i, j, k) in sorties:
    model.addConstr(
        sigma[i, j] + sigma[j, k] <= e + M * (1 - y[i, j, k]),
        name=f"endurance_{i}_{j}_{k}"
    )

# (C16) Initial conditions
model.addConstr(t[0]  == 0, name="t0")
model.addConstr(td[0] == 0, name="td0")

# ─────────────────────────────────────────────
# 4. SOLVE
# ─────────────────────────────────────────────

model.optimize()

# ─────────────────────────────────────────────
# 5. EXTRACT & PRINT RESULTS
# ─────────────────────────────────────────────

result = {"status": "unknown", "obj": None}

if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
    obj_minutes = round(model.ObjVal * 60, 2)
    result["status"] = "optimal" if model.Status == GRB.OPTIMAL else "time_limit"
    result["obj"] = obj_minutes

    print(f"\n{'='*60}")
    print(f"  FSTSP OPTIMAL SOLUTION (Flow-Based Formulation)")
    print(f"{'='*60}")
    print(f"  Makespan (total delivery time): {obj_minutes:.2f} minutes\n")

    # Truck route
    truck_edges = [(i, j) for (i, j) in x if x[i, j].X > 0.5]
    route = [0]
    while route[-1] != 5:
        nxt = [j for (i, j) in truck_edges if i == route[-1]]
        if not nxt:
            break
        route.append(nxt[0])

    print("  TRUCK ROUTE:")
    for idx in range(len(route) - 1):
        nd = route[idx]
        print(f"    Node {nd}  (t={t[nd].X*60:.2f} min) → ", end="")
    print(f"Node {route[-1]}  (t={t[route[-1]].X*60:.2f} min)")

    # Drone sorties
    active_sorties = [(i, j, k) for (i, j, k) in sorties if y[i, j, k].X > 0.5]
    print("\n  DRONE SORTIES:")
    if active_sorties:
        for (i, j, k) in active_sorties:
            flight_min = (sigma[i,j] + sigma[j,k]) * 60
            print(f"    Launch at node {i} → Deliver to node {j} → Retrieve at node {k} "
                  f" [flight={flight_min:.1f} min]")
    else:
        print("    No drone sorties used.")

    # Flow values (diagnostic)
    print("\n  FLOW VALUES (SCF subtour elimination):")
    for (i, j) in truck_arcs:
        if f[i, j].X > 0.01:
            print(f"    f[{i},{j}] = {f[i,j].X:.2f}")

    print(f"\n  Result: {json.dumps(result)}")
    print(f"{'='*60}")

else:
    print(f"No solution found. Gurobi status: {model.Status}")

print(json.dumps(result))
