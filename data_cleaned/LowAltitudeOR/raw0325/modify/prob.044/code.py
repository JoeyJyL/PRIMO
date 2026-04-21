"""
Flying Sidekick Traveling Salesman Problem (FSTSP)
Solved using Gurobi MIP formulation.
Based on Murray & Chu (2015), extended from Liu, Li & Khojandi (2022).

Modification: Mutual exclusion — drone cannot serve both Customer 1 and Customer 4.
"""

import gurobipy as gp
from gurobipy import GRB
import math
import itertools

# ─────────────────────────────────────────────
# 1. PROBLEM DATA
# ─────────────────────────────────────────────

# Node coordinates (km): 0=start depot, 1-4=customers, 5=end depot
coords = {
    0: (0.0, 0.0),   # Start depot
    1: (3.0, 4.0),   # Customer 1
    2: (6.0, 1.0),   # Customer 2
    3: (8.0, 5.0),   # Customer 3
    4: (2.0, 7.0),   # Customer 4
    5: (0.0, 0.0),   # End depot
}

# Sets
all_nodes  = list(coords.keys())          # {0,1,2,3,4,5}
customers  = [1, 2, 3, 4]                 # C
drone_elig = [1, 2, 3, 4]                 # C' (all customers)
N0         = [0, 1, 2, 3, 4]             # departure nodes
Nplus      = [1, 2, 3, 4, 5]             # arrival nodes (truck)
# Drone retrieval must be a customer node (truck), NOT the end depot
drone_ret  = [1, 2, 3, 4]                # valid retrieval nodes for drone
c          = len(customers)               # 4

# Speed & timing parameters
truck_speed  = 40.2336   # km/h  (25 mph)
drone_speed  = 80.4672   # km/h  (50 mph)
e            = 0.5       # drone endurance (hours)
s_L          = 1 / 60    # launch time (hours)
s_R          = 1 / 60    # recovery time (hours)
M            = 1000      # big-M constant

# ─────────────────────────────────────────────
# 2. DISTANCE / TRAVEL TIME MATRICES
# ─────────────────────────────────────────────

def manhattan(a, b):
    return abs(coords[a][0] - coords[b][0]) + abs(coords[a][1] - coords[b][1])

def euclidean(a, b):
    return math.sqrt((coords[a][0]-coords[b][0])**2 + (coords[a][1]-coords[b][1])**2)

# Truck travel time (Manhattan / truck_speed)
s  = {(i, j): manhattan(i, j) / truck_speed  for i in N0     for j in Nplus if i != j}
# Drone travel time (Euclidean / drone_speed)
sigma = {(i, j): euclidean(i, j) / drone_speed for i in N0   for j in all_nodes if i != j}

# Valid drone sorties: (i, j, k) where i=launch, j=customer, k=retrieval
# Drone must return to the TRUCK (a customer node), NOT the end depot (node 5)
sorties = [
    (i, j, k)
    for i in N0
    for j in drone_elig
    for k in drone_ret          # <-- retrieval restricted to customer nodes only
    if i != j and j != k and i != k
    and sigma.get((i, j), 1e9) + sigma.get((j, k), 1e9) <= e
]

print(f"Number of valid drone sorties: {len(sorties)}")
for s_tuple in sorties:
    i, j, k = s_tuple
    flight = sigma[i,j] + sigma[j,k]
    print(f"  Launch:{i} -> Drone-deliver:{j} -> Retrieve:{k}  "
          f"(flight={flight*60:.1f} min, limit={e*60:.0f} min)")

# ─────────────────────────────────────────────
# 3. GUROBI MODEL
# ─────────────────────────────────────────────

model = gp.Model("FSTSP")
model.setParam("OutputFlag", 1)
model.setParam("TimeLimit", 300)   # 5-minute time limit

# ── Decision Variables ──────────────────────

# x[i,j] = 1 if truck travels directly from i to j
x = model.addVars(
    [(i, j) for i in N0 for j in Nplus if i != j],
    vtype=GRB.BINARY, name="x"
)

# y[i,j,k] = 1 if drone launches at i, serves customer j, retrieved at k
y = model.addVars(sorties, vtype=GRB.BINARY, name="y")

# t[j]  = truck arrival time at node j
t = model.addVars(all_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="t")

# td[j] = drone arrival time at node j
td = model.addVars(all_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="td")

# p[i,j] = 1 if customer i is visited before j in the truck route
p = model.addVars(
    [(i, j) for i in customers for j in customers if i != j],
    vtype=GRB.BINARY, name="p"
)

# u[i] = position of node i in truck route (MTZ subtour elimination)
u = model.addVars(customers, lb=1, ub=c + 2, vtype=GRB.CONTINUOUS, name="u")

# ── Objective ────────────────────────────────
# Minimize makespan = truck return time to end depot (node 5)
model.setObjective(t[5], GRB.MINIMIZE)

# ── Constraints ──────────────────────────────

# (C1) Each customer served exactly once (truck OR drone)
for j in customers:
    model.addConstr(
        gp.quicksum(x[i, j] for i in N0 if i != j) +
        gp.quicksum(y[i, j, k] for (ii, jj, kk) in sorties if jj == j for i in [ii] for k in [kk])
        == 1,
        name=f"serve_{j}"
    )

# (C2) Truck departs depot exactly once
model.addConstr(gp.quicksum(x[0, j] for j in Nplus) == 1, name="depot_depart")

# (C3) Truck returns to end depot exactly once
model.addConstr(gp.quicksum(x[i, 5] for i in N0) == 1, name="depot_return")

# (C4) Flow conservation for truck at each customer node
for j in customers:
    model.addConstr(
        gp.quicksum(x[i, j] for i in N0 if i != j) ==
        gp.quicksum(x[j, k] for k in Nplus if k != j),
        name=f"flow_{j}"
    )

# (C5) MTZ subtour elimination
for i in customers:
    for j in [jj for jj in Nplus if jj != i and jj in customers]:
        model.addConstr(
            u[i] - u[j] + 1 <= (c + 2) * (1 - x[i, j]),
            name=f"mtz_{i}_{j}"
        )

# (C6) Drone launches from any node at most once
for i in N0:
    launches = [(ii, jj, kk) for (ii, jj, kk) in sorties if ii == i]
    if launches:
        model.addConstr(gp.quicksum(y[ii, jj, kk] for (ii, jj, kk) in launches) <= 1,
                        name=f"drone_launch_{i}")

# (C7) Drone retrieval at any node at most once
for k in Nplus:
    retrievals = [(ii, jj, kk) for (ii, jj, kk) in sorties if kk == k]
    if retrievals:
        model.addConstr(gp.quicksum(y[ii, jj, kk] for (ii, jj, kk) in retrievals) <= 1,
                        name=f"drone_retrieve_{k}")

# (C8) Truck must visit BOTH launch node i AND retrieval node k.
for (i, j, k) in sorties:
    # Retrieval node k must always be visited by the truck
    in_k = gp.quicksum(x[l, k] for l in N0 if l != k and (l, k) in x)
    model.addConstr(y[i, j, k] <= in_k, name=f"truck_visit_ret_{i}_{j}_{k}")
    # Launch node i must also be visited by the truck (depot i=0 is trivially visited)
    if i in customers:
        in_i = gp.quicksum(x[h, i] for h in N0 if h != i)
        model.addConstr(y[i, j, k] <= in_i, name=f"truck_visit_launch_{i}_{j}_{k}")

# (C9) Ordering: if drone launches at i (customer), retrieves at k → truck visits i before k
for (i, j, k) in sorties:
    if i in customers and k in customers:
        model.addConstr(
            u[k] - u[i] >= 1 - (c + 2) * (1 - y[i, j, k]),
            name=f"order_{i}_{j}_{k}"
        )

# (C10) Time sync at launch node: drone departs when truck arrives
for i in customers:
    active_launches = [ijk for ijk in sorties if ijk[0] == i]
    if active_launches:
        launch_sum = gp.quicksum(y[ii, jj, kk] for (ii, jj, kk) in active_launches)
        model.addConstr(td[i] >= t[i] - M * (1 - launch_sum), name=f"tsync_launch_lo_{i}")
        model.addConstr(td[i] <= t[i] + M * (1 - launch_sum), name=f"tsync_launch_hi_{i}")

# (C11) Time sync at retrieval node: truck waits for drone (or drone waits for truck)
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
            # Extra time for launch / recovery at node k
            launch_at_k = gp.quicksum(y[k, jj, ll] for (ii, jj, ll) in sorties if ii == k) if k in N0 else gp.LinExpr(0)
            ret_at_k    = gp.quicksum(y[ii, jj, k]  for (ii, jj, kk) in sorties if kk == k)
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
    # Total drone air time = sigma_ij + sigma_jk ≤ e
    model.addConstr(
        sigma[i, j] + sigma[j, k] <= e + M * (1 - y[i, j, k]),
        name=f"endurance_{i}_{j}_{k}"
    )

# (C16) Initial conditions
model.addConstr(t[0]  == 0, name="t0")
model.addConstr(td[0] == 0, name="td0")

# (C17) Ordering variable consistency
for i in customers:
    for j in customers:
        if i != j:
            model.addConstr(p[i, j] + p[j, i] == 1, name=f"pij_sym_{i}_{j}")

# Depot always first in ordering
for j in customers:
    model.addConstr(u[j] >= 2, name=f"u_lb_{j}")   # depot is position 1

# (C18) Mutual exclusion: drone cannot serve both Customer 3 and Customer 4
model.addConstr(
    gp.quicksum(y[i, j, k] for (i, j, k) in sorties if j == 3) +
    gp.quicksum(y[i, j, k] for (i, j, k) in sorties if j == 4) <= 1,
    name="mutual_excl_3_4"
)

# ─────────────────────────────────────────────
# 4. SOLVE
# ─────────────────────────────────────────────

model.optimize()

# ─────────────────────────────────────────────
# 5. EXTRACT & PRINT RESULTS
# ─────────────────────────────────────────────
result = {"status": "unknown", "obj": None}



if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
    result["status"] = "optimal" if model.status == GRB.OPTIMAL else "time_limit"
    result["obj"] = round(model.ObjBound*60,2)
    print("\n" + "="*60)
    print("  FSTSP OPTIMAL SOLUTION")
    print("="*60)
    print(f"  Makespan (total delivery time): {t[5].X*60:.2f} minutes\n")

    # Truck route
    truck_edges = [(i, j) for (i, j) in x if x[i, j].X > 0.5]
    # Build ordered route
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
            print(f"    Launch at node {i} (t={td[i].X*60:.2f} min) "
                  f"→ Deliver to node {j} → Retrieve at node {k} "
                  f"(t={td[k].X*60:.2f} min)  [flight={flight_min:.1f} min]")
    else:
        print("    No drone sorties used.")

    # Service summary
    print("\n  CUSTOMER SERVICE SUMMARY:")
    for cust in customers:
        served_by_truck = any(x[i, cust].X > 0.5 for i in N0 if (i, cust) in x)
        served_by_drone = any(y[i, cust, k].X > 0.5 for (i, jj, k) in sorties if jj == cust)
        if served_by_truck:
            print(f"    Customer {cust}: served by TRUCK  (arrival t={t[cust].X*60:.2f} min)")
        elif served_by_drone:
            print(f"    Customer {cust}: served by DRONE  (arrival t={td[cust].X*60:.2f} min)")
        else:
            print(f"    Customer {cust}: ⚠ NOT SERVED (check model)")

    print(f"\n status: {result}  |  Obj: {model.ObjBound*60:.2f} min")
    print("="*60)

else:
    print(f"No solution found. Gurobi status: {model.Status}")
