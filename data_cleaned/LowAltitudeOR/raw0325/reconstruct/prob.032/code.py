"""
Air Taxi Skyport Location Problem — Gurobi MILP Solver
=======================================================
Select p=2 skyports from 5 candidates to maximize air taxi revenue.
Big-M formulation with revenue flow variables per model.txt.
"""

import math
import gurobipy as gp
from gurobipy import GRB

# ============================================================
# Data
# ============================================================
origins = [1, 2, 3, 4, 5]
airports = ['A', 'B']
p = 2  # skyports to open

# Monthly demand D[i][j]
D = {
    1: {'A': 150, 'B': 100},
    2: {'A': 200, 'B': 180},
    3: {'A': 120, 'B': 160},
    4: {'A': 180, 'B': 130},
    5: {'A': 100, 'B': 220},
}

# Ground direct: time(min), dist(miles)
ground_direct = {
    (1,'A'): (40, 12), (1,'B'): (55, 18),
    (2,'A'): (50, 15), (2,'B'): (35, 10),
    (3,'A'): (45, 14), (3,'B'): (60, 20),
    (4,'A'): (30,  9), (4,'B'): (50, 16),
    (5,'A'): (65, 22), (5,'B'): (25,  8),
}

# Ground access: time(min) / dist(miles) from origin i to skyport k
ground_access_time = {
    (1,1):0,  (1,2):15, (1,3):20, (1,4):10, (1,5):25,
    (2,1):15, (2,2):0,  (2,3):12, (2,4):18, (2,5):22,
    (3,1):20, (3,2):12, (3,3):0,  (3,4):15, (3,5):18,
    (4,1):10, (4,2):18, (4,3):15, (4,4):0,  (4,5):20,
    (5,1):25, (5,2):22, (5,3):18, (5,4):20, (5,5):0,
}
ground_access_dist = {
    (1,1):0, (1,2):5, (1,3):7, (1,4):3, (1,5):9,
    (2,1):5, (2,2):0, (2,3):4, (2,4):6, (2,5):8,
    (3,1):7, (3,2):4, (3,3):0, (3,4):5, (3,5):6,
    (4,1):3, (4,2):6, (4,3):5, (4,4):0, (4,5):7,
    (5,1):9, (5,2):8, (5,3):6, (5,4):7, (5,5):0,
}

# Aerial distance (skyport k to airport j, air miles)
aerial_dist = {
    (1,'A'):8.0,  (1,'B'):12.0,
    (2,'A'):10.0, (2,'B'):7.0,
    (3,'A'):9.0,  (3,'B'):13.0,
    (4,'A'):6.0,  (4,'B'):11.0,
    (5,'A'):14.0, (5,'B'):5.0,
}

# Fare parameters
base_fee = 3.0
ground_per_mile = 1.50
ground_per_min = 0.30
air_per_mile = 5.73
transfer_cost = 4.50  # $0.30/min × 15 min

print("=" * 70)
print("AIR TAXI SKYPORT LOCATION — GUROBI MILP SOLVER")
print("=" * 70)

# ============================================================
# Pre-compute: utilities, probabilities, revenues
# ============================================================

# Ground taxi fare & utility for direct trip (i,j)
ground_fare = {}
ground_utility = {}
for i in origins:
    for j in airports:
        t, d = ground_direct[(i,j)]
        fare = base_fee + ground_per_mile * d + ground_per_min * t
        ground_fare[(i,j)] = fare
        # V_ground = 0.0313 × Time - 0.0125 × Fare
        ground_utility[(i,j)] = 0.0313 * t - 0.0125 * fare

# Air taxi: for each (i, k, j) combination
air_fare_total = {}  # total air fare for (i,k,j)
air_utility = {}
air_prob = {}  # logit probability
R_coeff = {}   # revenue coefficient

print("\nPre-computing logit probabilities and revenue coefficients...")
print(f"{'(i,k,j)':>10} {'P_air':>8} {'AirFare':>9} {'Revenue':>10}")

for i in origins:
    for k in origins:
        for j in airports:
            # Ground access fare (origin i to skyport k)
            ga_time = ground_access_time[(i,k)]
            ga_dist = ground_access_dist[(i,k)]
            ga_fare = base_fee + ground_per_mile * ga_dist + ground_per_min * ga_time
            if i == k:
                ga_fare = 0.0  # no ground access if at skyport
            
            # Flight fare (skyport k to airport j)
            a_dist = aerial_dist[(k,j)]
            flight_fare = air_per_mile * a_dist
            
            # Total air fare
            total_air = ga_fare + transfer_cost + flight_fare
            air_fare_total[(i,k,j)] = total_air
            
            # V_air = 0.018 × AerialDist - 0.0213 × TotalAirFare
            V_air = 0.018 * a_dist - 0.0213 * total_air
            air_utility[(i,k,j)] = V_air
            
            V_ground = ground_utility[(i,j)]
            
            # Logit probability
            exp_air = math.exp(V_air)
            exp_ground = math.exp(V_ground)
            prob = exp_air / (exp_air + exp_ground)
            air_prob[(i,k,j)] = prob
            
            # Revenue = total_air_fare × probability × demand
            rev = total_air * prob * D[i][j]
            R_coeff[(i,k,j)] = rev
            
            if rev > 50:  # only print significant
                print(f"  ({i},{k},{j}): P={prob:.4f}, Fare=${total_air:.2f}, Rev=${rev:.2f}")

# ============================================================
# Gurobi Model (Big-M formulation from model.txt)
# ============================================================
print(f"\n{'=' * 70}")
print("Building Gurobi MILP (Big-M formulation)...")
print(f"{'=' * 70}")

model = gp.Model("SkyportLocation")
model.setParam('OutputFlag', 1)

# --- Decision Variables ---
# y[k]: 1 if skyport k is opened
y = model.addVars(origins, vtype=GRB.BINARY, name="y")

# x[i,k,j]: 1 if origin i assigned to skyport k for airport j
x = model.addVars(origins, origins, airports, vtype=GRB.BINARY, name="x")

# f[i,k,j]: revenue flow variable
f = model.addVars(origins, origins, airports, lb=0.0, name="f")

# --- Objective: max Σ f[i,k,j] ---
model.setObjective(
    gp.quicksum(f[i,k,j] for i in origins for k in origins for j in airports),
    GRB.MAXIMIZE
)

# --- Constraints ---

# 1. Single allocation: each (i,j) assigned to exactly one skyport
for i in origins:
    for j in airports:
        model.addConstr(
            gp.quicksum(x[i,k,j] for k in origins) == 1,
            name=f"alloc_{i}_{j}"
        )

# 2. Big-M revenue bound (skyport opening): f ≤ R × y_k
for i in origins:
    for k in origins:
        for j in airports:
            model.addConstr(
                f[i,k,j] <= R_coeff[(i,k,j)] * y[k],
                name=f"bigM_{i}_{k}_{j}"
            )

# 3. Revenue activation: f ≤ R × x
for i in origins:
    for k in origins:
        for j in airports:
            model.addConstr(
                f[i,k,j] <= R_coeff[(i,k,j)] * x[i,k,j],
                name=f"activate_{i}_{k}_{j}"
            )

# 4. Revenue completeness: f ≥ R × (x + y - 1)
for i in origins:
    for k in origins:
        for j in airports:
            model.addConstr(
                f[i,k,j] >= R_coeff[(i,k,j)] * (x[i,k,j] + y[k] - 1),
                name=f"complete_{i}_{k}_{j}"
            )

# 5. Budget: exactly p skyports
model.addConstr(
    gp.quicksum(y[k] for k in origins) == p,
    name="budget"
)

# --- Solve ---
n_binary = len(origins) + len(origins)*len(origins)*len(airports)
n_cont = len(origins)*len(origins)*len(airports)
n_constr = (len(origins)*len(airports) + 3*len(origins)**2*len(airports) + 1)
print(f"\nVariables: {n_binary} binary + {n_cont} continuous")
print(f"Constraints: {n_constr}")
print("\nSolving...")
model.optimize()

# ============================================================
# Output
# ============================================================
print(f"\n{'=' * 70}")
print("★ OPTIMAL SOLUTION")
print(f"{'=' * 70}")

if model.status == GRB.OPTIMAL:
    print(f"\n  Total Revenue: ${model.objVal:,.2f}/month")
    
    # Which skyports opened
    opened = [k for k in origins if y[k].X > 0.5]
    print(f"\n  Opened Skyports: {opened}")
    
    # Allocation details
    print(f"\n  Allocation (origin → skyport → airport):")
    total_by_skyport = {k: 0.0 for k in opened}
    total_demand_served = 0
    total_air_pax = 0.0
    
    for j in airports:
        print(f"\n    Airport {j}:")
        for i in origins:
            for k in origins:
                if x[i,k,j].X > 0.5:
                    rev = f[i,k,j].X
                    prob = air_prob[(i,k,j)]
                    demand = D[i][j]
                    air_pax = prob * demand
                    fare = air_fare_total[(i,k,j)]
                    total_by_skyport[k] += rev
                    total_demand_served += demand
                    total_air_pax += air_pax
                    print(f"      Origin {i} → Skyport {k}: "
                          f"demand={demand}, P_air={prob:.4f}, "
                          f"air_pax={air_pax:.1f}, fare=${fare:.2f}, "
                          f"revenue=${rev:,.2f}")
    
    print(f"\n  Revenue by skyport:")
    for k in opened:
        print(f"    Skyport {k}: ${total_by_skyport[k]:,.2f}")
    
    print(f"\n  {'─' * 50}")
    print(f"  SUMMARY:")
    print(f"    Skyports opened: {opened}")
    print(f"    Total monthly revenue: ${model.objVal:,.2f}")
    print(f"    Total demand (all modes): {total_demand_served} trips")
    print(f"    Expected air taxi passengers: {total_air_pax:.0f} trips")
    print(f"    Air taxi mode share: {total_air_pax/total_demand_served*100:.1f}%")
    
    # Compare all C(5,2)=10 skyport pairs
    print(f"\n  {'─' * 50}")
    print(f"  ALL SKYPORT PAIR COMPARISON:")
    
    import itertools
    pair_revs = []
    for pair in itertools.combinations(origins, 2):
        # For each pair, find best allocation
        rev = 0
        for i in origins:
            for j in airports:
                best_k_rev = -1
                for k in pair:
                    r = R_coeff[(i,k,j)]
                    if r > best_k_rev:
                        best_k_rev = r
                rev += best_k_rev
        pair_revs.append((pair, rev))
    
    pair_revs.sort(key=lambda x: -x[1])
    for pair, rev in pair_revs:
        marker = " ★" if set(pair) == set(opened) else ""
        print(f"    Skyports {pair}: ${rev:,.2f}{marker}")

elif model.status == GRB.INFEASIBLE:
    print("\n  INFEASIBLE!")
else:
    print(f"\n  Status: {model.status}")