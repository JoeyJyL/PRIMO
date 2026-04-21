"""
Truck-Drone Collaborative Delivery Problem

Goal: Minimize makespan for delivering to 8 customers
- Truck visits some customers
- Drone launches from truck, serves up to 3 customers, returns to truck
- Each customer served exactly once
"""

import gurobipy as gp
from gurobipy import GRB
import math

# ============================================================
# Data
# ============================================================

# Depot at origin, customers 1-8, return depot 9
depot_start = 0
depot_end = 9
customers = list(range(1, 9))
n_customers = 8

# Customer coordinates and parcel weights
customer_data = {
    1: (4, 10, 0.8),
    2: (8, 2, 1.0),
    3: (12, 0, 0.5),
    4: (10, 9, 0.7),
    5: (16, 3, 1.2),
    6: (20, 1, 1.5),
    7: (6, -2, 0.6),
    8: (14, 7, 0.9),
}

# All node coordinates (depot at origin)
coords = {0: (0, 0)}
coords.update({i: (customer_data[i][0], customer_data[i][1]) for i in customers})
coords[9] = (0, 0)  # Return depot

# Parcel weights
parcel_weight = {i: customer_data[i][2] for i in customers}

# Vehicle parameters
truck_speed = 50  # km/h
drone_speed = 75  # km/h
drone_max_payload = 3  # kg
drone_max_energy = 500  # Wh
alpha_p = 80  # W (base power)
beta_p = 30  # W/kg (power per kg payload)

# Calculate distances
def euclidean_dist(i, j):
    return math.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)

# All nodes
all_nodes = [0] + customers + [9]
C0 = [0] + customers  # Nodes from which can depart
C_plus = customers + [9]  # Nodes to which can arrive

# Distance matrix
dist = {(i, j): euclidean_dist(i, j) for i in all_nodes for j in all_nodes if i != j}

# Big-M
M = 1000

print("="*70)
print("Truck-Drone Collaborative Delivery Problem")
print("="*70)
print(f"Customers: {n_customers}")
print(f"Truck speed: {truck_speed} km/h, Drone speed: {drone_speed} km/h")
print(f"Drone max payload: {drone_max_payload} kg, Max energy: {drone_max_energy} Wh")

# ============================================================
# Model
# ============================================================

model = gp.Model("TruckDrone")

# Decision Variables

# x[i,j]: 1 if truck travels from i to j
x = model.addVars([(i, j) for i in C0 for j in C_plus if i != j], 
                  vtype=GRB.BINARY, name="x")

# z[i]: 1 if customer i is served by drone
z = model.addVars(customers, vtype=GRB.BINARY, name="z")

# L[i]: 1 if drone launches from node i
L = model.addVars(C0, vtype=GRB.BINARY, name="L")

# R[j]: 1 if drone returns at node j
R = model.addVars(C_plus, vtype=GRB.BINARY, name="R")

# t[i]: truck arrival time at node i
t = model.addVars(all_nodes, lb=0, name="t")

# q[i]: truck departure time from node i
q = model.addVars(all_nodes, lb=0, name="q")

# u[i]: position in truck tour (MTZ)
u = model.addVars(customers, lb=1, ub=n_customers, name="u")

# Drone variables
D_drone = model.addVar(lb=0, name="D_drone")  # Total drone flight distance
T_drone = model.addVar(lb=0, name="T_drone")  # Total drone flight time
T_ret = model.addVar(lb=0, name="T_ret")      # Drone return time

# Makespan
T = model.addVar(lb=0, name="T")

# h: whether drone is used (0 or 1)
h = model.addVar(vtype=GRB.BINARY, name="h")

# ============================================================
# Objective: Minimize makespan
# ============================================================

model.setObjective(T, GRB.MINIMIZE)

# ============================================================
# Constraints
# ============================================================

# Makespan >= truck return time and drone return time
model.addConstr(T >= t[9], "Makespan_truck")
model.addConstr(T >= T_ret, "Makespan_drone")

# --- Customer Assignment ---
# Each customer served exactly once by truck or drone
for i in customers:
    model.addConstr(
        gp.quicksum(x[j, i] for j in C0 if j != i) + z[i] == 1,
        f"Assign_{i}"
    )

# --- Truck Routing ---
# Truck leaves depot exactly once
model.addConstr(gp.quicksum(x[0, j] for j in customers) == 1, "TruckLeaveDepot")

# Truck returns to depot exactly once
model.addConstr(gp.quicksum(x[i, 9] for i in customers) == 1, "TruckReturnDepot")

# Flow conservation at customer nodes
for j in customers:
    model.addConstr(
        gp.quicksum(x[i, j] for i in C0 if i != j) == 
        gp.quicksum(x[j, k] for k in C_plus if k != j),
        f"FlowCons_{j}"
    )

# --- Subtour Elimination (MTZ) ---
for i in customers:
    for j in customers:
        if i != j:
            model.addConstr(
                u[i] - u[j] + n_customers * x[i, j] <= n_customers - 1,
                f"MTZ_{i}_{j}"
            )

# --- Truck Timing ---
model.addConstr(t[0] == 0, "TruckStart")
model.addConstr(q[0] == 0, "TruckDepart0")

# Truck arrival time
for i in C0:
    for j in C_plus:
        if i != j:
            model.addConstr(
                t[j] >= q[i] + dist[i, j] / truck_speed - M * (1 - x[i, j]),
                f"TruckTime_{i}_{j}"
            )

# Truck departure time >= arrival time
for i in C_plus:
    model.addConstr(q[i] >= t[i], f"Depart_{i}")

# --- Drone Sortie ---
# Number of launch points = number of return points = h
model.addConstr(gp.quicksum(L[i] for i in C0) == h, "LaunchCount")
model.addConstr(gp.quicksum(R[j] for j in C_plus) == h, "ReturnCount")

# Drone can only launch/return at truck-visited nodes
for i in C0:
    if i == 0:
        model.addConstr(L[i] <= 1, "LaunchDepot")  # Can always launch from depot
    else:
        model.addConstr(L[i] <= gp.quicksum(x[j, i] for j in C0 if j != i), f"LaunchAtTruck_{i}")

for j in C_plus:
    if j == 9:
        model.addConstr(R[j] <= 1, "ReturnDepot")  # Can always return to depot
    else:
        model.addConstr(R[j] <= gp.quicksum(x[j, k] for k in C_plus if k != j), f"ReturnAtTruck_{j}")

# Max 3 customers served by drone per sortie
model.addConstr(gp.quicksum(z[i] for i in customers) <= 3 * h, "MaxDroneCustomers")

# If no drone used, no customers served by drone
model.addConstr(gp.quicksum(z[i] for i in customers) >= h, "MinDroneCustomers")

# --- Drone Payload ---
model.addConstr(
    gp.quicksum(parcel_weight[i] * z[i] for i in customers) <= drone_max_payload,
    "DronePayload"
)

# --- Drone Distance and Energy ---
# Drone distance: from launch to all drone customers and back to return
# Simplified: calculate based on selected customers
# We need auxiliary variables for drone route

# For simplicity, we approximate drone distance as:
# distance from launch point to drone customers (TSP) plus return
# This is a simplification - for exact model would need more variables

# Create auxiliary variables for drone customer sequence
# Approximate: sum of distances from depot to each drone customer and back
drone_dist_approx = model.addVar(lb=0, name="drone_dist_approx")

# Simplified: drone flies from launch, visits each drone customer, returns to rendezvous
# Approximate total distance
model.addConstr(
    drone_dist_approx >= gp.quicksum(2 * dist[0, i] * z[i] for i in customers) - M * (1 - h),
    "DroneDistApprox"
)

model.addConstr(D_drone >= drone_dist_approx, "DroneDistLink")

# Drone flight time
model.addConstr(T_drone == D_drone / drone_speed, "DroneTime")

# Drone energy constraint
# Power = alpha_p + beta_p * payload
# Energy = Power * Time (in hours) -> convert to Wh
drone_payload = gp.quicksum(parcel_weight[i] * z[i] for i in customers)
model.addConstr(
    (alpha_p + beta_p * drone_max_payload) * T_drone <= drone_max_energy / 1000,  # Convert Wh to kWh, time in h
    "DroneEnergy"
)

# --- Synchronization ---
# Truck waits for drone at rendezvous point
for j in C_plus:
    model.addConstr(q[j] >= T_ret - M * (1 - R[j]), f"Sync_{j}")

# Drone return time
for i in C0:
    model.addConstr(
        T_ret >= q[i] + T_drone - M * (1 - L[i]),
        f"DroneRet_{i}"
    )

# --- Launch before return in truck tour ---
for i in customers:
    for j in customers:
        if i != j:
            model.addConstr(
                u[i] <= u[j] - 1 + M * (2 - L[i] - R[j]),
                f"LaunchBeforeReturn_{i}_{j}"
            )

# If drone launches from depot (0), it can return anywhere
# If drone returns to depot (9), it was launched from somewhere

# ============================================================
# Solve
# ============================================================

model.Params.OutputFlag = 1
model.Params.TimeLimit = 120
model.Params.MIPGap = 0.01
model.optimize()

# ============================================================
# Results
# ============================================================

print("\n" + "="*70)
print("RESULTS")
print("="*70)

if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
    result = model.objVal
    
    print(f"\nMinimum Makespan: {result:.4f} hours = {result * 60:.2f} minutes")
    
    # Truck route
    print("\n--- Truck Route ---")
    truck_route = [0]
    current = 0
    while current != 9:
        for j in C_plus:
            if j != current and (current, j) in x and x[current, j].X > 0.5:
                truck_route.append(j)
                current = j
                break
    print(f"Route: {' -> '.join(map(str, truck_route))}")
    
    # Truck-served customers
    truck_customers = [i for i in customers if z[i].X < 0.5]
    print(f"Truck serves customers: {truck_customers}")
    
    # Drone-served customers
    drone_customers = [i for i in customers if z[i].X > 0.5]
    print(f"\n--- Drone ---")
    print(f"Drone serves customers: {drone_customers}")
    
    if h.X > 0.5:
        launch_node = [i for i in C0 if L[i].X > 0.5]
        return_node = [j for j in C_plus if R[j].X > 0.5]
        print(f"Launch from: {launch_node}")
        print(f"Return to: {return_node}")
        print(f"Drone flight distance: {D_drone.X:.2f} km")
        print(f"Drone flight time: {T_drone.X:.4f} hours = {T_drone.X * 60:.2f} minutes")
    
    # Timing
    print("\n--- Timing ---")
    print(f"Truck return time: {t[9].X:.4f} hours = {t[9].X * 60:.2f} minutes")
    print(f"Drone return time: {T_ret.X:.4f} hours = {T_ret.X * 60:.2f} minutes")
    print(f"Makespan: {T.X:.4f} hours = {T.X * 60:.2f} minutes")
    
    print(f"\nresult = {result}")
else:
    result = None
    print(f"Optimization failed with status: {model.status}")