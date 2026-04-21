"""
UAM eVTOL Fleet Total Cost of Ownership (TCO) Optimization
WITH Fast-Charging Mandate (150kW or 350kW only, 50kW excluded)

Goal: Select optimal vehicle concept, battery capacity, and charging infrastructure
to minimize daily TCO while ensuring all flights can be completed.
"""

import gurobipy as gp
from gurobipy import GRB

# ============================================================
# Parameters
# ============================================================

# Vehicle concepts
# k=0: TiltWing, k=1: LiftCruise
vehicle_types = [0, 1]
empty_weight = {0: 363, 1: 700}  # Empty weights (kg) - Vahana ~363kg, Cora ~700kg
max_weight = {0: 815, 1: 1224}   # Max takeoff weight (kg)
max_pax = {0: 1, 1: 2}
max_speed = {0: 200, 1: 180}

# Energy consumption parameters (per kg)
hover_power = {0: 0.055, 1: 0.042}  # kW/kg
cruise_energy = {0: 0.00028, 1: 0.00035}  # kWh/km/kg

# Charging infrastructure (FAST-CHARGING MANDATE: r=0 excluded)
# r=0: 50kW (EXCLUDED), r=1: 150kW, r=2: 350kW
charger_types = [1, 2]  # Only 150kW and 350kW permitted
charger_power = {1: 150, 2: 350}  # kW
charger_hw_cost = {1: 75000, 2: 140000}
charger_install_cost = {1: 47781, 2: 65984}

# Battery parameters
energy_density = 0.4  # kWh/kg
power_density = 3.0   # kW/kg
usable_fraction = 0.8
battery_cost_per_kwh = 285  # $/kWh
cycle_life = 2000

# Operations
n_vehicles = 2
n_stations = 4
parking_slots = 2
cruise_speed = 150  # kph
hover_time = 60 / 3600  # 60 seconds in hours
turnaround_time = 3 / 60  # 3 minutes in hours
pax_weight = 90  # kg

# Cost parameters
vehicle_cost_per_kg = 770  # $/kg
vehicle_life = 13  # years
residual_value = 0.30
ops_days_per_year = 330
maintenance_per_vehicle = 51300  # $/year
charger_life = 13  # years
energy_price = 0.25  # $/kWh
maneuver_factor = 1.0  # anti-collision factor

# Flight sequences
# Vehicle 0: trips 0, 1, 2
# Vehicle 1: trips 3, 4, 5
trips = {
    # trip_id: (vehicle, distance_km, pax, deadhead_after, dh_distance, time_gap_hours)
    0: (0, 18, 1, False, 0, 0.5),
    1: (0, 25, 1, True, 12, 0.8),
    2: (0, 15, 1, False, 0, 0),
    3: (1, 22, 1, True, 8, 0.6),
    4: (1, 30, 1, False, 0, 0.4),
    5: (1, 20, 1, False, 0, 0),
}

sequences = {
    0: [0, 1, 2],  # Vehicle 0's trips
    1: [3, 4, 5],  # Vehicle 1's trips
}

print("="*70)
print("UAM eVTOL Fleet TCO Optimization")
print("WITH FAST-CHARGING MANDATE (150kW/350kW only)")
print("="*70)

# ============================================================
# Model
# ============================================================

model = gp.Model("UAM_TCO_FastCharge")

# Decision Variables
# Binary: vehicle concept selection
y_k = model.addVars(vehicle_types, vtype=GRB.BINARY, name="y_k")

# Binary: charger type selection (only 150kW and 350kW)
y_r = model.addVars(charger_types, vtype=GRB.BINARY, name="y_r")

# Integer: battery weight (kg)
Q_w = model.addVar(lb=50, ub=300, vtype=GRB.INTEGER, name="Q_w")

# Continuous: SOC before each trip (kWh)
q = model.addVars(trips.keys(), lb=0, name="q")

# Continuous: energy recharged before each trip (kWh)
w = model.addVars(trips.keys(), lb=0, name="w")

# Continuous: additional recharge before deadheading (kWh)
u = model.addVars(trips.keys(), lb=0, name="u")

# Auxiliary: linearization variables
total_mass = model.addVars(vehicle_types, lb=0, name="total_mass")

# Selected charger power
selected_power = model.addVar(lb=0, name="selected_power")

# ============================================================
# Constraints
# ============================================================

# Selection constraints
model.addConstr(gp.quicksum(y_k[k] for k in vehicle_types) == 1, "SelectOneVehicle")
model.addConstr(gp.quicksum(y_r[r] for r in charger_types) == 1, "SelectOneCharger")

# Link selected charger power
model.addConstr(selected_power == gp.quicksum(charger_power[r] * y_r[r] for r in charger_types), "ChargerPower")

# Big-M for linearization
M_mass = 1000
M_energy = 500

# Total mass when vehicle k is selected
for k in vehicle_types:
    model.addConstr(total_mass[k] <= (empty_weight[k] + Q_w) + M_mass * (1 - y_k[k]), f"Mass_ub_{k}")
    model.addConstr(total_mass[k] >= (empty_weight[k] + Q_w) - M_mass * (1 - y_k[k]), f"Mass_lb_{k}")
    model.addConstr(total_mass[k] <= M_mass * y_k[k], f"Mass_zero_{k}")

# Function to calculate trip energy
def trip_energy_expr(trip_id, k, with_pax=True):
    _, dist, pax, _, _, _ = trips[trip_id]
    pax_mass = pax_weight * pax if with_pax else 0
    
    hover_energy = hover_power[k] * (empty_weight[k] + pax_mass) * hover_time * 2
    cruise_e = cruise_energy[k] * (empty_weight[k] + pax_mass) * dist
    hover_battery = hover_power[k] * hover_time * 2 * Q_w
    cruise_battery = cruise_energy[k] * dist * Q_w
    
    return hover_energy + cruise_e + hover_battery + cruise_battery

# For each sequence, set up constraints
for seq_id, trip_list in sequences.items():
    for idx, trip_id in enumerate(trip_list):
        veh, dist, pax, dh_after, dh_dist, time_gap = trips[trip_id]
        
        # Maximum SOC constraint
        model.addConstr(
            q[trip_id] + w[trip_id] <= usable_fraction * Q_w * energy_density,
            f"MaxSOC_{trip_id}"
        )
        
        # Minimum energy for trip
        for k in vehicle_types:
            E_trip = trip_energy_expr(trip_id, k, with_pax=True)
            model.addConstr(
                q[trip_id] + w[trip_id] >= E_trip - M_energy * (1 - y_k[k]),
                f"MinEnergy_{trip_id}_{k}"
            )
        
        # Recharge limits and SOC linkage for consecutive trips
        if idx > 0:
            prev_trip_id = trip_list[idx - 1]
            _, _, _, prev_dh, prev_dh_dist, prev_time_gap = trips[prev_trip_id]
            
            if prev_dh:
                recharge_time = turnaround_time
            else:
                recharge_time = prev_time_gap
            
            model.addConstr(
                w[trip_id] <= selected_power * recharge_time,
                f"RechargeLimit_{trip_id}"
            )
            
            for k in vehicle_types:
                E_prev = trip_energy_expr(prev_trip_id, k, with_pax=True)
                
                if prev_dh:
                    E_dh = (hover_power[k] * empty_weight[k] * hover_time * 2 +
                            cruise_energy[k] * empty_weight[k] * prev_dh_dist +
                            hover_power[k] * hover_time * 2 * Q_w +
                            cruise_energy[k] * prev_dh_dist * Q_w)
                    
                    model.addConstr(
                        q[trip_id] <= q[prev_trip_id] + w[prev_trip_id] - E_prev + u[prev_trip_id] - E_dh + M_energy * (1 - y_k[k]),
                        f"SOCLink_{prev_trip_id}_{trip_id}_{k}"
                    )
                    
                    dh_flight_time = hover_time * 2 + prev_dh_dist / cruise_speed
                    u_max_time = prev_time_gap - turnaround_time - dh_flight_time
                    if u_max_time > 0:
                        model.addConstr(u[prev_trip_id] <= selected_power * u_max_time, f"ULimit_{prev_trip_id}")
                    else:
                        model.addConstr(u[prev_trip_id] == 0, f"UZero_{prev_trip_id}")
                else:
                    model.addConstr(
                        q[trip_id] <= q[prev_trip_id] + w[prev_trip_id] - E_prev + M_energy * (1 - y_k[k]),
                        f"SOCLink_{prev_trip_id}_{trip_id}_{k}"
                    )
                    model.addConstr(u[prev_trip_id] == 0, f"UZero_{prev_trip_id}")
        else:
            model.addConstr(q[trip_id] == usable_fraction * Q_w * energy_density, f"InitSOC_{trip_id}")
            model.addConstr(w[trip_id] == 0, f"InitW_{trip_id}")

# Last trips don't need u
for seq_id, trip_list in sequences.items():
    last_trip = trip_list[-1]
    model.addConstr(u[last_trip] == 0, f"LastU_{last_trip}")

# Max takeoff weight constraint
for k in vehicle_types:
    model.addConstr(
        empty_weight[k] + Q_w + pax_weight * max_pax[k] <= max_weight[k] + M_mass * (1 - y_k[k]),
        f"MaxWeight_{k}"
    )

# Min battery power constraint
for k in vehicle_types:
    min_power_approx = hover_power[k] * max_weight[k]
    model.addConstr(
        Q_w * power_density >= min_power_approx - M_mass * (1 - y_k[k]),
        f"MinPower_{k}"
    )

# ============================================================
# Objective Function
# ============================================================

# Daily fleet cost
daily_vehicle_cost = gp.quicksum(
    y_k[k] * (n_vehicles * vehicle_cost_per_kg * empty_weight[k] * (1 - residual_value) / (vehicle_life * ops_days_per_year))
    for k in vehicle_types
)

# Battery depreciation cost
total_energy_used = gp.quicksum(w[i] + u[i] for i in trips.keys())
battery_daily_cost = total_energy_used * battery_cost_per_kwh / cycle_life

# Daily maintenance cost
daily_maintenance = n_vehicles * maintenance_per_vehicle / ops_days_per_year

# Daily infrastructure cost
daily_infra_cost = gp.quicksum(
    y_r[r] * (n_stations * parking_slots * (charger_hw_cost[r] + charger_install_cost[r]) / (charger_life * ops_days_per_year))
    for r in charger_types
)

# Daily energy cost
daily_energy_cost = model.addVar(lb=0, name="daily_energy")

for k in vehicle_types:
    total_trip_energy = 0
    for trip_id in trips.keys():
        total_trip_energy += trip_energy_expr(trip_id, k, with_pax=True)
    
    for seq_id, trip_list in sequences.items():
        for idx, trip_id in enumerate(trip_list[:-1]):
            _, _, _, dh_after, dh_dist, _ = trips[trip_id]
            if dh_after:
                total_trip_energy += (hover_power[k] * empty_weight[k] * hover_time * 2 +
                                     cruise_energy[k] * empty_weight[k] * dh_dist +
                                     hover_power[k] * hover_time * 2 * Q_w +
                                     cruise_energy[k] * dh_dist * Q_w)
    
    model.addConstr(
        daily_energy_cost >= energy_price * maneuver_factor * total_trip_energy - M_energy * (1 - y_k[k]),
        f"EnergyCost_{k}"
    )

# Total objective
total_cost = daily_vehicle_cost + battery_daily_cost + daily_maintenance + daily_infra_cost + daily_energy_cost

model.setObjective(total_cost, GRB.MINIMIZE)

# ============================================================
# Solve
# ============================================================

model.Params.OutputFlag = 1
model.Params.MIPGap = 0.01
model.optimize()

# ============================================================
# Results
# ============================================================

print("\n" + "="*70)
print("OPTIMIZATION RESULTS")
print("="*70)

if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
    result = model.objVal
    
    print(f"\nMinimum Daily TCO: ${result:.2f}")
    
    # Selected vehicle
    for k in vehicle_types:
        if y_k[k].X > 0.5:
            veh_name = "TiltWing" if k == 0 else "LiftCruise"
            print(f"\nSelected Vehicle: {veh_name} (k={k})")
            print(f"  Empty Weight: {empty_weight[k]} kg")
    
    # Selected charger
    for r in charger_types:
        if y_r[r].X > 0.5:
            print(f"\nSelected Charger: {charger_power[r]} kW (r={r})")
            print(f"  (Fast-charging mandate satisfied: ≥150 kW)")
    
    # Battery
    print(f"\nBattery Weight: {Q_w.X:.0f} kg")
    print(f"Battery Capacity: {Q_w.X * energy_density:.1f} kWh")
    print(f"Usable Capacity: {Q_w.X * energy_density * usable_fraction:.1f} kWh")
    
    # Cost breakdown
    print("\n--- Cost Breakdown (Daily) ---")
    veh_cost = sum(y_k[k].X * (n_vehicles * vehicle_cost_per_kg * empty_weight[k] * (1 - residual_value) / (vehicle_life * ops_days_per_year)) for k in vehicle_types)
    print(f"  Vehicle Depreciation: ${veh_cost:.2f}")
    print(f"  Battery Depreciation: ${battery_daily_cost.getValue():.2f}")
    print(f"  Maintenance: ${daily_maintenance:.2f}")
    infra_cost = sum(y_r[r].X * (n_stations * parking_slots * (charger_hw_cost[r] + charger_install_cost[r]) / (charger_life * ops_days_per_year)) for r in charger_types)
    print(f"  Infrastructure: ${infra_cost:.2f}")
    print(f"  Energy: ${daily_energy_cost.X:.2f}")
    
    print(f"\nresult = {result}")
else:
    result = None
    print(f"Optimization failed with status: {model.status}")