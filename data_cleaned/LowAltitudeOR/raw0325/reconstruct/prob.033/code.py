"""
UAM Fleet TCO Optimization — Gurobi MILP Solver
=================================================
Minimize daily Total Cost of Ownership for eVTOL operations.
McCormick linearization for bilinear terms y_k * Q_w.
"""

import math
import gurobipy as gp
from gurobipy import GRB

# ============================================================
# Data
# ============================================================

# Vehicle concepts
vehicles = {
    0: {'name': 'TiltWing', 'hover_kw_kg': 0.055, 'cruise_kwh_km_kg': 0.00028,
        'empty_weight': 450, 'max_pax': 1, 'max_speed': 200, 'max_tow': 900},
    1: {'name': 'LiftCruise', 'hover_kw_kg': 0.042, 'cruise_kwh_km_kg': 0.00035,
        'empty_weight': 600, 'max_pax': 2, 'max_speed': 180, 'max_tow': 1200},
}

# Charger types
chargers = {
    0: {'name': '50kW', 'power_kw': 50, 'hw_cost': 28401, 'inst_cost': 45506},
    1: {'name': '150kW', 'power_kw': 150, 'hw_cost': 75000, 'inst_cost': 47781},
    2: {'name': '350kW', 'power_kw': 350, 'hw_cost': 140000, 'inst_cost': 65984},
}

# Operations
n_vehicles = 2
n_stations = 4
n_slots = 2  # parking slots per station
cruise_speed = 150  # kph
hover_time = 60 / 3600  # 60 seconds in hours
turnaround = 3 / 60  # 3 minutes in hours
pax_weight = 90  # kg
maneuver_factor = 1.0  # η^man (no extra specified, assume 1.0)

# Battery
energy_density = 0.4  # kWh/kg
power_density = 3.0   # kW/kg
usable_fraction = 0.8
battery_cost_per_kwh = 285  # $/kWh
energy_price = 0.25  # $/kWh
cycle_life = 2000

# Cost
vehicle_cost_per_kg = 770  # $/kg empty weight
vehicle_life = 13  # years
residual_value = 0.30
ops_days = 330
maintenance_per_vehicle_year = 51300
charger_life = 13  # years

# Battery weight bounds
Qw_L = 50   # min battery weight (kg)
Qw_U = 300  # max battery weight (kg)

# Flight sequences
# Vehicle 0: trips 0, 1, 2
# Vehicle 1: trips 3, 4, 5
sequences = {
    0: {  # Vehicle 0
        'trips': [
            {'id': 0, 'dist': 18, 'pax': 1, 'dh_after': False, 'dh_dist': 0},
            {'id': 1, 'dist': 25, 'pax': 1, 'dh_after': True,  'dh_dist': 12},
            {'id': 2, 'dist': 15, 'pax': 1, 'dh_after': False, 'dh_dist': 0},
        ],
        'gaps': [0.5, 0.8],  # time gap after trip 0, after trip 1
    },
    1: {  # Vehicle 1
        'trips': [
            {'id': 3, 'dist': 22, 'pax': 1, 'dh_after': True,  'dh_dist': 8},
            {'id': 4, 'dist': 30, 'pax': 1, 'dh_after': False, 'dh_dist': 0},
            {'id': 5, 'dist': 20, 'pax': 1, 'dh_after': False, 'dh_dist': 0},
        ],
        'gaps': [0.6, 0.4],
    },
}

print("=" * 70)
print("UAM FLEET TCO OPTIMIZATION — GUROBI MILP")
print("=" * 70)

# ============================================================
# Precompute energy coefficients
# ============================================================
# E_trip = η * (α * Qw + β) * y_k, where:
#   α_k = e_hover * t_hover_total + e_cruise * dist  (per-kg-battery coefficient)
#   β_k = (EW + pax*pax_weight) * (same energy rate)  (fixed-mass coefficient)
# With McCormick: E_trip = η * α_k * P_k + η * β_k * y_k

def energy_coeffs(k_idx, dist_km, n_pax):
    """Compute energy coefficients α (battery-dependent) and β (fixed) for a flight."""
    veh = vehicles[k_idx]
    hover_total = hover_time * 2  # takeoff + landing
    alpha = veh['hover_kw_kg'] * hover_total + veh['cruise_kwh_km_kg'] * dist_km
    fixed_mass = veh['empty_weight'] + pax_weight * n_pax
    beta = fixed_mass * (veh['hover_kw_kg'] * hover_total + veh['cruise_kwh_km_kg'] * dist_km)
    return alpha, beta

def deadhead_coeffs(k_idx, dh_dist_km):
    """Deadhead: no passengers."""
    return energy_coeffs(k_idx, dh_dist_km, 0)

# ============================================================
# Gurobi Model
# ============================================================
print("\nBuilding model...")
model = gp.Model("UAM_TCO")
model.setParam('OutputFlag', 1)

K = [0, 1]
R = [0, 1, 2]

# --- Decision Variables ---
Qw = model.addVar(lb=Qw_L, ub=Qw_U, vtype=GRB.INTEGER, name="Qw")
y_k = model.addVars(K, vtype=GRB.BINARY, name="y_k")
y_r = model.addVars(R, vtype=GRB.BINARY, name="y_r")

# McCormick: P_k = y_k * Qw
P_k = model.addVars(K, lb=0, ub=Qw_U, name="P_k")

# Trip/deadhead energy
all_trip_ids = []
all_dh_ids = []
for s, seq in sequences.items():
    for trip in seq['trips']:
        all_trip_ids.append((s, trip['id']))
        if trip['dh_after']:
            all_dh_ids.append((s, trip['id']))

E_trip = model.addVars(all_trip_ids, lb=0, name="E_trip")
E_dh = model.addVars(all_dh_ids, lb=0, name="E_dh")

# SOC and recharge variables
q = model.addVars(all_trip_ids, lb=0, name="q")   # SOC before trip
w = model.addVars(all_trip_ids, lb=0, name="w")   # recharge before trip
u = {}  # recharge before deadhead
for s, tid in all_dh_ids:
    u[s, tid] = model.addVar(lb=0, name=f"u_{s}_{tid}")

# --- Constraints ---

# 1. Exactly one vehicle type and one charger type
model.addConstr(gp.quicksum(y_k[k] for k in K) == 1, "one_vehicle")
model.addConstr(gp.quicksum(y_r[r] for r in R) == 1, "one_charger")

# 2. McCormick envelope for P_k = y_k * Qw
for k in K:
    model.addConstr(P_k[k] >= Qw_L * y_k[k], f"mc_lb1_{k}")
    model.addConstr(P_k[k] <= Qw_U * y_k[k], f"mc_ub1_{k}")
    model.addConstr(P_k[k] >= Qw - Qw_U * (1 - y_k[k]), f"mc_lb2_{k}")
    model.addConstr(P_k[k] <= Qw - Qw_L * (1 - y_k[k]), f"mc_ub2_{k}")

# 3. Energy consumption definitions
for s, seq in sequences.items():
    for trip in seq['trips']:
        tid = trip['id']
        # Trip energy
        model.addConstr(
            E_trip[s, tid] == gp.quicksum(
                maneuver_factor * energy_coeffs(k, trip['dist'], trip['pax'])[0] * P_k[k] +
                maneuver_factor * energy_coeffs(k, trip['dist'], trip['pax'])[1] * y_k[k]
                for k in K
            ),
            f"Etrip_{s}_{tid}"
        )
        # Deadhead energy
        if trip['dh_after']:
            model.addConstr(
                E_dh[s, tid] == gp.quicksum(
                    maneuver_factor * deadhead_coeffs(k, trip['dh_dist'])[0] * P_k[k] +
                    maneuver_factor * deadhead_coeffs(k, trip['dh_dist'])[1] * y_k[k]
                    for k in K
                ),
                f"Edh_{s}_{tid}"
            )

# 4. Max SOC (battery capacity)
for s, tid in all_trip_ids:
    model.addConstr(
        q[s, tid] + w[s, tid] <= usable_fraction * energy_density * Qw,
        f"max_soc_{s}_{tid}"
    )

# 5. Min energy for each trip
for s, tid in all_trip_ids:
    model.addConstr(
        q[s, tid] + w[s, tid] >= E_trip[s, tid],
        f"min_energy_{s}_{tid}"
    )

# 6. SOC linkage between consecutive trips
for s, seq in sequences.items():
    trips = seq['trips']
    gaps = seq['gaps']
    for idx in range(len(trips) - 1):
        i_trip = trips[idx]
        j_trip = trips[idx + 1]
        i_id = i_trip['id']
        j_id = j_trip['id']
        has_dh = i_trip['dh_after']
        
        if has_dh:
            model.addConstr(
                q[s, j_id] <= q[s, i_id] + w[s, i_id] - E_trip[s, i_id]
                + u[s, i_id] - E_dh[s, i_id],
                f"soc_link_{s}_{i_id}_{j_id}"
            )
        else:
            model.addConstr(
                q[s, j_id] <= q[s, i_id] + w[s, i_id] - E_trip[s, i_id],
                f"soc_link_{s}_{i_id}_{j_id}"
            )

# 7. Recharge limits (charger power × available time)
for s, seq in sequences.items():
    trips = seq['trips']
    gaps = seq['gaps']
    
    # First trip: no recharge before it (departs with initial SOC)
    model.addConstr(w[s, trips[0]['id']] == 0, f"no_w_first_{s}")
    
    # Subsequent trips
    for idx in range(1, len(trips)):
        j_trip = trips[idx]
        i_trip = trips[idx - 1]
        j_id = j_trip['id']
        gap = gaps[idx - 1]
        has_dh = i_trip['dh_after']
        dh_dist = i_trip['dh_dist']
        
        if has_dh:
            # Recharge time = turnaround only (dh takes flight time from gap)
            recharge_time = turnaround
        else:
            # Recharge time = full gap
            recharge_time = gap
        
        # w_j <= s_r * recharge_time for selected r
        model.addConstr(
            w[s, j_id] <= gp.quicksum(
                chargers[r]['power_kw'] * recharge_time * y_r[r] for r in R
            ),
            f"w_limit_{s}_{j_id}"
        )

# 8. Recharge before deadhead
for s, seq in sequences.items():
    trips = seq['trips']
    gaps = seq['gaps']
    for idx in range(len(trips) - 1):
        i_trip = trips[idx]
        if i_trip['dh_after']:
            i_id = i_trip['id']
            gap = gaps[idx]
            dh_dist = i_trip['dh_dist']
            # Time for deadhead flight
            dh_flight_time = hover_time * 2 + dh_dist / cruise_speed
            # Available recharge time before deadhead
            avail_time = gap - turnaround - dh_flight_time
            if avail_time < 0:
                avail_time = 0
            
            model.addConstr(
                u[s, i_id] <= gp.quicksum(
                    chargers[r]['power_kw'] * avail_time * y_r[r] for r in R
                ),
                f"u_limit_{s}_{i_id}"
            )

# 9. Initial SOC = full usable capacity
for s, seq in sequences.items():
    first_id = seq['trips'][0]['id']
    model.addConstr(
        q[s, first_id] == usable_fraction * energy_density * Qw,
        f"init_soc_{s}"
    )

# 10. Vehicle feasibility (applied via big-M with y_k)
for k in K:
    veh = vehicles[k]
    # Max TOW: EW + Qw + pax_weight * max_pax <= max_tow
    model.addConstr(
        veh['empty_weight'] + Qw + pax_weight * 1 <= veh['max_tow'] + Qw_U * (1 - y_k[k]),
        f"tow_{k}"
    )
    # Max speed
    if cruise_speed > veh['max_speed']:
        model.addConstr(y_k[k] == 0, f"speed_{k}")
    # Min battery power (for hover): Qw * power_density >= hover_kw_kg * max_tow
    # This ensures enough power for takeoff at max weight
    min_power_needed = veh['hover_kw_kg'] * veh['max_tow']  # kW needed
    min_qw_for_power = min_power_needed / power_density
    model.addConstr(
        Qw >= min_qw_for_power - Qw_U * (1 - y_k[k]),
        f"min_power_{k}"
    )

# --- Objective Function ---

# Fleet acquisition cost (daily)
fleet_acq = gp.quicksum(
    n_vehicles * vehicles[k]['empty_weight'] * vehicle_cost_per_kg * (1 - residual_value)
    / (vehicle_life * ops_days) * y_k[k]
    for k in K
)

# Battery depreciation cost (per kWh recharged, amortized over cycle life)
# Each kWh recharged costs battery_cost_per_kwh / cycle_life
battery_dep = (battery_cost_per_kwh / cycle_life) * gp.quicksum(
    w[s, tid] + (u[s, tid] if (s, tid) in u else 0)
    for s, tid in all_trip_ids
)

# Maintenance (daily)
maint_daily = n_vehicles * maintenance_per_vehicle_year / ops_days

# Infrastructure (daily)
infra = gp.quicksum(
    n_stations * n_slots * (chargers[r]['hw_cost'] + chargers[r]['inst_cost'])
    / (charger_life * ops_days) * y_r[r]
    for r in R
)

# Energy cost
energy_cost = energy_price * gp.quicksum(
    E_trip[s, tid] for s, tid in all_trip_ids
) + energy_price * gp.quicksum(
    E_dh[s, tid] for s, tid in all_dh_ids
)

# Battery acquisition cost (daily amortization)
# Battery cost = Qw * energy_density * battery_cost_per_kwh, amortized over life
# Battery life ≈ cycle_life / (trips_per_day) days... simplified: use vehicle life
battery_acq = n_vehicles * energy_density * battery_cost_per_kwh * Qw / (vehicle_life * ops_days)

model.setObjective(
    fleet_acq + battery_dep + maint_daily + infra + energy_cost + battery_acq,
    GRB.MINIMIZE
)

# --- Solve ---
print("\nSolving...")
model.optimize()

# ============================================================
# Output
# ============================================================
print(f"\n{'=' * 70}")
print("★ OPTIMAL SOLUTION")
print(f"{'=' * 70}")

if model.status == GRB.OPTIMAL:
    print(f"\n  Daily TCO: ${model.objVal:.2f}")
    
    # Selected vehicle
    sel_k = [k for k in K if y_k[k].X > 0.5][0]
    sel_r = [r for r in R if y_r[r].X > 0.5][0]
    qw_val = round(Qw.X)
    
    print(f"\n  Vehicle: {vehicles[sel_k]['name']} (k={sel_k})")
    print(f"    Empty weight: {vehicles[sel_k]['empty_weight']} kg")
    print(f"    Battery weight: {qw_val} kg")
    print(f"    Battery capacity: {qw_val * energy_density:.1f} kWh "
          f"(usable: {qw_val * energy_density * usable_fraction:.1f} kWh)")
    print(f"    TOW: {vehicles[sel_k]['empty_weight'] + qw_val + pax_weight} kg "
          f"(max: {vehicles[sel_k]['max_tow']} kg)")
    
    print(f"\n  Charger: {chargers[sel_r]['name']} (r={sel_r}), {chargers[sel_r]['power_kw']} kW")
    
    # Cost breakdown
    fleet_val = sum(
        n_vehicles * vehicles[k]['empty_weight'] * vehicle_cost_per_kg * (1 - residual_value)
        / (vehicle_life * ops_days) * y_k[k].X for k in K
    )
    bat_dep_val = (battery_cost_per_kwh / cycle_life) * sum(
        w[s, tid].X + (u[s, tid].X if (s, tid) in u else 0)
        for s, tid in all_trip_ids
    )
    infra_val = sum(
        n_stations * n_slots * (chargers[r]['hw_cost'] + chargers[r]['inst_cost'])
        / (charger_life * ops_days) * y_r[r].X for r in R
    )
    energy_val = energy_price * (
        sum(E_trip[s, tid].X for s, tid in all_trip_ids) +
        sum(E_dh[s, tid].X for s, tid in all_dh_ids)
    )
    bat_acq_val = n_vehicles * energy_density * battery_cost_per_kwh * qw_val / (vehicle_life * ops_days)
    
    print(f"\n  COST BREAKDOWN (daily):")
    print(f"    Fleet acquisition:  ${fleet_val:.2f}")
    print(f"    Battery acquisition: ${bat_acq_val:.2f}")
    print(f"    Battery degradation: ${bat_dep_val:.2f}")
    print(f"    Maintenance:        ${maint_daily:.2f}")
    print(f"    Infrastructure:     ${infra_val:.2f}")
    print(f"    Energy:             ${energy_val:.2f}")
    print(f"    ─────────────────────────")
    print(f"    TOTAL:              ${model.objVal:.2f}")
    
    # Flight details
    print(f"\n  FLIGHT SEQUENCE DETAILS:")
    for s, seq in sequences.items():
        print(f"\n    Vehicle {s}:")
        for trip in seq['trips']:
            tid = trip['id']
            soc_before = q[s, tid].X
            recharge = w[s, tid].X
            e_trip_val = E_trip[s, tid].X
            soc_after = soc_before + recharge - e_trip_val
            
            print(f"      Trip {tid}: {trip['dist']}km, {trip['pax']}PAX")
            print(f"        SOC before: {soc_before:.2f} kWh, "
                  f"recharge: +{recharge:.2f} kWh, "
                  f"energy used: {e_trip_val:.2f} kWh, "
                  f"SOC after: {soc_after:.2f} kWh")
            
            if trip['dh_after'] and (s, tid) in u:
                e_dh_val = E_dh[s, tid].X
                u_val = u[s, tid].X
                soc_after_dh = soc_after + u_val - e_dh_val
                print(f"        Deadhead: {trip['dh_dist']}km, "
                      f"recharge before DH: +{u_val:.2f} kWh, "
                      f"DH energy: {e_dh_val:.2f} kWh, "
                      f"SOC after DH: {soc_after_dh:.2f} kWh")
    
    # Verification
    print(f"\n  VERIFICATION:")
    print(f"    Vehicle type: {vehicles[sel_k]['name']}  ✓")
    print(f"    Charger type: {chargers[sel_r]['name']}  ✓")
    cap = qw_val * energy_density * usable_fraction
    print(f"    Usable capacity: {cap:.1f} kWh")
    
    all_ok = True
    for s, tid in all_trip_ids:
        avail = q[s, tid].X + w[s, tid].X
        needed = E_trip[s, tid].X
        if avail < needed - 0.01:
            print(f"    SOC VIOLATION: trip {tid}: avail={avail:.2f} < need={needed:.2f}")
            all_ok = False
    if all_ok:
        print(f"    All trips have sufficient energy  ✓")
    
    tow = vehicles[sel_k]['empty_weight'] + qw_val + pax_weight
    print(f"    TOW check: {tow} ≤ {vehicles[sel_k]['max_tow']}  "
          f"{'✓' if tow <= vehicles[sel_k]['max_tow'] else '✗'}")

elif model.status == GRB.INFEASIBLE:
    print("\n  INFEASIBLE!")
    model.computeIIS()
    for c in model.getConstrs():
        if c.IISConstr:
            print(f"    IIS: {c.ConstrName}")
else:
    print(f"\n  Status: {model.status}")