# -*- coding: utf-8 -*-
"""
Air Taxi Skyport Location Problem — Revenue Maximization
Single-allocation p-hub median with elastic demand (binary logit choice model)
Select p=2 skyports from 5 candidates to maximize total air taxi revenue.
"""

import gurobipy as gp
from gurobipy import GRB
import math

def solve_skyport():
    # =========================================================
    # 1. DATA
    # =========================================================
    L = [1, 2, 3, 4, 5]       # origins & candidate skyport locations
    J = ['A', 'B']             # airports
    p = 2                       # skyports to open

    # Demand D[i,j] (monthly trips)
    D = {
        (1,'A'):150, (1,'B'):100,
        (2,'A'):200, (2,'B'):180,
        (3,'A'):120, (3,'B'):160,
        (4,'A'):180, (4,'B'):130,
        (5,'A'):100, (5,'B'):220,
    }

    # Ground direct travel: (time_min, dist_miles)
    ground = {
        (1,'A'):(40,12), (1,'B'):(55,18),
        (2,'A'):(50,15), (2,'B'):(35,10),
        (3,'A'):(45,14), (3,'B'):(60,20),
        (4,'A'):(30,9),  (4,'B'):(50,16),
        (5,'A'):(65,22), (5,'B'):(25,8),
    }

    # Ground access i->k: (time_min, dist_miles)
    access_data = {
        (1,1):(0,0),  (1,2):(15,5), (1,3):(20,7), (1,4):(10,3), (1,5):(25,9),
        (2,1):(15,5), (2,2):(0,0),  (2,3):(12,4), (2,4):(18,6), (2,5):(22,8),
        (3,1):(20,7), (3,2):(12,4), (3,3):(0,0),  (3,4):(15,5), (3,5):(18,6),
        (4,1):(10,3), (4,2):(18,6), (4,3):(15,5), (4,4):(0,0),  (4,5):(20,7),
        (5,1):(25,9), (5,2):(22,8), (5,3):(18,6), (5,4):(20,7), (5,5):(0,0),
    }

    # Aerial distance k->j (miles)
    aerial = {
        (1,'A'):8.0,  (1,'B'):12.0,
        (2,'A'):10.0, (2,'B'):7.0,
        (3,'A'):9.0,  (3,'B'):13.0,
        (4,'A'):6.0,  (4,'B'):11.0,
        (5,'A'):14.0, (5,'B'):5.0,
    }

    # Fare parameters
    base_fee = 3.0
    per_mile = 1.50
    per_min = 0.30
    flight_rate = 5.73      # $/air-mile
    transfer_cost = 4.50    # $0.30/min * 15 min

    # =========================================================
    # 2. PRE-COMPUTE PARAMETERS
    # =========================================================

    # Ground taxi fare
    f_ground = {}
    for (i, j), (t, d) in ground.items():
        f_ground[i, j] = base_fee + per_mile * d + per_min * t

    # Ground access fare (origin to skyport)
    f_access = {}
    for (i, k), (t, d) in access_data.items():
        if i == k:
            f_access[i, k] = 0.0  # no cost if at skyport
        else:
            f_access[i, k] = base_fee + per_mile * d + per_min * t

    # Flight fare (skyport to airport)
    f_flight = {}
    for (k, j), d in aerial.items():
        f_flight[k, j] = flight_rate * d

    # Total air fare and revenue
    f_total_air = {}   # total fare passenger pays
    revenue = {}       # operator revenue per passenger
    for i in L:
        for k in L:
            for j in J:
                f_total_air[i, k, j] = f_access[i, k] + transfer_cost + f_flight[k, j]
                revenue[i, k, j] = f_access[i, k] + f_flight[k, j]  # excl. transfer cost

    # Ground utility
    V_ground = {}
    for (i, j), (t, d) in ground.items():
        V_ground[i, j] = 0.0313 * t - 0.0125 * f_ground[i, j]

    # Air utility and choice probability
    V_air = {}
    theta = {}  # choice probability
    for i in L:
        for k in L:
            for j in J:
                d_aerial = aerial[k, j]
                V_air[i, k, j] = 0.018 * d_aerial - 0.0213 * f_total_air[i, k, j]
                exp_air = math.exp(V_air[i, k, j])
                exp_gnd = math.exp(V_ground[i, j])
                theta[i, k, j] = exp_air / (exp_air + exp_gnd)

    # Objective coefficient: revenue * theta * demand
    obj_coeff = {}
    for i in L:
        for k in L:
            for j in J:
                obj_coeff[i, k, j] = revenue[i, k, j] * theta[i, k, j] * D[i, j]

    # Print summary
    print("=== Revenue coefficient summary (top 10) ===")
    sorted_coeffs = sorted(obj_coeff.items(), key=lambda x: -x[1])
    for (i, k, j), val in sorted_coeffs[:10]:
        print(f"  ({i},{k},{j}): rev=${revenue[i,k,j]:.2f} * theta={theta[i,k,j]:.4f} * D={D[i,j]} = ${val:.2f}")

    total_demand = sum(D.values())
    print(f"\nTotal monthly demand: {total_demand} trips")

    # =========================================================
    # 3. MODEL
    # =========================================================
    model = gp.Model("Skyport_REV")
    model.setParam("OutputFlag", 1)

    # Decision variables
    y = model.addVars(L, vtype=GRB.BINARY, name="y")
    x = model.addVars(
        [(i, k, j) for i in L for k in L for j in J],
        vtype=GRB.BINARY, name="x"
    )

    # Objective: maximize total revenue
    model.setObjective(
        gp.quicksum(obj_coeff[i, k, j] * x[i, k, j] for i in L for k in L for j in J),
        GRB.MAXIMIZE
    )

    # (1) Single allocation: each (i,j) assigned to exactly one skyport
    for i in L:
        for j in J:
            model.addConstr(
                gp.quicksum(x[i, k, j] for k in L) == 1,
                f"alloc_{i}_{j}"
            )

    # (2) Hub linking: can only route via open skyport
    for i in L:
        for k in L:
            for j in J:
                model.addConstr(x[i, k, j] <= y[k], f"link_{i}_{k}_{j}")

    # (3) Budget: exactly p skyports
    model.addConstr(gp.quicksum(y[k] for k in L) == p, "budget")

    # =========================================================
    # 4. SOLVE
    # =========================================================
    model.optimize()

    # =========================================================
    # 5. RESULTS
    # =========================================================
    if model.SolCount > 0:
        print(f"\n{'='*65}")
        print("AIR TAXI SKYPORT LOCATION — OPTIMAL SOLUTION")
        print(f"{'='*65}")
        print(f"Maximum Revenue = ${model.ObjVal:,.2f}/month")

        # Selected skyports
        selected = [k for k in L if y[k].X > 0.5]
        print(f"\nSelected Skyports: {selected}")
        for k in selected:
            total_rev = sum(obj_coeff[i, k, j] * x[i, k, j].X for i in L for j in J)
            print(f"  Skyport {k}: revenue = ${total_rev:,.2f}/month")

        # Allocation details
        print(f"\nAllocation (origin -> skyport -> airport):")
        for i in L:
            for j in J:
                for k in L:
                    if x[i, k, j].X > 0.5:
                        pax = theta[i, k, j] * D[i, j]
                        rev = obj_coeff[i, k, j]
                        print(f"  ({i}->{k}->{j}): demand={D[i,j]}, P_air={theta[i,k,j]:.4f}, "
                              f"exp_pax={pax:.1f}, fare=${revenue[i,k,j]:.2f}, rev=${rev:.2f}")

        # Revenue by airport
        print(f"\nRevenue by airport:")
        for j in J:
            rev_j = sum(obj_coeff[i, k, j] * x[i, k, j].X for i in L for k in L)
            print(f"  Airport {j}: ${rev_j:,.2f}/month")

        # Revenue by origin
        print(f"\nRevenue by origin zone:")
        for i in L:
            rev_i = sum(obj_coeff[i, k, j] * x[i, k, j].X for k in L for j in J)
            print(f"  Zone {i}: ${rev_i:,.2f}/month")

        # Expected passenger summary
        print(f"\nExpected passengers:")
        total_pax = 0
        for i in L:
            for j in J:
                for k in L:
                    if x[i, k, j].X > 0.5:
                        pax = theta[i, k, j] * D[i, j]
                        total_pax += pax
        print(f"  Total expected air taxi passengers: {total_pax:.1f}/month")
        print(f"  Total demand: {total_demand}/month")
        print(f"  Air taxi mode share: {total_pax/total_demand*100:.1f}%")

    return model


if __name__ == "__main__":
    solve_skyport()