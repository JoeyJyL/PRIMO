# -*- coding: utf-8 -*-
"""
Multi-Visit Drone Routing Problem for Pickup and Delivery (MDRP-PD) - CORRECTED
Based on: Meng et al. (2023), Transportation Research Part E 169 (2023) 102990.

Fixes applied:
  1. Node 3 coordinates: (4.0, 5.0) -> (4.0, 4.0) per question.txt
  2. Energy service term: P*u/3600 -> P*u (units already Wh)
  3. Energy arc bounds: loose [alpha*w0*r, alpha*(w0+W)*r] -> exact alpha*(w0+w_total)*r
  4. Removed nonlinear term in truck load constraint
"""

import gurobipy as gp
from gurobipy import GRB
import math


def solve_mdrp_pd():
    # =========================================================
    # 1. DATA (CORRECTED)
    # =========================================================
    N_c = [1, 2, 3]
    N_o = [0, 1, 2, 3]
    N_e = [1, 2, 3, 4]
    N_d = [1, 2, 3]
    N_all = [0, 1, 2, 3, 4]

    # FIX 1: Node 3 coordinates corrected from (4,5) to (4,4)
    coords = {
        0: (0.0, 0.0),
        1: (2.0, 3.0),
        2: (5.0, 1.0),
        3: (4.0, 4.0),   # FIXED: was (4.0, 5.0)
        4: (0.0, 0.0),
    }

    def dist(i, j):
        xi, yi = coords[i]
        xj, yj = coords[j]
        return math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

    r = {}
    for i in N_o:
        for j in N_e:
            if i != j:
                r[i, j] = dist(i, j)

    d_demand = {1: 1.0, 2: 0.8, 3: 0.0}
    p_demand = {1: 0.5, 2: 0.0, 3: 1.2}

    Q = 10.0; v_T = 30.0; c_T = 0.78; c_0 = 22.0
    u_truck = {1: 1/12, 2: 1/30, 3: 1/20}

    W = 3.0; w_0 = 6.0; alpha = 3.5; E = 504.0
    P_power = 1008.0; c_F = 0.00248; BR = 1/60
    u_drone = {1: 1/12, 2: 1/60, 3: 1/30}

    M_big = 1e4
    A = [(i, j) for i in N_o for j in N_e if i != j]

    # =========================================================
    # 2. MODEL
    # =========================================================
    model = gp.Model("MDRP_PD_corrected")
    model.setParam("OutputFlag", 1)
    model.setParam("TimeLimit", 300)

    # =========================================================
    # 3. DECISION VARIABLES
    # =========================================================
    x0 = model.addVars(A, vtype=GRB.BINARY, name="x0")
    x1 = model.addVars(A, vtype=GRB.BINARY, name="x1")
    x2 = model.addVars(A, vtype=GRB.BINARY, name="x2")
    z1 = model.addVars(N_c, vtype=GRB.BINARY, name="z1")
    z2 = model.addVars(N_d, vtype=GRB.BINARY, name="z2")

    s1  = model.addVars(N_all, lb=0.0, name="s1")
    s1p = model.addVars(N_all, lb=0.0, name="s1p")
    s2  = model.addVars(N_all, lb=0.0, name="s2")
    s2p = model.addVars(N_all, lb=0.0, name="s2p")

    e_arc  = model.addVars(A, lb=0.0, ub=E, name="e_arc")
    e_node = model.addVars(N_o, lb=0.0, ub=E, name="e_node")

    w_total = model.addVars(N_o, lb=0.0, ub=W, name="w_total")
    w_del   = model.addVars(N_o, lb=0.0, ub=W, name="w_del")
    v_truck = model.addVars(N_o, lb=0.0, ub=Q, name="v_truck")

    # =========================================================
    # 4. OBJECTIVE
    # =========================================================
    truck_cost = gp.quicksum(
        c_T * r[i, j] * (x0[i, j] + x1[i, j])
        for (i, j) in A if not (i == 0 and j == 4)
    )
    drone_energy_cost = gp.quicksum(c_F * e_arc[i, j] for (i, j) in A)
    fixed_cost = c_0 * (1 - x0[0, 4])

    model.setObjective(truck_cost + drone_energy_cost + fixed_cost, GRB.MINIMIZE)

    # =========================================================
    # 5. CONSTRAINTS
    # =========================================================

    # --- Routing ---
    model.addConstr(
        gp.quicksum(x0[0, j] + x1[0, j] for j in N_e if (0, j) in A) == 1,
        "truck_depart")
    model.addConstr(
        gp.quicksum(x0[i, 4] + x1[i, 4] for i in N_o if (i, 4) in A) == 1,
        "truck_return")
    model.addConstr(
        gp.quicksum(x0[0, j] + x2[0, j] for j in N_e if (0, j) in A) == 1,
        "drone_depart")
    model.addConstr(
        gp.quicksum(x0[i, 4] + x2[i, 4] for i in N_o if (i, 4) in A) == 1,
        "drone_return")

    for (i, j) in A:
        if not (i == 0 and j == 4):
            model.addConstr(x0[i, j] + x1[i, j] + x2[i, j] <= 1,
                            f"arc_usage_{i}_{j}")

    # Flow conservation - truck
    for j in N_c:
        in_arcs  = [(i, j) for i in N_o if (i, j) in A]
        out_arcs = [(j, l) for l in N_e if (j, l) in A]
        model.addConstr(
            gp.quicksum(x0[i, j] + x1[i, j] for (i, j) in in_arcs) == z1[j],
            f"truck_flow_in_{j}")
        model.addConstr(
            gp.quicksum(x0[j, l] + x1[j, l] for (j, l) in out_arcs) == z1[j],
            f"truck_flow_out_{j}")

    # Flow conservation - drone
    for j in N_c:
        in_arcs  = [(i, j) for i in N_o if (i, j) in A]
        out_arcs = [(j, l) for l in N_e if (j, l) in A]
        model.addConstr(
            gp.quicksum(x0[i, j] + x2[i, j] for (i, j) in in_arcs) ==
            gp.quicksum(x0[j, l] + x2[j, l] for (j, l) in out_arcs),
            f"drone_flow_{j}")

    # Drone service requires visit
    for j in N_d:
        in_x2  = [(i, j) for i in N_o if (i, j) in A]
        out_x2 = [(j, l) for l in N_e if (j, l) in A]
        model.addConstr(
            2 * z2[j] <= gp.quicksum(x2[i, j] for (i, j) in in_x2) +
                          gp.quicksum(x2[j, l] for (j, l) in out_x2),
            f"drone_serve_{j}")

    # Each customer served exactly once
    for j in N_c:
        model.addConstr(z1[j] + z2[j] == 1, f"serve_once_{j}")

    # Launch/retrieval requires truck
    for i in N_c:
        out_x2 = [(i, j) for j in N_e if (i, j) in A]
        model.addConstr(
            z1[i] >= gp.quicksum(x2[i, j] for (i, j) in out_x2) - z2[i],
            f"truck_launch_{i}")
    for i in N_c:
        in_x2 = [(j, i) for j in N_o if (j, i) in A]
        model.addConstr(
            z1[i] >= gp.quicksum(x2[j, i] for (j, i) in in_x2) - z2[i],
            f"truck_retrieve_{i}")

    # Drone visit limits
    for i in N_c:
        out_x2 = [(i, j) for j in N_e if (i, j) in A]
        model.addConstr(gp.quicksum(x2[i, j] for (i, j) in out_x2) <= 1,
                        f"drone_leave_{i}")
    for i in N_c:
        in_x2 = [(j, i) for j in N_o if (j, i) in A]
        model.addConstr(gp.quicksum(x2[j, i] for (j, i) in in_x2) <= 1,
                        f"drone_arrive_{i}")

    # --- Timing ---
    for (i, j) in A:
        tt = r[i, j] / v_T
        model.addConstr(s1[j] >= s1p[i] + tt - M_big * (1 - x0[i, j] - x1[i, j]),
                        f"truck_time_lb_{i}_{j}")
        model.addConstr(s1[j] <= s1p[i] + tt + M_big * (1 - x0[i, j] - x1[i, j]),
                        f"truck_time_ub_{i}_{j}")

    for i in N_c:
        in_x2 = [(j, i) for j in N_d if (j, i) in A]
        model.addConstr(
            s1p[i] >= s1[i] + u_truck[i] * z1[i]
            + BR * gp.quicksum(x2[j, i] for (j, i) in in_x2),
            f"truck_depart_time_{i}")

    for (i, j) in A:
        avg_w = w_0 + W / 2
        dt = alpha * avg_w * r[i, j] / P_power
        model.addConstr(s2[j] >= s2p[i] + dt - M_big * (1 - x2[i, j]),
                        f"drone_time_lb_{i}_{j}")
        model.addConstr(s2[j] <= s2p[i] + dt + M_big * (1 - x2[i, j]),
                        f"drone_time_ub_{i}_{j}")

    for i in N_d:
        model.addConstr(s2p[i] >= s2[i] + u_drone[i] - M_big * (1 - z2[i]),
                        f"drone_depart_time_{i}")
        model.addConstr(s2p[i] <= s2[i] + u_drone[i] + M_big * (1 - z2[i]),
                        f"drone_depart_time_ub_{i}")

    # Sync: launch
    for i in N_o:
        out_x2    = [(i, j) for j in N_d if (i, j) in A]
        out_truck = [(i, j) for j in N_e if (i, j) in A]
        if out_x2 and out_truck:
            model.addConstr(
                s1p[i] >= s2p[i] - M_big * (
                    2 - gp.quicksum(x0[i, j] + x1[i, j] for (i, j) in out_truck)
                      - gp.quicksum(x2[i, j] for (i, j) in out_x2)),
                f"sync_launch_{i}")

    # Sync: retrieval
    for i in N_c:
        in_x2 = [(j, i) for j in N_d if (j, i) in A]
        if in_x2:
            model.addConstr(
                s1p[i] >= s2[i] + BR - M_big * (
                    2 - gp.quicksum(x2[j, i] for (j, i) in in_x2) - z1[i]),
                f"sync_retrieve_{i}")

    # --- Drone energy (FIX 2 & 3) ---
    for (i, j) in A:
        model.addConstr(e_arc[i, j] <= E * x2[i, j],
                        f"energy_zero_{i}_{j}")

    # FIX 3: Exact energy = alpha * (w_0 + w_total[i]) * r[i,j]
    # Linearized with big-M:
    for (i, j) in A:
        if i in N_o:
            model.addConstr(
                e_arc[i, j] >= alpha * w_0 * r[i, j] * x2[i, j]
                + alpha * r[i, j] * w_total[i]
                - M_big * (1 - x2[i, j]),
                f"energy_exact_lb_{i}_{j}")
            model.addConstr(
                e_arc[i, j] <= alpha * w_0 * r[i, j]
                + alpha * r[i, j] * w_total[i]
                + M_big * (1 - x2[i, j]),
                f"energy_exact_ub_{i}_{j}")

    # Energy reset at launch nodes
    for i in N_o:
        out_truck = [(i, j) for j in N_e if (i, j) in A]
        out_x2   = [(i, j) for j in N_d if (i, j) in A]
        model.addConstr(
            e_node[i] <= E * (
                2 - gp.quicksum(x0[i, j] + x1[i, j] for (i, j) in out_truck)
                  - gp.quicksum(x2[i, j] for (i, j) in out_x2)),
            f"energy_reset_{i}")

    # FIX 2: Energy tracking — P_power * u_drone is already in Wh (W × h = Wh)
    # REMOVED the /3600 divisor
    for i in N_o:
        for j in N_d:
            if (i, j) in A:
                model.addConstr(
                    e_node[j] >= e_node[i] + e_arc[i, j]
                    + P_power * u_drone[j]    # FIXED: was P_power * u_drone[j] / 3600
                    - M_big * (2 - x2[i, j] - z2[j]),
                    f"energy_cons_{i}_{j}")

    # Battery capacity at retrieval
    for i in N_d:
        for j in N_e:
            if (i, j) in A:
                in_truck = [(l, j) for l in N_o if (l, j) in A]
                model.addConstr(
                    e_node[i] + e_arc[i, j] <= E + M_big * (
                        2 - x2[i, j]
                        - gp.quicksum(x0[l, j] + x1[l, j] for (l, j) in in_truck)),
                    f"energy_cap_{i}_{j}")

    # --- Payload ---
    for i in N_o:
        out_truck = [(i, j) for j in N_e if (i, j) in A]
        out_x2   = [(i, j) for j in N_d if (i, j) in A]
        launch = 2 - gp.quicksum(x0[i, j] + x1[i, j] for (i, j) in out_truck) \
                   - gp.quicksum(x2[i, j] for (i, j) in out_x2)
        model.addConstr(w_total[i] <= w_del[i] + W * launch,
                        f"payload_total_ub_{i}")
        model.addConstr(w_total[i] >= w_del[i] - W * launch,
                        f"payload_total_lb_{i}")

    for i in N_d:
        for j in N_e:
            if (i, j) in A:
                in_truck = [(l, j) for l in N_o if (l, j) in A]
                model.addConstr(
                    w_del[i] <= W * (
                        2 - x2[i, j]
                        - gp.quicksum(x0[l, j] + x1[l, j] for (l, j) in in_truck)),
                    f"del_weight_zero_{i}_{j}")

    for i in N_o:
        for j in N_d:
            if (i, j) in A:
                model.addConstr(
                    w_del[j] >= w_del[i] - d_demand[j]
                    - M_big * (2 - x2[i, j] - z2[j]),
                    f"del_track_lb_{i}_{j}")
                model.addConstr(
                    w_del[j] <= w_del[i] - d_demand[j]
                    + M_big * (2 - x2[i, j] - z2[j]),
                    f"del_track_ub_{i}_{j}")

    for i in N_o:
        for j in N_d:
            if (i, j) in A:
                model.addConstr(
                    w_total[j] >= w_total[i] - d_demand[j] + p_demand[j]
                    - M_big * (2 - x2[i, j] - z2[j]),
                    f"total_track_lb_{i}_{j}")
                model.addConstr(
                    w_total[j] <= w_total[i] - d_demand[j] + p_demand[j]
                    + M_big * (2 - x2[i, j] - z2[j]),
                    f"total_track_ub_{i}_{j}")

    # --- Truck load (FIX 4: removed nonlinear term) ---
    total_delivery = sum(d_demand[i] for i in N_c)
    model.addConstr(v_truck[0] == total_delivery, "truck_init_load")

    for j in N_c:
        for i in N_o:
            if (i, j) in A:
                model.addConstr(
                    v_truck[j] >= v_truck[i] + p_demand[j] * z1[j] - d_demand[j] * z1[j]
                    - M_big * (1 - x0[i, j] - x1[i, j]),
                    f"truck_load_lb_{i}_{j}")
                model.addConstr(
                    v_truck[j] <= v_truck[i] + p_demand[j] * z1[j] - d_demand[j] * z1[j]
                    + M_big * (1 - x0[i, j] - x1[i, j]),
                    f"truck_load_ub_{i}_{j}")

    # Initial times
    model.addConstr(s1[0] == 0, "truck_start")
    model.addConstr(s1p[0] == 0, "truck_start_dep")
    model.addConstr(s2[0] == 0, "drone_start")
    model.addConstr(s2p[0] == 0, "drone_start_dep")

    # =========================================================
    # 6. SOLVE
    # =========================================================
    model.optimize()

    # =========================================================
    # 7. RESULTS
    # =========================================================
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        print("\n" + "=" * 60)
        print("CORRECTED MODEL - SOLUTION")
        print("=" * 60)
        print(f"Objective (Total Cost) = ${model.ObjVal:.6f}")

        print("\n--- Customer Assignments ---")
        for j in N_c:
            server = "TRUCK" if z1[j].X > 0.5 else "DRONE"
            print(f"  Customer {j}: served by {server}")

        print("\n--- Truck Route ---")
        truck_arcs = [(i, j) for (i, j) in A if x0[i, j].X + x1[i, j].X > 0.5]
        for (i, j) in truck_arcs:
            mode = "with drone" if x0[i, j].X > 0.5 else "alone"
            print(f"  {i} -> {j} ({mode}, {r[i,j]:.2f} km)")

        print("\n--- Drone Flights ---")
        drone_arcs = [(i, j) for (i, j) in A if x2[i, j].X > 0.5]
        for (i, j) in drone_arcs:
            print(f"  {i} -> {j} ({r[i,j]:.2f} km, energy={e_arc[i,j].X:.2f} Wh)")

        print("\n--- Cost Breakdown ---")
        tc = sum(c_T * r[i, j] * (x0[i, j].X + x1[i, j].X)
                 for (i, j) in A if not (i == 0 and j == 4))
        de = sum(c_F * e_arc[i, j].X for (i, j) in A)
        fc = c_0 * (1 - x0[0, 4].X)
        print(f"  Fixed deployment cost: ${fc:.4f}")
        print(f"  Truck travel cost:     ${tc:.4f}")
        print(f"  Drone energy cost:     ${de:.4f}")
        print(f"  Total cost:            ${model.ObjVal:.6f}")

        total_energy = sum(e_arc[i,j].X for (i,j) in A)
        print(f"\n  Total drone energy: {total_energy:.2f} Wh / {E} Wh capacity")

    elif model.status == GRB.INFEASIBLE:
        print("INFEASIBLE")
        model.computeIIS()
        for c in model.getConstrs():
            if c.IISConstr:
                print(f"  IIS: {c.ConstrName}")
    else:
        print(f"Status: {model.status}")

    return model


if __name__ == "__main__":
    solve_mdrp_pd()