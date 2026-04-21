# -*- coding: utf-8 -*-
"""
Multi-Visit Drone Routing Problem for Pickup and Delivery (MDRP-PD)
Based on the paper: "The multi-visit drone routing problem for pickup and
delivery services" by Meng, Guo, Li, and Liu (2023).
Transportation Research Part E 169 (2023) 102990.

Modified: Customer 3 must be served by drone (drone service mandate).
"""

import gurobipy as gp
from gurobipy import GRB
import math
import json


def solve_mdrp_pd():
    """
    Solve the MDRP-PD problem using Gurobi MILP.
    Customer 3 is forced to be served by drone.
    """

    # =========================================================
    # 1. DATA INITIALIZATION
    # =========================================================

    N_c = [1, 2, 3]
    N_o = [0, 1, 2, 3]
    N_e = [1, 2, 3, 4]
    N_d = [1, 2, 3]
    N_all = [0, 1, 2, 3, 4]

    K = 1
    k = 1

    coords = {
        0: (0.0, 0.0),
        1: (2.0, 3.0),
        2: (5.0, 1.0),
        3: (4.0, 5.0),
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

    Q = 10.0
    v_T = 30.0
    c_T = 0.78
    c_0 = 22.0

    u_truck = {1: 1/12, 2: 1/30, 3: 1/20}

    W = 3.0
    w_0 = 6.0
    alpha = 3.5
    E = 504.0
    P_power = 1008.0
    c_F = 0.00248
    BR = 1/60

    u_drone = {1: 1/12, 2: 1/60, 3: 1/30}

    M_big = 1000.0

    A = [(i, j) for i in N_o for j in N_e if i != j]

    # =========================================================
    # 2. BUILD GUROBI MODEL
    # =========================================================

    model = gp.Model("MDRP_PD")
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

    s1 = model.addVars(N_all, vtype=GRB.CONTINUOUS, lb=0.0, name="s1")
    s1p = model.addVars(N_all, vtype=GRB.CONTINUOUS, lb=0.0, name="s1p")
    s2 = model.addVars(N_all, vtype=GRB.CONTINUOUS, lb=0.0, name="s2")
    s2p = model.addVars(N_all, vtype=GRB.CONTINUOUS, lb=0.0, name="s2p")

    e_arc = model.addVars(A, vtype=GRB.CONTINUOUS, lb=0.0, ub=E, name="e_arc")
    e_node = model.addVars(N_o, vtype=GRB.CONTINUOUS, lb=0.0, ub=E, name="e_node")
    w_total = model.addVars(N_o, vtype=GRB.CONTINUOUS, lb=0.0, ub=W, name="w_total")
    w_del = model.addVars(N_o, vtype=GRB.CONTINUOUS, lb=0.0, ub=W, name="w_del")
    v_truck = model.addVars(N_o, vtype=GRB.CONTINUOUS, lb=0.0, ub=Q, name="v_truck")

    # =========================================================
    # 4. OBJECTIVE FUNCTION
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

    # --- Routing constraints ---

    model.addConstr(
        gp.quicksum(x0[0, j] + x1[0, j] for j in N_e if (0, j) in A) == 1,
        name="truck_depart"
    )
    model.addConstr(
        gp.quicksum(x0[i, 4] + x1[i, 4] for i in N_o if (i, 4) in A) == 1,
        name="truck_return"
    )
    model.addConstr(
        gp.quicksum(x0[0, j] + x2[0, j] for j in N_e if (0, j) in A) == 1,
        name="drone_depart"
    )
    model.addConstr(
        gp.quicksum(x0[i, 4] + x2[i, 4] for i in N_o if (i, 4) in A) == 1,
        name="drone_return"
    )

    for (i, j) in A:
        if not (i == 0 and j == 4):
            model.addConstr(
                x0[i, j] + x1[i, j] + x2[i, j] <= 1,
                name=f"arc_usage_{i}_{j}"
            )

    for j in N_c:
        in_arcs = [(i, j) for i in N_o if (i, j) in A]
        out_arcs = [(j, l) for l in N_e if (j, l) in A]
        model.addConstr(
            gp.quicksum(x0[i, j] + x1[i, j] for (i, j) in in_arcs) == z1[j],
            name=f"truck_flow_in_{j}"
        )
        model.addConstr(
            gp.quicksum(x0[j, l] + x1[j, l] for (j, l) in out_arcs) == z1[j],
            name=f"truck_flow_out_{j}"
        )

    for j in N_c:
        in_arcs = [(i, j) for i in N_o if (i, j) in A]
        out_arcs = [(j, l) for l in N_e if (j, l) in A]
        model.addConstr(
            gp.quicksum(x0[i, j] + x2[i, j] for (i, j) in in_arcs) ==
            gp.quicksum(x0[j, l] + x2[j, l] for (j, l) in out_arcs),
            name=f"drone_flow_{j}"
        )

    for j in N_d:
        in_arcs_x2 = [(i, j) for i in N_o if (i, j) in A]
        out_arcs_x2 = [(j, l) for l in N_e if (j, l) in A]
        model.addConstr(
            2 * z2[j] <= gp.quicksum(x2[i, j] for (i, j) in in_arcs_x2) +
                          gp.quicksum(x2[j, l] for (j, l) in out_arcs_x2),
            name=f"drone_serve_{j}"
        )

    for j in N_c:
        model.addConstr(z1[j] + z2[j] == 1, name=f"serve_once_{j}")

    for i in N_c:
        out_x2 = [(i, j) for j in N_e if (i, j) in A]
        model.addConstr(
            z1[i] >= gp.quicksum(x2[i, j] for (i, j) in out_x2) - z2[i],
            name=f"truck_launch_{i}"
        )

    for i in N_c:
        in_x2 = [(j, i) for j in N_o if (j, i) in A]
        model.addConstr(
            z1[i] >= gp.quicksum(x2[j, i] for (j, i) in in_x2) - z2[i],
            name=f"truck_retrieve_{i}"
        )

    for i in N_c:
        out_x2 = [(i, j) for j in N_e if (i, j) in A]
        model.addConstr(
            gp.quicksum(x2[i, j] for (i, j) in out_x2) <= 1,
            name=f"drone_leave_{i}"
        )

    for i in N_c:
        in_x2 = [(j, i) for j in N_o if (j, i) in A]
        model.addConstr(
            gp.quicksum(x2[j, i] for (j, i) in in_x2) <= 1,
            name=f"drone_arrive_{i}"
        )

    # --- Timing constraints ---

    for (i, j) in A:
        travel_time = r[i, j] / v_T
        model.addConstr(
            s1[j] >= s1p[i] + travel_time - M_big * (1 - x0[i, j] - x1[i, j]),
            name=f"truck_time_lb_{i}_{j}"
        )
        model.addConstr(
            s1[j] <= s1p[i] + travel_time + M_big * (1 - x0[i, j] - x1[i, j]),
            name=f"truck_time_ub_{i}_{j}"
        )

    for i in N_c:
        in_x2 = [(j, i) for j in N_d if (j, i) in A]
        model.addConstr(
            s1p[i] >= s1[i] + u_truck[i] * z1[i] + BR * gp.quicksum(x2[j, i] for (j, i) in in_x2),
            name=f"truck_depart_time_{i}"
        )

    for (i, j) in A:
        avg_weight = w_0 + W / 2
        drone_travel_time = alpha * avg_weight * r[i, j] / P_power
        model.addConstr(
            s2[j] >= s2p[i] + drone_travel_time - M_big * (1 - x2[i, j]),
            name=f"drone_time_lb_{i}_{j}"
        )
        model.addConstr(
            s2[j] <= s2p[i] + drone_travel_time + M_big * (1 - x2[i, j]),
            name=f"drone_time_ub_{i}_{j}"
        )

    for i in N_d:
        model.addConstr(
            s2p[i] >= s2[i] + u_drone[i] - M_big * (1 - z2[i]),
            name=f"drone_depart_time_{i}"
        )
        model.addConstr(
            s2p[i] <= s2[i] + u_drone[i] + M_big * (1 - z2[i]),
            name=f"drone_depart_time_ub_{i}"
        )

    for i in N_o:
        out_x2 = [(i, j) for j in N_d if (i, j) in A]
        out_truck = [(i, j) for j in N_e if (i, j) in A]
        if out_x2 and out_truck:
            model.addConstr(
                s1p[i] >= s2p[i] - M_big * (2 - gp.quicksum(x0[i, j] + x1[i, j] for (i, j) in out_truck)
                                              - gp.quicksum(x2[i, j] for (i, j) in out_x2)),
                name=f"sync_launch_{i}"
            )

    for i in N_c:
        in_x2 = [(j, i) for j in N_d if (j, i) in A]
        if in_x2:
            model.addConstr(
                s1p[i] >= s2[i] + BR - M_big * (2 - gp.quicksum(x2[j, i] for (j, i) in in_x2) - z1[i]),
                name=f"sync_retrieve_{i}"
            )

    # --- Drone energy constraints ---

    for (i, j) in A:
        model.addConstr(e_arc[i, j] <= E * x2[i, j], name=f"energy_ub_{i}_{j}")

    for (i, j) in A:
        model.addConstr(
            e_arc[i, j] >= alpha * w_0 * r[i, j] * x2[i, j],
            name=f"energy_lb_{i}_{j}"
        )
        model.addConstr(
            e_arc[i, j] <= alpha * (w_0 + W) * r[i, j] * x2[i, j],
            name=f"energy_ub2_{i}_{j}"
        )

    for i in N_o:
        out_truck = [(i, j) for j in N_e if (i, j) in A]
        out_x2 = [(i, j) for j in N_d if (i, j) in A]
        model.addConstr(
            e_node[i] <= E * (2 - gp.quicksum(x0[i, j] + x1[i, j] for (i, j) in out_truck)
                               - gp.quicksum(x2[i, j] for (i, j) in out_x2)),
            name=f"energy_reset_{i}"
        )

    for i in N_o:
        for j in N_d:
            if (i, j) in A:
                model.addConstr(
                    e_node[j] >= e_node[i] + e_arc[i, j] + P_power * u_drone[j] / 3600
                    - M_big * (2 - x2[i, j] - z2[j]),
                    name=f"energy_cons_{i}_{j}"
                )

    for i in N_d:
        for j in N_e:
            if (i, j) in A:
                in_truck = [(l, j) for l in N_o if (l, j) in A]
                model.addConstr(
                    e_node[i] + e_arc[i, j] <= E + M_big * (
                        2 - x2[i, j] - gp.quicksum(x0[l, j] + x1[l, j] for (l, j) in in_truck)
                    ),
                    name=f"energy_cap_{i}_{j}"
                )

    # --- Drone payload constraints ---

    for i in N_o:
        out_truck = [(i, j) for j in N_e if (i, j) in A]
        out_x2 = [(i, j) for j in N_d if (i, j) in A]
        launch_indicator = 2 - gp.quicksum(x0[i, j] + x1[i, j] for (i, j) in out_truck) \
                             - gp.quicksum(x2[i, j] for (i, j) in out_x2)
        model.addConstr(
            w_total[i] <= w_del[i] + W * launch_indicator,
            name=f"payload_total_ub_{i}"
        )
        model.addConstr(
            w_total[i] >= w_del[i] - W * launch_indicator,
            name=f"payload_total_lb_{i}"
        )

    for i in N_d:
        for j in N_e:
            if (i, j) in A:
                in_truck = [(l, j) for l in N_o if (l, j) in A]
                model.addConstr(
                    w_del[i] <= W * (2 - x2[i, j] - gp.quicksum(x0[l, j] + x1[l, j] for (l, j) in in_truck)),
                    name=f"del_weight_zero_{i}_{j}"
                )

    for i in N_o:
        for j in N_d:
            if (i, j) in A:
                model.addConstr(
                    w_del[j] >= w_del[i] - d_demand[j] - M_big * (2 - x2[i, j] - z2[j]),
                    name=f"del_track_lb_{i}_{j}"
                )
                model.addConstr(
                    w_del[j] <= w_del[i] - d_demand[j] + M_big * (2 - x2[i, j] - z2[j]),
                    name=f"del_track_ub_{i}_{j}"
                )

    for i in N_o:
        for j in N_d:
            if (i, j) in A:
                model.addConstr(
                    w_total[j] >= w_total[i] - d_demand[j] + p_demand[j] - M_big * (2 - x2[i, j] - z2[j]),
                    name=f"total_track_lb_{i}_{j}"
                )
                model.addConstr(
                    w_total[j] <= w_total[i] - d_demand[j] + p_demand[j] + M_big * (2 - x2[i, j] - z2[j]),
                    name=f"total_track_ub_{i}_{j}"
                )

    # --- Truck load constraints ---

    total_delivery = sum(d_demand[i] for i in N_c)
    total_drone_delivery = gp.quicksum(d_demand[j] * z2[j] for j in N_d)
    drone_load_from_depot = gp.quicksum(w_total[0] * x2[0, j] for j in N_d if (0, j) in A)

    model.addConstr(
        v_truck[0] >= total_delivery - total_drone_delivery - drone_load_from_depot
        - M_big * (1 - gp.quicksum(x0[0, j] + x1[0, j] for j in N_e if (0, j) in A)),
        name="truck_load_depot_lb"
    )
    model.addConstr(
        v_truck[0] <= Q * gp.quicksum(x0[0, j] + x1[0, j] for j in N_e if (0, j) in A),
        name="truck_load_depot_ub"
    )

    for j in N_c:
        for i in N_o:
            if (i, j) in A:
                model.addConstr(
                    v_truck[j] >= v_truck[i] + p_demand[j] - d_demand[j]
                    - M_big * (1 - x0[i, j] - x1[i, j]),
                    name=f"truck_load_lb_{i}_{j}"
                )
                model.addConstr(
                    v_truck[j] <= v_truck[i] + p_demand[j] - d_demand[j]
                    + M_big * (1 - x0[i, j] - x1[i, j]),
                    name=f"truck_load_ub_{i}_{j}"
                )

    for i in N_o:
        model.addConstr(v_truck[i] >= 0, name=f"truck_load_nn_{i}")
        model.addConstr(v_truck[i] <= Q, name=f"truck_load_cap_{i}")

    model.addConstr(s1[0] == 0, name="truck_start")
    model.addConstr(s1p[0] == 0, name="truck_start_dep")
    model.addConstr(s2[0] == 0, name="drone_start")
    model.addConstr(s2p[0] == 0, name="drone_start_dep")

    # --- Mandatory drone service for Customer 3 ---
    model.addConstr(z2[3] == 1, name="mandatory_drone_customer3")

    # =========================================================
    # 6. SOLVE
    # =========================================================

    model.optimize()

    # =========================================================
    # 7. EXTRACT AND DISPLAY RESULTS
    # =========================================================

    result = {"status": "unknown", "obj": None}

    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        if model.SolCount > 0:
            result["status"] = "optimal" if model.status == GRB.OPTIMAL else "time_limit"
            result["obj"] = round(model.ObjVal, 6)

            print("\n" + "=" * 60)
            print("SOLUTION FOUND")
            print("=" * 60)
            print(f"Objective (Total Cost) = ${model.ObjVal:.6f}")

            print("\n--- Customer Assignments ---")
            for j in N_c:
                if z1[j].X > 0.5:
                    print(f"  Customer {j}: served by TRUCK")
                elif z2[j].X > 0.5:
                    forced = " [MANDATORY]" if j == 3 else ""
                    print(f"  Customer {j}: served by DRONE{forced}")

            print("\n--- Truck Route ---")
            truck_arcs = [(i, j) for (i, j) in A if x0[i, j].X + x1[i, j].X > 0.5]
            print(f"  Truck arcs: {truck_arcs}")

            print("\n--- Drone Flights ---")
            drone_arcs = [(i, j) for (i, j) in A if x2[i, j].X > 0.5]
            print(f"  Drone arcs: {drone_arcs}")

            print("\n--- Cost Breakdown ---")
            truck_travel = sum(c_T * r[i, j] * (x0[i, j].X + x1[i, j].X)
                               for (i, j) in A if not (i == 0 and j == 4))
            drone_energy = sum(c_F * e_arc[i, j].X for (i, j) in A)
            fixed = c_0 * (1 - x0[0, 4].X)
            print(f"  Fixed deployment cost: ${fixed:.4f}")
            print(f"  Truck travel cost:     ${truck_travel:.4f}")
            print(f"  Drone energy cost:     ${drone_energy:.4f}")
            print(f"  Total cost:            ${model.ObjVal:.6f}")

    else:
        result["status"] = f"status_{model.status}"
        print(f"\nNo solution found. Solver status = {model.status}")

    return result


if __name__ == "__main__":
    result = solve_mdrp_pd()
    print(f"\nResult: {result}")
