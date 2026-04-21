# -*- coding: utf-8 -*-
"""
UAV Traffic Management - Full Optimization (FO) Model
Reformulated with Auxiliary Variable Delay Decomposition.

The piecewise linear delay cost is reformulated by introducing auxiliary
variables d1[i] and d2[i] that decompose total delay into two segments,
replacing the original epigraph (lower-bound) formulation.
"""

import gurobipy as gp
from gurobipy import GRB
import json


def solve_uav_utm_fo():
    """
    Solve the UAV UTM Full Optimization (FO) model.
    Minimizes total cost = sum of delay costs + sum of path costs,
    subject to conflict-free scheduling constraints.
    """

    # =========================================================
    # 1. DATA INITIALIZATION
    # =========================================================

    # Set of flights
    F = [1, 2, 3, 4]

    # Set of paths: 0 = optimal path, 1 = alternative path
    P = [0, 1]

    # Scheduled departure times (seconds)
    s = {1: 0, 2: 10, 3: 5, 4: 20}

    # Path costs c[i][j]: extra cost ($) of flight i taking path j
    c = {
        (1, 0): 0.0, (1, 1): 0.5,
        (2, 0): 0.0, (2, 1): 0.8,
        (3, 0): 0.0, (3, 1): 0.3,
        (4, 0): 0.0, (4, 1): 0.6,
    }

    # Unit delay cost r[i] ($/second) when delay <= d
    r = {1: 0.004, 2: 0.002, 3: 0.005, 4: 0.003}

    # Delay cost threshold (seconds)
    d = 300.0

    # Unit delay cost R[i] when delay > d: R_i = 2 * r_i
    R = {i: 2 * r[i] for i in F}

    # Safety separation buffer (seconds)
    t0 = 10.0

    # Big-M constant
    M = 10000.0

    # Spatial conflict pairs
    conflicts = {
        "C1": {"i": 1, "j": 0, "m": 2, "n": 0, "b": 30, "e": 60, "B": 20, "E": 50},
        "C2": {"i": 1, "j": 0, "m": 3, "n": 0, "b": 50, "e": 80, "B": 40, "E": 70},
        "C3": {"i": 2, "j": 1, "m": 4, "n": 0, "b": 25, "e": 55, "B": 15, "E": 45},
        "C4": {"i": 3, "j": 1, "m": 4, "n": 1, "b": 35, "e": 65, "B": 30, "E": 60},
    }
    C = list(conflicts.keys())

    # =========================================================
    # 2. BUILD GUROBI MODEL
    # =========================================================

    model = gp.Model("UAV_UTM_FO_AuxVar")
    model.setParam("OutputFlag", 1)

    # =========================================================
    # 3. DECISION VARIABLES
    # =========================================================

    # I[i,j] = 1 if flight i takes path j
    I = model.addVars(F, P, vtype=GRB.BINARY, name="I")

    # t[i] = assigned departure time of flight i (seconds)
    t = model.addVars(F, vtype=GRB.CONTINUOUS, lb=0.0, name="t")

    # Auxiliary delay decomposition variables
    # d1[i]: delay in segment 1 (0 to d seconds), cheaper rate r_i
    d1 = model.addVars(F, vtype=GRB.CONTINUOUS, lb=0.0, name="d1")
    # d2[i]: delay in segment 2 (beyond d seconds), more expensive rate R_i
    d2 = model.addVars(F, vtype=GRB.CONTINUOUS, lb=0.0, name="d2")

    # D[i] = delay cost of flight i ($)
    D = model.addVars(F, vtype=GRB.CONTINUOUS, lb=0.0, name="D")

    # z[c] = 1 if both flights in conflict c take their conflicting paths
    z = model.addVars(C, vtype=GRB.BINARY, name="z")

    # X_c = 1 if flight i passes the conflict region before flight m
    X_var = model.addVars(C, vtype=GRB.BINARY, name="X")

    # x_c = 1 if flight m passes the conflict region before flight i
    x_var = model.addVars(C, vtype=GRB.BINARY, name="x")

    # =========================================================
    # 4. OBJECTIVE FUNCTION
    # =========================================================

    model.setObjective(
        gp.quicksum(D[i] for i in F) +
        gp.quicksum(c[i, j] * I[i, j] for i in F for j in P),
        GRB.MINIMIZE
    )

    # =========================================================
    # 5. CONSTRAINTS
    # =========================================================

    # (1) Auxiliary variable delay decomposition
    # Replaces the original piecewise linear epigraph (two lower-bound constraints)
    for i in F:
        # Total delay = d1 + d2
        model.addConstr(d1[i] + d2[i] == t[i] - s[i], name=f"delay_decomp_{i}")
        # d1 bounded by threshold d
        model.addConstr(d1[i] <= d, name=f"d1_upper_{i}")
        # Delay cost = weighted sum of segments
        model.addConstr(D[i] == r[i] * d1[i] + R[i] * d2[i], name=f"delay_cost_{i}")

    # (2) Conflict resolution: flight i passes before flight m
    for cid in C:
        cf = conflicts[cid]
        fi = cf["i"]
        fm = cf["m"]
        e_c = cf["e"]
        B_c = cf["B"]
        model.addConstr(
            t[fi] + e_c + t0 <= M * (1 - X_var[cid]) + B_c + t[fm],
            name=f"conflict_i_first_{cid}"
        )

    # (3) Conflict resolution: flight m passes before flight i
    for cid in C:
        cf = conflicts[cid]
        fi = cf["i"]
        fm = cf["m"]
        b_c = cf["b"]
        E_c = cf["E"]
        model.addConstr(
            t[fm] + E_c + t0 <= M * (1 - x_var[cid]) + b_c + t[fi],
            name=f"conflict_m_first_{cid}"
        )

    # (4) Linearization of z_c = I[i,j] * I[m,n]
    for cid in C:
        cf = conflicts[cid]
        fi, pj = cf["i"], cf["j"]
        fm, pn = cf["m"], cf["n"]
        model.addConstr(z[cid] <= I[fi, pj], name=f"z_ub1_{cid}")
        model.addConstr(z[cid] <= I[fm, pn], name=f"z_ub2_{cid}")
        model.addConstr(z[cid] + 1 >= I[fi, pj] + I[fm, pn], name=f"z_lb_{cid}")

    # (5) Conflict ordering: if both flights take conflicting paths, one must pass first
    for cid in C:
        model.addConstr(
            X_var[cid] + x_var[cid] == z[cid],
            name=f"conflict_order_{cid}"
        )

    # (6) Departure time feasibility: t_i >= s_i
    for i in F:
        model.addConstr(t[i] >= s[i], name=f"depart_feasibility_{i}")

    # (7) Path selection: each flight must choose exactly one path
    for i in F:
        model.addConstr(
            gp.quicksum(I[i, j] for j in P) == 1,
            name=f"path_selection_{i}"
        )

    # =========================================================
    # 6. SOLVE
    # =========================================================

    model.optimize()

    # =========================================================
    # 7. EXTRACT AND DISPLAY RESULTS
    # =========================================================

    result = {"status": "unknown", "obj": None}

    if model.status == GRB.OPTIMAL:
        result["status"] = "optimal"
        result["obj"] = round(model.ObjVal, 6)

        print("\n" + "=" * 60)
        print("OPTIMAL SOLUTION FOUND (Auxiliary Variable Reformulation)")
        print("=" * 60)
        print(f"Objective (Total System Cost) = ${model.ObjVal:.6f}")

        print("\n--- Path Assignments ---")
        for i in F:
            for j in P:
                if I[i, j].X > 0.5:
                    print(f"  Flight {i}: Path {j} (cost = ${c[i, j]:.2f})")

        print("\n--- Departure Times, Delay Decomposition, and Costs ---")
        total_delay_cost = 0.0
        for i in F:
            assigned_t = t[i].X
            delay = assigned_t - s[i]
            d1_val = d1[i].X
            d2_val = d2[i].X
            delay_cost = D[i].X
            total_delay_cost += delay_cost
            print(f"  Flight {i}: scheduled={s[i]}s, assigned={assigned_t:.2f}s, "
                  f"delay={delay:.2f}s (d1={d1_val:.2f}, d2={d2_val:.2f}), "
                  f"delay_cost=${delay_cost:.6f}")

        print("\n--- Conflict Resolution ---")
        for cid in C:
            cf = conflicts[cid]
            fi, pj = cf["i"], cf["j"]
            fm, pn = cf["m"], cf["n"]
            z_val = z[cid].X
            X_val = X_var[cid].X
            x_val = x_var[cid].X
            print(f"  {cid} (F{fi}p{pj} vs F{fm}p{pn}): "
                  f"z={z_val:.0f}, X(i_first)={X_val:.0f}, x(m_first)={x_val:.0f}")

        total_path_cost = sum(c[i, j] * I[i, j].X for i in F for j in P)
        print(f"\n--- Cost Breakdown ---")
        print(f"  Total Delay Cost:  ${total_delay_cost:.6f}")
        print(f"  Total Path Cost:   ${total_path_cost:.6f}")
        print(f"  Total System Cost: ${model.ObjVal:.6f}")

    else:
        result["status"] = f"status_{model.status}"
        print(f"\nNo optimal solution found. Solver status = {model.status}")

    return result


if __name__ == "__main__":
    result = solve_uav_utm_fo()
    with open("answer.json", "w") as f:
        json.dump(result, f, indent=4)
    print(f"\nSaved answer.json: {result}")
