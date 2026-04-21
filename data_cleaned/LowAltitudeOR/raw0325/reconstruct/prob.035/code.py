import gurobipy as gp
from gurobipy import GRB
import json
import math

def solve_uam_trajectory_aoi():
    """
    UAM Trajectory & AoI Optimization — Reformulated with Auxiliary Variables.
    Introduces cumulative energy tracking E_cum[m,t], per-slot AoI
    contribution c[m,t], and sum auxiliary S_aoi[m] for decomposed averaging.
    """

    # ================================================================
    # 1. DATA
    # ================================================================
    M = 2
    N = 4
    T_total = 6
    aircraft = list(range(M))
    time_slots = list(range(T_total))

    h_aircraft = 300.0
    h_bs = 30.0

    bs_pos = {0: (0, 0), 1: (500, 0), 2: (0, 500), 3: (500, 500)}
    start_pos = {0: (50, 50), 1: (450, 50)}
    dest_pos = {0: (450, 450), 1: (50, 450)}

    directions = {
        0: (100, 0), 1: (100, 100), 2: (0, 100), 3: (-100, 100),
        4: (-100, 0), 5: (-100, -100), 6: (0, -100), 7: (100, -100),
    }
    n_dirs = len(directions)

    dt = 100.0 / 35.0
    P_cruise = 120.0
    P_trans = 0.001
    P_max = 50.0

    E_move = P_cruise * (dt / 3600.0)
    E_comm = P_trans * (dt / 3600.0)

    AoI_max = 8
    D_col = 40.0
    x_min, x_max_val = 0, 500
    y_min, y_max_val = 0, 500

    # ================================================================
    # 2. BUILD MODEL
    # ================================================================
    model = gp.Model("UAM_Traj_AoI_AuxVar")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 120)

    # --- Direction choice ---
    psi = model.addVars(aircraft, time_slots, range(n_dirs),
                        vtype=GRB.BINARY, name="psi")

    # --- Positions ---
    pos_x = model.addVars(aircraft, range(T_total + 1),
                          lb=x_min, ub=x_max_val, name="px")
    pos_y = model.addVars(aircraft, range(T_total + 1),
                          lb=y_min, ub=y_max_val, name="py")

    # --- Bandwidth ---
    beta_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    n_beta = len(beta_levels)
    beta_sel = model.addVars(aircraft, time_slots, range(n_beta),
                             vtype=GRB.BINARY, name="bsel")
    beta = model.addVars(aircraft, time_slots, lb=0, ub=1, name="beta")

    # --- AoI ---
    aoi = model.addVars(aircraft, range(T_total + 1), lb=0, ub=AoI_max,
                        vtype=GRB.INTEGER, name="aoi")
    tx_done = model.addVars(aircraft, time_slots, vtype=GRB.BINARY, name="txdone")

    # --- Auxiliary: cumulative energy ---
    E_cum = model.addVars(aircraft, range(T_total + 1), lb=0, ub=P_max, name="Ecum")

    # --- Auxiliary: remaining power (at final time) ---
    P_remain = model.addVars(aircraft, lb=0, ub=P_max, name="Prem")

    # --- Auxiliary: per-slot AoI contributions and sum ---
    c_aoi = model.addVars(aircraft, range(T_total + 1), lb=0, ub=AoI_max, name="c_aoi")
    S_aoi = model.addVars(aircraft, lb=0, name="S_aoi")
    aoi_avg = model.addVars(aircraft, lb=0, name="aoi_avg")

    # ================================================================
    # CONSTRAINTS
    # ================================================================

    # Direction selection
    for m in aircraft:
        for t in time_slots:
            model.addConstr(
                gp.quicksum(psi[m, t, d] for d in range(n_dirs)) == 1,
                f"one_dir_{m}_{t}")

    # Bandwidth selection
    for m in aircraft:
        for t in time_slots:
            model.addConstr(
                gp.quicksum(beta_sel[m, t, b] for b in range(n_beta)) == 1,
                f"one_beta_{m}_{t}")
            model.addConstr(
                beta[m, t] == gp.quicksum(
                    beta_levels[b] * beta_sel[m, t, b] for b in range(n_beta)),
                f"beta_val_{m}_{t}")

    # Bandwidth sharing
    for t in time_slots:
        model.addConstr(
            gp.quicksum(beta[m, t] for m in aircraft) <= 1.0,
            f"bw_cap_{t}")

    # Initial positions
    for m in aircraft:
        model.addConstr(pos_x[m, 0] == start_pos[m][0], f"init_x_{m}")
        model.addConstr(pos_y[m, 0] == start_pos[m][1], f"init_y_{m}")

    # Position update
    for m in aircraft:
        for t in time_slots:
            model.addConstr(
                pos_x[m, t + 1] == pos_x[m, t] + gp.quicksum(
                    directions[d][0] * psi[m, t, d] for d in range(n_dirs)),
                f"upd_x_{m}_{t}")
            model.addConstr(
                pos_y[m, t + 1] == pos_y[m, t] + gp.quicksum(
                    directions[d][1] * psi[m, t, d] for d in range(n_dirs)),
                f"upd_y_{m}_{t}")

    # Destination
    for m in aircraft:
        model.addConstr(pos_x[m, T_total] >= dest_pos[m][0] - 50, f"dest_xlo_{m}")
        model.addConstr(pos_x[m, T_total] <= dest_pos[m][0] + 50, f"dest_xhi_{m}")
        model.addConstr(pos_y[m, T_total] >= dest_pos[m][1] - 50, f"dest_ylo_{m}")
        model.addConstr(pos_y[m, T_total] <= dest_pos[m][1] + 50, f"dest_yhi_{m}")

    # --- Cumulative energy (auxiliary decomposition) ---
    for m in aircraft:
        model.addConstr(E_cum[m, 0] == 0, f"init_Ecum_{m}")
        for t in time_slots:
            model.addConstr(
                E_cum[m, t + 1] == E_cum[m, t] + E_move + E_comm,
                f"Ecum_upd_{m}_{t}")

    # Remaining power derived from cumulative energy
    for m in aircraft:
        model.addConstr(
            P_remain[m] == P_max - E_cum[m, T_total],
            f"Prem_def_{m}")

    # --- AoI tracking ---
    for m in aircraft:
        model.addConstr(aoi[m, 0] == 0, f"init_aoi_{m}")
        for t in time_slots:
            # tx_done linkage with beta threshold
            model.addConstr(beta[m, t] >= 0.4 - (1 - tx_done[m, t]),
                            f"txdone_a_{m}_{t}")
            model.addConstr(beta[m, t] <= 0.39 + tx_done[m, t],
                            f"txdone_b_{m}_{t}")
            # AoI update
            model.addConstr(aoi[m, t + 1] >= aoi[m, t] + 1 - AoI_max * tx_done[m, t],
                            f"aoi_inc_{m}_{t}")
            model.addConstr(aoi[m, t + 1] <= aoi[m, t] + 1,
                            f"aoi_max_inc_{m}_{t}")
            model.addConstr(aoi[m, t + 1] <= AoI_max * (1 - tx_done[m, t]),
                            f"aoi_reset_{m}_{t}")

    # AoI threshold
    for m in aircraft:
        for t in range(T_total + 1):
            model.addConstr(aoi[m, t] <= AoI_max, f"aoi_thresh_{m}_{t}")

    # --- AoI auxiliary: per-slot contribution and sum ---
    for m in aircraft:
        for t in range(T_total + 1):
            model.addConstr(c_aoi[m, t] == aoi[m, t], f"c_aoi_def_{m}_{t}")
        model.addConstr(
            S_aoi[m] == gp.quicksum(c_aoi[m, t] for t in range(T_total + 1)),
            f"S_aoi_def_{m}")
        model.addConstr(
            aoi_avg[m] == S_aoi[m] / (T_total + 1),
            f"aoi_avg_def_{m}")

    # --- Collision avoidance ---
    for t in range(T_total + 1):
        for m1 in range(M):
            for m2 in range(m1 + 1, M):
                dx_abs = model.addVar(lb=0, name=f"dx_{m1}_{m2}_{t}")
                dy_abs = model.addVar(lb=0, name=f"dy_{m1}_{m2}_{t}")
                model.addConstr(dx_abs >= pos_x[m1, t] - pos_x[m2, t],
                                f"dxa_{m1}_{m2}_{t}")
                model.addConstr(dx_abs >= pos_x[m2, t] - pos_x[m1, t],
                                f"dxb_{m1}_{m2}_{t}")
                model.addConstr(dy_abs >= pos_y[m1, t] - pos_y[m2, t],
                                f"dya_{m1}_{m2}_{t}")
                model.addConstr(dy_abs >= pos_y[m2, t] - pos_y[m1, t],
                                f"dyb_{m1}_{m2}_{t}")
                model.addConstr(dx_abs + dy_abs >= D_col,
                                f"col_{m1}_{m2}_{t}")

    # ================================================================
    # OBJECTIVE
    # ================================================================
    obj = gp.quicksum(
        P_remain[m] / P_max + (AoI_max - aoi_avg[m]) / AoI_max
        for m in aircraft
    )
    model.setObjective(obj, GRB.MAXIMIZE)

    # ================================================================
    # SOLVE
    # ================================================================
    model.optimize()

    # ================================================================
    # OUTPUT
    # ================================================================
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        obj_val = round(model.ObjVal, 4)
        status = "optimal" if model.status == GRB.OPTIMAL else "feasible"
        result = {"status": status, "obj": obj_val}

        print(f"=== Objective Value: {obj_val:.4f} ===\n")

        for m in aircraft:
            print(f"Aircraft {m}:")
            traj = []
            for t in range(T_total + 1):
                traj.append((round(pos_x[m, t].X), round(pos_y[m, t].X)))
            print(f"  Trajectory: {traj}")
            print(f"  Cumulative Energy: {E_cum[m, T_total].X:.4f} kWh")
            print(f"  Remaining Power: {P_remain[m].X:.4f} kWh "
                  f"({P_remain[m].X / P_max * 100:.1f}%)")
            print(f"  S_AoI: {S_aoi[m].X:.2f}, Avg AoI: {aoi_avg[m].X:.2f}")

            bw_alloc = [round(beta[m, t].X, 2) for t in time_slots]
            print(f"  BW allocation: {bw_alloc}")

            aoi_vals = [int(round(aoi[m, t].X)) for t in range(T_total + 1)]
            print(f"  AoI profile: {aoi_vals}")
            print()

        return result
    else:
        print(f"Status: {model.status}")
        return {"status": "infeasible"}


if __name__ == "__main__":
    result = solve_uam_trajectory_aoi()
    print(json.dumps(result, indent=4))
