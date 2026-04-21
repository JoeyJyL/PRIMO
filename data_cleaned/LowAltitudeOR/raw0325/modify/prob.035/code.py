import gurobipy as gp
from gurobipy import GRB
import json
import math

def solve_uam_trajectory_aoi():
    """
    Simplified UAM Trajectory & AoI Optimization based on:
    "Cooperative Urban Air Mobility Trajectory Design for Power and AoI
     Optimization" by Kim et al. (IEEE TVT, 2025).

    Simplified: 2 UAM aircraft, 4 base stations, 6 time slots on a 
    discretized 2D grid. Each aircraft must reach its destination while 
    maximizing remaining power and minimizing average AoI, subject to 
    collision avoidance and bandwidth allocation constraints.
    """

    # ================================================================
    # 1. DATA
    # ================================================================
    M = 2          # number of UAM aircraft
    N = 4          # number of base stations
    T_total = 6   # time slots
    aircraft = list(range(M))
    bs_set = list(range(N))
    time_slots = list(range(T_total))

    # Grid: 500m x 500m, discretized into 100m cells
    # Aircraft fly at fixed altitude h=300m
    h_aircraft = 300.0  # m
    h_bs = 30.0         # m (BS antenna height)

    # BS positions (x, y) in meters
    bs_pos = {
        0: (0, 0),
        1: (500, 0),
        2: (0, 500),
        3: (500, 500),
    }

    # Aircraft start and destination positions
    start_pos = {0: (50, 50), 1: (450, 50)}
    dest_pos = {0: (450, 450), 1: (50, 450)}

    # 8 possible directions + hover: dx, dy offsets per time slot
    # Each step = 100m (speed ~35 m/s * ~2.86s per slot ≈ 100m)
    directions = {
        0: (100, 0),    # East
        1: (100, 100),  # NE
        2: (0, 100),    # North
        3: (-100, 100), # NW
        4: (-100, 0),   # West
        5: (-100, -100),# SW
        6: (0, -100),   # South
        7: (100, -100), # SE
    }
    n_dirs = len(directions)

    # Power model (simplified from Eq. 5-7)
    # Hovering power ~ 150 kW for JOBY S4 (from paper params)
    # Cruise at 35 m/s ~ 120 kW
    # Simplified: P_prop per slot for moving = 120 kW, hover = 150 kW
    # Time per slot: assume 3 seconds => energy per slot in kWh
    dt = 100.0 / 35.0  # seconds per slot (~2.86s)
    P_cruise = 120.0    # kW while moving
    P_hover = 150.0     # kW while hovering (not used — aircraft must move)
    P_trans = 0.001     # kW communication power (1W)
    P_max = 50.0        # kWh total battery

    # Energy per moving slot (kWh)
    E_move = P_cruise * (dt / 3600.0)  # ~0.0953 kWh per slot
    E_comm = P_trans * (dt / 3600.0)   # negligible

    # Communication model (simplified from Eq. 1-4)
    c = 3e8          # speed of light
    f = 2.8e9        # frequency Hz
    P_tx = 1.0       # transmit power W
    G_tx = 2.0       # transmitter gain
    G_rx = 2.0       # receiver gain
    noise_dBm = -174.0
    B_max = 5e6      # Hz max bandwidth
    sigma2 = 10**((noise_dBm - 30) / 10) * B_max  # noise power in W

    alpha_max = 200e6 * 8  # 200 MB in bits (data to transmit per cycle)
    AoI_max = 8            # max AoI threshold (time slots)

    # Collision distances
    D_col = 40.0     # m between aircraft
    D_ob = 20.0      # m to obstacles (not modeled here for simplicity)

    # Grid bounds
    x_min, x_max = 0, 500
    y_min, y_max = 0, 500

    # ================================================================
    # 2. PRECOMPUTE: data rate for each possible position to each BS
    # ================================================================
    def compute_snr(x_a, y_a, x_b, y_b):
        """SNR between aircraft at (x_a, y_a, h_aircraft) and BS at (x_b, y_b, h_bs)"""
        dist_3d = math.sqrt((x_a - x_b)**2 + (y_a - y_b)**2 + (h_aircraft - h_bs)**2)
        if dist_3d < 1:
            dist_3d = 1
        fspl = (4 * math.pi * f * dist_3d / c)**2
        P_rec = P_tx * G_tx * G_rx / fspl
        snr = P_rec / sigma2
        return snr

    def data_rate(snr, beta):
        """Data rate in bits/s given SNR and bandwidth fraction beta"""
        return beta * B_max * math.log2(1 + snr)

    # ================================================================
    # 3. BUILD MODEL
    # ================================================================
    model = gp.Model("UAM_Traj_AoI")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 120)

    # --- Decision Variables ---
    # Direction choice for each aircraft at each time slot
    # psi[m, t, d] = 1 if aircraft m chooses direction d at time t
    psi = model.addVars(aircraft, time_slots, range(n_dirs),
                        vtype=GRB.BINARY, name="psi")

    # Position variables (continuous, derived from direction choices)
    pos_x = model.addVars(aircraft, range(T_total + 1), lb=x_min, ub=x_max, name="px")
    pos_y = model.addVars(aircraft, range(T_total + 1), lb=y_min, ub=y_max, name="py")

    # Bandwidth allocation: beta[m, t] in {0, 0.2, 0.4, 0.6, 0.8, 1.0}
    beta_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    n_beta = len(beta_levels)
    beta_sel = model.addVars(aircraft, time_slots, range(n_beta),
                             vtype=GRB.BINARY, name="bsel")
    beta = model.addVars(aircraft, time_slots, lb=0, ub=1, name="beta")

    # AoI tracking
    aoi = model.addVars(aircraft, range(T_total + 1), lb=0, ub=AoI_max, 
                        vtype=GRB.INTEGER, name="aoi")

    # Remaining data to transmit
    alpha_rem = model.addVars(aircraft, range(T_total + 1), lb=0, name="alpha")

    # Binary: did transmission complete at slot t?
    tx_done = model.addVars(aircraft, time_slots, vtype=GRB.BINARY, name="txdone")

    # Remaining power
    P_remain = model.addVars(aircraft, range(T_total + 1), lb=0, ub=P_max, name="Prem")

    # Average AoI per aircraft
    aoi_avg = model.addVars(aircraft, lb=0, name="aoi_avg")

    # --- Constraints ---

    # Each aircraft chooses exactly one direction per time slot
    for m in aircraft:
        for t in time_slots:
            model.addConstr(gp.quicksum(psi[m, t, d] for d in range(n_dirs)) == 1,
                            f"one_dir_{m}_{t}")

    # Each aircraft chooses exactly one beta level per time slot
    for m in aircraft:
        for t in time_slots:
            model.addConstr(gp.quicksum(beta_sel[m, t, b] for b in range(n_beta)) == 1,
                            f"one_beta_{m}_{t}")
            model.addConstr(beta[m, t] == gp.quicksum(
                beta_levels[b] * beta_sel[m, t, b] for b in range(n_beta)),
                f"beta_val_{m}_{t}")

    # Bandwidth: total beta across aircraft per BS <= 0.6 (reduced due to maintenance)
    # Simplified: all aircraft share one BS pool at each time slot
    for t in time_slots:
        model.addConstr(gp.quicksum(beta[m, t] for m in aircraft) <= 0.6,
                        f"bw_cap_{t}")

    # Initial positions
    for m in aircraft:
        model.addConstr(pos_x[m, 0] == start_pos[m][0], f"init_x_{m}")
        model.addConstr(pos_y[m, 0] == start_pos[m][1], f"init_y_{m}")

    # Position update from direction choice
    for m in aircraft:
        for t in time_slots:
            model.addConstr(
                pos_x[m, t+1] == pos_x[m, t] + gp.quicksum(
                    directions[d][0] * psi[m, t, d] for d in range(n_dirs)),
                f"upd_x_{m}_{t}")
            model.addConstr(
                pos_y[m, t+1] == pos_y[m, t] + gp.quicksum(
                    directions[d][1] * psi[m, t, d] for d in range(n_dirs)),
                f"upd_y_{m}_{t}")

    # Destination constraint: aircraft must reach destination by final slot
    for m in aircraft:
        # Allow being within 50m of destination
        model.addConstr(pos_x[m, T_total] >= dest_pos[m][0] - 50, f"dest_xlo_{m}")
        model.addConstr(pos_x[m, T_total] <= dest_pos[m][0] + 50, f"dest_xhi_{m}")
        model.addConstr(pos_y[m, T_total] >= dest_pos[m][1] - 50, f"dest_ylo_{m}")
        model.addConstr(pos_y[m, T_total] <= dest_pos[m][1] + 50, f"dest_yhi_{m}")

    # Power tracking
    for m in aircraft:
        model.addConstr(P_remain[m, 0] == P_max, f"init_P_{m}")
        for t in time_slots:
            model.addConstr(
                P_remain[m, t+1] == P_remain[m, t] - E_move - E_comm,
                f"P_upd_{m}_{t}")

    # AoI tracking (simplified: AoI resets when beta >= threshold)
    # Since computing exact data rate requires knowing position (nonlinear),
    # we use a simplified model: AoI resets if beta[m,t] >= 0.4
    for m in aircraft:
        model.addConstr(aoi[m, 0] == 0, f"init_aoi_{m}")
        for t in time_slots:
            # tx_done[m,t] = 1 if sufficient bandwidth allocated
            model.addConstr(beta[m, t] >= 0.4 - (1 - tx_done[m, t]),
                            f"txdone_a_{m}_{t}")
            model.addConstr(beta[m, t] <= 0.39 + tx_done[m, t],
                            f"txdone_b_{m}_{t}")
            # AoI update: if tx_done, reset to 0; else increment
            model.addConstr(aoi[m, t+1] >= aoi[m, t] + 1 - AoI_max * tx_done[m, t],
                            f"aoi_inc_{m}_{t}")
            model.addConstr(aoi[m, t+1] <= aoi[m, t] + 1,
                            f"aoi_max_inc_{m}_{t}")
            model.addConstr(aoi[m, t+1] <= AoI_max * (1 - tx_done[m, t]),
                            f"aoi_reset_{m}_{t}")

    # AoI threshold constraint (Eq. 11e)
    for m in aircraft:
        for t in range(T_total + 1):
            model.addConstr(aoi[m, t] <= AoI_max, f"aoi_thresh_{m}_{t}")

    # Average AoI
    for m in aircraft:
        model.addConstr(
            aoi_avg[m] == gp.quicksum(aoi[m, t] for t in range(T_total + 1)) / (T_total + 1),
            f"aoi_avg_{m}")

    # Collision avoidance (Eq. 11f): |pos_m - pos_mu| > D_col
    # Linearized: for each pair, at each time, Manhattan distance > D_col
    # Use auxiliary variables for absolute differences
    for t in range(T_total + 1):
        for m1 in range(M):
            for m2 in range(m1 + 1, M):
                dx_abs = model.addVar(lb=0, name=f"dx_{m1}_{m2}_{t}")
                dy_abs = model.addVar(lb=0, name=f"dy_{m1}_{m2}_{t}")
                model.addConstr(dx_abs >= pos_x[m1, t] - pos_x[m2, t], f"dxa_{m1}_{m2}_{t}")
                model.addConstr(dx_abs >= pos_x[m2, t] - pos_x[m1, t], f"dxb_{m1}_{m2}_{t}")
                model.addConstr(dy_abs >= pos_y[m1, t] - pos_y[m2, t], f"dya_{m1}_{m2}_{t}")
                model.addConstr(dy_abs >= pos_y[m2, t] - pos_y[m1, t], f"dyb_{m1}_{m2}_{t}")
                model.addConstr(dx_abs + dy_abs >= D_col, f"col_{m1}_{m2}_{t}")

    # ================================================================
    # 4. OBJECTIVE (Eq. 11a)
    # ================================================================
    # Maximize: sum_m [P_remain_m / P_max + (AoI_max - AoI_avg_m) / AoI_max]
    obj = gp.quicksum(
        P_remain[m, T_total] / P_max + (AoI_max - aoi_avg[m]) / AoI_max
        for m in aircraft
    )
    model.setObjective(obj, GRB.MAXIMIZE)

    # ================================================================
    # 5. SOLVE
    # ================================================================
    model.optimize()

    # ================================================================
    # 6. OUTPUT
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
                x_val = pos_x[m, t].X
                y_val = pos_y[m, t].X
                traj.append((round(x_val), round(y_val)))
            print(f"  Trajectory: {traj}")
            print(f"  Remaining Power: {P_remain[m, T_total].X:.4f} kWh "
                  f"({P_remain[m, T_total].X / P_max * 100:.1f}%)")
            print(f"  Avg AoI: {aoi_avg[m].X:.2f} slots")

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