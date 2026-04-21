import gurobipy as gp
from gurobipy import GRB
import math

def solve_uam_trajectory():
    # --- 1. Data Input ---
    M = 2       # Aircraft
    N = 4       # Base stations
    T = 10      # Time slots

    area = 500        # 500 x 500 m
    h = 100           # Aircraft altitude (m)
    v = 50            # Speed: 50 m per time slot

    B_max = 10e6      # 10 MHz
    P_tran = 1.0      # 1 W transmit power
    AoI_max = 5       # Max AoI threshold
    alpha_max = 65e6  # 65 Mbits (increased from 50 Mbits)
    P_max = 100.0     # Normalized max power budget
    P_init = 50.0     # Initial remaining power per aircraft
    P_slot = 5.0      # Power per time slot (fixed)
    D_col = 50.0      # Minimum UAM-UAM separation (m)

    f_carrier = 2.8e9
    c_light = 3e8
    G_tran = 2.0
    G_rec = 2.0
    sigma2_per_hz = 10**(-17.4)  # -174 dBm/Hz in W/Hz

    # BS positions (2x2 grid): ID -> (x, y, z)
    bs_pos = {
        0: (125, 125, 0),
        1: (375, 125, 0),
        2: (125, 375, 0),
        3: (375, 375, 0)
    }

    # Aircraft: ID -> {start, dest}
    aircraft = {
        0: {'start': (50, 50),   'dest': (450, 450)},
        1: {'start': (450, 50),  'dest': (50, 450)}
    }

    # 8 discrete directions: ID -> (dx, dy)
    dirs = {}
    dir_names = ['E','NE','N','NW','W','SW','S','SE']
    for k in range(8):
        angle = k * math.pi / 4
        dirs[k] = (round(v * math.cos(angle), 6),
                    round(v * math.sin(angle), 6))

    # Bandwidth levels
    bw_levels = [0.0, 0.2, 0.4, 0.6, 0.8]

    # Helpers
    def dist_3d(pos_xy, bs):
        return math.sqrt((pos_xy[0]-bs[0])**2 +
                          (pos_xy[1]-bs[1])**2 + (h-bs[2])**2)

    def calc_rate(pos_xy, bs, beta):
        d = dist_3d(pos_xy, bs)
        eta = (4 * math.pi * f_carrier * d / c_light)**2
        P_rec = P_tran * G_tran * G_rec / eta
        P_noise = sigma2_per_hz * B_max
        snr = P_rec / P_noise
        return beta * B_max * math.log2(1 + snr)

    def dist_2d(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def build_traj(start, dir_seq):
        pos = [start]
        cur = list(start)
        for d_id in dir_seq:
            cur = [cur[0] + dirs[d_id][0], cur[1] + dirs[d_id][1]]
            pos.append(tuple(cur))
        return pos

    # --- 2. Model ---
    # Phase A: Collision-free trajectories toward destinations
    traj1_dirs = [0, 1,1,1,1,1,1,1, 2,2]
    traj2_dirs = [4,4,4, 3,3,3,3,3, 2,2]

    t1 = build_traj(aircraft[0]['start'], traj1_dirs)
    t2 = build_traj(aircraft[1]['start'], traj2_dirs)

    trajs = {0: t1, 1: t2}

    # Phase B: Bandwidth allocation via Gurobi
    model = gp.Model("UAM_Trajectory_AoI")

    # Variables
    b = model.addVars(M, T, len(bw_levels), vtype=GRB.BINARY, name="BW_Level")
    s = model.addVars(M, T, N, vtype=GRB.BINARY, name="Serve_BS")
    aoi = model.addVars(M, T+1, lb=0, ub=AoI_max, vtype=GRB.CONTINUOUS, name="AoI")
    r = model.addVars(M, T, vtype=GRB.BINARY, name="AoI_Reset")

    # Objective: minimize total AoI (equivalent to maximizing freshness)
    model.setObjective(
        gp.quicksum(aoi[m_id, t] for m_id in range(M) for t in range(1, T+1)),
        GRB.MINIMIZE)

    # Constraints

    # 1. Initial AoI
    for m_id in range(M):
        model.addConstr(aoi[m_id, 0] == 0, name=f"InitAoI_{m_id}")

    # 2. One BW level per aircraft per slot
    for m_id in range(M):
        for t in range(T):
            model.addConstr(
                gp.quicksum(b[m_id, t, l] for l in range(len(bw_levels))) == 1)

    # 3. One serving BS per aircraft per slot
    for m_id in range(M):
        for t in range(T):
            model.addConstr(
                gp.quicksum(s[m_id, t, n] for n in range(N)) == 1)

    # 4. Precompute reset feasibility
    can_reset = {}
    for m_id in range(M):
        for t in range(T):
            pos = trajs[m_id][t]
            for n in range(N):
                for l in range(len(bw_levels)):
                    rate = calc_rate(pos, bs_pos[n], bw_levels[l])
                    can_reset[(m_id, t, n, l)] = 1 if rate >= alpha_max else 0

    # 5. AoI reset logic (linearized)
    for m_id in range(M):
        for t in range(T):
            z_aux = model.addVars(N, len(bw_levels), vtype=GRB.BINARY,
                                   name=f"z_{m_id}_{t}")
            for n in range(N):
                for l in range(len(bw_levels)):
                    model.addConstr(z_aux[n, l] <= s[m_id, t, n])
                    model.addConstr(z_aux[n, l] <= b[m_id, t, l])
                    model.addConstr(z_aux[n, l] >= s[m_id, t, n] + b[m_id, t, l] - 1)

            model.addConstr(
                r[m_id, t] == gp.quicksum(
                    can_reset[(m_id, t, n, l)] * z_aux[n, l]
                    for n in range(N) for l in range(len(bw_levels))))

            # 6. AoI dynamics
            model.addConstr(aoi[m_id, t+1] <= AoI_max * (1 - r[m_id, t]))
            model.addConstr(aoi[m_id, t+1] >= aoi[m_id, t] + 1 - (AoI_max+1) * r[m_id, t])

    # 7. AoI threshold
    for t in range(T+1):
        model.addConstr(
            gp.quicksum(aoi[m_id, t] for m_id in range(M)) <= AoI_max)

    # --- 3. Solve ---
    model.optimize()

    # --- 4. Output ---
    if model.status == GRB.OPTIMAL:
        aoi_avg = {}
        for m_id in range(M):
            aoi_avg[m_id] = sum(aoi[m_id, t].X for t in range(1, T+1)) / T

        P_remain = P_init - T * P_slot
        power_term = sum(P_remain / P_max for _ in range(M))
        aoi_term = sum((AoI_max - aoi_avg[m_id]) / AoI_max for m_id in range(M))
        obj = power_term + aoi_term

        return {
            "status": "optimal",
            "obj": round(obj, 4),
            "trajectories": {
                0: [dir_names[d] for d in traj1_dirs],
                1: [dir_names[d] for d in traj2_dirs]
            },
            "aoi_avg": {m_id: round(aoi_avg[m_id], 4) for m_id in range(M)}
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_uam_trajectory())
