import gurobipy as gp
from gurobipy import GRB
import math


def solve_uam_trajectory():
    # --- 1. Data Input ---
    M = 2
    N = 4
    T = 10

    h = 100
    v = 50

    B_max = 10e6
    P_tran = 1.0
    AoI_max = 5
    alpha_max = 50e6
    P_max = 100.0
    P_init = 50.0
    P_slot = 5.0

    f_carrier = 2.8e9
    c_light = 3e8
    G_tran = 2.0
    G_rec = 2.0
    sigma2_per_hz = 10**(-17.4)

    bs_pos = {
        0: (125, 125, 0),
        1: (375, 125, 0),
        2: (125, 375, 0),
        3: (375, 375, 0)
    }

    aircraft = {
        0: {'start': (50, 50), 'dest': (450, 450)},
        1: {'start': (450, 50), 'dest': (50, 450)}
    }

    dirs = {}
    dir_names = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
    for k in range(8):
        angle = k * math.pi / 4
        dirs[k] = (round(v * math.cos(angle), 6),
                    round(v * math.sin(angle), 6))

    bw_levels = [0.0, 0.2, 0.4, 0.6, 0.8]
    L = len(bw_levels)

    def dist_3d(pos_xy, bs):
        return math.sqrt((pos_xy[0] - bs[0])**2 +
                         (pos_xy[1] - bs[1])**2 + (h - bs[2])**2)

    def calc_rate(pos_xy, bs, beta):
        d = dist_3d(pos_xy, bs)
        eta = (4 * math.pi * f_carrier * d / c_light)**2
        P_rec = P_tran * G_tran * G_rec / eta
        P_noise = sigma2_per_hz * B_max
        snr = P_rec / P_noise
        return beta * B_max * math.log2(1 + snr)

    def build_traj(start, dir_seq):
        pos = [start]
        cur = list(start)
        for d_id in dir_seq:
            cur = [cur[0] + dirs[d_id][0], cur[1] + dirs[d_id][1]]
            pos.append(tuple(cur))
        return pos

    # Pre-built collision-free trajectories (same as original)
    traj1_dirs = [0, 1, 1, 1, 1, 1, 1, 1, 2, 2]
    traj2_dirs = [4, 4, 4, 3, 3, 3, 3, 3, 2, 2]

    trajs = {
        0: build_traj(aircraft[0]['start'], traj1_dirs),
        1: build_traj(aircraft[1]['start'], traj2_dirs)
    }

    # --- 2. Model ---
    model = gp.Model("UAM_Trajectory_CombinedAssign")
    model.setParam('OutputFlag', 0)

    # Combined assignment variable: a[m, t, n, l]
    a = model.addVars(M, T, N, L, vtype=GRB.BINARY, name="Assign")

    # AoI and reset variables
    aoi = model.addVars(M, T + 1, lb=0, ub=AoI_max,
                        vtype=GRB.CONTINUOUS, name="AoI")
    r = model.addVars(M, T, vtype=GRB.BINARY, name="Reset")

    # Objective: minimize total AoI
    model.setObjective(
        gp.quicksum(aoi[m_id, t]
                    for m_id in range(M) for t in range(1, T + 1)),
        GRB.MINIMIZE)

    # Initial AoI
    for m_id in range(M):
        model.addConstr(aoi[m_id, 0] == 0, name=f"InitAoI_{m_id}")

    # Precompute reset feasibility
    can_reset = {}
    for m_id in range(M):
        for t in range(T):
            pos = trajs[m_id][t]
            for n in range(N):
                for l in range(L):
                    rate = calc_rate(pos, bs_pos[n], bw_levels[l])
                    can_reset[(m_id, t, n, l)] = 1 if rate >= alpha_max else 0

    for m_id in range(M):
        for t in range(T):
            # Constraint 3: Exactly one (BS, BW) assignment per slot
            model.addConstr(
                gp.quicksum(a[m_id, t, n, l]
                            for n in range(N) for l in range(L)) == 1,
                name=f"OneAssign_{m_id}_{t}")

            # Constraint 4: AoI reset via combined assignment (no product
            # linearization needed)
            model.addConstr(
                r[m_id, t] == gp.quicksum(
                    can_reset[(m_id, t, n, l)] * a[m_id, t, n, l]
                    for n in range(N) for l in range(L)),
                name=f"Reset_{m_id}_{t}")

            # Constraint 5: AoI dynamics
            model.addConstr(
                aoi[m_id, t + 1] <= AoI_max * (1 - r[m_id, t]),
                name=f"AoI_UB_{m_id}_{t}")
            model.addConstr(
                aoi[m_id, t + 1] >= aoi[m_id, t] + 1
                - (AoI_max + 1) * r[m_id, t],
                name=f"AoI_LB_{m_id}_{t}")

    # Constraint 6: AoI threshold
    for t in range(T + 1):
        model.addConstr(
            gp.quicksum(aoi[m_id, t] for m_id in range(M)) <= AoI_max,
            name=f"AoIThresh_{t}")

    # --- 3. Solve ---
    model.optimize()

    # --- 4. Output ---
    if model.status == GRB.OPTIMAL:
        aoi_avg = {}
        for m_id in range(M):
            aoi_avg[m_id] = sum(
                aoi[m_id, t].X for t in range(1, T + 1)) / T

        P_remain = P_init - T * P_slot
        power_term = sum(P_remain / P_max for _ in range(M))
        aoi_term = sum(
            (AoI_max - aoi_avg[m_id]) / AoI_max for m_id in range(M))
        obj = power_term + aoi_term

        return {
            "status": "optimal",
            "obj": round(obj, 4),
            "trajectories": {
                0: [dir_names[d] for d in traj1_dirs],
                1: [dir_names[d] for d in traj2_dirs]
            },
            "aoi_avg": {m_id: round(aoi_avg[m_id], 4)
                        for m_id in range(M)}
        }
    else:
        return {"status": "infeasible"}


if __name__ == "__main__":
    print(solve_uam_trajectory())
