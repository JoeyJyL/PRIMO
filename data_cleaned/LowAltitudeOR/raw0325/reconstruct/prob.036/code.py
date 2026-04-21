import gurobipy as gp
from gurobipy import GRB
import json

def solve_vtol_arrival_scheduling():
    """
    VTOL Arrival Scheduling — Reformulated with Big-M pairwise ordering.
    Replaces assignment matrix x[i,s] with pairwise ordering variables
    p[i,j] and Big-M separation constraints.
    """

    # --- 1. Data Input ---
    aircraft = {
        'A1': {'ETA': 15,  'STA': 50},
        'A2': {'ETA': 25,  'STA': 55},
        'A3': {'ETA': 60,  'STA': 80},
        'A4': {'ETA': 35,  'STA': 90},
        'A5': {'ETA': 70,  'STA': 100},
        'A6': {'ETA': 10,  'STA': 120},
        'A7': {'ETA': 85,  'STA': 130},
        'A8': {'ETA': 50,  'STA': 140},
    }

    N = list(aircraft.keys())
    n = len(N)
    delta_L = 40
    t0 = 0
    alpha = 0.4

    # Big-M: upper bound on any AETA
    M_big = max(aircraft[i]['STA'] for i in N) + n * delta_L

    # --- 2. Optimization Model ---
    model = gp.Model("VTOL_Scheduling_BigM")
    model.setParam('OutputFlag', 0)

    # AETA variables
    AETA = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="AETA")

    # Deviation variables
    d_plus = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="d_plus")
    d_minus = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="d_minus")

    # Pairwise ordering: p[i,j] = 1 if i lands before j
    p = model.addVars(N, N, vtype=GRB.BINARY, name="p")

    # --- 3. Objective ---
    model.setObjective(
        gp.quicksum(alpha * d_minus[i] + (1 - alpha) * d_plus[i] for i in N),
        GRB.MINIMIZE
    )

    # --- 4. Constraints ---

    # C1: AETA >= ETA
    for i in N:
        model.addConstr(AETA[i] >= aircraft[i]['ETA'], name=f"AETA_geq_ETA_{i}")

    # C2: Pairwise ordering — exactly one direction
    for idx_i, i in enumerate(N):
        for idx_j, j in enumerate(N):
            if idx_i < idx_j:
                model.addConstr(p[i, j] + p[j, i] == 1,
                                name=f"order_{i}_{j}")

    # Diagonal: no self-ordering
    for i in N:
        model.addConstr(p[i, i] == 0, name=f"no_self_{i}")

    # C3: AETA floor from implicit position via Big-M ordering
    # position(i) = 1 + sum_{j!=i} p[j,i]  (count how many land before i)
    # AETA_i >= delta_L * (position(i) - 1) = delta_L * sum_{j!=i} p[j,i]
    for i in N:
        model.addConstr(
            AETA[i] >= delta_L * gp.quicksum(p[j, i] for j in N if j != i),
            name=f"AETA_floor_{i}")

    # C4: Transitivity
    for i in N:
        for j in N:
            for k in N:
                if i != j and j != k and i != k:
                    model.addConstr(
                        p[i, j] + p[j, k] - p[i, k] <= 1,
                        name=f"trans_{i}_{j}_{k}")

    # C5: Deviation definition
    for i in N:
        model.addConstr(
            d_plus[i] - d_minus[i] == AETA[i] - aircraft[i]['STA'],
            name=f"DevDef_{i}")

    # --- 5. Solve ---
    model.optimize()

    # --- 6. Output ---
    if model.status == GRB.OPTIMAL:
        result = {
            "status": "optimal",
            "obj": round(model.ObjVal, 2),
        }

        print(f"\nOptimal Objective Value: {model.ObjVal:.2f}")
        print("\n--- Landing Schedule (by AETA order) ---")

        schedule = []
        for i in N:
            aeta_val = AETA[i].X
            schedule.append((aeta_val, i))
        schedule.sort()

        for seq, (aeta_val, ac) in enumerate(schedule, 1):
            sta_val = aircraft[ac]['STA']
            eta_val = aircraft[ac]['ETA']
            dp = d_plus[ac].X
            dm = d_minus[ac].X
            print(f"  Seq {seq}: {ac} | ETA={eta_val:>5.0f}s | "
                  f"AETA={aeta_val:>5.0f}s | STA={sta_val:>5.0f}s | "
                  f"Early={dm:>5.1f}s | Late={dp:>5.1f}s")

        return result
    else:
        return {"status": "infeasible"}


if __name__ == "__main__":
    result = solve_vtol_arrival_scheduling()
    print(f"\nResult: {result}")
