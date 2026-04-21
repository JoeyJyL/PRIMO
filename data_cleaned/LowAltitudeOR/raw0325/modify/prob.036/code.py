import gurobipy as gp
from gurobipy import GRB

def solve_vtol_arrival_scheduling():
    # --- 1. Data Input ---
    # Aircraft data: ETA and STA in seconds
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

    N = list(aircraft.keys())       # Set of aircraft
    n = len(N)
    S = list(range(1, n + 1))       # Sequence positions {1, 2, ..., 8}

    delta_L = 40    # Minimum landing separation time (seconds)
    t0 = 0          # Current time (seconds)
    alpha = 0.4     # Early operation weight

    # Precompute the sequence-based time for each position s
    # seq_time[s] = delta_L * (s - 1) + t0
    seq_time = {s: delta_L * (s - 1) + t0 for s in S}

    # --- 2. Optimization Model ---
    model = gp.Model("VTOL_Arrival_Scheduling")

    # Decision variables
    # x[i, s] = 1 if aircraft i is assigned to sequence position s
    x = model.addVars(N, S, vtype=GRB.BINARY, name="x")

    # AETA[i]: adjusted estimated time of arrival for aircraft i
    AETA = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="AETA")

    # d_plus[i]: late deviation (AETA_i - STA_i if positive)
    # d_minus[i]: early deviation (STA_i - AETA_i if positive)
    d_plus = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="d_plus")
    d_minus = model.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="d_minus")

    # --- 3. Objective Function ---
    # Minimize weighted absolute deviation
    model.setObjective(
        gp.quicksum(alpha * d_minus[i] + (1 - alpha) * d_plus[i] for i in N),
        GRB.MINIMIZE
    )

    # --- 4. Constraints ---

    # C1: Each aircraft is assigned exactly one sequence position
    for i in N:
        model.addConstr(
            gp.quicksum(x[i, s] for s in S) == 1,
            name=f"OneSeqPerAircraft_{i}"
        )

    # C2: Each sequence position is assigned exactly one aircraft
    for s in S:
        model.addConstr(
            gp.quicksum(x[i, s] for i in N) == 1,
            name=f"OneAircraftPerSeq_{s}"
        )

    # C3: AETA >= ETA
    for i in N:
        model.addConstr(
            AETA[i] >= aircraft[i]['ETA'],
            name=f"AETA_geq_ETA_{i}"
        )

    # C4: AETA >= sequence-based time
    # AETA_i >= sum_s x[i,s] * seq_time[s]
    for i in N:
        model.addConstr(
            AETA[i] >= gp.quicksum(x[i, s] * seq_time[s] for s in S),
            name=f"AETA_geq_SeqTime_{i}"
        )

    # C5: Deviation definition: d_plus - d_minus = AETA - STA
    for i in N:
        model.addConstr(
            d_plus[i] - d_minus[i] == AETA[i] - aircraft[i]['STA'],
            name=f"DevDef_{i}"
        )

    # C6: Diplomatic Priority Constraint - A6 must land in positions 1, 2, or 3
    model.addConstr(
        gp.quicksum(x['A6', s] for s in [1, 2, 3]) == 1,
        name="DiplomaticPriority_A6"
    )

    # --- 5. Solve ---
    model.optimize()

    # --- 6. Output ---
    if model.status == GRB.OPTIMAL:
        result = {
            "status": "optimal",
            "obj": round(model.ObjVal, 2),
        }

        print(f"\nOptimal Objective Value: {model.ObjVal:.2f}")
        print("\n--- Landing Schedule ---")

        schedule = []
        for i in N:
            for s in S:
                if x[i, s].X > 0.5:
                    schedule.append((s, i))
                    break

        schedule.sort()
        for seq_pos, ac in schedule:
            aeta_val = AETA[ac].X
            sta_val = aircraft[ac]['STA']
            eta_val = aircraft[ac]['ETA']
            dp = d_plus[ac].X
            dm = d_minus[ac].X
            print(f"  Seq {seq_pos}: {ac} | ETA={eta_val:>5.0f}s | "
                  f"AETA={aeta_val:>5.0f}s | STA={sta_val:>5.0f}s | "
                  f"Early={dm:>5.1f}s | Late={dp:>5.1f}s")

        return result
    else:
        return {"status": "infeasible"}


if __name__ == "__main__":
    result = solve_vtol_arrival_scheduling()
    print(f"\nResult: {result}")
