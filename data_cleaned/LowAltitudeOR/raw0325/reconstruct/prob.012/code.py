import gurobipy as gp
from gurobipy import GRB
import math
import json

def solve_drone_network_design():
    # --- 1. Data Input ---
    bases = {
        'DB1': (10, 20),
        'DB2': (30, 15),
        'DB3': (25, 40),
        'DB4': (50, 25)
    }

    otrs = {
        'O1': (12, 18, 0.8),
        'O2': (28, 20, 1.2),
        'O3': (35, 12, 0.6),
        'O4': (22, 38, 0.9),
        'O5': (48, 30, 0.7),
        'O6': (52, 22, 1.0)
    }

    p = 5           # Total drones available
    q = 4           # Maximum drone bases to open
    M = 2           # Maximum drones per base
    v = 30.0        # Drone speed
    r = 25.0        # Drone coverage radius
    E_xi = 0.4      # Expected non-travel service time
    epsilon = 0.01  # Stability margin

    J = list(bases.keys())
    I = list(otrs.keys())

    # --- 2. Pre-processing ---
    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    d = {}
    E_S = {}
    I_j = {j: [] for j in J}
    J_i = {i_: [] for i_ in I}

    for j in J:
        for i_ in I:
            dist = euclidean_distance(bases[j], (otrs[i_][0], otrs[i_][1]))
            d[i_, j] = dist
            E_S[i_, j] = dist / v + E_xi
            if dist <= r:
                I_j[j].append(i_)
                J_i[i_].append(j)

    lam = {i_: otrs[i_][2] for i_ in I}
    total_lambda = sum(lam.values())

    # --- 3. Model ---
    model = gp.Model("Drone_Network_AggCap_IntK")
    model.setParam('TimeLimit', 300)

    # Decision Variables
    x = model.addVars(J, vtype=GRB.BINARY, name="x")

    y = {}
    for i_ in I:
        for j in J_i[i_]:
            y[i_, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i_}_{j}")

    # Direct integer variable K_j (replaces gamma decomposition)
    K = model.addVars(J, vtype=GRB.INTEGER, lb=0, ub=M, name="K")

    # Auxiliary
    eta = model.addVars(J, vtype=GRB.CONTINUOUS, lb=0, name="eta")
    ws = model.addVars(J, vtype=GRB.CONTINUOUS, lb=0, name="ws")

    # --- 4. Constraints ---

    # 1. Assignment: each OTR assigned to exactly one base
    for i_ in I:
        if J_i[i_]:
            model.addConstr(
                gp.quicksum(y[i_, j] for j in J_i[i_]) == 1,
                name=f"assign_{i_}")

    # 2. Aggregated capacity linking (replaces individual y_{ij} <= x_j)
    num_otrs = len(I)
    for j in J:
        if I_j[j]:
            model.addConstr(
                gp.quicksum(y[i_, j] for i_ in I_j[j]) <= num_otrs * x[j],
                name=f"agg_cap_{j}")

    # 3. Maximum bases
    model.addConstr(gp.quicksum(x[j] for j in J) <= q, name="max_bases")

    # 4. Drone deployment via direct integer K_j with linking
    for j in J:
        model.addConstr(K[j] >= x[j], name=f"K_lb_{j}")
        model.addConstr(K[j] <= M * x[j], name=f"K_ub_{j}")

    # 5. Total drones
    model.addConstr(gp.quicksum(K[j] for j in J) == p, name="total_drones")

    # 6. Arrival rate definition
    for j in J:
        model.addConstr(
            eta[j] == gp.quicksum(lam[i_] * y[i_, j] for i_ in I_j[j]),
            name=f"eta_{j}")

    # 7. Weighted service time
    for j in J:
        model.addConstr(
            ws[j] == gp.quicksum(lam[i_] * y[i_, j] * E_S[i_, j] for i_ in I_j[j]),
            name=f"ws_{j}")

    # 8. Stability
    for j in J:
        model.addConstr(ws[j] <= K[j] - epsilon, name=f"stability_{j}")

    # --- 5. Objective ---
    flight_time = gp.quicksum(
        lam[i_] * d[i_, j] * y[i_, j] / v / total_lambda
        for i_ in I for j in J_i[i_]
    )
    queueing_penalty = 0.5 * gp.quicksum(ws[j] for j in J) / total_lambda

    model.setObjective(flight_time + queueing_penalty, GRB.MINIMIZE)

    # --- 6. Solve ---
    model.optimize()

    # --- 7. Output ---
    if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
        opened_bases = [j for j in J if x[j].X > 0.5]

        drones_at_base = {}
        for j in J:
            kv = round(K[j].X)
            if kv > 0:
                drones_at_base[j] = kv

        assignments = {}
        for i_ in I:
            for j in J_i[i_]:
                if y[i_, j].X > 0.5:
                    assignments[i_] = j

        total_response = 0
        for i_ in I:
            j = assignments[i_]
            total_response += lam[i_] * d[i_, j] / v
        avg_flight_time = total_response / total_lambda

        result = {
            "status": "optimal" if model.status == GRB.OPTIMAL else "feasible",
            "obj": round(model.ObjVal, 4),
            "avg_flight_time_hours": round(avg_flight_time, 4),
            "opened_bases": opened_bases,
            "drones_at_base": drones_at_base,
            "assignments": assignments
        }
        return result
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    result = solve_drone_network_design()
    print(json.dumps(result, indent=4))
