import gurobipy as gp
from gurobipy import GRB
import math


def solve_drone_delivery():
    # ========================================
    # 1. PARAMETERS
    # ========================================
    F = 500.0
    Q = 3.0
    v = 6.0
    tau = 60.0
    alpha = 0.217
    beta = 0.185
    xi = 650.0
    epsilon = 0.1
    T = 600.0
    M_drones = 5

    locations = {
        0: (0, 0), 1: (100, 200), 2: (300, 100),
        3: (200, 300), 4: (400, 250), 5: (150, 400)
    }
    demands = {1: 0.8, 2: 1.2, 3: 0.6, 4: 1.0, 5: 0.9}

    N = list(locations.keys())
    N0 = [i for i in N if i != 0]

    def euclidean_distance(loc1, loc2):
        return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

    dist = {}
    for i in N:
        for j in N:
            if i != j:
                dist[i, j] = euclidean_distance(locations[i], locations[j])

    max_routes = len(N0)
    R = list(range(max_routes))
    K = 10000.0

    # All arcs (i,j) with i != j
    arcs = [(i, j) for i in N for j in N if i != j]
    # Arcs where j != 0 (not returning to depot)
    arcs_j_nonzero = [(i, j) for i, j in arcs if j != 0]

    # ========================================
    # 2. CREATE MODEL
    # ========================================
    model = gp.Model("DroneDelivery_McCormick")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 300)

    # ========================================
    # 3. ORIGINAL DECISION VARIABLES
    # ========================================
    x = model.addVars(N, N, R, vtype=GRB.BINARY, name="x")
    u = model.addVars(R, vtype=GRB.BINARY, name="u")
    y = model.addVars(N, R, vtype=GRB.CONTINUOUS, lb=0, name="y")
    q = model.addVars(R, vtype=GRB.CONTINUOUS, lb=0, name="q")
    z = model.addVars(R, vtype=GRB.CONTINUOUS, lb=0, name="z")
    t = model.addVars(N, R, vtype=GRB.CONTINUOUS, lb=0, name="t")
    a = model.addVars(R, vtype=GRB.CONTINUOUS, lb=0, name="a")
    n = model.addVar(vtype=GRB.INTEGER, lb=1, ub=M_drones, name="n")

    # ========================================
    # 4. McCORMICK AUXILIARY VARIABLES
    # ========================================
    # wq[i,j,r] = q[r] * x[i,j,r]
    wq = {}
    for i, j in arcs:
        for r in R:
            wq[i, j, r] = model.addVar(lb=0, ub=Q, name=f"wq_{i}_{j}_{r}")

    # wyA[i,j,r] = y[i,r] * x[i,j,r]
    wyA = {}
    for i, j in arcs:
        for r in R:
            wyA[i, j, r] = model.addVar(lb=0, ub=Q, name=f"wyA_{i}_{j}_{r}")

    # wyB[i,j,r] = y[j,r] * x[i,j,r]  (only for j != 0)
    wyB = {}
    for i, j in arcs_j_nonzero:
        for r in R:
            wyB[i, j, r] = model.addVar(lb=0, ub=Q, name=f"wyB_{i}_{j}_{r}")

    # ========================================
    # 5. OBJECTIVE FUNCTION
    # ========================================
    model.setObjective(
        F * n + epsilon * gp.quicksum(z[r] for r in R),
        GRB.MINIMIZE
    )

    # ========================================
    # 6. ORIGINAL CONSTRAINTS (C1-C7)
    # ========================================

    # C1: Each customer visited exactly once
    for i in N0:
        model.addConstr(
            gp.quicksum(x[i, j, r] for j in N if j != i for r in R) == 1,
            name=f"visit_{i}"
        )

    # C2: Flow conservation
    for i in N:
        for r in R:
            model.addConstr(
                gp.quicksum(x[i, j, r] for j in N if j != i) ==
                gp.quicksum(x[j, i, r] for j in N if j != i),
                name=f"flow_{i}_{r}"
            )

    # C3: Route starts at depot
    for r in R:
        model.addConstr(
            gp.quicksum(x[0, j, r] for j in N0) == u[r],
            name=f"start_{r}"
        )

    # C4: Route ends at depot
    for r in R:
        model.addConstr(
            gp.quicksum(x[i, 0, r] for i in N0) == u[r],
            name=f"end_{r}"
        )

    # C5: Payload at depot
    for r in R:
        model.addConstr(
            y[0, r] == gp.quicksum(demands[j] * x[0, j, r] for j in N0),
            name=f"payload_depot_{r}"
        )

    # C6: Payload decrease
    for r in R:
        for i in N0:
            for j in N0:
                if i != j:
                    model.addConstr(
                        y[i, r] >= y[j, r] + demands[j] * x[i, j, r]
                        - K * (1 - x[i, j, r]),
                        name=f"payload_{i}_{j}_{r}"
                    )

    # C7: Capacity
    for r in R:
        for i in N:
            model.addConstr(
                q[r] + y[i, r] <= Q * u[r],
                name=f"capacity_{i}_{r}"
            )

    # ========================================
    # 7. McCORMICK ENVELOPE CONSTRAINTS (C8-C10)
    # ========================================

    # C8: wq[i,j,r] = q[r] * x[i,j,r]
    for i, j in arcs:
        for r in R:
            model.addConstr(wq[i, j, r] <= Q * x[i, j, r],
                            name=f"mq_ub1_{i}_{j}_{r}")
            model.addConstr(wq[i, j, r] <= q[r],
                            name=f"mq_ub2_{i}_{j}_{r}")
            model.addConstr(wq[i, j, r] >= q[r] - Q * (1 - x[i, j, r]),
                            name=f"mq_lb_{i}_{j}_{r}")

    # C9: wyA[i,j,r] = y[i,r] * x[i,j,r]
    for i, j in arcs:
        for r in R:
            model.addConstr(wyA[i, j, r] <= Q * x[i, j, r],
                            name=f"mya_ub1_{i}_{j}_{r}")
            model.addConstr(wyA[i, j, r] <= y[i, r],
                            name=f"mya_ub2_{i}_{j}_{r}")
            model.addConstr(wyA[i, j, r] >= y[i, r] - Q * (1 - x[i, j, r]),
                            name=f"mya_lb_{i}_{j}_{r}")

    # C10: wyB[i,j,r] = y[j,r] * x[i,j,r]  (j != 0 only)
    for i, j in arcs_j_nonzero:
        for r in R:
            model.addConstr(wyB[i, j, r] <= Q * x[i, j, r],
                            name=f"myb_ub1_{i}_{j}_{r}")
            model.addConstr(wyB[i, j, r] <= y[j, r],
                            name=f"myb_ub2_{i}_{j}_{r}")
            model.addConstr(wyB[i, j, r] >= y[j, r] - Q * (1 - x[i, j, r]),
                            name=f"myb_lb_{i}_{j}_{r}")

    # ========================================
    # 8. LINEARIZED ENERGY CONSTRAINT (C11)
    # ========================================
    for r in R:
        energy_expr = gp.LinExpr()
        for i in N:
            for j in N:
                if i != j:
                    tt = dist[i, j] / v + tau
                    # alpha * tt * wq (battery contribution)
                    energy_expr += alpha * tt * wq[i, j, r]
                    # alpha/2 * tt * wyA (departure payload contribution)
                    energy_expr += alpha * tt / 2.0 * wyA[i, j, r]
                    # alpha/2 * tt * wyB (arrival payload, j != 0 only)
                    if j != 0:
                        energy_expr += alpha * tt / 2.0 * wyB[i, j, r]
                    # beta * tt * x (base power contribution)
                    energy_expr += beta * tt * x[i, j, r]

        model.addConstr(z[r] == energy_expr, name=f"energy_{r}")

    # ========================================
    # 9. REMAINING CONSTRAINTS (C12-C17)
    # ========================================

    # C12: Battery energy
    for r in R:
        model.addConstr(xi * q[r] >= z[r], name=f"battery_{r}")

    # C13: Time progression
    for r in R:
        for i in N:
            for j in N0:
                if i != j:
                    tt = dist[i, j] / v + tau
                    model.addConstr(
                        t[j, r] >= t[i, r] + tt - K * (1 - x[i, j, r]),
                        name=f"time_{i}_{j}_{r}"
                    )

    # C14: Arrival time
    for r in R:
        for i in N0:
            tt = dist[i, 0] / v + tau
            model.addConstr(
                a[r] >= t[i, r] + tt - K * (1 - x[i, 0, r]),
                name=f"arrival_{i}_{r}"
            )

    # C15: Time limit
    for r in R:
        model.addConstr(a[r] <= T + K * (1 - u[r]), name=f"timelimit_{r}")

    # C16: Drone count
    model.addConstr(n >= gp.quicksum(u[r] for r in R), name="drone_count")

    # C17: Route ordering (symmetry breaking)
    for r in range(len(R) - 1):
        model.addConstr(u[r] >= u[r + 1], name=f"order_{r}")

    # ========================================
    # 10. SOLVE
    # ========================================
    model.optimize()

    if model.status == GRB.OPTIMAL:
        result = {"status": "optimal", "obj": round(model.ObjVal, 2)}
        num_drones = int(n.X + 0.5)
        energy_cost = sum(z[r].X for r in R) * epsilon

        print(f"\nOptimal: ${model.ObjVal:.2f}")
        print(f"  Drones: {num_drones}, Drone cost: ${F * num_drones:.2f}")
        print(f"  Energy cost: ${energy_cost:.2f}")

        for r in R:
            if u[r].X > 0.5:
                route = [0]
                current = 0
                visited = {0}
                while True:
                    next_loc = None
                    for j in N:
                        if j not in visited and x[current, j, r].X > 0.5:
                            next_loc = j
                            break
                    if next_loc is None or next_loc == 0:
                        break
                    route.append(next_loc)
                    visited.add(next_loc)
                    current = next_loc
                route.append(0)
                print(f"  Route {r}: {' -> '.join(map(str, route))}, "
                      f"battery={q[r].X:.3f}kg, energy={z[r].X:.2f}kJ")

        return result
    elif model.status == GRB.TIME_LIMIT and model.SolCount > 0:
        result = {"status": "time_limit", "obj": round(model.ObjVal, 2)}
        print(f"Time limit, best: ${model.ObjVal:.2f}")
        return result
    else:
        print(f"Status: {model.status}")
        return {"status": "infeasible"}


if __name__ == "__main__":
    result = solve_drone_delivery()
    print(f"\nResult: {result}")
