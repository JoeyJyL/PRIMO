import gurobipy as gp
from gurobipy import GRB
import math
import json

def solve_drone_latency_location_routing():
    """
    Solve the Drone Latency Location Routing Problem.
    Uses linearized product auxiliary variables (McCormick envelope) for wait time
    propagation, replacing the Big-M time propagation constraint.
    """

    # --- 1. Data Input ---
    fcs = {
        'FC1': (50, 50),
        'FC2': (20, 10)
    }

    customers = {
        'C1': (30, 40, 0.5),
        'C2': (60, 30, 0.4),
        'C3': (45, 70, 0.6),
        'C4': (70, 60, 0.5)
    }

    # Parameters
    k = 2
    Q = 2.0
    DS = 2
    phi = 2
    drone_speed = 40
    rho = 1.0
    uncertainty_ratio = 0.2

    n = len(customers)

    C = list(customers.keys())
    D = list(fcs.keys())

    # --- 2. Pre-processing ---
    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def get_coords(node):
        if node in fcs:
            return fcs[node]
        else:
            return (customers[node][0], customers[node][1])

    t_robust = {}
    for i in D + C:
        for j in D + C:
            if i != j:
                dist = euclidean_distance(get_coords(i), get_coords(j))
                t_bar = dist / drone_speed * 60
                t_robust[i, j] = t_bar * (1 + rho * uncertainty_ratio)

    d = {c: customers[c][2] for c in C}

    U = 1000  # Upper bound for McCormick linearization

    # --- 3. Model ---
    model = gp.Model("Drone_Latency_LRP_ProductLinearization")
    model.setParam('TimeLimit', 300)
    model.setParam('OutputFlag', 0)

    # Decision Variables
    z = model.addVars(D, vtype=GRB.BINARY, name="z")
    x = model.addVars([(i, j) for i in D + C for j in D + C if i != j],
                      vtype=GRB.BINARY, name="x")
    a = model.addVars(C, vtype=GRB.INTEGER, lb=1, ub=n, name="a")
    w = model.addVars(C, vtype=GRB.CONTINUOUS, lb=0, name="w")
    load = model.addVars([(i, j) for i in D + C for j in D + C if i != j],
                         vtype=GRB.CONTINUOUS, lb=0, ub=Q, name="load")

    # Auxiliary variables: delta[i,j] for customer->customer arcs
    # delta[i,j] linearizes (w[i] + t_robust[i,j]) * x[i,j]
    delta = {}
    for i in C:
        for j in C:
            if i != j:
                delta[i, j] = model.addVar(lb=0, vtype=GRB.CONTINUOUS,
                                           name=f"delta_{i}_{j}")

    # --- 4. Objective ---
    model.setObjective(gp.quicksum(w[i] for i in C), GRB.MINIMIZE)

    # --- 5. Constraints ---

    # Each customer visited exactly once
    for j in C:
        model.addConstr(
            gp.quicksum(x[i, j] for i in D + C if i != j) == 1,
            name=f"visit_{j}")

    # Flow conservation at customers
    for i in C:
        model.addConstr(
            gp.quicksum(x[j, i] for j in D + C if j != i) ==
            gp.quicksum(x[i, j] for j in D + C if j != i),
            name=f"flow_{i}")

    # Exactly k routes
    model.addConstr(
        gp.quicksum(x[s, j] for s in D for j in C) == k,
        name="routes_start")
    model.addConstr(
        gp.quicksum(x[i, s] for i in C for s in D) == k,
        name="routes_end")

    # FC opening
    for s in D:
        for j in C:
            model.addConstr(x[s, j] <= z[s], name=f"fc_open_{s}_{j}")
    for i in C:
        for s in D:
            model.addConstr(x[i, s] <= z[s], name=f"fc_return_{i}_{s}")

    # Max FCs
    model.addConstr(gp.quicksum(z[s] for s in D) <= DS, name="max_fcs")

    # FC capacity
    for s in D:
        model.addConstr(
            gp.quicksum(x[s, j] for j in C) <= phi,
            name=f"fc_cap_{s}")

    # MTZ subtour elimination
    for i in C:
        for j in C:
            if i != j:
                model.addConstr(
                    a[j] >= a[i] + 1 - n * (1 - x[i, j]),
                    name=f"mtz_{i}_{j}")

    # Position reset from FC
    for s in D:
        for j in C:
            model.addConstr(
                a[j] <= 1 + n * (1 - x[s, j]),
                name=f"pos_reset_{s}_{j}")

    # McCormick linearization for customer->customer delta
    for i in C:
        for j in C:
            if i != j:
                tau = t_robust[i, j]
                model.addConstr(delta[i, j] <= U * x[i, j],
                                name=f"delta_ub1_{i}_{j}")
                model.addConstr(delta[i, j] <= w[i] + tau,
                                name=f"delta_ub2_{i}_{j}")
                model.addConstr(delta[i, j] >= w[i] + tau - U * (1 - x[i, j]),
                                name=f"delta_lb_{i}_{j}")

    # Arrival time definition via auxiliary variables
    for j in C:
        model.addConstr(
            w[j] == gp.quicksum(t_robust[s, j] * x[s, j] for s in D) +
                    gp.quicksum(delta[i, j] for i in C if i != j),
            name=f"time_defn_{j}")

    # No inter-depot arcs
    for d1 in D:
        for d2 in D:
            if d1 != d2:
                model.addConstr(x[d1, d2] == 0)

    # Payload constraints
    for s in D:
        for j in C:
            model.addConstr(load[s, j] <= Q * x[s, j], name=f"load_cap_fc_{s}_{j}")
            model.addConstr(load[s, j] >= d[j] * x[s, j], name=f"load_min_fc_{s}_{j}")

    for i in C:
        for j in C:
            if i != j:
                model.addConstr(load[i, j] <= Q * x[i, j], name=f"load_cap_{i}_{j}")

    for i in C:
        for s in D:
            model.addConstr(load[i, s] == 0, name=f"load_return_{i}_{s}")

    for i in C:
        model.addConstr(
            gp.quicksum(load[s, i] for s in D) +
            gp.quicksum(load[j, i] for j in C if j != i) -
            gp.quicksum(load[i, j] for j in C if j != i) -
            gp.quicksum(load[i, s] for s in D) == d[i],
            name=f"load_flow_{i}")

    # --- 6. Solve ---
    model.optimize()

    # --- 7. Output ---
    if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
        opened_fcs = [s for s in D if z[s].X > 0.5]

        routes = []
        used_arcs = [(i, j) for i in D + C for j in D + C
                     if i != j and x[i, j].X > 0.5]

        for s in D:
            for j in C:
                if (s, j) in x and x[s, j].X > 0.5:
                    route = [s, j]
                    current = j
                    while True:
                        next_node = None
                        for (a_i, a_j) in used_arcs:
                            if a_i == current and a_j not in route:
                                next_node = a_j
                                break
                        if next_node is None or next_node in D:
                            if next_node is not None:
                                route.append(next_node)
                            break
                        route.append(next_node)
                        current = next_node
                    routes.append(route)

        total_latency = sum(w[i].X for i in C)

        result = {
            "status": "optimal" if model.status == GRB.OPTIMAL else "feasible",
            "obj": round(total_latency, 2),
            "opened_fcs": opened_fcs,
            "routes": routes
        }
        return result
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    result = solve_drone_latency_location_routing()
    print(json.dumps(result, indent=4))
