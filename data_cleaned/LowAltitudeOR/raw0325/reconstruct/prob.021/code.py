import math
import json
import gurobipy as gp
from gurobipy import GRB

def solve_drone_charging_network():
    # 1. Data
    # Nodes: 0=Source, 1=Dest, 2=CS1(Direct), 3=CS2(North), 4=CS3(South)
    nodes = {
        0: (0, 0),
        1: (22, 0),
        2: (11, 0),
        3: (11, 5),
        4: (11, -5)
    }

    fixed_costs = {
        0: 0.0, 1: 0.0,
        2: 500.0,
        3: 100.0,
        4: 100.0
    }

    max_range = 15.0
    travel_cost_per_km = 1.0

    # 2. Build feasible arc set (distance <= max_range)
    def get_dist(i, j):
        p1, p2 = nodes[i], nodes[j]
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    arcs = []
    dist = {}
    for i in nodes:
        for j in nodes:
            if i != j:
                d = get_dist(i, j)
                if d <= max_range:
                    arcs.append((i, j))
                    dist[(i, j)] = d

    cost = {(i, j): dist[(i, j)] * travel_cost_per_km for (i, j) in arcs}

    N = list(nodes.keys())
    O = 0  # Origin
    D = 1  # Destination

    # 3. Build Gurobi Model
    m = gp.Model("drone_charging")
    m.setParam('OutputFlag', 0)

    # Decision variables
    y = {i: m.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in N}
    x = {(i, j): m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}") for (i, j) in arcs}

    # Auxiliary variables: w[i,j] = y[i] * y[j] (linearized product)
    w = {(i, j): m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"w_{i}_{j}") for (i, j) in arcs}

    m.update()

    # Objective: minimize fixed costs + travel costs
    m.setObjective(
        gp.quicksum(fixed_costs[i] * y[i] for i in N) +
        gp.quicksum(cost[(i, j)] * x[(i, j)] for (i, j) in arcs),
        GRB.MINIMIZE
    )

    # Constraints
    # 1. Flow balance
    for i in N:
        outflow = gp.quicksum(x[(i, j)] for j in N if (i, j) in x)
        inflow = gp.quicksum(x[(j, i)] for j in N if (j, i) in x)
        if i == O:
            m.addConstr(outflow - inflow == 1, name=f"flow_{i}")
        elif i == D:
            m.addConstr(outflow - inflow == -1, name=f"flow_{i}")
        else:
            m.addConstr(outflow - inflow == 0, name=f"flow_{i}")

    # 2. Linearized product constraints: w[i,j] represents y[i] * y[j]
    for (i, j) in arcs:
        m.addConstr(w[(i, j)] <= y[i], name=f"w_le_yi_{i}_{j}")
        m.addConstr(w[(i, j)] <= y[j], name=f"w_le_yj_{i}_{j}")
        m.addConstr(w[(i, j)] >= y[i] + y[j] - 1, name=f"w_ge_sum_{i}_{j}")

    # 3. Arc activation via linearized product
    for (i, j) in arcs:
        m.addConstr(x[(i, j)] <= w[(i, j)], name=f"arc_act_{i}_{j}")

    # 4. Fix origin and destination as open
    m.addConstr(y[O] == 1, name="fix_origin")
    m.addConstr(y[D] == 1, name="fix_dest")

    # Optimize
    m.optimize()

    if m.status == GRB.OPTIMAL:
        # Reconstruct path
        path = []
        current = O
        path.append(current)
        while current != D:
            for j in N:
                if (current, j) in x and x[(current, j)].X > 0.5:
                    path.append(j)
                    current = j
                    break

        open_stations = [n for n in path if n not in [O, D]]
        total_dist = sum(dist[(path[k], path[k + 1])] for k in range(len(path) - 1))

        result = {
            "status": "optimal",
            "obj": round(m.ObjVal, 2),
            "path": path,
            "open_stations": open_stations,
            "details": {
                "travel_distance": round(total_dist, 2),
                "fixed_cost": sum(fixed_costs[n] for n in open_stations)
            }
        }
        return result
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    result = solve_drone_charging_network()
    print(result)
