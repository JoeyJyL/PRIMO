import gurobipy as gp
from gurobipy import GRB
import math

def solve_latency_lrp():
    # --- 1. Data Input ---
    # Nodes: 0=FC1, 1=FC2, 2=C1, 3=C2, 4=C3, 5=C4
    depots = {
        0: {'loc': (0, 0), 'fixed': 20},
        1: {'loc': (10, 0), 'fixed': 20}
    }

    customers = {
        2: {'loc': (1, 0)},
        3: {'loc': (2, 0)},
        4: {'loc': (9, 0)},
        5: {'loc': (8, 0)}
    }

    all_nodes = {**{k: v['loc'] for k, v in depots.items()},
                 **{k: v['loc'] for k, v in customers.items()}}
    node_ids = list(all_nodes.keys())
    depot_ids = list(depots.keys())
    cust_ids = list(customers.keys())

    def get_time(i, j):
        p1, p2 = all_nodes[i], all_nodes[j]
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    times = {(i, j): get_time(i, j) for i in node_ids for j in node_ids}

    U = 1000  # Upper bound for McCormick linearization

    # --- 2. Model ---
    model = gp.Model("Latency_LRP_ProductLinearization")
    model.setParam('OutputFlag', 0)

    # Variables
    y = model.addVars(depot_ids, vtype=GRB.BINARY, name="OpenDepot")
    x = model.addVars(node_ids, node_ids, vtype=GRB.BINARY, name="Route")
    t = model.addVars(node_ids, vtype=GRB.CONTINUOUS, name="ArrivalTime")

    # Auxiliary variables: delta[i,j] linearizes (t[i] + tau[i,j]) * x[i,j]
    # Only defined for arcs where j is a customer
    delta = {}
    for i in node_ids:
        for j in cust_ids:
            if i != j:
                delta[i, j] = model.addVar(lb=0, vtype=GRB.CONTINUOUS,
                                           name=f"delta_{i}_{j}")

    # Objective: Fixed Cost + Sum of Arrival Times (Latency)
    obj_fixed = gp.quicksum(depots[d]['fixed'] * y[d] for d in depot_ids)
    obj_latency = gp.quicksum(t[c] for c in cust_ids)
    model.setObjective(obj_fixed + obj_latency, GRB.MINIMIZE)

    # --- Constraints ---

    # 1. Degree Constraints (Single visit per customer)
    for c in cust_ids:
        model.addConstr(
            gp.quicksum(x[i, c] for i in node_ids if i != c) == 1,
            name=f"In_{c}")
        model.addConstr(
            gp.quicksum(x[c, j] for j in node_ids if j != c) == 1,
            name=f"Out_{c}")

    # 2. Depot Logic
    for d in depot_ids:
        model.addConstr(
            gp.quicksum(x[d, j] for j in cust_ids) <= len(cust_ids) * y[d],
            name=f"DepotOpen_{d}")
        model.addConstr(
            gp.quicksum(x[d, j] for j in cust_ids) ==
            gp.quicksum(x[j, d] for j in cust_ids),
            name=f"DepotBal_{d}")

    # 3. Depot start time
    for d in depot_ids:
        model.addConstr(t[d] == 0, name=f"StartTime_{d}")

    # 4. Linearized product constraints (McCormick envelope)
    # delta[i,j] = (t[i] + tau[i,j]) * x[i,j]
    for i in node_ids:
        for j in cust_ids:
            if i != j:
                tau = times[i, j]
                # Upper bound: delta <= U * x (zero when arc not used)
                model.addConstr(delta[i, j] <= U * x[i, j],
                                name=f"delta_ub1_{i}_{j}")
                # Upper bound: delta <= t[i] + tau
                model.addConstr(delta[i, j] <= t[i] + tau,
                                name=f"delta_ub2_{i}_{j}")
                # Lower bound: delta >= (t[i] + tau) - U*(1 - x[i,j])
                model.addConstr(delta[i, j] >= t[i] + tau - U * (1 - x[i, j]),
                                name=f"delta_lb1_{i}_{j}")

    # 5. Arrival time definition via auxiliary variables
    # t[j] = sum_i delta[i,j] for each customer j
    for j in cust_ids:
        model.addConstr(
            t[j] == gp.quicksum(delta[i, j] for i in node_ids if i != j),
            name=f"TimeDefn_{j}")

    # 6. No Inter-Depot arcs
    for d1 in depot_ids:
        for d2 in depot_ids:
            model.addConstr(x[d1, d2] == 0)

    # --- 3. Solve ---
    model.optimize()

    # --- 4. Output ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "open_fc": [d for d in depot_ids if y[d].X > 0.5],
            "latencies": {c: t[c].X for c in cust_ids}
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_latency_lrp())
