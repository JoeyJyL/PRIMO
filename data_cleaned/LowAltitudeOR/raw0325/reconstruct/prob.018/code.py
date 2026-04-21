import gurobipy as gp
from gurobipy import GRB
import math

def solve_double_layer_network():
    # --- 1. Data Input ---
    supply = (0, 0)

    hubs = {
        1: {'loc': (5, 5),  'fixed': 500},
        2: {'loc': (5, -5), 'fixed': 500},
        3: {'loc': (10, 0), 'fixed': 600}
    }

    customers = {
        1: {'loc': (6, 6),  'dem': 10},
        2: {'loc': (6, 4),  'dem': 10},
        3: {'loc': (6, -6), 'dem': 10},
        4: {'loc': (11, 1), 'dem': 10}
    }

    c_trans = 1.0
    alpha_congestion = 2.0

    hub_ids = list(hubs.keys())
    cust_ids = list(customers.keys())

    def get_dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # --- 2. Model (Indicator Constraint + Variable Elimination of L_j) ---
    model = gp.Model("DoubleLayerUAV_Indicator")

    # Variables (L_j eliminated)
    y = model.addVars(hub_ids, vtype=GRB.BINARY, name="OpenHub")
    x = model.addVars(hub_ids, cust_ids, vtype=GRB.BINARY, name="Assign")

    # Objective
    # 1. Fixed Cost
    obj_fixed = gp.quicksum(hubs[j]['fixed'] * y[j] for j in hub_ids)

    # 2. Transport Cost: S -> H -> C
    obj_trans = gp.quicksum(
        customers[k]['dem'] * (get_dist(supply, hubs[j]['loc']) +
                               get_dist(hubs[j]['loc'], customers[k]['loc'])) *
        c_trans * x[j, k]
        for j in hub_ids for k in cust_ids)

    # 3. Congestion Cost: alpha * (sum_k D_k x_{jk})^2  — L_j substituted
    congestion_exprs = []
    for j in hub_ids:
        load_expr = gp.quicksum(customers[k]['dem'] * x[j, k] for k in cust_ids)
        congestion_exprs.append(alpha_congestion * load_expr * load_expr)
    obj_cong = gp.quicksum(congestion_exprs)

    model.setObjective(obj_fixed + obj_trans + obj_cong, GRB.MINIMIZE)

    # Constraints

    # 1. Assignment
    for k in cust_ids:
        model.addConstr(
            gp.quicksum(x[j, k] for j in hub_ids) == 1,
            name=f"Assign_{k}")

    # 2. Indicator constraint: y_j = 0 => x_{jk} = 0
    for j in hub_ids:
        for k in cust_ids:
            model.addGenConstrIndicator(
                y[j], False, x[j, k], GRB.EQUAL, 0.0,
                name=f"Ind_{j}_{k}")

    # --- 3. Solve ---
    model.optimize()

    # --- 4. Output ---
    if model.status == GRB.OPTIMAL:
        assignments = {}
        for k in cust_ids:
            for j in hub_ids:
                if x[j, k].X > 0.5:
                    assignments[k] = j

        hub_loads = {}
        for j in hub_ids:
            if y[j].X > 0.5:
                hub_loads[j] = sum(customers[k]['dem'] * x[j, k].X for k in cust_ids)

        return {
            "status": "optimal",
            "obj": round(model.ObjVal, 2),
            "open_hubs": [j for j in hub_ids if y[j].X > 0.5],
            "assignments": assignments,
            "hub_loads": hub_loads
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_double_layer_network())
