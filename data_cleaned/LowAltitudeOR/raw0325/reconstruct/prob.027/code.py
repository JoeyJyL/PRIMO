import gurobipy as gp
from gurobipy import GRB


def solve_uam_vertiport():
    # ========== Data Input ==========
    M = [1, 2, 3, 4, 5]
    vertiport_names = {
        1: "Downtown", 2: "Tech Park", 3: "Airport",
        4: "University", 5: "Shopping Mall"
    }

    f = {1: 5000, 2: 4500, 3: 6000, 4: 4000, 5: 5500}

    TR = [1, 2, 3, 4, 5, 6]
    od_info = {
        1: ("Zone A", "Zone B"), 2: ("Zone A", "Zone C"),
        3: ("Zone B", "Zone C"), 4: ("Zone C", "Zone D"),
        5: ("Zone A", "Zone D"), 6: ("Zone B", "Zone D")
    }

    d = {1: 120, 2: 80, 3: 60, 4: 100, 5: 50, 6: 40}
    R = {1: 150, 2: 100, 3: 80, 4: 120, 5: 180, 6: 140}

    c = {}
    for k in M:
        for m in M:
            if k != m:
                c[k, m] = 20 + abs(k - m) * 10

    p = 3

    # ========== Build Model ==========
    model = gp.Model("UAM_Vertiport_Siting_ProductLinearization")
    model.setParam('OutputFlag', 0)

    # Decision variables
    x = model.addVars(M, vtype=GRB.BINARY, name="x")

    y = {}
    for tr in TR:
        for k in M:
            for m in M:
                if k != m:
                    y[tr, k, m] = model.addVar(vtype=GRB.BINARY,
                                               name=f"y_{tr}_{k}_{m}")

    # Auxiliary variables: h[k,m] = x[k] * x[m] (linearized product)
    h = {}
    for k in M:
        for m in M:
            if k != m:
                h[k, m] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS,
                                       name=f"h_{k}_{m}")

    # Objective: max profit
    profit = gp.quicksum(
        d[tr] * (R[tr] - c[k, m]) * y[tr, k, m]
        for tr in TR for k in M for m in M if k != m
    )
    fixed_cost = gp.quicksum(f[i] * x[i] for i in M)
    model.setObjective(profit - fixed_cost, GRB.MAXIMIZE)

    # Constraint 1: Exactly p vertiports
    model.addConstr(gp.quicksum(x[i] for i in M) == p, name="SitingBudget")

    # Constraint 2: Linearized product constraints (McCormick envelope)
    for k in M:
        for m in M:
            if k != m:
                model.addConstr(h[k, m] <= x[k], name=f"h_le_xk_{k}_{m}")
                model.addConstr(h[k, m] <= x[m], name=f"h_le_xm_{k}_{m}")
                model.addConstr(h[k, m] >= x[k] + x[m] - 1,
                                name=f"h_ge_sum_{k}_{m}")

    # Constraint 3: Arc activation via linearized product
    for tr in TR:
        for k in M:
            for m in M:
                if k != m:
                    model.addConstr(y[tr, k, m] <= h[k, m],
                                    name=f"ArcAct_{tr}_{k}_{m}")

    # Constraint 4: Single routing per OD pair
    for tr in TR:
        model.addConstr(
            gp.quicksum(y[tr, k, m] for k in M for m in M if k != m) <= 1,
            name=f"SingleRoute_{tr}"
        )

    # ========== Solve ==========
    model.optimize()

    # ========== Output ==========
    if model.status == GRB.OPTIMAL:
        selected = [i for i in M if x[i].X > 0.5]
        total_fixed = sum(f[i] for i in selected)

        served_count = 0
        total_revenue = 0
        total_op_cost = 0

        for tr in TR:
            for k in M:
                for m in M:
                    if k != m and y[tr, k, m].X > 0.5:
                        total_revenue += d[tr] * R[tr]
                        total_op_cost += d[tr] * c[k, m]
                        served_count += 1

        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "selected_vertiports": selected,
            "served_od_pairs": served_count,
            "total_fixed_cost": total_fixed
        }
    else:
        return {"status": "failed"}


if __name__ == "__main__":
    result = solve_uam_vertiport()
    print(result)
