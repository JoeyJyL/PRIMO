import gurobipy as gp
from gurobipy import GRB
import math

def solve_choice_based_location():
    # --- 1. Data Input ---
    skyports = [1, 2, 3, 4]
    max_open = 2

    beta_time = -0.5
    beta_cost = -0.1

    routes = [
        {'id': 1, 'o': 1, 'd': 3, 'dem': 2000, 'gt': 60, 'gc': 10, 'at': 15, 'ac': 50, 'acc': 0},
        {'id': 2, 'o': 2, 'd': 3, 'dem': 1500, 'gt': 50, 'gc': 8,  'at': 12, 'ac': 45, 'acc': 0},
        {'id': 3, 'o': 1, 'd': 4, 'dem': 500,  'gt': 40, 'gc': 12, 'at': 20, 'ac': 40, 'acc': 2}
    ]

    # --- 2. Pre-calculate Logit Probabilities ---
    route_uam_demand = {}
    for r in routes:
        u_ground = beta_time * r['gt'] + beta_cost * r['gc']
        u_air    = beta_time * r['at'] + beta_cost * r['ac'] + r['acc']
        exp_g = math.exp(u_ground)
        exp_a = math.exp(u_air)
        prob_uam = exp_a / (exp_a + exp_g)
        route_uam_demand[r['id']] = r['dem'] * prob_uam

    # --- 3. Model (Variable Elimination + MIN General Constraint) ---
    model = gp.Model("Choice_Skyport_MinGC")

    # Skyport open variables
    y = model.addVars(skyports, vtype=GRB.BINARY, name="Open")

    # Product-indicator variables (replace route activation x_r)
    p = model.addVars([r['id'] for r in routes], vtype=GRB.BINARY, name="Prod")

    # Objective
    model.setObjective(
        gp.quicksum(route_uam_demand[r['id']] * p[r['id']] for r in routes),
        GRB.MAXIMIZE)

    # Budget constraint
    model.addConstr(gp.quicksum(y[k] for k in skyports) == max_open, name="Budget")

    # MIN general constraint: p_r = min(y_origin, y_dest)
    for r in routes:
        model.addGenConstrMin(
            p[r['id']],
            [y[r['o']], y[r['d']]],
            name=f"MinGC_{r['id']}")

    # --- 4. Solve ---
    model.optimize()

    # --- 5. Output ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "open_skyports": [k for k in skyports if y[k].X > 0.5],
            "active_routes": [r['id'] for r in routes if p[r['id']].X > 0.5],
            "market_capture": {r['id']: route_uam_demand[r['id']] for r in routes}
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_choice_based_location())
