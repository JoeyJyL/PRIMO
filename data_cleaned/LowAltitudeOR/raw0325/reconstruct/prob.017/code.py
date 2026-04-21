import gurobipy as gp
from gurobipy import GRB

def solve_baycity_uam():
    # --- 1. Data Input ---
    Budget = 30.0

    verts = {
        1: 10.0,  # Downtown
        2: 12.0,  # Island
        3: 8.0,   # Tech
        4: 6.0    # Suburb
    }

    rs_routes = {
        'A': {'ends': (1, 2), 'dem': 500},
        'B': {'ends': (1, 3), 'dem': 300}
    }

    # ODM zones: each covered by exactly one vertiport (variable eliminated)
    odm_zones = {
        'Z1': {'cov_vert': 1, 'dem': 100},
        'Z2': {'cov_vert': 2, 'dem': 50},
        'Z3': {'cov_vert': 3, 'dem': 80},
        'Z4': {'cov_vert': 4, 'dem': 150}
    }

    # --- 2. Model ---
    model = gp.Model("BayCity_UAM_AndGC")

    # Variables (w_z eliminated)
    x = model.addVars(verts.keys(), vtype=GRB.BINARY, name="OpenVert")
    y = model.addVars(rs_routes.keys(), vtype=GRB.BINARY, name="RS_Active")

    # Objective: RS demand + ODM demand (w_z replaced by x_{v_z})
    rs_obj = gp.quicksum(r['dem'] * y[rid] for rid, r in rs_routes.items())
    odm_obj = gp.quicksum(z['dem'] * x[z['cov_vert']] for z in odm_zones.values())
    model.setObjective(rs_obj + odm_obj, GRB.MAXIMIZE)

    # Constraints

    # 1. Budget
    model.addConstr(
        gp.quicksum(verts[i] * x[i] for i in verts) <= Budget,
        name="Budget")

    # 2. AND general constraint: y_r = and(x_i, x_j)
    for rid, r in rs_routes.items():
        u, v = r['ends']
        model.addGenConstrAnd(
            y[rid], [x[u], x[v]],
            name=f"AndGC_{rid}")

    # --- 3. Solve ---
    model.optimize()

    # --- 4. Output ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "open_vertiports": [i for i in verts if x[i].X > 0.5],
            "active_rs_routes": [rid for rid in rs_routes if y[rid].X > 0.5],
            "covered_odm_zones": [zid for zid, z in odm_zones.items()
                                  if x[z['cov_vert']].X > 0.5]
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_baycity_uam())
