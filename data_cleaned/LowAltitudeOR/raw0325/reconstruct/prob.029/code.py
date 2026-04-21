import gurobipy as gp
from gurobipy import GRB
import math


def solve_vertiport_mmd():
    # --- 1. Data Input ---
    p = 3
    Cap = 288
    DroneRange = 30
    LastMile = 10
    d_min = 2.44

    warehouses = {
        'W1': (10, 20), 'W2': (30, 10), 'W3': (50, 25)
    }

    verts = {
        'V1': (15, 25), 'V2': (20, 15), 'V3': (25, 30), 'V4': (35, 20),
        'V5': (40, 30), 'V6': (45, 15), 'V7': (55, 20), 'V8': (30, 35)
    }

    dests = {
        'D1': (12, 22), 'D2': (18, 18), 'D3': (22, 28), 'D4': (28, 12),
        'D5': (32, 25), 'D6': (38, 18), 'D7': (42, 28), 'D8': (48, 22),
        'D9': (52, 30), 'D10': (58, 18)
    }

    W = {
        'W1': {'D1': 50, 'D2': 30, 'D3': 40, 'D4': 20, 'D5': 15,
                'D6': 10, 'D7': 5,  'D8': 5,  'D9': 0,  'D10': 0},
        'W2': {'D1': 20, 'D2': 40, 'D3': 25, 'D4': 60, 'D5': 35,
                'D6': 45, 'D7': 30, 'D8': 20, 'D9': 15, 'D10': 10},
        'W3': {'D1': 10, 'D2': 15, 'D3': 20, 'D4': 25, 'D5': 30,
                'D6': 35, 'D7': 50, 'D8': 55, 'D9': 60, 'D10': 45}
    }

    def dist(a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    # Precompute coverage
    cov = {}
    for k in verts:
        for i in warehouses:
            for j in dests:
                d_wk = dist(warehouses[i], verts[k])
                d_kd = dist(verts[k], dests[j])
                cov[(k, i, j)] = 1 if (d_wk <= DroneRange and d_kd <= LastMile) else 0

    # Pairwise vertiport distances
    vert_dist = {}
    for k1 in verts:
        for k2 in verts:
            if k1 < k2:
                vert_dist[(k1, k2)] = dist(verts[k1], verts[k2])

    # --- 2. Model ---
    model = gp.Model("Vertiport_MMD_BigM")
    model.setParam('OutputFlag', 0)

    # Variables
    z = model.addVars(verts.keys(), vtype=GRB.BINARY, name="Open")

    # Flow variables f[k,i,j]: actual packages/day (Big-M reformulation)
    f = {}
    for k in verts:
        for i in warehouses:
            for j in dests:
                ub = W[i][j] if cov[(k, i, j)] == 1 else 0
                f[(k, i, j)] = model.addVar(
                    lb=0.0, ub=ub, vtype=GRB.CONTINUOUS,
                    name=f"f_{k}_{i}_{j}")

    # Objective: maximize total packages served
    model.setObjective(
        gp.quicksum(f[(k, i, j)]
                    for k in verts for i in warehouses for j in dests),
        GRB.MAXIMIZE)

    # Constraint 1: Exactly p vertiports
    model.addConstr(gp.quicksum(z[k] for k in verts) == p, name="VertCount")

    # Constraint 2: Big-M coverage linking
    for k in verts:
        for i in warehouses:
            for j in dests:
                bigM = W[i][j] * cov[(k, i, j)]
                model.addConstr(f[(k, i, j)] <= bigM * z[k],
                                name=f"BigM_{k}_{i}_{j}")

    # Constraint 3: Capacity
    for k in verts:
        model.addConstr(
            gp.quicksum(f[(k, i, j)] for i in warehouses for j in dests)
            <= Cap * z[k],
            name=f"Cap_{k}")

    # Constraint 4: Demand conservation
    for i in warehouses:
        for j in dests:
            model.addConstr(
                gp.quicksum(f[(k, i, j)] for k in verts) <= W[i][j],
                name=f"DemCons_{i}_{j}")

    # Constraint 5: Safety distance
    for (k1, k2), d_val in vert_dist.items():
        if d_val < d_min:
            model.addConstr(z[k1] + z[k2] <= 1,
                            name=f"SafeDist_{k1}_{k2}")

    # --- 3. Solve ---
    model.optimize()

    # --- 4. Output ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "open_vertiports": [k for k in verts if z[k].X > 0.5],
            "demand_served_by_vert": {
                k: sum(f[(k, i, j)].X for i in warehouses for j in dests)
                for k in verts if z[k].X > 0.5
            }
        }
    else:
        return {"status": "infeasible"}


if __name__ == "__main__":
    print(solve_vertiport_mmd())
