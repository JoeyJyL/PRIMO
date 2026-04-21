import gurobipy as gp
from gurobipy import GRB

def solve_baycity_uam():
    # --- 1. Data Input ---
    Budget = 28.0  # Reduced from 30M due to budget cut
    
    # Vertiports: ID -> Cost
    verts = {
        1: 10.0, # Downtown
        2: 12.0, # Island
        3: 8.0,  # Tech
        4: 6.0   # Suburb
    }
    
    # RS Routes: ID -> {ends: (u, v), demand}
    rs_routes = {
        'A': {'ends': (1, 2), 'dem': 500},
        'B': {'ends': (1, 3), 'dem': 300}
    }
    
    # ODM Zones: ID -> {covered_by: [v], demand}
    odm_zones = {
        'Z1': {'cov': [1], 'dem': 100},
        'Z2': {'cov': [2], 'dem': 50},
        'Z3': {'cov': [3], 'dem': 80},
        'Z4': {'cov': [4], 'dem': 150}
    }
    
    # --- 2. Model ---
    model = gp.Model("BayCity_UAM")
    
    # Variables
    x = model.addVars(verts.keys(), vtype=GRB.BINARY, name="OpenVert")
    y = model.addVars(rs_routes.keys(), vtype=GRB.BINARY, name="RS_Active")
    w = model.addVars(odm_zones.keys(), vtype=GRB.BINARY, name="ODM_Covered")
    
    # Objective
    rs_obj = gp.quicksum(r['dem'] * y[rid] for rid, r in rs_routes.items())
    odm_obj = gp.quicksum(z['dem'] * w[zid] for zid, z in odm_zones.items())
    model.setObjective(rs_obj + odm_obj, GRB.MAXIMIZE)
    
    # Constraints
    
    # 1. Budget
    model.addConstr(gp.quicksum(verts[i] * x[i] for i in verts) <= Budget, name="Budget")
    
    # 2. RS Logic
    for rid, r in rs_routes.items():
        u, v = r['ends']
        model.addConstr(y[rid] <= x[u])
        model.addConstr(y[rid] <= x[v])
        
    # 3. ODM Logic
    for zid, z in odm_zones.items():
        model.addConstr(w[zid] <= gp.quicksum(x[v] for v in z['cov']))
        
    # --- 3. Solve ---
    model.optimize()
    
    # --- 4. Output ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "open_vertiports": [i for i in verts if x[i].X > 0.5],
            "active_rs_routes": [rid for rid in rs_routes if y[rid].X > 0.5],
            "covered_odm_zones": [zid for zid in odm_zones if w[zid].X > 0.5]
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_baycity_uam())
