
import gurobipy as gp
from gurobipy import GRB

def solve_vertiport_location():
    # --- 1. Data Input ---
    # Parameters
    Budget = 25
    
    # Vertiports: ID -> Cost
    vertiports = {
        1: 10, # CBD
        2: 15, # PEK
        3: 12  # PKX
    }
    
    # RS Routes: ID -> {endpoints: (u, v), demand: d}
    rs_routes = {
        'A': {'ends': (1, 2), 'dem': 100}, # CBD <-> PEK
        'B': {'ends': (1, 3), 'dem': 110}  # CBD <-> PKX
    }
    
    # ODM Zones: ID -> {covered_by: v, demand: d}
    odm_zones = {
        'Z1': {'cov': 1, 'dem': 20},
        'Z2': {'cov': 2, 'dem': 30},
        'Z3': {'cov': 3, 'dem': 30}
    }
    
    # --- 2. Model ---
    model = gp.Model("UAM_Network_Beijing")
    
    # Variables
    x = model.addVars(vertiports.keys(), vtype=GRB.BINARY, name="OpenVert")
    y = model.addVars(rs_routes.keys(), vtype=GRB.BINARY, name="ActiveRoute")
    w = model.addVars(odm_zones.keys(), vtype=GRB.BINARY, name="ServeZone")
    
    # Objective: Maximize Demand
    rs_obj = gp.quicksum(r['dem'] * y[rid] for rid, r in rs_routes.items())
    odm_obj = gp.quicksum(z['dem'] * w[zid] for zid, z in odm_zones.items())
    
    model.setObjective(rs_obj + odm_obj, GRB.MAXIMIZE)
    
    # Constraints
    
    # 1. Budget
    model.addConstr(gp.quicksum(vertiports[i] * x[i] for i in vertiports) <= Budget, name="Budget")
    
    # 2. RS Logic (Both endpoints must be open)
    for rid, r in rs_routes.items():
        u, v = r['ends']
        model.addConstr(y[rid] <= x[u], name=f"RS_{rid}_end1")
        model.addConstr(y[rid] <= x[v], name=f"RS_{rid}_end2")
        
    # 3. ODM Logic (Associated vertiport must be open)
    for zid, z in odm_zones.items():
        v = z['cov']
        model.addConstr(w[zid] <= x[v], name=f"ODM_{zid}")
        
    # --- 3. Solve ---
    model.optimize()
    
    # --- 4. Output ---
    if model.status == GRB.OPTIMAL:
        open_sites = [i for i in vertiports if x[i].X > 0.5]
        active_routes = [rid for rid in rs_routes if y[rid].X > 0.5]
        served_zones = [zid for zid in odm_zones if w[zid].X > 0.5]
        
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "open_vertiports": open_sites,
            "rs_routes": active_routes,
            "odm_zones": served_zones
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_vertiport_location())
