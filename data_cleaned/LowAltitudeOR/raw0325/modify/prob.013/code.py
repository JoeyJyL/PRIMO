
import gurobipy as gp
from gurobipy import GRB
import math

def solve_3df_mom():
    # --- 1. Data Input ---
    # Candidates: ID -> (x, y, cost, noise)
    candidates = {
        1: {'loc': (2, 2), 'cost': 100, 'noise': 5},
        2: {'loc': (2, 8), 'cost': 120, 'noise': 8},
        3: {'loc': (8, 2), 'cost': 110, 'noise': 4},
        4: {'loc': (8, 8), 'cost': 130, 'noise': 6},
        5: {'loc': (5, 5), 'cost': 90,  'noise': 3}
    }
    
    # Zones: ID -> (x, y, demand)
    zones = {
        1: {'loc': (1, 1), 'dem': 50},
        2: {'loc': (1, 9), 'dem': 40},
        3: {'loc': (9, 1), 'dem': 60},
        4: {'loc': (9, 9), 'dem': 45}
    }
    
    # Parameters
    P_max = 3
    R_max = 4.0
    
    # Weights
    W_cov = 10.0
    W_cost = 1.0
    W_noise = 2.0
    
    cand_ids = list(candidates.keys())
    zone_ids = list(zones.keys())

    # Distance & Coverage
    def get_dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # a[i,j] = 1 if j covers i
    a = {}
    for i in zone_ids:
        for j in cand_ids:
            if get_dist(zones[i]['loc'], candidates[j]['loc']) <= R_max:
                a[i, j] = 1
            else:
                a[i, j] = 0

    # --- 2. Model ---
    model = gp.Model("CityFly_Location")
    
    # Variables
    x = model.addVars(cand_ids, vtype=GRB.BINARY, name="Open")
    y = model.addVars(zone_ids, vtype=GRB.BINARY, name="Covered")
    
    # Objective Terms
    term_cov = gp.quicksum(zones[i]['dem'] * y[i] for i in zone_ids)
    term_cost = gp.quicksum(candidates[j]['cost'] * x[j] for j in cand_ids)
    term_noise = gp.quicksum(candidates[j]['noise'] * x[j] for j in cand_ids)
    
    model.setObjective(W_cov * term_cov - W_cost * term_cost - W_noise * term_noise, GRB.MAXIMIZE)
    
    # Constraints
    
    # 1. Max Vertiports
    model.addConstr(gp.quicksum(x[j] for j in cand_ids) <= P_max, name="MaxP")
    
    # 2. Coverage Logic
    for i in zone_ids:
        model.addConstr(y[i] <= gp.quicksum(a[i, j] * x[j] for j in cand_ids), name=f"CoverLogic_{i}")
    
    # 3. Equity Mandate: Zone Z2 must be covered
    model.addConstr(y[2] == 1, name="EquityMandate_Z2")
        
    # --- 3. Solve ---
    model.optimize()
    
    # --- 4. Output ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "open_vertiports": [j for j in cand_ids if x[j].X > 0.5],
            "covered_zones": [i for i in zone_ids if y[i].X > 0.5]
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_3df_mom())
