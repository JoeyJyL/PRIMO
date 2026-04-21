import gurobipy as gp
from gurobipy import GRB
import math

def solve_mcflpd():
    # --- 1. Data Input ---
    # Parameters
    R_max = 40.0  # Total Daily Range Budget (km)
    P_budget = 1  # Open 1 station
    
    # Nodes
    sites = {
        1: (0, 0),   # Site A
        2: (20, 0)   # Site B
    }
    
    villages = {
        3: {'loc': (0, 4),  'pop': 10}, # V1
        4: {'loc': (0, 8),  'pop': 15}, # V2
        5: {'loc': (3, 0),  'pop': 20}, # V3
        6: {'loc': (18, 0), 'pop': 30}, # V4
        7: {'loc': (20, 6), 'pop': 25}  # V5
    }
    
    site_ids = list(sites.keys())
    village_ids = list(villages.keys())

    # Distance Helper (Euclidean)
    def get_dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # Pre-calculate One-way distances d_ij
    d = {}
    for j in site_ids:
        for i in village_ids:
            d[i, j] = get_dist(villages[i]['loc'], sites[j])

    # --- 2. Model ---
    model = gp.Model("MCFLPD")

    # Variables
    y = model.addVars(site_ids, vtype=GRB.BINARY, name="OpenSite")
    x = model.addVars(village_ids, site_ids, vtype=GRB.BINARY, name="Serve")

    # Objective: Maximize Population Served
    model.setObjective(gp.quicksum(villages[i]['pop'] * x[i, j] 
                                   for i in village_ids for j in site_ids), GRB.MAXIMIZE)

    # Constraints

    # 1. Facility Budget
    model.addConstr(gp.quicksum(y[j] for j in site_ids) == P_budget, name="Budget")

    # 2. Assignment Validity
    for i in village_ids:
        for j in site_ids:
            model.addConstr(x[i, j] <= y[j], name=f"Link_{i}_{j}")
            
    # 3. Each village served at most once
    for i in village_ids:
        model.addConstr(gp.quicksum(x[i, j] for j in site_ids) <= 1, name=f"OneServe_{i}")

    # 4. Energy/Range Constraint (The Core Constraint)
    # Sum of Round Trips (2 * d_ij) <= R_max for each site
    for j in site_ids:
        model.addConstr(gp.quicksum(2 * d[i, j] * x[i, j] for i in village_ids) <= R_max, name=f"Range_{j}")

    # --- 3. Solve ---
    model.optimize()

    # --- 4. Output ---
    if model.status == GRB.OPTIMAL:
        selected_site = [j for j in site_ids if y[j].X > 0.5]
        served_villages = [i for i in village_ids if sum(x[i, j].X for j in site_ids) > 0.5]
        
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "selected_site": selected_site,
            "served_villages": served_villages,
            "range_used": sum(2 * d[i, j] * x[i,j].X for i in village_ids for j in site_ids)
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_mcflpd())
