import gurobipy as gp
from gurobipy import GRB
import math

def solve_mcflpd():
    # --- 1. Data Input ---
    R_max = 40.0
    P_budget = 1
    
    sites = {
        1: (0, 0),
        2: (20, 0)
    }
    
    villages = {
        3: {'loc': (0, 4),  'pop': 10},
        4: {'loc': (0, 8),  'pop': 15},
        5: {'loc': (3, 0),  'pop': 20},
        6: {'loc': (18, 0), 'pop': 30},
        7: {'loc': (20, 6), 'pop': 25}
    }
    
    site_ids = list(sites.keys())
    village_ids = list(villages.keys())

    def get_dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    d = {}
    for j in site_ids:
        for i in village_ids:
            d[i, j] = get_dist(villages[i]['loc'], sites[j])

    # --- 2. Model ---
    model = gp.Model("MCFLPD_EnergyAux")

    # Variables
    y = model.addVars(site_ids, vtype=GRB.BINARY, name="OpenSite")
    x = model.addVars(village_ids, site_ids, vtype=GRB.BINARY, name="Serve")
    E = model.addVars(site_ids, vtype=GRB.CONTINUOUS, lb=0, name="Energy")

    # Objective: Maximize Population Served
    model.setObjective(
        gp.quicksum(villages[i]['pop'] * x[i, j] for i in village_ids for j in site_ids),
        GRB.MAXIMIZE
    )

    # Constraints

    # 1. Facility Budget
    model.addConstr(gp.quicksum(y[j] for j in site_ids) == P_budget, name="Budget")

    # 2. Energy Definition (auxiliary variable)
    for j in site_ids:
        model.addConstr(
            E[j] == gp.quicksum(2 * d[i, j] * x[i, j] for i in village_ids),
            name=f"EnergyDef_{j}"
        )

    # 3. Merged Energy-Activation Constraint (replaces individual x[i,j] <= y[j])
    for j in site_ids:
        model.addConstr(E[j] <= R_max * y[j], name=f"EnergyActivation_{j}")

    # 4. Each village served at most once
    for i in village_ids:
        model.addConstr(gp.quicksum(x[i, j] for j in site_ids) <= 1, name=f"OneServe_{i}")

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
            "energy_used": {j: E[j].X for j in site_ids if y[j].X > 0.5}
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_mcflpd())
