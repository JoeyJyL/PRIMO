
import gurobipy as gp
from gurobipy import GRB
import math

def solve_charging_station_deployment():
    # --- 1. Data Input ---
    R_max = 5.0
    
    candidates = {
        1: (2, 2), # CS1
        2: (2, 8), # CS2
        3: (8, 2), # CS3
        4: (8, 8)  # CS4
    }
    
    targets = {
        1: (1, 1), # T1
        2: (1, 9), # T2
        3: (9, 1), # T3
        4: (9, 9), # T4
        5: (5, 5), # T5 (Center)
        6: (2, 5)  # T6 (Left Edge)
    }
    
    cand_ids = list(candidates.keys())
    target_ids = list(targets.keys())

    # Distance Helper
    def get_dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # Pre-calculate Coverage Matrix a[i, j]
    # 1 if candidate j covers target i
    a = {}
    for i in target_ids:
        for j in cand_ids:
            dist = get_dist(targets[i], candidates[j])
            if dist <= R_max:
                a[i, j] = 1
            else:
                a[i, j] = 0

    # --- 2. Model ---
    model = gp.Model("UAV_Station_Coverage")
    
    # Variables: Open station j?
    x = model.addVars(cand_ids, vtype=GRB.BINARY, name="x")

    # Objective: Minimize number of stations
    model.setObjective(gp.quicksum(x[j] for j in cand_ids), GRB.MINIMIZE)

    # Constraints
    
    # 1. Coverage: Each target i must be covered by at least one open station
    for i in target_ids:
        model.addConstr(gp.quicksum(a[i, j] * x[j] for j in cand_ids) >= 1, name=f"Cover_{i}")

    # --- 3. Solve ---
    model.optimize()

    # --- 4. Output ---
    if model.status == GRB.OPTIMAL:
        open_stations = [j for j in cand_ids if x[j].X > 0.5]
        
        # Determine who covers whom (for reporting)
        coverage_map = {i: [] for i in target_ids}
        for i in target_ids:
            for j in open_stations:
                if a[i, j] == 1:
                    coverage_map[i].append(j)
                    
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "open_stations": open_stations,
            "coverage_details": coverage_map
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_charging_station_deployment())
