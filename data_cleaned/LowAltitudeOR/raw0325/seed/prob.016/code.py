
import gurobipy as gp
from gurobipy import GRB
import math

def solve_choice_based_location():
    # --- 1. Data Input ---
    # Candidates
    skyports = [1, 2, 3, 4] # S1, S2, S3, S4
    max_open = 2
    
    # Utility Coeffs
    beta_time = -0.5
    beta_cost = -0.1
    
    # Routes: id -> {o, d, demand, ground_t, ground_c, air_t, air_c, acc}
    routes = [
        {'id': 1, 'o': 1, 'd': 3, 'dem': 2000, 'gt': 60, 'gc': 10, 'at': 15, 'ac': 50, 'acc': 0},  # S1->S3
        {'id': 2, 'o': 2, 'd': 3, 'dem': 1500, 'gt': 50, 'gc': 8,  'at': 12, 'ac': 45, 'acc': 0},  # S2->S3
        {'id': 3, 'o': 1, 'd': 4, 'dem': 500,  'gt': 40, 'gc': 12, 'at': 20, 'ac': 40, 'acc': 2}   # S1->S4
    ]
    
    # --- 2. Pre-calculate Probabilities (Logit) ---
    route_uam_demand = {}
    
    for r in routes:
        # Calculate Utilities
        u_ground = beta_time * r['gt'] + beta_cost * r['gc']
        u_air    = beta_time * r['at'] + beta_cost * r['ac'] + r['acc']
        
        # Calculate Prob UAM
        # P = exp(Ua) / (exp(Ua) + exp(Ug))
        exp_g = math.exp(u_ground)
        exp_a = math.exp(u_air)
        prob_uam = exp_a / (exp_a + exp_g)
        
        expected_pax = r['dem'] * prob_uam
        route_uam_demand[r['id']] = expected_pax
        
        print(f"Route {r['id']} (S{r['o']}->S{r['d']}): Ug={u_ground:.2f}, Ua={u_air:.2f}, Prob={prob_uam:.2%}, ExpPax={expected_pax:.2f}")

    # --- 3. Model ---
    model = gp.Model("Choice_Skyport")
    
    # Variables
    x = model.addVars(skyports, vtype=GRB.BINARY, name="Open")
    y = model.addVars([r['id'] for r in routes], vtype=GRB.BINARY, name="Active")
    
    # Objective: Maximize Expected UAM Pax
    model.setObjective(gp.quicksum(route_uam_demand[r['id']] * y[r['id']] for r in routes), GRB.MAXIMIZE)
    
    # Constraints
    
    # 1. Budget
    model.addConstr(gp.quicksum(x[k] for k in skyports) == max_open, name="Budget")
    
    # 2. Topology
    for r in routes:
        rid = r['id']
        origin = r['o']
        dest = r['d']
        model.addConstr(y[rid] <= x[origin], name=f"Link_{rid}_O")
        model.addConstr(y[rid] <= x[dest],   name=f"Link_{rid}_D")
        
    # --- 4. Solve ---
    model.optimize()
    
    # --- 5. Output ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "open_skyports": [k for k in skyports if x[k].X > 0.5],
            "active_routes": [r['id'] for r in routes if y[r['id']].X > 0.5],
            "market_capture": {r['id']: route_uam_demand[r['id']] for r in routes}
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_choice_based_location())
