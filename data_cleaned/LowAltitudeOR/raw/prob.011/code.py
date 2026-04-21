
import gurobipy as gp
from gurobipy import GRB
import math

def solve_robust_uam_network():
    # --- 1. Data Input ---
    # Coefficients for Utility: U = beta_t * Time + beta_c * Cost
    beta_t = -0.1
    beta_c = -0.05
    
    # Vertiports: Fixed Cost
    vertiports = {
        1: 1000, # Downtown
        2: 800,  # Tech Park
        3: 1200  # Airport
    }
    
    # Routes: (u, v) -> Data
    # Dem_nom, Dem_unc, G_time, G_cost, A_time, A_ticket, A_opcost
    routes = {
        (1, 2): {'dn': 500, 'du': 100, 'gt': 45, 'gc': 30, 'at': 10, 'ap': 80, 'ac': 40},
        (1, 3): {'dn': 800, 'du': 200, 'gt': 60, 'gc': 50, 'at': 15, 'ap': 100, 'ac': 50},
        (2, 3): {'dn': 300, 'du': 50,  'gt': 40, 'gc': 40, 'at': 12, 'ap': 90, 'ac': 45}
    }
    
    # --- 2. Pre-processing (Mode Choice & Robustness) ---
    route_params = {}
    for (u, v), d in routes.items():
        # Robust Demand (Box Uncertainty: Worst Case)
        dem_robust = d['dn'] - d['du']
        
        # Utilities
        u_g = beta_t * d['gt'] + beta_c * d['gc']
        u_a = beta_t * d['at'] + beta_c * d['ap'] # Utility uses Ticket Price
        
        # Logit Probability
        # P = exp(Ua) / (exp(Ua) + exp(Ug))
        exp_g = math.exp(u_g)
        exp_a = math.exp(u_a)
        prob_a = exp_a / (exp_a + exp_g)
        
        # Net Margin per Passenger (Revenue - OpCost)
        margin = d['ap'] - d['ac']
        
        # Expected Revenue if Route is Active
        revenue = dem_robust * prob_a * margin
        
        route_params[(u, v)] = revenue

    # --- 3. Optimization Model ---
    model = gp.Model("Robust_UAM")
    
    # Variables
    y = model.addVars(vertiports.keys(), vtype=GRB.BINARY, name="Open")
    z = model.addVars(routes.keys(), vtype=GRB.BINARY, name="Active")
    
    # Objective: Total Revenue - Total Fixed Cost
    obj_rev = gp.quicksum(route_params[k] * z[k] for k in routes)
    obj_cost = gp.quicksum(vertiports[i] * y[i] for i in vertiports)
    
    model.setObjective(obj_rev - obj_cost, GRB.MAXIMIZE)
    
    # Constraints
    # Route Logic: z_uv <= y_u, z_uv <= y_v
    for (u, v) in routes:
        model.addConstr(z[u, v] <= y[u], name=f"Link_{u}_{v}_src")
        model.addConstr(z[u, v] <= y[v], name=f"Link_{u}_{v}_dst")
        
    # --- 4. Solve ---
    model.optimize()
    
    # --- 5. Output ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "open_vertiports": [i for i in vertiports if y[i].X > 0.5],
            "active_routes": [str(k) for k in routes if z[k].X > 0.5]
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_robust_uam_network())
