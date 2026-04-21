import gurobipy as gp
from gurobipy import GRB

def solve_stochastic_sizing():
    # --- 1. Data Input ---
    scenarios = {
        'Normal': {'prob': 0.6, 'drones': 2},
        'Peak':   {'prob': 0.4, 'drones': 5}
    }
    
    cost_fixed = 200.0
    cost_pile = 50.0
    cost_wait_hr = 100.0  # Increased from 60 to 100 (higher urgency penalty)
    time_charge = 1.0
    
    max_cap = 6 # Sufficient upper bound
    
    # --- 2. Helper: Pre-calculate Wait Times ---
    # Returns total wait hours for n drones with c piles
    def calc_wait_time(n, c):
        if c == 0: return 1e6 # Penalty for 0 capacity if drones exist
        total_wait = 0
        for i in range(n):
            batch_index = i // c
            wait_time = batch_index * time_charge
            total_wait += wait_time
        return total_wait

    # --- 3. Model ---
    model = gp.Model("Stochastic_Charging")
    
    # Stage 1: Investment
    y = model.addVar(vtype=GRB.BINARY, name="Open")
    k = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_cap, name="Capacity")
    
    # Auxiliary for selecting capacity level (to linearize wait lookup)
    z = model.addVars(range(max_cap + 1), vtype=GRB.BINARY, name="CapSelect")
    
    # Stage 2: Operation Costs per Scenario
    wait_cost = model.addVars(scenarios.keys(), vtype=GRB.CONTINUOUS, name="WaitCost")
    
    # Constraints
    
    # Link k and z
    model.addConstr(gp.quicksum(c * z[c] for c in range(max_cap + 1)) == k, name="Link_k_z")
    model.addConstr(gp.quicksum(z[c] for c in range(max_cap + 1)) == 1, name="SelectOneCap")
    
    # Link y and k (If open, at least 1 pile; if closed, 0 piles)
    model.addConstr(k <= max_cap * y, name="ForceClosed")
    model.addConstr(k >= y, name="MinCapIfOpen")
    
    # Calculate Wait Costs via Lookup
    for s_name, s_data in scenarios.items():
        n = s_data['drones']
        # Expression: sum( z[c] * precalculated_cost(n, c) )
        expr = gp.quicksum(
            cost_wait_hr * calc_wait_time(n, c) * z[c] 
            for c in range(max_cap + 1)
        )
        model.addConstr(wait_cost[s_name] == expr, name=f"SetCost_{s_name}")
        
    # Objective: Invest + Expected Wait
    obj_invest = cost_fixed * y + cost_pile * k
    obj_exp_wait = gp.quicksum(scenarios[s]['prob'] * wait_cost[s] for s in scenarios)
    
    model.setObjective(obj_invest + obj_exp_wait, GRB.MINIMIZE)
    
    # --- 4. Solve ---
    model.optimize()
    
    # --- 5. Output ---
    if model.status == GRB.OPTIMAL:
        opt_k = int(k.X)
        
        # Recalculate details for reporting
        wait_times = {}
        for s_name, s_data in scenarios.items():
            wait_times[s_name] = calc_wait_time(s_data['drones'], opt_k)
            
        return {
            "status": "optimal",
            "obj": round(model.ObjVal, 2),
            "optimal_capacity": opt_k,
            "station_open": True if y.X > 0.5 else False,
            "details": {
                "investment_cost": cost_fixed + cost_pile * opt_k,
                "expected_wait_cost": round(obj_exp_wait.getValue(), 2),
                "wait_hours_normal": wait_times['Normal'],
                "wait_hours_peak": wait_times['Peak']
            }
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_stochastic_sizing())