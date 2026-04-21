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
    cost_wait_hr = 60.0
    time_charge = 1.0

    max_cap = 6
    max_drones = max(s['drones'] for s in scenarios.values())
    max_batches = max_drones  # Worst case: k=1, need max_drones batches (0..max_drones-1)

    # --- 2. Model ---
    model = gp.Model("Stochastic_Charging_BatchAux")
    model.setParam('OutputFlag', 0)

    # Stage 1: Investment
    y = model.addVar(vtype=GRB.BINARY, name="Open")
    k = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_cap, name="Capacity")

    # Auxiliary variables: g[s,b] = number of drones served in batch b for scenario s
    g = {}
    for s_name in scenarios:
        for b in range(max_batches):
            g[s_name, b] = model.addVar(lb=0, vtype=GRB.CONTINUOUS,
                                        name=f"g_{s_name}_{b}")

    # Stage 2: Wait cost per scenario
    wait_cost = model.addVars(scenarios.keys(), vtype=GRB.CONTINUOUS, lb=0,
                              name="WaitCost")

    # --- Constraints ---

    # Link y and k
    model.addConstr(k <= max_cap * y, name="ForceClosed")
    model.addConstr(k >= y, name="MinCapIfOpen")

    # Batch constraints for each scenario
    for s_name, s_data in scenarios.items():
        n_drones = s_data['drones']

        # Total drones across all batches = N_s
        model.addConstr(
            gp.quicksum(g[s_name, b] for b in range(max_batches)) == n_drones,
            name=f"TotalDrones_{s_name}"
        )

        # Each batch serves at most k drones
        for b in range(max_batches):
            model.addConstr(g[s_name, b] <= k, name=f"BatchCap_{s_name}_{b}")

        # Wait cost = C_wait * sum_b (b * time_charge * g[s,b])
        model.addConstr(
            wait_cost[s_name] == cost_wait_hr * gp.quicksum(
                b * time_charge * g[s_name, b] for b in range(max_batches)
            ),
            name=f"SetCost_{s_name}"
        )

    # --- Objective ---
    obj_invest = cost_fixed * y + cost_pile * k
    obj_exp_wait = gp.quicksum(scenarios[s]['prob'] * wait_cost[s] for s in scenarios)
    model.setObjective(obj_invest + obj_exp_wait, GRB.MINIMIZE)

    # --- Solve ---
    model.optimize()

    # --- Output ---
    if model.status == GRB.OPTIMAL:
        opt_k = int(k.X)

        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "optimal_capacity": opt_k,
            "station_open": True if y.X > 0.5 else False,
            "details": {
                "investment_cost": cost_fixed + cost_pile * opt_k,
                "expected_wait_cost": obj_exp_wait.getValue(),
                "batch_details": {
                    s_name: {b: round(g[s_name, b].X, 2)
                             for b in range(max_batches) if g[s_name, b].X > 0.01}
                    for s_name in scenarios
                }
            }
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_stochastic_sizing())
