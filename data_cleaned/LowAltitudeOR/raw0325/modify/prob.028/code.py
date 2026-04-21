import gurobipy as gp
from gurobipy import GRB

def solve_spatio_temporal_uam():
    # --- 1. Data Input ---
    verts = [1, 2, 3] # 1=City, 2=North, 3=South
    periods = ['AM', 'Mid', 'PM']
    
    # Demand: t -> (i,j) -> vol
    demand = {
        'AM': {(2,1): 30, (3,1): 20},
        'Mid': {(1,2): 10, (2,1): 10, (1,3): 10, (3,1): 10},
        'PM': {(1,2): 30, (1,3): 20}
    }
    
    # Parameters
    t_fly = 0.5
    t_chg = 0.5  # Increased from 0.3 to 0.5 (battery degradation)
    min_idle = 1.0
    
    # Costs
    c_fleet = 500.0
    c_space = {1: 100.0, 2: 20.0, 3: 20.0} # V1 Expensive
    c_reloc = 5.0
    
    # --- 2. Model ---
    model = gp.Model("UAM_Heterogeneity")
    
    # Variables
    r = model.addVars(periods, verts, verts, vtype=GRB.CONTINUOUS, name="Reloc")
    w = model.addVars(periods, verts, lb=min_idle, vtype=GRB.CONTINUOUS, name="Idle")
    
    fleet = model.addVar(vtype=GRB.CONTINUOUS, name="Fleet")
    spaces = model.addVars(verts, vtype=GRB.CONTINUOUS, name="Spaces")
    
    # --- 3. Constraints ---
    for t in periods:
        # Per Period Logic
        total_flights_t = 0
        
        for i in verts:
            # Flow Balance
            inflow = 0
            outflow = 0
            
            # Calc Inflow (Service + Reloc)
            for j in verts:
                d_in = demand[t].get((j,i), 0)
                inflow += d_in + r[t, j, i]
            
            # Calc Outflow
            for j in verts:
                d_out = demand[t].get((i,j), 0)
                outflow += d_out + r[t, i, j]
                total_flights_t += d_out + r[t, i, j]
                
            model.addConstr(inflow == outflow, name=f"Bal_{t}_{i}")
            
            # Space Requirement at i in t
            req_space = w[t, i] + (inflow * t_chg)
            model.addConstr(spaces[i] >= req_space, name=f"DimSpace_{t}_{i}")
            
        # Fleet Requirement in t
        req_fleet = (total_flights_t * (t_fly + t_chg)) + gp.quicksum(w[t, i] for i in verts)
        model.addConstr(fleet >= req_fleet, name=f"DimFleet_{t}")
        
    # --- 4. Objective ---
    obj_inv = c_fleet * fleet + gp.quicksum(c_space[i] * spaces[i] for i in verts)
    obj_ops = gp.quicksum(c_reloc * r[t, i, j] for t in periods for i in verts for j in verts)
    
    model.setObjective(obj_inv + obj_ops, GRB.MINIMIZE)
    
    # --- 5. Solve ---
    model.optimize()
    
    # --- 6. Output ---
    if model.status == GRB.OPTIMAL:
        reloc_vols = {t: sum(r[t,i,j].X for i in verts for j in verts) for t in periods}
        
        return {
            "status": "optimal",
            "obj": round(model.ObjVal, 2),
            "fleet_size": fleet.X,
            "spaces": {i: spaces[i].X for i in verts},
            "relocation_flights": reloc_vols
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_spatio_temporal_uam())