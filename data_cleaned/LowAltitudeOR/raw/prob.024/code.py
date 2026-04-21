import gurobipy as gp
from gurobipy import GRB

def solve_uam_planning():
    # --- 1. Data Input ---
    # Nodes: 1=Res, 2=City, 3=Air
    verts = [1, 2, 3]
    periods = ['AM', 'PM']
    
    # Demand n[t][i][j]
    demand = {
        'AM': {(1,2): 20, (3,2): 5},
        'PM': {(2,1): 20, (2,3): 5}
    }
    
    # Parameters
    flight_time = 0.5
    charge_time = 0.5
    min_idle = 1.0
    
    # Costs (Scaled for toy instance)
    c_fleet = 100.0  # Per vehicle
    c_space = 20.0   # Per parking spot
    c_reloc = 5.0    # Per relocation flight
    
    # --- 2. Model ---
    model = gp.Model("UAM_Framework")
    
    # Variables
    # Operational (per t)
    # r[t, i, j]: Relocation flow
    r = model.addVars(periods, verts, verts, vtype=GRB.CONTINUOUS, name="Reloc")
    # w[t, i]: Idle drones
    w = model.addVars(periods, verts, vtype=GRB.CONTINUOUS, name="Idle")
    
    # Planning (Global Max)
    fleet_size = model.addVar(vtype=GRB.CONTINUOUS, name="MaxFleet")
    spaces = model.addVars(verts, vtype=GRB.CONTINUOUS, name="MaxSpace")
    
    # Helper expressions for resource usage per period
    # We add constraints to link per-period usage to global max variables
    
    obj_ops_cost = 0
    
    for t in periods:
        # 1. Calculate Flow Balance & Relocation Cost
        for i in verts:
            # Inflow: Service Arriving + Reloc Arriving
            inflow = 0
            for j in verts:
                dem_in = demand[t].get((j,i), 0)
                inflow += dem_in + r[t, j, i]
            
            # Outflow: Service Departing + Reloc Departing
            outflow = 0
            for j in verts:
                dem_out = demand[t].get((i,j), 0)
                outflow += dem_out + r[t, i, j]
                
            # Constraint: Steady State Flow Conservation
            model.addConstr(inflow == outflow, name=f"Bal_{t}_{i}")
            
            # Constraint: Min Idle
            model.addConstr(w[t, i] >= min_idle, name=f"MinIdle_{t}_{i}")
            
        # 2. Calculate Fleet Needed in t
        # Fleet_t = (Total Flights * flight_time) + (Total Flights * charge_time) + Total Idle
        # Note: Total Flights = Sum(Demand) + Sum(Reloc)
        total_flights_t = 0
        for i in verts:
            for j in verts:
                total_flights_t += demand[t].get((i,j), 0) + r[t, i, j]
        
        needed_fleet_t = (total_flights_t * (flight_time + charge_time)) + gp.quicksum(w[t, i] for i in verts)
        
        # Link to Global Fleet
        model.addConstr(fleet_size >= needed_fleet_t, name=f"SetMaxFleet_{t}")
        
        # 3. Calculate Spaces Needed at i in t
        # Spaces_it = Idle + Charging (Assuming charging happens at destination after arrival)
        # Charging_at_i = (Arrivals_at_i) * charge_time? 
        # Actually in steady state, N_charging = Rate_in * T_charge.
        for i in verts:
            arrivals_rate = 0
            for j in verts:
                arrivals_rate += demand[t].get((j,i), 0) + r[t, j, i]
            
            needed_space_it = w[t, i] + (arrivals_rate * charge_time)
            
            # Link to Global Space
            model.addConstr(spaces[i] >= needed_space_it, name=f"SetMaxSpace_{t}_{i}")
            
        # Accumulate Operational Cost
        obj_ops_cost += gp.quicksum(c_reloc * r[t, i, j] for i in verts for j in verts)

    # --- Objective ---
    obj_inv_fleet = c_fleet * fleet_size
    obj_inv_space = gp.quicksum(c_space * spaces[i] for i in verts)
    
    model.setObjective(obj_inv_fleet + obj_inv_space + obj_ops_cost, GRB.MINIMIZE)
    
    # --- Solve ---
    model.optimize()
    
    # --- Output ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "fleet_size": fleet_size.X,
            "spaces": {i: spaces[i].X for i in verts},
            "relocation_summary": {t: sum(r[t,i,j].X for i in verts for j in verts) for t in periods}
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_uam_planning())