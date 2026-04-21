import gurobipy as gp
from gurobipy import GRB

def solve_uam_planning():
    # --- 1. Data Input ---
    verts = [1, 2, 3]
    periods = ['AM', 'PM']

    # Demand n[t][i][j]
    demand = {
        'AM': {(1, 2): 20, (3, 2): 5},
        'PM': {(2, 1): 20, (2, 3): 5}
    }

    # Parameters
    flight_time = 0.5
    charge_time = 0.5
    min_idle = 1.0

    # Costs
    c_fleet = 100.0
    c_space = 20.0
    c_reloc = 5.0

    # --- 2. Model ---
    model = gp.Model("UAM_AuxiliaryVariables")
    model.setParam('OutputFlag', 0)

    # Decision Variables
    r = model.addVars(periods, verts, verts, vtype=GRB.CONTINUOUS, name="Reloc")
    w = model.addVars(periods, verts, vtype=GRB.CONTINUOUS, name="Idle")
    fleet_size = model.addVar(vtype=GRB.CONTINUOUS, name="MaxFleet")
    spaces = model.addVars(verts, vtype=GRB.CONTINUOUS, name="MaxSpace")

    # Auxiliary Variables: in-flight drones and charging drones
    p = model.addVars(periods, verts, verts, vtype=GRB.CONTINUOUS, name="InFlight")
    q = model.addVars(periods, verts, vtype=GRB.CONTINUOUS, name="Charging")

    # --- Objective ---
    obj_inv_fleet = c_fleet * fleet_size
    obj_inv_space = gp.quicksum(c_space * spaces[i] for i in verts)
    obj_ops_cost = gp.quicksum(c_reloc * r[t, i, j]
                               for t in periods for i in verts for j in verts)
    model.setObjective(obj_inv_fleet + obj_inv_space + obj_ops_cost, GRB.MINIMIZE)

    # --- Constraints ---
    for t in periods:
        # 1. Flow Conservation
        for i in verts:
            inflow = gp.quicksum(demand[t].get((j, i), 0) + r[t, j, i]
                                 for j in verts)
            outflow = gp.quicksum(demand[t].get((i, j), 0) + r[t, i, j]
                                  for j in verts)
            model.addConstr(inflow == outflow, name=f"Bal_{t}_{i}")

        # 2. In-flight auxiliary definition: p[t,i,j] = (demand + reloc) * flight_time
        for i in verts:
            for j in verts:
                dem = demand[t].get((i, j), 0)
                model.addConstr(p[t, i, j] == (dem + r[t, i, j]) * flight_time,
                                name=f"InFlight_{t}_{i}_{j}")

        # 3. Charging auxiliary definition: q[t,i] = sum_j (arrivals at i) * charge_time
        for i in verts:
            arrivals = gp.quicksum(demand[t].get((j, i), 0) + r[t, j, i]
                                   for j in verts)
            model.addConstr(q[t, i] == arrivals * charge_time,
                            name=f"Charging_{t}_{i}")

        # 4. Fleet sizing (decomposed with auxiliary variables)
        total_inflight = gp.quicksum(p[t, i, j] for i in verts for j in verts)
        total_charging = gp.quicksum(q[t, i] for i in verts)
        total_idle = gp.quicksum(w[t, i] for i in verts)
        model.addConstr(fleet_size >= total_inflight + total_charging + total_idle,
                        name=f"SetMaxFleet_{t}")

        # 5. Space capacity (using charging auxiliary)
        for i in verts:
            model.addConstr(spaces[i] >= w[t, i] + q[t, i],
                            name=f"SetMaxSpace_{t}_{i}")

        # 6. Minimum idle
        for i in verts:
            model.addConstr(w[t, i] >= min_idle, name=f"MinIdle_{t}_{i}")

    # --- Solve ---
    model.optimize()

    # --- Output ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "fleet_size": fleet_size.X,
            "spaces": {i: spaces[i].X for i in verts},
            "relocation_summary": {t: sum(r[t, i, j].X for i in verts for j in verts)
                                   for t in periods},
            "inflight_total": {t: sum(p[t, i, j].X for i in verts for j in verts)
                               for t in periods},
            "charging_total": {t: sum(q[t, i].X for i in verts)
                               for t in periods}
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_uam_planning())
