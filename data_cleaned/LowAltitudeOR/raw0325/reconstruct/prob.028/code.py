import gurobipy as gp
from gurobipy import GRB


def solve_spatio_temporal_uam():
    # --- 1. Data Input ---
    verts = [1, 2, 3]  # 1=City, 2=North, 3=South
    periods = ['AM', 'Mid', 'PM']

    demand = {
        'AM':  {(2, 1): 30, (3, 1): 20},
        'Mid': {(1, 2): 10, (2, 1): 10, (1, 3): 10, (3, 1): 10},
        'PM':  {(1, 2): 30, (1, 3): 20}
    }

    t_fly = 0.5
    t_chg = 0.3
    min_idle = 1.0

    c_fleet = 500.0
    c_space = {1: 100.0, 2: 20.0, 3: 20.0}
    c_reloc = 5.0

    # --- 2. Model ---
    model = gp.Model("UAM_AuxiliaryFlowDecomposition")
    model.setParam('OutputFlag', 0)

    # Decision variables
    r = model.addVars(periods, verts, verts,
                      vtype=GRB.CONTINUOUS, name="Reloc")
    w = model.addVars(periods, verts,
                      lb=min_idle, vtype=GRB.CONTINUOUS, name="Idle")
    fleet = model.addVar(vtype=GRB.CONTINUOUS, name="Fleet")
    spaces = model.addVars(verts, vtype=GRB.CONTINUOUS, name="Spaces")

    # Auxiliary variables
    A = model.addVars(periods, verts,
                      vtype=GRB.CONTINUOUS, name="Arrival")
    O = model.addVars(periods, verts,
                      vtype=GRB.CONTINUOUS, name="Depart")
    Q = model.addVars(periods, verts,
                      vtype=GRB.CONTINUOUS, name="Charging")
    F = model.addVars(periods,
                      vtype=GRB.CONTINUOUS, name="TotalFlights")

    # --- 3. Constraints ---
    for t in periods:
        for i in verts:
            # Auxiliary definition: Arrival rate
            inflow_expr = gp.quicksum(
                demand[t].get((j, i), 0) + r[t, j, i] for j in verts
            )
            model.addConstr(A[t, i] == inflow_expr,
                            name=f"DefA_{t}_{i}")

            # Auxiliary definition: Departure rate
            outflow_expr = gp.quicksum(
                demand[t].get((i, j), 0) + r[t, i, j] for j in verts
            )
            model.addConstr(O[t, i] == outflow_expr,
                            name=f"DefO_{t}_{i}")

            # Auxiliary definition: Charging drones
            model.addConstr(Q[t, i] == t_chg * A[t, i],
                            name=f"DefQ_{t}_{i}")

            # Flow balance: A == O
            model.addConstr(A[t, i] == O[t, i],
                            name=f"Bal_{t}_{i}")

            # Space capacity
            model.addConstr(spaces[i] >= w[t, i] + Q[t, i],
                            name=f"DimSpace_{t}_{i}")

        # Auxiliary definition: Total flight volume
        model.addConstr(F[t] == gp.quicksum(O[t, i] for i in verts),
                        name=f"DefF_{t}")

        # Fleet dimensioning
        model.addConstr(
            fleet >= F[t] * (t_fly + t_chg)
                     + gp.quicksum(w[t, i] for i in verts),
            name=f"DimFleet_{t}"
        )

    # --- 4. Objective ---
    obj_inv = c_fleet * fleet + gp.quicksum(
        c_space[i] * spaces[i] for i in verts
    )
    obj_ops = gp.quicksum(
        c_reloc * r[t, i, j]
        for t in periods for i in verts for j in verts
    )
    model.setObjective(obj_inv + obj_ops, GRB.MINIMIZE)

    # --- 5. Solve ---
    model.optimize()

    # --- 6. Output ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "fleet_size": fleet.X,
            "spaces": {i: spaces[i].X for i in verts},
        }
    else:
        return {"status": "infeasible"}


if __name__ == "__main__":
    print(solve_spatio_temporal_uam())
