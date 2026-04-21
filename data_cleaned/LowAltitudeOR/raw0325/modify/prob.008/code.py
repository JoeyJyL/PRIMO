
import gurobipy as gp
from gurobipy import GRB

def solve_uam_network():
    # --- Data ---
    # Vertiports: id -> (name, cost)
    vertiports = {1: ("CBD", 10), 2: ("PEK", 15), 3: ("PKX", 12)}
    budget = 25

    # RS Routes: (endpoint1, endpoint2, demand)
    rs_routes = {"A": (1, 2, 100), "B": (1, 3, 110)}

    # ODM Zones: (vertiport_needed, demand)
    odm_zones = {1: (1, 20), 2: (2, 30), 3: (3, 30)}

    # --- Model ---
    model = gp.Model("UAM_Network")

    # Variables
    x = model.addVars(vertiports.keys(), vtype=GRB.BINARY, name="build")
    y = model.addVars(rs_routes.keys(), vtype=GRB.BINARY, name="rs_route")
    w = model.addVars(odm_zones.keys(), vtype=GRB.BINARY, name="odm_zone")

    # Objective: Maximize total demand served
    model.setObjective(
        gp.quicksum(rs_routes[r][2] * y[r] for r in rs_routes) +
        gp.quicksum(odm_zones[z][1] * w[z] for z in odm_zones),
        GRB.MAXIMIZE
    )

    # --- Constraints ---

    # 1. Budget
    model.addConstr(
        gp.quicksum(vertiports[i][1] * x[i] for i in vertiports) <= budget,
        name="budget"
    )

    # 2. RS route activation (AND logic)
    for r in rs_routes:
        ep1, ep2, _ = rs_routes[r]
        model.addConstr(y[r] <= x[ep1], name=f"rs_{r}_ep1")
        model.addConstr(y[r] <= x[ep2], name=f"rs_{r}_ep2")

    # 3. ODM zone service
    for z in odm_zones:
        vp, _ = odm_zones[z]
        model.addConstr(w[z] <= x[vp], name=f"odm_{z}")

    # 4. Airspace mutual exclusion: V1 and V3 cannot both be built
    model.addConstr(x[1] + x[3] <= 1, name="airspace_exclusion")

    # --- Solve ---
    model.optimize()

    if model.status == GRB.OPTIMAL:
        built = [i for i in vertiports if x[i].X > 0.5]
        active_rs = [r for r in rs_routes if y[r].X > 0.5]
        active_odm = [z for z in odm_zones if w[z].X > 0.5]

        return {
            "status": "optimal",
            "obj": round(model.ObjVal, 2),
            "built_vertiports": built,
            "active_rs_routes": active_rs,
            "active_odm_zones": active_odm
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_uam_network())
