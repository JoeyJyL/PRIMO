import gurobipy as gp
from gurobipy import GRB
import json
import math

def solve_crp_td():
    """
    Collaborative Truck-Drone Routing Problem — Reformulated with
    Auxiliary Variables for arc-time decomposition.
    Introduces explicit tau_arc[i,j] variables for truck travel time
    per arc, replacing implicit Big-M timing propagation.
    """

    # ================================================================
    # 1. DATA
    # ================================================================
    n = 8
    depot_s = 0
    depot_e = n + 1

    C = list(range(1, n + 1))
    V = [depot_s] + C + [depot_e]

    coords = {
        0: (0, 0),
        1: (4, 10),
        2: (8, 2),
        3: (12, 0),
        4: (10, 9),
        5: (16, 3),
        6: (20, 1),
        7: (6, -2),
        8: (14, 7),
        9: (0, 0),
    }

    parcel_wt = {1: 0.8, 2: 1.0, 3: 0.5, 4: 0.7, 5: 1.2, 6: 1.5, 7: 0.6, 8: 0.9}

    v_t = 50.0
    v_d = 75.0
    Q_drone = 3.0
    E_drone = 500.0
    alpha_p = 80.0
    beta_p = 30.0

    def dist(i, j):
        return math.sqrt((coords[i][0] - coords[j][0])**2 +
                         (coords[i][1] - coords[j][1])**2)

    d = {(i, j): dist(i, j) for i in V for j in V}

    M = 100.0

    # ================================================================
    # 2. MODEL
    # ================================================================
    model = gp.Model("CRP_TD_AuxVar")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 300)

    # --- Truck routing variables ---
    truck_arcs = [(i, j) for i in [depot_s] + C for j in C + [depot_e] if i != j]
    x = model.addVars(truck_arcs, vtype=GRB.BINARY, name="x")

    # z[i] = 1 if customer i served by drone
    z = model.addVars(C, vtype=GRB.BINARY, name="z")

    # Drone sortie: launch/return nodes
    L = model.addVars([depot_s] + C, vtype=GRB.BINARY, name="L")
    R = model.addVars(C + [depot_e], vtype=GRB.BINARY, name="R")

    # Truck arrival and departure times
    t = model.addVars(V, lb=0, name="t")
    q = model.addVars(V, lb=0, name="q")

    # Subtour elimination
    u = model.addVars(C, lb=1, ub=n, vtype=GRB.CONTINUOUS, name="u")

    # Drone variables
    drone_return_time = model.addVar(lb=0, name="drone_ret")
    drone_dist = model.addVar(lb=0, name="drone_dist")
    drone_time = model.addVar(lb=0, name="drone_time")
    has_sortie = model.addVar(vtype=GRB.BINARY, name="has_sortie")

    # Auxiliary: arc travel time variables
    tau_arc = model.addVars(truck_arcs, lb=0, name="tau")

    # Makespan
    T = model.addVar(lb=0, name="T")

    # --- Objective ---
    model.setObjective(T, GRB.MINIMIZE)
    model.addConstr(T >= t[depot_e], "makespan_truck")
    model.addConstr(T >= drone_return_time, "makespan_drone")

    # ================================================================
    # TRUCK ROUTING CONSTRAINTS
    # ================================================================

    model.addConstr(
        gp.quicksum(x[depot_s, j] for j in C if (depot_s, j) in x) == 1, "depart")
    model.addConstr(
        gp.quicksum(x[i, depot_e] for i in C if (i, depot_e) in x) == 1, "arrive")

    for j in C:
        model.addConstr(
            gp.quicksum(x[i, j] for i in [depot_s] + C if i != j and (i, j) in x) ==
            gp.quicksum(x[j, k] for k in C + [depot_e] if k != j and (j, k) in x),
            f"flow_{j}")

    for i in C:
        model.addConstr(
            gp.quicksum(x[j, i] for j in [depot_s] + C if j != i and (j, i) in x) == 1 - z[i],
            f"truck_visit_{i}")

    for i in C:
        model.addConstr(
            gp.quicksum(x[j, i] for j in [depot_s] + C if j != i and (j, i) in x) >= L[i],
            f"launch_visit_{i}")
        model.addConstr(
            gp.quicksum(x[j, i] for j in [depot_s] + C if j != i and (j, i) in x) >= R[i],
            f"return_visit_{i}")

    # MTZ subtour elimination
    for i in C:
        for j in C:
            if i != j and (i, j) in x:
                model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, f"mtz_{i}_{j}")

    # ================================================================
    # TRUCK TIMING via ARC-TIME AUXILIARY VARIABLES
    # ================================================================

    model.addConstr(t[depot_s] == 0, "t0")
    model.addConstr(q[depot_s] == 0, "q0")

    # Arc time: tau_arc[i,j] = (d[i,j]/v_t) * x[i,j]
    # Linearized: tau_arc[i,j] <= (d[i,j]/v_t) * x[i,j]
    #             tau_arc[i,j] >= 0
    for (i, j) in truck_arcs:
        travel = d[i, j] / v_t
        model.addConstr(tau_arc[i, j] <= travel * x[i, j], f"tau_ub_{i}_{j}")
        model.addConstr(tau_arc[i, j] >= travel * x[i, j] - travel * (1 - x[i, j]),
                        f"tau_lb_{i}_{j}")

    # Timing propagation using auxiliary arc times
    for (i, j) in truck_arcs:
        model.addConstr(
            t[j] >= q[i] + tau_arc[i, j] - M * (1 - x[i, j]),
            f"ttime_{i}_{j}")

    # Departure: truck waits for drone at return node
    for i in C:
        model.addConstr(q[i] >= t[i], f"depart_t_{i}")
        model.addConstr(q[i] >= drone_return_time - M * (1 - R[i]), f"depart_d_{i}")

    # ================================================================
    # DRONE SORTIE CONSTRAINTS
    # ================================================================

    model.addConstr(gp.quicksum(L[i] for i in [depot_s] + C) == has_sortie, "one_launch")
    model.addConstr(gp.quicksum(R[j] for j in C + [depot_e]) == has_sortie, "one_return")

    model.addConstr(gp.quicksum(z[i] for i in C) >= has_sortie, "drone_cust")
    model.addConstr(gp.quicksum(z[i] for i in C) <= n * has_sortie, "no_orphan")
    model.addConstr(gp.quicksum(z[i] for i in C) <= 3, "max_drone_cust")

    # Payload
    model.addConstr(
        gp.quicksum(parcel_wt[i] * z[i] for i in C) <= Q_drone, "drone_payload")

    # Drone flight time
    model.addConstr(drone_time * v_d == drone_dist, "dt_dd")

    # Drone distance bounds
    model.addConstr(
        drone_dist <= gp.quicksum(2 * d[0, i] * z[i] for i in C), "drone_dist_ub")
    model.addConstr(
        drone_dist >= gp.quicksum(0.5 * d[0, i] * z[i] for i in C), "drone_dist_lb")

    # Energy
    model.addConstr(
        (alpha_p + beta_p * Q_drone) * drone_time <= E_drone, "drone_energy")

    # Drone return time
    for i in [depot_s] + C:
        if i == depot_s:
            model.addConstr(
                drone_return_time >= drone_time - M * (1 - L[i]), f"dret_{i}")
        else:
            model.addConstr(
                drone_return_time >= t[i] + drone_time - M * (1 - L[i]), f"dret_{i}")

    # Launch before return in truck tour
    for i in C:
        for j in C:
            if i != j:
                model.addConstr(
                    u[i] <= u[j] - 1 + M * (2 - L[i] - R[j]),
                    f"LbeforeR_{i}_{j}")

    # ================================================================
    # 4. SOLVE
    # ================================================================
    model.optimize()

    # ================================================================
    # 5. OUTPUT
    # ================================================================
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        obj_val = round(model.ObjVal, 2)
        status = "optimal" if model.status == GRB.OPTIMAL else "feasible"
        result = {"status": status, "obj": obj_val}

        print(f"=== Completion Time: {obj_val:.4f} hours ({obj_val*60:.1f} min) ===")

        # Truck route
        truck_route = [depot_s]
        current = depot_s
        while current != depot_e:
            for j in C + [depot_e]:
                if j != current and (current, j) in x and x[current, j].X > 0.5:
                    truck_route.append(j)
                    current = j
                    break
            else:
                break
        print(f"\nTruck Route: {' -> '.join(str(nd) for nd in truck_route)}")

        print(f"\nCustomer Assignments:")
        drone_custs = []
        for c in C:
            mode = "Drone" if z[c].X > 0.5 else "Truck"
            if z[c].X > 0.5:
                drone_custs.append(c)
            print(f"  Customer {c} ({parcel_wt[c]}kg): {mode}")

        if has_sortie.X > 0.5:
            launch = [i for i in [depot_s] + C if L[i].X > 0.5]
            ret = [j for j in C + [depot_e] if R[j].X > 0.5]
            print(f"\nDrone Sortie: Launch from {launch}, Return at {ret}")
            print(f"  Drone customers: {drone_custs}")
            print(f"  Drone distance: {drone_dist.X:.2f} km")
            print(f"  Drone time: {drone_time.X:.4f} h ({drone_time.X*60:.1f} min)")

        print(f"\nTruck arrival at depot: {t[depot_e].X:.4f} h")

        # Print arc times
        print(f"\nActive arc travel times:")
        for (i, j) in truck_arcs:
            if x[i, j].X > 0.5:
                print(f"  Arc ({i}->{j}): tau={tau_arc[i,j].X:.4f} h, "
                      f"dist={d[i,j]:.2f} km")

        return result
    else:
        print(f"Status: {model.status}")
        return {"status": "infeasible"}


if __name__ == "__main__":
    result = solve_crp_td()
    print(f"\n{json.dumps(result, indent=4)}")
