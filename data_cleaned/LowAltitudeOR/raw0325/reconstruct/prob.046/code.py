"""
Robust Vehicle Routing Problem with Drones (VRPD) — Indicator Constraint Reformulation
Replaces Big-M time propagation with Gurobi indicator constraints.
Based on Sun, Wu & Zhang (2026).
"""

import gurobipy as gp
from gurobipy import GRB
import math
import json

def solve_robust_vrpd():
    # --- Parameter Setup ---
    v_truck = 30.0
    v_drone = 60.0
    L_K = 100.0
    L_D = 10.0
    L_R = 1
    T_D = 30.0 / 60  # hours
    B   = 1000.0
    t_bar = 10.0      # Big-M retained only for sync constraints

    eta_K = 0.3
    eta_D = 0.3

    coords = {
        0: (0.0, 0.0),
        1: (3.0, 0.0),
        2: (6.0, 0.0),
        3: (4.0, 3.0),
        4: (9.0, 0.0),
    }

    tasks = {
        (1, 1): {'dw': 5.0, 'dt': 0.0,    'p': 2},
        (2, 1): {'dw': 3.0, 'dt': 0.0,    'p': 1},
        (3, 2): {'dw': 0.0, 'dt': 5.0/60, 'p': 3},
    }

    o_s, o_e = 0, 4
    V_k  = [0, 1, 2, 4]
    V_d  = [0, 1, 2, 3, 4]
    V_cust_k = [1, 2]
    V_cust_d_only = [3]
    K    = [1]
    D    = [1]
    M    = [1, 2]

    def dist(a, b):
        return math.sqrt((coords[a][0]-coords[b][0])**2 + (coords[a][1]-coords[b][1])**2)

    tK = {(i,j): dist(i,j)/v_truck for i in V_k for j in V_k if i != j}
    tD = {(i,j): dist(i,j)/v_drone for i in V_d for j in V_d if i != j}
    hat_tK = {(i,j): eta_K * tK[i,j] for (i,j) in tK}
    hat_tD = {(i,j): eta_D * tD[i,j] for (i,j) in tD}

    print("=" * 55)
    print("Solving Deterministic VRPD — Indicator Constraint Formulation")
    print("=" * 55)

    model = gp.Model("RobustVRPD_Indicator")
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', 300)

    # Decision variables
    x = model.addVars([(i,j,k) for i in V_k for j in V_k if i!=j for k in K],
                      vtype=GRB.BINARY, name="x")
    y = model.addVars([(i,j,d) for i in V_d for j in V_d if i!=j for d in D],
                      vtype=GRB.BINARY, name="y")
    c = model.addVars([(i,d,m) for i in V_d for d in D for m in M],
                      vtype=GRB.BINARY, name="c")
    g = model.addVars([(i,m) for i in V_d for m in M if (i,m) in tasks],
                      vtype=GRB.BINARY, name="g")
    taK = model.addVars([(i,k) for i in V_k for k in K], lb=0.0, name="taK")
    tdK = model.addVars([(i,k) for i in V_k for k in K], lb=0.0, name="tdK")
    taD = model.addVars([(i,d) for i in V_d for d in D], lb=0.0, name="taD")
    tdD = model.addVars([(i,d) for i in V_d for d in D], lb=0.0, name="tdD")
    t = model.addVars([(i,m) for (i,m) in tasks], lb=0.0, name="t")
    wD = model.addVars([(i,d) for i in V_d for d in D], lb=0.0, name="wD")
    vD = model.addVars([(i,d) for i in V_d for d in D], lb=0.0, name="vD")
    wK = model.addVars([(i,k) for i in V_k for k in K], lb=0.0, name="wK")

    # Objective
    obj = (gp.quicksum(tasks[i,m]['p'] * t[i,m] for (i,m) in tasks) +
           B * gp.quicksum(tasks[i,m]['p'] * g[i,m] for (i,m) in tasks))
    model.setObjective(obj, GRB.MINIMIZE)

    # (C1) Truck flow
    for k in K:
        model.addConstr(gp.quicksum(x[o_s,j,k] for j in V_k if j != o_s) == 1,
                        name=f"truck_depart_{k}")
        model.addConstr(gp.quicksum(x[i,o_e,k] for i in V_k if i != o_e) == 1,
                        name=f"truck_return_{k}")
    for j in V_cust_k:
        for k in K:
            model.addConstr(gp.quicksum(x[i,j,k] for i in V_k if i != j) ==
                            gp.quicksum(x[j,i,k] for i in V_k if i != j),
                            name=f"truck_flow_{j}_{k}")

    # (C2) Drone flow
    for d in D:
        model.addConstr(gp.quicksum(y[o_s,j,d] for j in V_d if j != o_s) == 1,
                        name=f"drone_depart_{d}")
        model.addConstr(gp.quicksum(y[i,o_e,d] for i in V_d if i != o_e) == 1,
                        name=f"drone_return_{d}")
    for j in [n for n in V_d if n not in [o_s, o_e]]:
        for d in D:
            model.addConstr(gp.quicksum(y[i,j,d] for i in V_d if i != j) ==
                            gp.quicksum(y[j,i,d] for i in V_d if i != j),
                            name=f"drone_flow_{j}_{d}")

    # (C3) Task coverage
    for j in V_cust_k:
        for m in M:
            if (j,m) in tasks:
                model.addConstr(gp.quicksum(x[i,j,k] for i in V_k if i != j for k in K) +
                                gp.quicksum(c[j,d,m] for d in D) + g[j,m] == 1,
                                name=f"task_truck_{j}_{m}")
    for i in V_cust_d_only:
        for m in M:
            if (i,m) in tasks:
                model.addConstr(gp.quicksum(c[i,d,m] for d in D) + g[i,m] == 1,
                                name=f"task_drone_{i}_{m}")

    # (C4) Drone capacity
    for i in V_d:
        for d in D:
            model.addConstr(wD[i,d] <= L_D * gp.quicksum(y[j,i,d] for j in V_d if j != i),
                            name=f"drone_cap_ub_{i}_{d}")
            model.addConstr(wD[i,d] >= gp.quicksum(tasks.get((i,m),{'dw':0})['dw'] * c[i,d,m]
                                                     for m in M),
                            name=f"drone_cap_lb_{i}_{d}")

    # (C5) Drone flight duration (robust, upper bound)
    for j in V_d:
        for d in D:
            truck_visit_j = (gp.quicksum(x[i,j,k] for i in V_k if i != j for k in K)
                             if j in V_k else 0)
            model.addConstr(
                vD[j,d] >= gp.quicksum((tD[i,j] + hat_tD[i,j]) * y[i,j,d] for i in V_d if i != j) +
                            gp.quicksum(tasks.get((j,m),{'dt':0})['dt'] * c[j,d,m] for m in M) -
                            t_bar * truck_visit_j,
                name=f"drone_flight_lo_{j}_{d}"
            )
            model.addConstr(vD[j,d] <= T_D * gp.quicksum(y[i,j,d] for i in V_d if i != j),
                            name=f"drone_flight_ub_{j}_{d}")

    # (C6) Drone flight duration propagation — INDICATOR CONSTRAINT (replaces Big-M)
    for i in V_d:
        for j in V_d:
            if i != j:
                for d in D:
                    truck_depart_i = (gp.quicksum(x[i,l,k] for l in V_k if l != i for k in K)
                                      if i in V_k else 0)
                    # Indicator: y[i,j,d] == 1 => vD[j,d] >= vD[i,d] + travel + service - reset
                    model.addGenConstrIndicator(
                        y[i,j,d], True,
                        vD[j,d] - vD[i,d]
                        - gp.quicksum(tasks.get((j,m),{'dt':0})['dt'] * c[j,d,m] for m in M)
                        + t_bar * truck_depart_i,
                        GRB.GREATER_EQUAL,
                        tD[i,j] + hat_tD[i,j],
                        name=f"drone_flight_prop_{i}_{j}_{d}"
                    )

    # (C7) Truck time propagation — INDICATOR CONSTRAINT (replaces Big-M)
    for i in V_k:
        for j in V_k:
            if i != j:
                for k in K:
                    model.addGenConstrIndicator(
                        x[i,j,k], True,
                        taK[j,k] - tdK[i,k],
                        GRB.GREATER_EQUAL,
                        tK[i,j],
                        name=f"truck_arrive_{i}_{j}_{k}"
                    )

    # (C8) Drone time propagation — INDICATOR CONSTRAINT (replaces Big-M)
    for i in V_d:
        for j in V_d:
            if i != j:
                for d in D:
                    model.addGenConstrIndicator(
                        y[i,j,d], True,
                        taD[j,d] - tdD[i,d],
                        GRB.GREATER_EQUAL,
                        tD[i,j],
                        name=f"drone_arrive_{i}_{j}_{d}"
                    )

    # (C9) Truck departure >= arrival
    for i in V_k:
        for k in K:
            model.addConstr(tdK[i,k] >= taK[i,k], name=f"truck_dept_ge_arr_{i}_{k}")
            model.addConstr(tdK[i,k] <= t_bar * gp.quicksum(x[l,i,k] for l in V_k if l != i),
                            name=f"truck_dept_active_{i}_{k}")

    # (C10) Drone departure after task service
    for j in V_d:
        for d in D:
            model.addConstr(
                tdD[j,d] >= taD[j,d] + gp.quicksum(tasks.get((j,m),{'dt':0})['dt'] * c[j,d,m]
                                                     for m in M),
                name=f"drone_dept_{j}_{d}"
            )

    # (C11) Time synchronization (Big-M retained — involves conjunctions of binaries)
    for i in V_cust_k:
        for d in D:
            sum_yd_out = gp.quicksum(y[i,j,d] for j in V_d if j != i)
            sum_xk_in  = gp.quicksum(x[l,i,k] for l in V_k if l != i for k in K)
            model.addConstr(
                tdD[i,d] >= gp.quicksum(taK[i,k] for k in K) + t_bar * (sum_yd_out + sum_xk_in - 2),
                name=f"sync_dept_lo_{i}_{d}"
            )
            model.addConstr(
                tdD[i,d] <= gp.quicksum(tdK[i,k] for k in K) - t_bar * (sum_yd_out + sum_xk_in - 2),
                name=f"sync_dept_hi_{i}_{d}"
            )
            sum_yd_in = gp.quicksum(y[j,i,d] for j in V_d if j != i)
            model.addConstr(
                taD[i,d] >= gp.quicksum(taK[i,k] for k in K) + t_bar * (sum_yd_in + sum_xk_in - 2),
                name=f"sync_arr_lo_{i}_{d}"
            )
            model.addConstr(
                taD[i,d] <= gp.quicksum(tdK[i,k] for k in K) - t_bar * (sum_yd_in + sum_xk_in - 2),
                name=f"sync_arr_hi_{i}_{d}"
            )

    # (C12) Service start time — INDICATOR CONSTRAINT (replaces Big-M)
    for (i,m) in tasks:
        if i in V_cust_k:
            model.addConstr(t[i,m] >= gp.quicksum(taK[i,k] for k in K),
                            name=f"svc_truck_{i}_{m}")
        for d in D:
            model.addGenConstrIndicator(
                c[i,d,m], True,
                t[i,m] - taD[i,d],
                GRB.GREATER_EQUAL,
                0.0,
                name=f"svc_drone_{i}_{m}_{d}"
            )

    # (C13) Depot initial time = 0
    for k in K:
        model.addConstr(taK[o_s,k] == 0, name=f"depot_start_truck_{k}")
        model.addConstr(tdK[o_s,k] == 0, name=f"depot_start_tdK_{k}")
    for d in D:
        model.addConstr(taD[o_s,d] == 0, name=f"depot_start_drone_{d}")
        model.addConstr(tdD[o_s,d] == 0, name=f"depot_start_tdD_{d}")

    # (C14) No departure from end depot
    for k in K:
        model.addConstr(gp.quicksum(x[o_e,j,k] for j in V_k if j != o_e) == 0,
                        name=f"no_depart_end_truck_{k}")
    for d in D:
        model.addConstr(gp.quicksum(y[o_e,j,d] for j in V_d if j != o_e) == 0,
                        name=f"no_depart_end_drone_{d}")

    # (C15) Visit once per vehicle
    for j in V_d:
        for k in K:
            if (o_s,j,k) in x:
                model.addConstr(gp.quicksum(x[i,j,k] for i in V_k if i != j) <= 1,
                                name=f"truck_visit_once_{j}_{k}")
    for j in V_d:
        for d in D:
            model.addConstr(gp.quicksum(y[i,j,d] for i in V_d if i != j) <= 1,
                            name=f"drone_visit_once_{j}_{d}")

    # (C16) Drone serves task only if visiting node
    for j in V_d:
        for d in D:
            for m in M:
                model.addConstr(c[j,d,m] <= gp.quicksum(y[i,j,d] for i in V_d if i != j),
                                name=f"drone_serve_link_{j}_{d}_{m}")

    # Solve
    model.optimize()

    # --- Results ---
    print("\n" + "=" * 55)
    print("Solution Results")
    print("=" * 55)

    result = {"status": "unknown", "obj": None}

    if model.status == GRB.OPTIMAL:
        result = {"status": "optimal", "obj": round(model.ObjVal, 2)}
        print(f"\nOptimal objective value: {model.ObjVal:.4f} (hours)")

        print("\nTruck route:")
        truck_edges = [(i,j,k) for (i,j,k) in x if x[i,j,k].X > 0.5]
        route = [o_s]
        while route[-1] != o_e:
            nxt = [j for (i,j,k) in truck_edges if i == route[-1]]
            if not nxt: break
            route.append(nxt[0])
        for nd in route:
            arr = taK[nd,1].X*60 if nd in V_k else 0
            arrow = " -> " if nd != route[-1] else ""
            print(f"  Node {nd} (arr={arr:.2f} min){arrow}", end="")
        print()

        print("\nDrone route:")
        drone_edges = [(i,j,d) for (i,j,d) in y if y[i,j,d].X > 0.5]
        droute = [o_s]
        while droute[-1] != o_e:
            nxt = [j for (i,j,d) in drone_edges if i == droute[-1]]
            if not nxt: break
            droute.append(nxt[0])
        for nd in droute:
            arr = taD[nd,1].X*60
            arrow = " -> " if nd != droute[-1] else ""
            print(f"  Node {nd} (arr={arr:.2f} min){arrow}", end="")
        print()

        print("\nTask service summary:")
        for (i,m) in tasks:
            info = tasks[i,m]
            typ = "delivery" if m == 1 else "surv"
            by_truck = any(x[l,i,k].X > 0.5 for l in V_k if l!=i for k in K if (l,i,k) in x)
            by_drone = any(c[i,d,m].X > 0.5 for d in D)
            srv = "TRUCK" if by_truck else ("DRONE" if by_drone else "UNSERVED")
            svc_t = t[i,m].X * 60
            print(f"  Node {i}, {typ} (p={info['p']}): {srv}, start={svc_t:.2f} min")
    else:
        print(f"Model status: {model.status}")

    print(f"\n{json.dumps(result)}")
    return result


if __name__ == "__main__":
    result = solve_robust_vrpd()
