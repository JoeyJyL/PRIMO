import gurobipy as gp
from gurobipy import GRB
import math

def solve_robust_vrpd():
    # --- Parameter Setup ---
    v_truck = 30.0   # truck speed (km/h)
    v_drone = 60.0   # drone speed (km/h)
    L_K = 100.0      # truck weight capacity (kg)
    L_D = 10.0       # drone weight capacity (kg)
    L_R = 1          # max drones a truck can carry on a link
    T_D = 30.0 / 60  # max drone flight duration (hours)
    B   = 1000.0     # penalty for unserved task
    t_bar = 10.0     # big-M for time constraints (hours, >> max makespan)

    # Uncertainty parameters (deterministic baseline: Gamma=0)
    # To solve deterministic version with Gurobi, set Gamma_K=Gamma_D=0
    # Maximum deviation coefficients
    eta_K = 0.3
    eta_D = 0.3
    # For deterministic solve (nominal travel times), deviations = 0
    # Robust worst-case accounted for via uncertainty budgets in analysis

    # Node coordinates (km)
    # Node 0: start depot, Node 4: end depot, Nodes 1-3: customers
    coords = {
        0: (0.0, 0.0),   # start depot
        1: (3.0, 0.0),   # customer 1 (delivery, truck-reachable)
        2: (6.0, 0.0),   # customer 2 (delivery, truck-reachable)
        3: (4.0, 3.0),   # customer 3 (surveillance, truck-UNreachable)
        4: (9.0, 0.0),   # end depot
    }

    # Task definitions: (type, weight_demand_kg, time_demand_h, priority)
    tasks = {
        (1, 1): {'dw': 5.0, 'dt': 0.0,      'p': 2},  # node 1, delivery
        (2, 1): {'dw': 3.0, 'dt': 0.0,      'p': 1},  # node 2, delivery
        (3, 2): {'dw': 0.0, 'dt': 5.0/60,   'p': 3},  # node 3, surveillance
    }

    # Sets
    o_s, o_e = 0, 4                           # start/end depot
    V_k  = [0, 1, 2, 4]                       # truck-reachable nodes
    V_d  = [0, 1, 2, 3, 4]                    # drone-reachable nodes
    V_cust_k = [1, 2]                          # truck-reachable customers
    V_cust_d_only = [3]                        # drone-only customers
    K    = [1]                                 # truck indices
    D    = [1]                                 # drone indices
    M    = [1, 2]                              # task types

    def dist(a, b):
        return math.sqrt((coords[a][0]-coords[b][0])**2 + (coords[a][1]-coords[b][1])**2)

    # Nominal travel times
    tK = {(i,j): dist(i,j)/v_truck for i in V_k for j in V_k if i != j}
    tD = {(i,j): dist(i,j)/v_drone for i in V_d for j in V_d if i != j}

    # Maximum deviations (hat_t = eta * t_nominal)
    hat_tK = {(i,j): eta_K * tK[i,j] for (i,j) in tK}
    hat_tD = {(i,j): eta_D * tD[i,j] for (i,j) in tD}

    # --- Print Preprocessing Info ---
    print("=" * 55)
    print("Preprocessing Info")
    print("=" * 55)

    print("\nNominal truck travel times (min):")
    for (i,j), v in sorted(tK.items()):
        print(f"  ({i},{j}): {v*60:.2f}")

    print("\nNominal drone travel times (min) [selected]:")
    for (i,j), v in sorted(tD.items()):
        if 3 in (i,j):
            print(f"  ({i},{j}): {v*60:.2f}")

    print(f"\nTruck-reachable: {V_k}")
    print(f"Drone-reachable: {V_d}")
    print(f"Truck-only customers: {V_cust_k}")
    print(f"Drone-only customers: {V_cust_d_only}")

    print("\nTask list:")
    for (i,m), info in tasks.items():
        typ = "delivery" if m == 1 else "surveillance"
        print(f"  Node {i}, type {m} ({typ}): dw={info['dw']}kg, dt={info['dt']*60:.1f}min, priority={info['p']}")

    # --- Build Gurobi Model ---
    print("\n" + "=" * 55)
    print("Solving Deterministic VRPD (Gurobi)")
    print("=" * 55)

    model = gp.Model("RobustVRPD")
    model.setParam('OutputFlag', 0)

    # Decision variables
    # x[i,j,k]: truck k travels link (i,j)
    x = model.addVars([(i,j,k) for i in V_k for j in V_k if i!=j for k in K],
                      vtype=GRB.BINARY, name="x")
    # y[i,j,d]: drone d travels link (i,j)
    y = model.addVars([(i,j,d) for i in V_d for j in V_d if i!=j for d in D],
                      vtype=GRB.BINARY, name="y")
    # c[i,d,m]: drone d serves task m at node i
    c = model.addVars([(i,d,m) for i in V_d for d in D for m in M],
                      vtype=GRB.BINARY, name="c")
    # g[i,m]: task m at node i is unserved
    g = model.addVars([(i,m) for i in V_d for m in M if (i,m) in tasks],
                      vtype=GRB.BINARY, name="g")
    # Arrival and departure times for truck
    taK = model.addVars([(i,k) for i in V_k for k in K], lb=0.0, name="taK")
    tdK = model.addVars([(i,k) for i in V_k for k in K], lb=0.0, name="tdK")
    # Arrival and departure times for drone
    taD = model.addVars([(i,d) for i in V_d for d in D], lb=0.0, name="taD")
    tdD = model.addVars([(i,d) for i in V_d for d in D], lb=0.0, name="tdD")
    # Service start time
    t = model.addVars([(i,m) for (i,m) in tasks], lb=0.0, name="t")
    # Drone cumulative weight and flight duration
    wD = model.addVars([(i,d) for i in V_d for d in D], lb=0.0, name="wD")
    vD = model.addVars([(i,d) for i in V_d for d in D], lb=0.0, name="vD")
    # Truck cumulative weight
    wK = model.addVars([(i,k) for i in V_k for k in K], lb=0.0, name="wK")

    # Objective: minimize priority cost + penalty for unserved tasks
    obj = (gp.quicksum(tasks[i,m]['p'] * t[i,m] for (i,m) in tasks) +
           B * gp.quicksum(tasks[i,m]['p'] * g[i,m] for (i,m) in tasks))
    model.setObjective(obj, GRB.MINIMIZE)

    # ── Constraints ──────────────────────────────────────────

    # (C1) Truck flow: depart depot once, return to end depot once
    for k in K:
        model.addConstr(gp.quicksum(x[o_s,j,k] for j in V_k if j != o_s) == 1,
                        name=f"truck_depart_{k}")
        model.addConstr(gp.quicksum(x[i,o_e,k] for i in V_k if i != o_e) == 1,
                        name=f"truck_return_{k}")

    # (C2) Truck flow conservation at customer nodes
    for j in V_cust_k:
        for k in K:
            model.addConstr(gp.quicksum(x[i,j,k] for i in V_k if i != j) ==
                            gp.quicksum(x[j,i,k] for i in V_k if i != j),
                            name=f"truck_flow_{j}_{k}")

    # (C3) Drone flow: depart depot once, return to end depot once
    for d in D:
        model.addConstr(gp.quicksum(y[o_s,j,d] for j in V_d if j != o_s) == 1,
                        name=f"drone_depart_{d}")
        model.addConstr(gp.quicksum(y[i,o_e,d] for i in V_d if i != o_e) == 1,
                        name=f"drone_return_{d}")

    # (C4) Drone flow conservation at all non-depot nodes
    for j in [n for n in V_d if n not in [o_s, o_e]]:
        for d in D:
            model.addConstr(gp.quicksum(y[i,j,d] for i in V_d if i != j) ==
                            gp.quicksum(y[j,i,d] for i in V_d if i != j),
                            name=f"drone_flow_{j}_{d}")

    # (C5) Task status at truck-reachable customer nodes
    for j in V_cust_k:
        for m in M:
            if (j,m) in tasks:
                model.addConstr(gp.quicksum(x[i,j,k] for i in V_k if i != j for k in K) +
                                gp.quicksum(c[j,d,m] for d in D) + g[j,m] == 1,
                                name=f"task_status_truck_{j}_{m}")

    # (C6) Task status at drone-only nodes
    for i in V_cust_d_only:
        for m in M:
            if (i,m) in tasks:
                model.addConstr(gp.quicksum(c[i,d,m] for d in D) + g[i,m] == 1,
                                name=f"task_status_drone_{i}_{m}")

    # (C7) Drone capacity: cumulative weight tracking
    for i in V_d:
        for d in D:
            model.addConstr(wD[i,d] <= L_D * gp.quicksum(y[j,i,d] for j in V_d if j != i),
                            name=f"drone_cap_ub_{i}_{d}")
            model.addConstr(wD[i,d] >= gp.quicksum(tasks.get((i,m),{'dw':0})['dw'] * c[i,d,m]
                                                     for m in M),
                            name=f"drone_cap_lb_{i}_{d}")

    # (C8) Drone flight duration (using max uncertain travel time for robustness)
    for j in V_d:
        for d in D:
            # First leg: minimum accumulative flight time at j
            # Guard: x[i,j,k] only exists when j in V_k (truck-reachable)
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

    # (C9) Drone flight duration propagation
    for i in V_d:
        for j in V_d:
            if i != j:
                for d in D:
                    # Guard: x[i,l,k] only exists when i in V_k (truck-reachable)
                    truck_depart_i = (gp.quicksum(x[i,l,k] for l in V_k if l != i for k in K)
                                      if i in V_k else 0)
                    model.addConstr(
                        vD[j,d] >= vD[i,d] + (tD[i,j] + hat_tD[i,j]) * y[i,j,d] +
                                   gp.quicksum(tasks.get((j,m),{'dt':0})['dt'] * c[j,d,m] for m in M) +
                                   t_bar * (y[i,j,d] - 1) -
                                   t_bar * truck_depart_i,
                        name=f"drone_flight_prop_{i}_{j}_{d}"
                    )

    # (C10) Truck arrival time propagation
    for i in V_k:
        for j in V_k:
            if i != j:
                for k in K:
                    model.addConstr(
                        taK[j,k] >= tdK[i,k] + tK[i,j] * x[i,j,k] + t_bar * (x[i,j,k] - 1),
                        name=f"truck_arrive_{i}_{j}_{k}"
                    )

    # (C11) Truck departure >= arrival
    for i in V_k:
        for k in K:
            model.addConstr(tdK[i,k] >= taK[i,k], name=f"truck_dept_ge_arr_{i}_{k}")
            model.addConstr(tdK[i,k] <= t_bar * gp.quicksum(x[l,i,k] for l in V_k if l != i),
                            name=f"truck_dept_active_{i}_{k}")

    # (C12) Drone arrival time propagation (nominal travel time for drone timing)
    for i in V_d:
        for j in V_d:
            if i != j:
                for d in D:
                    model.addConstr(
                        taD[j,d] >= tdD[i,d] + tD[i,j] * y[i,j,d] + t_bar * (y[i,j,d] - 1),
                        name=f"drone_arrive_{i}_{j}_{d}"
                    )

    # (C13) Drone departure after task service time
    for j in V_d:
        for d in D:
            model.addConstr(
                tdD[j,d] >= taD[j,d] + gp.quicksum(tasks.get((j,m),{'dt':0})['dt'] * c[j,d,m]
                                                     for m in M),
                name=f"drone_dept_{j}_{d}"
            )

    # (C14) Time synchronization: drone departs/arrives at truck node when truck is present
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

    # (C15) Service start time
    for (i,m) in tasks:
        if i in V_cust_k:
            model.addConstr(t[i,m] >= gp.quicksum(taK[i,k] for k in K),
                            name=f"svc_truck_{i}_{m}")
        for d in D:
            model.addConstr(t[i,m] >= taD[i,d] + t_bar * (c[i,d,m] - 1),
                            name=f"svc_drone_{i}_{m}_{d}")

    # (C16) Depot initial time = 0
    for k in K:
        model.addConstr(taK[o_s,k] == 0, name=f"depot_start_truck_{k}")
        model.addConstr(tdK[o_s,k] == 0, name=f"depot_start_tdK_{k}")
    for d in D:
        model.addConstr(taD[o_s,d] == 0, name=f"depot_start_drone_{d}")
        model.addConstr(tdD[o_s,d] == 0, name=f"depot_start_tdD_{d}")

    # (C17) No self-loops and no travel from end depot
    for k in K:
        model.addConstr(gp.quicksum(x[o_e,j,k] for j in V_k if j != o_e) == 0,
                        name=f"no_depart_end_truck_{k}")
    for d in D:
        model.addConstr(gp.quicksum(y[o_e,j,d] for j in V_d if j != o_e) == 0,
                        name=f"no_depart_end_drone_{d}")

    # (C18) Each node visited at most once per vehicle
    for j in V_d:
        for k in K:
            if (o_s,j,k) in x:
                model.addConstr(gp.quicksum(x[i,j,k] for i in V_k if i != j) <= 1,
                                name=f"truck_visit_once_{j}_{k}")
    for j in V_d:
        for d in D:
            model.addConstr(gp.quicksum(y[i,j,d] for i in V_d if i != j) <= 1,
                            name=f"drone_visit_once_{j}_{d}")

    # (C19) Drone serves task only if it visits the node
    for j in V_d:
        for d in D:
            for m in M:
                model.addConstr(c[j,d,m] <= gp.quicksum(y[i,j,d] for i in V_d if i != j),
                                name=f"drone_serve_link_{j}_{d}_{m}")

    # Solve
    model.optimize()

    # --- Print Results ---
    print("\n" + "=" * 55)
    print("Solution Results")
    print("=" * 55)

    if model.status == GRB.OPTIMAL:
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
            unserved = g[i,m].X > 0.5
            srv = "TRUCK" if by_truck else ("DRONE" if by_drone else "UNSERVED")
            svc_t = t[i,m].X * 60
            print(f"  Node {i}, {typ} (p={info['p']}): {srv}, start={svc_t:.2f} min")

        return {
            "status": "optimal",
            "obj": round(model.ObjVal,2)
        }
    else:
        print("Model infeasible or unbounded!")
        return {"status": "infeasible"}


if __name__ == "__main__":
    result = solve_robust_vrpd()
    print("\n" + "=" * 55)
    print(f"Final result: {result}")