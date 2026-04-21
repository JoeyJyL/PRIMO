import gurobipy as gp
from gurobipy import GRB
import math

def solve_drone_fleet_planning():
    # --- 1. Data Input ---
    truck_cost_km = 1.0
    drone_cost_km = 0.1
    drone_range = 10.0

    cdc = (0, 0)
    mds = {
        1: {'loc': (10, 10), 'fixed': 100},
        2: {'loc': (20, 0),  'fixed': 250}
    }
    customers = {
        3: (12, 12),
        4: (8, 8),
        5: (22, 2),
        6: (0, 15)
    }

    all_nodes = {0: cdc}
    for k, v in mds.items(): all_nodes[k] = v['loc']
    for k, v in customers.items(): all_nodes[k] = v

    node_ids = list(all_nodes.keys())
    N = len(node_ids)

    def get_dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    dist = {(i, j): get_dist(all_nodes[i], all_nodes[j]) for i in node_ids for j in node_ids}

    # --- 2. Model ---
    model = gp.Model("DroneFleet_Flow")

    # Variables
    y = model.addVars(mds.keys(), vtype=GRB.BINARY, name="OpenMD")
    z = model.addVars(customers.keys(), mds.keys(), vtype=GRB.BINARY, name="AssignDrone")
    v = model.addVars(customers.keys(), vtype=GRB.BINARY, name="AssignTruck")

    x = model.addVars(node_ids, node_ids, vtype=GRB.BINARY, name="TruckRoute")
    f = model.addVars(node_ids, node_ids, vtype=GRB.CONTINUOUS, lb=0, name="Flow")

    # --- Objective ---
    obj_fixed = gp.quicksum(mds[m]['fixed'] * y[m] for m in mds)

    obj_drone = 0
    for i in customers:
        for m in mds:
            d_im = get_dist(customers[i], mds[m]['loc'])
            obj_drone += 2 * d_im * drone_cost_km * z[i, m]

    obj_truck = gp.quicksum(dist[i, j] * truck_cost_km * x[i, j] for i in node_ids for j in node_ids)

    model.setObjective(obj_fixed + obj_drone + obj_truck, GRB.MINIMIZE)

    # --- Constraints ---

    # 1. Customer Assignment
    for i in customers:
        model.addConstr(gp.quicksum(z[i, m] for m in mds) + v[i] == 1, name=f"Assign_{i}")

    # 2. Drone Range & MD Logic
    for i in customers:
        for m in mds:
            d_im = get_dist(customers[i], mds[m]['loc'])
            if d_im > drone_range:
                model.addConstr(z[i, m] == 0)
            else:
                model.addConstr(z[i, m] <= y[m])

    # 3. Truck Routing Flow Conservation
    is_visited = model.addVars(node_ids, vtype=GRB.BINARY, name="IsVisited")

    model.addConstr(is_visited[0] == 1)
    for m in mds:
        model.addConstr(is_visited[m] == y[m])
    for i in customers:
        model.addConstr(is_visited[i] == v[i])

    for i in node_ids:
        model.addConstr(gp.quicksum(x[i, j] for j in node_ids if i != j) == is_visited[i])
        model.addConstr(gp.quicksum(x[j, i] for j in node_ids if i != j) == is_visited[i])

    # 4. Single-commodity flow subtour elimination (replacing MTZ)
    # Flow capacity
    for i in node_ids:
        for j in node_ids:
            if i != j:
                model.addConstr(f[i, j] <= (N - 1) * x[i, j], name=f"FlowCap_{i}_{j}")

    # Flow conservation at non-CDC nodes: each active node consumes 1 unit
    for i in node_ids:
        if i == 0:
            continue
        flow_in = gp.quicksum(f[j, i] for j in node_ids if j != i)
        flow_out = gp.quicksum(f[i, j] for j in node_ids if j != i)
        model.addConstr(flow_in - flow_out == is_visited[i], name=f"FlowCons_{i}")

    # CDC sources flow = total active non-CDC nodes
    cdc_out = gp.quicksum(f[0, j] for j in node_ids if j != 0)
    cdc_in = gp.quicksum(f[j, 0] for j in node_ids if j != 0)
    model.addConstr(
        cdc_out - cdc_in == gp.quicksum(y[m] for m in mds) + gp.quicksum(v[i] for i in customers),
        name="FlowSource_CDC"
    )

    # --- Solve ---
    model.optimize()

    # --- Output ---
    if model.status == GRB.OPTIMAL:
        open_mds = [m for m in mds if y[m].X > 0.5]
        drone_custs = [i for i in customers if sum(z[i, m].X for m in mds) > 0.5]
        truck_custs = [i for i in customers if v[i].X > 0.5]

        return {
            "status": "optimal",
            "obj": round(model.ObjVal, 2),
            "open_mds": open_mds,
            "truck_route_customers": truck_custs,
            "drone_served_customers": drone_custs
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_drone_fleet_planning())
