import gurobipy as gp
from gurobipy import GRB
import math
import json

def solve_charging_station_network():
    """
    Solve the Drone Charging Station Network Design Problem.
    Bi-objective: minimize path lengths and number of active stations.
    Big-M reformulation: removes auxiliary product linearization variables z_{esi},
    uses Big-M edge activation constraint y_{esi} <= u_{si} to ensure equivalence.
    """

    # --- 1. Data Input ---

    # Hub location
    hubs = {
        'H1': (0, 0)
    }

    # Candidate charging station locations
    stations = {
        'S1': (15, 10),
        'S2': (25, 5),
        'S3': (20, 25),
        'S4': (35, 20),
        'S5': (40, 35),
        'S6': (10, 30)
    }

    # Delivery points
    delivery_points = {
        'D1': (18, 8),
        'D2': (30, 10),
        'D3': (22, 28),
        'D4': (38, 25),
        'D5': (45, 38),
        'D6': (12, 35)
    }

    # Parameters
    R = 15.0        # One-way coverage radius
    two_R = 30.0    # Round-trip range (2R)
    theta = 0.5     # Weight for bi-objective (0.5 = balanced)

    # Sets
    S = list(hubs.keys())      # Hubs
    I = list(stations.keys())  # Candidate stations
    N = list(delivery_points.keys())  # Delivery points

    # --- 2. Pre-processing ---

    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def get_coords(node):
        if node in hubs:
            return hubs[node]
        elif node in stations:
            return stations[node]
        else:
            return delivery_points[node]

    # Build feasible edges (distance <= 2R)
    E = []
    edge_lengths = {}

    # Edges from hubs to stations
    for s in S:
        for i in I:
            dist = euclidean_distance(get_coords(s), get_coords(i))
            if dist <= two_R:
                edge = (s, i)
                E.append(edge)
                edge_lengths[edge] = dist

    # Edges between stations
    for i in I:
        for j in I:
            if i != j:
                dist = euclidean_distance(get_coords(i), get_coords(j))
                if dist <= two_R:
                    edge = (i, j)
                    E.append(edge)
                    edge_lengths[edge] = dist

    # Coverage matrix: c[i,n] = 1 if delivery point n is within R of station i
    c = {}
    for i in I:
        for n in N:
            dist = euclidean_distance(get_coords(i), get_coords(n))
            c[i, n] = 1 if dist <= R else 0

    # Normalizing constants
    beta1 = sum(edge_lengths.values()) if edge_lengths else 1
    beta2 = len(I)

    # --- 3. Optimization Model ---
    model = gp.Model("Charging_Station_Network_BigM")
    model.setParam('TimeLimit', 300)
    model.setParam('OutputFlag', 0)

    # Decision Variables

    # x[i]: 1 if station i is activated
    x = model.addVars(I, vtype=GRB.BINARY, name="x")

    # y[e,s,i]: 1 if edge e is used in path from hub s to station i
    y = {}
    for e in E:
        for s in S:
            for i in I:
                y[e, s, i] = model.addVar(vtype=GRB.BINARY, name=f"y_{e[0]}_{e[1]}_{s}_{i}")

    # u[s,i]: 1 if station i is a terminal station connected to hub s
    u = model.addVars(S, I, vtype=GRB.BINARY, name="u")

    # NOTE: No z[e,s,i] auxiliary variables needed — Big-M edge activation
    # constraint y[e,s,i] <= u[s,i] ensures y*u = y at every feasible point.

    # --- 4. Objective Function ---

    # Direct use of y[e,s,i] instead of product linearization z[e,s,i]
    obj_paths = gp.quicksum(edge_lengths[e] * y[e, s, i]
                            for e in E for s in S for i in I)
    obj_stations = gp.quicksum(x[i] for i in I)

    model.setObjective(
        (theta / beta1) * obj_paths + ((1 - theta) / beta2) * obj_stations,
        GRB.MINIMIZE
    )

    # --- 5. Constraints ---

    # Flow balance constraints
    for s in S:
        for i in I:
            for k in I:
                # Incoming edges to k
                incoming = gp.quicksum(y[e, s, i] for e in E if e[1] == k)
                # Outgoing edges from k
                outgoing = gp.quicksum(y[e, s, i] for e in E if e[0] == k)

                if i == k:
                    model.addConstr(incoming - outgoing == u[s, i],
                                   name=f"flow_{s}_{i}_{k}")
                else:
                    model.addConstr(incoming - outgoing == 0,
                                   name=f"flow_{s}_{i}_{k}")

    # Single outgoing edge from hub per path
    for s in S:
        for i in I:
            model.addConstr(
                gp.quicksum(y[e, s, i] for e in E if e[0] == s) <= 1,
                name=f"single_out_{s}_{i}"
            )

    # Edge activation via Big-M implication (replaces product linearization)
    for e in E:
        for s in S:
            for i in I:
                model.addConstr(y[e, s, i] <= u[s, i], name=f"edge_act_{e}_{s}_{i}")

    # Single hub assignment per terminal station
    for i in I:
        model.addConstr(gp.quicksum(u[s, i] for s in S) <= 1, name=f"single_hub_{i}")

    # Delivery point coverage
    for n in N:
        model.addConstr(
            gp.quicksum(c[i, n] * u[s, i] for s in S for i in I) >= 1,
            name=f"cover_{n}"
        )

    # Station activation linkage (disaggregated Big-M: per-edge, M=1)
    for e in E:
        for s in S:
            for i in I:
                model.addConstr(y[e, s, i] <= x[i], name=f"station_act_{e}_{s}_{i}")

    # Path existence: if u[s,i]=1, there must be at least one incoming edge
    for s in S:
        for i in I:
            direct_edges = [e for e in E if e[0] == s and e[1] == i]
            indirect_edges = [e for e in E if e[1] == i and e[0] in I]
            all_incoming = direct_edges + indirect_edges
            if all_incoming:
                model.addConstr(
                    gp.quicksum(y[e, s, i] for e in all_incoming) >= u[s, i],
                    name=f"path_exist_{s}_{i}"
                )

    # --- 6. Solve ---
    model.optimize()

    # --- 7. Output ---
    if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
        # Extract solution
        active_stations = [i for i in I if x[i].X > 0.5]
        terminal_stations = [(s, i) for s in S for i in I if u[s, i].X > 0.5]

        # Calculate individual objectives
        total_path_length = sum(edge_lengths[e] * y[e, s, i].X
                               for e in E for s in S for i in I)
        num_stations = sum(x[i].X for i in I)

        # Extract paths
        paths = []
        for s in S:
            for i in I:
                if u[s, i].X > 0.5:
                    path_edges = [e for e in E if y[e, s, i].X > 0.5]
                    if path_edges:
                        # Reconstruct path
                        path = [s]
                        current = s
                        visited = set()
                        while current != i and len(visited) < len(path_edges) + 1:
                            visited.add(current)
                            for e in path_edges:
                                if e[0] == current and e[1] not in visited:
                                    path.append(e[1])
                                    current = e[1]
                                    break
                        paths.append(path)

        result = {
            "status": "optimal" if model.status == GRB.OPTIMAL else "feasible",
            "obj": round(model.ObjVal, 2),
            "total_path_length": round(total_path_length, 2),
            "num_active_stations": int(num_stations),
            "active_stations": active_stations,
            "terminal_stations": [t[1] for t in terminal_stations],
            "paths": paths
        }
        return result
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    result = solve_charging_station_network()
    print(json.dumps(result, indent=4))
