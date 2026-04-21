import gurobipy as gp
from gurobipy import GRB
import math

def solve_skyport_congestion():
    # --- 1. Data Setup ---
    nodes = {0: (0, 0), 1: (10, 10), 2: (0, 10), 3: (10, 0)}
    candidates = list(nodes.keys())

    demands = [
        {'id': 'D1', 'o': 0, 'd': 1, 'vol': 100},  # Diagonal 1
        {'id': 'D2', 'o': 2, 'd': 3, 'vol': 100},  # Diagonal 2
        {'id': 'D3', 'o': 0, 'd': 2, 'vol': 50}    # Vertical
    ]

    fixed_cost_per_port = 500
    cost_time = 1.0
    cost_risk_unit = 0.1
    speed_g = 0.5
    speed_a = 2.0
    speed_access = 0.5

    def get_dist(i, j):
        p1, p2 = nodes[i], nodes[j]
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # --- 2. Pre-compute travel times ---
    # For this small instance, each node IS a candidate, so closest port = self
    # Ground time for each demand
    ground_times = {}
    air_times = {}
    for dem in demands:
        dist_g = get_dist(dem['o'], dem['d'])
        ground_times[dem['id']] = dist_g / speed_g

        # Air: access(origin->port_o) + flight(port_o->port_d) + egress(port_d->dest)
        # Since origin/dest ARE candidates, access/egress = 0 when those ports are open
        # port_o = dem['o'], port_d = dem['d']
        dist_air = get_dist(dem['o'], dem['d'])
        air_times[dem['id']] = dist_air / speed_a

    # Intersection check: D1 (0->1) and D2 (2->3) intersect
    intersecting_pairs = [('D1', 'D2')]

    # --- 3. Model ---
    model = gp.Model("Skyport_MILP")

    # Variables
    y = model.addVars(candidates, vtype=GRB.BINARY, name="Open")
    x = {}
    for dem in demands:
        x[dem['id']] = model.addVar(vtype=GRB.BINARY, name=f"Air_{dem['id']}")

    # McCormick linearization variable for D1*D2 product
    w = model.addVar(vtype=GRB.BINARY, name="w_D1_D2")

    # --- 4. Objective ---
    # Facility cost
    obj_facility = gp.quicksum(fixed_cost_per_port * y[i] for i in candidates)

    # Travel time cost (with z eliminated: z = 1 - x)
    # cost_t * sum[ tau_g + (tau_a - tau_g) * x ] * vol
    obj_travel = 0
    for dem in demands:
        tau_g = ground_times[dem['id']]
        tau_a = air_times[dem['id']]
        vol = dem['vol']
        obj_travel += cost_time * (tau_g * vol + (tau_a - tau_g) * vol * x[dem['id']])

    # Risk cost (linearized): C_risk * vol1 * vol2 * w
    vol_d1 = demands[0]['vol']  # D1
    vol_d2 = demands[1]['vol']  # D2
    obj_risk = cost_risk_unit * vol_d1 * vol_d2 * w

    model.setObjective(obj_facility + obj_travel + obj_risk, GRB.MINIMIZE)

    # --- 5. Constraints ---

    # Skyport availability: air mode requires both endpoint ports open
    for dem in demands:
        model.addConstr(x[dem['id']] <= y[dem['o']], name=f"PortO_{dem['id']}")
        model.addConstr(x[dem['id']] <= y[dem['d']], name=f"PortD_{dem['id']}")

    # McCormick linearization for w = x_D1 * x_D2
    model.addConstr(w <= x['D1'], name="McC_w_le_D1")
    model.addConstr(w <= x['D2'], name="McC_w_le_D2")
    model.addConstr(w >= x['D1'] + x['D2'] - 1, name="McC_w_ge")

    # --- 6. Solve ---
    model.optimize()

    # --- 7. Output ---
    if model.status == GRB.OPTIMAL:
        open_ports = [i for i in candidates if y[i].X > 0.5]
        modes = tuple(1 if x[dem['id']].X > 0.5 else 0 for dem in demands)
        return {
            "status": "optimal",
            "obj": round(model.ObjVal, 2),
            "open_ports": open_ports,
            "modes": modes  # 0=Ground, 1=Air
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_skyport_congestion())
