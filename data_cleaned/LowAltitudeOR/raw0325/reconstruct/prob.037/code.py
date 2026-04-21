import gurobipy as gp
from gurobipy import GRB

def solve_drone_delivery_allocation():
    # --- 1. Data Input ---

    # Drone fleet parameters
    K = 8           # Number of drones
    v = 40.0        # Drone speed (mph)
    c_w = 100.00    # Delivery delay cost per excess workload unit ($/unit)
    c_v = 1.20      # Variable cost per delivery ($/order)
    c_F = 0.50      # Fixed cost per drone per period ($/drone/hr)

    # Service zones: name -> distance (miles)
    zones = {
        'Z1': {'dist': 4},
        'Z2': {'dist': 7},
        'Z3': {'dist': 11},
    }

    # Compute service rate mu_i = v / (2 * s_i) for each zone
    for z_name, z_data in zones.items():
        z_data['mu'] = v / (2 * z_data['dist'])

    # Product categories: name -> unit profit, opportunity cost
    categories = {
        'P1': {'w': 2.00, 'c': 0.60},
        'P2': {'w': 4.50, 'c': 1.50},
        'P3': {'w': 8.00, 'c': 3.00},
    }

    # Demand matrix D[zone][category] (units/period)
    demand = {
        ('Z1', 'P1'): 18, ('Z1', 'P2'): 12, ('Z1', 'P3'): 7,
        ('Z2', 'P1'): 14, ('Z2', 'P2'): 9,  ('Z2', 'P3'): 5,
        ('Z3', 'P1'): 10, ('Z3', 'P2'): 6,  ('Z3', 'P3'): 3,
    }

    zone_names = list(zones.keys())
    cat_names = list(categories.keys())
    pairs = [(i, j) for i in zone_names for j in cat_names]

    # --- 2. Optimization Model ---
    model = gp.Model("Drone_Delivery_AuxVar")

    # Decision variables
    # a[i,j]: drone delivery offer rate for category j in zone i
    a = model.addVars(pairs, vtype=GRB.CONTINUOUS, lb=0, name="offer")

    # u[i,j]: unfulfilled demand for category j in zone i
    u = model.addVars(pairs, vtype=GRB.CONTINUOUS, lb=0, name="unfulfilled")

    # g[i,j]: auxiliary -- net contribution from (i,j) pair
    g = model.addVars(pairs, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="net_contrib")

    # WL[i]: auxiliary -- normalized workload contribution from zone i
    WL = model.addVars(zone_names, vtype=GRB.CONTINUOUS, lb=0, name="zone_workload")

    # L: auxiliary -- total normalized workload
    L = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="total_workload")

    # W: excess workload beyond capacity
    W = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="delay_workload")

    # --- 3. Objective Function ---
    # Maximize: sum g_ij - c_w * W - K * c_F
    model.setObjective(
        gp.quicksum(g[i, j] for i, j in pairs) - c_w * W - K * c_F,
        GRB.MAXIMIZE
    )

    # --- 4. Constraints ---

    # C1: Net contribution definition (auxiliary linking)
    # g_ij = (w_j - c_v) * a_ij - c_j * u_ij
    for i, j in pairs:
        model.addConstr(
            g[i, j] == (categories[j]['w'] - c_v) * a[i, j]
                        - categories[j]['c'] * u[i, j],
            name=f"net_contrib_def_{i}_{j}"
        )

    # C2: Offer rate bounded by demand
    for i, j in pairs:
        model.addConstr(
            a[i, j] <= demand[(i, j)],
            name=f"demand_cap_{i}_{j}"
        )

    # C3: Unfulfilled demand definition
    for i, j in pairs:
        model.addConstr(
            u[i, j] >= demand[(i, j)] - a[i, j],
            name=f"unfulfilled_def_{i}_{j}"
        )

    # C4: Zone workload definition (auxiliary linking)
    # WL_i = sum_j a_ij / (K * mu_i)
    for i in zone_names:
        model.addConstr(
            WL[i] == gp.quicksum(
                a[i, j] / (K * zones[i]['mu']) for j in cat_names
            ),
            name=f"zone_wl_def_{i}"
        )

    # C5: Total workload aggregation (auxiliary linking)
    # L = sum_i WL_i
    model.addConstr(
        L == gp.quicksum(WL[i] for i in zone_names),
        name="total_wl_def"
    )

    # C6: Excess workload definition
    # W >= L - 1
    model.addConstr(
        W >= L - 1,
        name="delay_workload_def"
    )

    # --- 5. Solve ---
    model.optimize()

    # --- 6. Output ---
    if model.status == GRB.OPTIMAL:
        result = {
            "status": "optimal",
            "obj": round(model.ObjVal, 2),
        }

        print(f"\nOptimal Objective Value (Net Profit): ${model.ObjVal:.2f}")
        print(f"Fixed Cost: ${K * c_F:.2f}")
        print(f"Total Workload (L): {L.X:.4f}")
        print(f"Delay Workload (W): {W.X:.4f}")
        print(f"Delay Cost: ${c_w * W.X:.2f}")

        print("\n--- Zone Workloads ---")
        for i in zone_names:
            print(f"  {i}: WL = {WL[i].X:.4f}")

        print("\n--- Drone Delivery Offer Rates ---")
        for i in zone_names:
            for j in cat_names:
                d = demand[(i, j)]
                offered = a[i, j].X
                unfulfilled = u[i, j].X
                contrib = g[i, j].X
                print(f"  {i}-{j}: demand={d}, offered={offered:.2f}, "
                      f"unfulfilled={unfulfilled:.2f}, g={contrib:.2f}")

        return result
    else:
        return {"status": "infeasible"}


if __name__ == "__main__":
    result = solve_drone_delivery_allocation()
    print(f"\nResult: {result}")
