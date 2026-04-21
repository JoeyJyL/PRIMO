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
    model = gp.Model("Drone_Delivery_Allocation")

    # Decision variables
    # a[i,j]: drone delivery offer rate for category j in zone i
    a = model.addVars(pairs, vtype=GRB.CONTINUOUS, lb=0, name="offer")

    # u[i,j]: unfulfilled demand for category j in zone i
    u = model.addVars(pairs, vtype=GRB.CONTINUOUS, lb=0, name="unfulfilled")

    # W: excess workload beyond capacity (delay workload)
    W = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="delay_workload")

    # --- 3. Objective Function ---
    # Maximize: sum (w_j - c_v) * a_ij - sum c_j * u_ij - c_w * W - K * c_F

    revenue_minus_var = gp.quicksum(
        (categories[j]['w'] - c_v) * a[i, j] for i, j in pairs
    )
    opportunity_cost = gp.quicksum(
        categories[j]['c'] * u[i, j] for i, j in pairs
    )
    delay_cost = c_w * W
    fixed_cost = K * c_F

    model.setObjective(
        revenue_minus_var - opportunity_cost - delay_cost - fixed_cost,
        GRB.MAXIMIZE
    )

    # --- 4. Constraints ---

    # C1: Offer rate bounded by demand
    for i, j in pairs:
        model.addConstr(
            a[i, j] <= demand[(i, j)],
            name=f"demand_cap_{i}_{j}"
        )

    # C2: Unfulfilled demand definition
    for i, j in pairs:
        model.addConstr(
            u[i, j] >= demand[(i, j)] - a[i, j],
            name=f"unfulfilled_def_{i}_{j}"
        )

    # C3: Excess workload (delay) constraint
    # Workload = sum a[i,j] / (K * mu_i), delay W >= workload - 1
    total_workload = gp.quicksum(
        a[i, j] / (K * zones[i]['mu']) for i, j in pairs
    )
    model.addConstr(
        W >= total_workload - 1,
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
        print(f"Fixed Cost: ${fixed_cost:.2f}")
        print(f"Delay Workload (W): {W.X:.4f}")
        print(f"Delay Cost: ${c_w * W.X:.2f}")

        print("\n--- Drone Delivery Offer Rates ---")
        print(f"{'Zone':<6} {'Cat':<6} {'Demand':<8} {'Offered':<10} "
              f"{'Unfulfilled':<12} {'NetMargin/u':<12} {'WorkloadUnit':<14}")
        for i in zone_names:
            for j in cat_names:
                d = demand[(i, j)]
                offered = a[i, j].X
                unfulfilled = u[i, j].X
                net_margin = categories[j]['w'] - c_v
                wl = offered / (K * zones[i]['mu'])
                print(f"{i:<6} {j:<6} {d:<8} {offered:<10.2f} "
                      f"{unfulfilled:<12.2f} ${net_margin:<11.2f} {wl:<14.4f}")

        total_wl = sum(
            a[i, j].X / (K * zones[i]['mu']) for i, j in pairs
        )
        print(f"\nTotal Workload: {total_wl:.4f} (capacity = 1.0)")

        total_offered = sum(a[i, j].X for i, j in pairs)
        total_demand = sum(demand[(i, j)] for i, j in pairs)
        print(f"Total Orders Offered: {total_offered:.1f} / {total_demand} demanded")

        return result
    else:
        return {"status": "infeasible"}


if __name__ == "__main__":
    result = solve_drone_delivery_allocation()
    print(f"\nResult: {result}")