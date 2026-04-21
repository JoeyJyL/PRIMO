
import gurobipy as gp
from gurobipy import GRB
import math
import json

def solve_vertiport_location():
    # --- Parameter Setup ---
    R = 3.0    # Service radius (km): demand point covered if vertiport within R
    r = 1.5    # Hub distance threshold (km): vertiport must be within r of a hub
    p = 3      # Maximum number of vertiports to open
    cost_per_vp = 500  # Construction + operational cost per vertiport (hundred USD)

    # Objective weights for weighted-sum scalarization
    w1 = 1.0    # Weight for ridership (maximized, negated in minimization)
    w2 = 0.001  # Weight for facility cost
    w3 = 0.01   # Weight for travel distance

    # Candidate vertiport sites
    vertiports = {
        1: (2.0, 7.0),   # V1
        2: (5.0, 8.0),   # V2
        3: (5.0, 4.0),   # V3
        4: (8.0, 7.0),   # V4
        5: (8.0, 3.0),   # V5
    }

    # Demand points: (x, y, ridership)
    demands = {
        1: (1.0, 8.0, 120),   # D1
        2: (3.0, 6.0, 200),   # D2
        3: (5.0, 9.0, 150),   # D3
        4: (6.0, 5.0, 180),   # D4
        5: (9.0, 8.0, 160),   # D5
        6: (9.0, 4.0, 140),   # D6
    }

    # Existing mobility hubs: (x, y)
    hubs = {
        1: (2.5, 7.5),   # H1: Metro Station
        2: (5.5, 8.5),   # H2: Train Station
        3: (4.5, 3.5),   # H3: Metro Station
        4: (8.5, 6.5),   # H4: Airport
    }

    vertiport_indices = list(vertiports.keys())
    demand_indices    = list(demands.keys())
    hub_indices       = list(hubs.keys())

    def get_dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Pre-compute distances: demand -> vertiport
    d = {}
    for i in demand_indices:
        for j in vertiport_indices:
            d[i, j] = get_dist(demands[i][:2], vertiports[j])

    # Pre-compute distances: vertiport -> hub
    l = {}
    for j in vertiport_indices:
        for h in hub_indices:
            l[j, h] = get_dist(vertiports[j], hubs[h])

    # Coverage eligibility parameter: a[i,j] = 1 if d[i,j] <= R
    a = {}
    for i in demand_indices:
        for j in vertiport_indices:
            a[i, j] = 1 if d[i, j] <= R else 0

    # Big-M constant for hub proximity constraints
    M = max(l[j, h] for j in vertiport_indices for h in hub_indices)

    # --- Print Preprocessing Info ---
    print("=" * 50)
    print("Preprocessing Info")
    print("=" * 50)

    print(f"\nBig-M constant: {M:.4f}")

    print("\nDistance matrix: Demand -> Vertiport (km):")
    print("      ", end="")
    for j in vertiport_indices:
        print(f"  V{j:<5}", end="")
    print()
    for i in demand_indices:
        print(f"D{i}    ", end="")
        for j in vertiport_indices:
            mark = "✓" if a[i, j] else " "
            print(f"{d[i,j]:5.2f}{mark} ", end="")
        print()

    print("\nVertiport-Hub distances (km):")
    for j in vertiport_indices:
        for h in hub_indices:
            eligible = "≤ r" if l[j, h] <= r else "> r"
            print(f"  V{j}-H{h}: {l[j,h]:.2f} ({eligible})")

    # --- Build Gurobi Model ---
    print("\n" + "=" * 50)
    print("Solving Optimization Model (Big-M Reformulation)")
    print("=" * 50)

    model = gp.Model("VertiportLocation_BigM")
    model.setParam('OutputFlag', 0)

    # Decision variables
    y = model.addVars(vertiport_indices, vtype=GRB.BINARY, name="y")
    x = model.addVars(demand_indices, vertiport_indices, vtype=GRB.BINARY, name="x")
    z = model.addVars(demand_indices, vertiport_indices, vtype=GRB.BINARY, name="z")
    # Auxiliary variables for Big-M hub proximity
    delta = model.addVars(vertiport_indices, hub_indices, vtype=GRB.BINARY, name="delta")

    # Objective: minimize -w1*g1 + w2*g2 + w3*g3
    g1 = gp.quicksum(demands[i][2] * z[i, j] for i in demand_indices for j in vertiport_indices)
    g2 = cost_per_vp * gp.quicksum(y[j] for j in vertiport_indices)
    g3 = gp.quicksum(d[i, j] * x[i, j] for i in demand_indices for j in vertiport_indices)
    model.setObjective(-w1 * g1 + w2 * g2 + w3 * g3, GRB.MINIMIZE)

    # Constraint 1: Big-M hub proximity — delta[j,h]=1 implies l[j,h] <= r
    for j in vertiport_indices:
        for h in hub_indices:
            model.addConstr(
                l[j, h] <= r + M * (1 - delta[j, h]),
                name=f"BigM_HubProx_{j}_{h}"
            )

    # Constraint 2: Hub certification — selected vertiport needs at least one certifying hub
    for j in vertiport_indices:
        model.addConstr(
            y[j] <= gp.quicksum(delta[j, h] for h in hub_indices),
            name=f"HubCert_{j}"
        )

    # Constraint 3: Each demand point assigned to exactly one vertiport
    for i in demand_indices:
        model.addConstr(
            gp.quicksum(x[i, j] for j in vertiport_indices) == 1,
            name=f"Assign_{i}"
        )

    # Constraint 4: Assignment only to selected vertiport
    for i in demand_indices:
        for j in vertiport_indices:
            model.addConstr(x[i, j] <= y[j], name=f"AssignLink_{i}_{j}")

    # Constraint 5: Combined Big-M coverage constraint (replaces separate linking + feasibility)
    for i in demand_indices:
        for j in vertiport_indices:
            model.addConstr(
                z[i, j] <= a[i, j] * y[j],
                name=f"CoverBigM_{i}_{j}"
            )

    # Constraint 6: Budget — at most p vertiports
    model.addConstr(
        gp.quicksum(y[j] for j in vertiport_indices) <= p,
        name="Budget"
    )

    # Solve
    model.optimize()

    # --- Print Results ---
    print("\n" + "=" * 50)
    print("Solution Results")
    print("=" * 50)

    if model.status == GRB.OPTIMAL:
        selected_vps  = [j for j in vertiport_indices if y[j].X > 0.5]
        covered_pairs = [(i, j) for i in demand_indices for j in vertiport_indices if z[i, j].X > 0.5]
        assigned_pairs= [(i, j) for i in demand_indices for j in vertiport_indices if x[i, j].X > 0.5]

        # Show hub certification results
        print("\nHub certification (delta) values:")
        for j in vertiport_indices:
            certs = [f"H{h}" for h in hub_indices if delta[j, h].X > 0.5]
            print(f"  V{j}: certified by {certs}")

        total_ridership = sum(demands[i][2] for (i, j) in covered_pairs)
        total_cost      = cost_per_vp * len(selected_vps)
        total_dist      = sum(d[i, j] for (i, j) in assigned_pairs)
        total_demand    = sum(demands[i][2] for i in demand_indices)

        print(f"\nOptimal objective value: {model.ObjVal:.4f}")

        print("\nSelected vertiports:")
        for j in selected_vps:
            certs = [f"H{h}" for h in hub_indices if delta[j, h].X > 0.5]
            print(f"  V{j}: {vertiports[j]}  [certified by: {certs}]")

        print("\nDemand coverage:")
        for i in demand_indices:
            assigned_to = next(j for (ii, jj) in assigned_pairs if ii == i for j in [jj])
            covered_by  = [j for (ii, jj) in covered_pairs if ii == i for j in [jj]]
            q = demands[i][2]
            cov_str = f"covered by V{covered_by}" if covered_by else "not covered"
            print(f"  D{i} (ridership={q}): assigned to V{assigned_to}  |  {cov_str}")

        print(f"\ng1 (ridership covered): {total_ridership}")
        print(f"g2 (facility cost):     {total_cost} hundred USD")
        print(f"g3 (travel distance):   {total_dist:.4f} km")
        print(f"Coverage rate: {total_ridership}/{total_demand} ({100*total_ridership/total_demand:.1f}%)")

        result = {
            "status": "optimal",
            "obj": round(model.ObjVal, 2)
        }
        print(f"\nResult: {result}")
        return result
    else:
        print("Model infeasible or not optimal!")
        result = {"status": "infeasible"}
        print(f"\nResult: {result}")
        return result


if __name__ == "__main__":
    result = solve_vertiport_location()
    with open("answer.json", "w") as f:
        json.dump(result, f, indent=4)
    print(f"\nSaved answer.json: {result}")
