import gurobipy as gp
from gurobipy import GRB
import math


def solve_uam_vertiport_network():
    # ========================================
    # 1. PARAMETERS
    # ========================================
    P = 3
    D_max = 15.0
    alpha = 3.5

    locations = {
        1: (2, 8), 2: (5, 12), 3: (8, 6), 4: (12, 10), 5: (15, 14),
        6: (18, 8), 7: (10, 16), 8: (6, 4), 9: (14, 4), 10: (20, 12)
    }

    construction_cost = {
        1: 2500000, 2: 2800000, 3: 2300000, 4: 2600000, 5: 2900000,
        6: 2400000, 7: 3000000, 8: 2200000, 9: 2350000, 10: 2700000
    }

    demands = {
        1: {'loc': (3, 10), 'demand': 50000},
        2: {'loc': (7, 8), 'demand': 80000},
        3: {'loc': (11, 13), 'demand': 60000},
        4: {'loc': (16, 10), 'demand': 70000},
        5: {'loc': (9, 5), 'demand': 55000},
        6: {'loc': (13, 7), 'demand': 65000},
        7: {'loc': (5, 15), 'demand': 45000},
        8: {'loc': (19, 9), 'demand': 75000}
    }

    J_prime = list(locations.keys())
    I = list(demands.keys())

    def euclidean_distance(loc1, loc2):
        return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

    # Distance between candidate locations
    dist_loc = {}
    for j in J_prime:
        for k in J_prime:
            if j != k:
                dist_loc[j, k] = euclidean_distance(locations[j], locations[k])

    # Distance from demand points to candidate locations
    dist_demand = {}
    for i in I:
        for j in J_prime:
            dist_demand[i, j] = euclidean_distance(demands[i]['loc'], locations[j])

    # Feasible directed arcs (within D_max)
    feasible_arcs = [(j, k) for j in J_prime for k in J_prime
                     if j != k and dist_loc[j, k] <= D_max]

    # Pre-compute incoming arcs for each node
    in_arcs = {k: [] for k in J_prime}
    for j, k in feasible_arcs:
        in_arcs[k].append((j, k))

    BigM_ord = P  # Big-M constant for MTZ ordering

    # ========================================
    # 2. CREATE MODEL
    # ========================================
    model = gp.Model("UAM_BigM_MTZ")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 300)

    # ========================================
    # 3. DECISION VARIABLES
    # ========================================

    # X[j]: open vertiport at location j
    X = model.addVars(J_prime, vtype=GRB.BINARY, name="X")

    # Y[i,j]: assign demand i to vertiport j
    Y = model.addVars(I, J_prime, vtype=GRB.BINARY, name="Y")

    # a[j,k]: directed arc in spanning arborescence
    a_arc = model.addVars(feasible_arcs, vtype=GRB.BINARY, name="arc")

    # r[j]: root indicator
    root = model.addVars(J_prime, vtype=GRB.BINARY, name="root")

    # ord[j]: MTZ ordering variable
    ord_var = model.addVars(J_prime, vtype=GRB.CONTINUOUS, lb=0, ub=P - 1,
                            name="ord")

    # ========================================
    # 4. OBJECTIVE FUNCTION
    # ========================================
    model.setObjective(
        alpha * gp.quicksum(
            demands[i]['demand'] * dist_demand[i, j] * Y[i, j]
            for i in I for j in J_prime
        ) +
        gp.quicksum(construction_cost[j] * X[j] for j in J_prime),
        GRB.MINIMIZE
    )

    # ========================================
    # 5. CONSTRAINTS
    # ========================================

    # C1: Exactly P vertiports
    model.addConstr(
        gp.quicksum(X[j] for j in J_prime) == P,
        name="facility_selection"
    )

    # C2: Each demand assigned to exactly one vertiport
    for i in I:
        model.addConstr(
            gp.quicksum(Y[i, j] for j in J_prime) == 1,
            name=f"demand_{i}"
        )

    # C3: Assignment only to selected vertiports
    for i in I:
        for j in J_prime:
            model.addConstr(Y[i, j] <= X[j], name=f"assign_feas_{i}_{j}")

    # =========================================
    # Big-M MTZ Connectivity Constraints
    # =========================================

    # C4: Total arcs = P - 1
    model.addConstr(
        gp.quicksum(a_arc[j, k] for j, k in feasible_arcs) == P - 1,
        name="total_arcs"
    )

    # C5: Arc only between selected vertiports
    for j, k in feasible_arcs:
        model.addConstr(a_arc[j, k] <= X[j], name=f"arc_src_{j}_{k}")
        model.addConstr(a_arc[j, k] <= X[k], name=f"arc_dst_{j}_{k}")

    # C6: Incoming arc balance (non-root: 1, root: 0)
    for k in J_prime:
        model.addConstr(
            gp.quicksum(a_arc[arc] for arc in in_arcs[k]) == X[k] - root[k],
            name=f"incoming_{k}"
        )

    # C7: Exactly one root
    model.addConstr(
        gp.quicksum(root[j] for j in J_prime) == 1,
        name="one_root"
    )

    # C8: Root must be selected
    for j in J_prime:
        model.addConstr(root[j] <= X[j], name=f"root_sel_{j}")

    # C9: MTZ ordering (Big-M)
    for j, k in feasible_arcs:
        model.addConstr(
            ord_var[k] >= ord_var[j] + 1 - BigM_ord * (1 - a_arc[j, k]),
            name=f"mtz_{j}_{k}"
        )

    # C10: Ordering bounds
    for j in J_prime:
        model.addConstr(
            ord_var[j] <= (P - 1) * X[j],
            name=f"ord_ub_{j}"
        )

    # ========================================
    # 6. SOLVE
    # ========================================
    model.optimize()

    if model.status == GRB.OPTIMAL:
        result = {"status": "optimal", "obj": round(model.ObjVal, 2)}

        print(f"\nOptimal Total Cost: ${model.ObjVal:,.2f}")

        selected = [j for j in J_prime if X[j].X > 0.5]
        constr_total = sum(construction_cost[j] for j in selected)
        transp_total = model.ObjVal - constr_total

        print(f"  Construction: ${constr_total:,.2f}")
        print(f"  Transportation: ${transp_total:,.2f}")

        print(f"\nSelected Vertiports:")
        for j in selected:
            is_root = root[j].X > 0.5
            print(f"  L{j}: {locations[j]}, ${construction_cost[j]:,}"
                  f"{' [ROOT]' if is_root else ''}, ord={ord_var[j].X:.1f}")

        print(f"\nArborescence Arcs:")
        for j, k in feasible_arcs:
            if a_arc[j, k].X > 0.5:
                print(f"  L{j} -> L{k} ({dist_loc[j, k]:.1f} km)")

        print(f"\nDemand Assignments:")
        for i in I:
            for j in J_prime:
                if Y[i, j].X > 0.5:
                    print(f"  D{i} -> L{j}: {dist_demand[i, j]:.2f} km, "
                          f"{demands[i]['demand']:,} pax")

        return result
    else:
        print(f"Status: {model.status}")
        return {"status": "infeasible"}


if __name__ == "__main__":
    result = solve_uam_vertiport_network()
    print(f"\nResult: {result}")
