# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB

def solve_munichair_stochastic_vot():
    # -------------------------
    # 1) Embedded data (from question/model; numeric route parameters are a consistent example)
    # -------------------------
    # Candidate vertiports
    N = ["V1", "V2", "V3"]

    # Must open exactly P=2 vertiports
    P = 2

    # Demand flows K (only one OD flow in the simplified question)
    K = ["Flow1"]
    Flow = {"Flow1": 100.0}  # pax

    # Scenarios S: VOT uncertainty
    S = ["Business", "Leisure"]
    Prob = {"Business": 0.5, "Leisure": 0.5}
    VOT = {"Business": 100.0, "Leisure": 20.0}  # $/hour

    # Feasible routing options (i -> j) for Flow1 per question narrative:
    # - Via V1 + V3  (interpreted as choosing link V1 -> V3)
    # - Via V2 + V3  (interpreted as choosing link V2 -> V3)
    # We only create x variables on these feasible links.
    L = [("V1", "V3"), ("V2", "V3")]

    # Route attributes (example numbers consistent with "fast but expensive" vs "slow but cheap")
    # You can replace these with your real numbers if you have them.
    ticket = {("V1", "V3"): 100.0, ("V2", "V3"): 60.0}   # $
    time_min = {("V1", "V3"): 20.0, ("V2", "V3"): 50.0}  # minutes

    # Precompute Generalized Cost GC[(i,j,k,s)] = ticket + (time_hr * VOT_s)
    GC = {}
    for (i, j) in L:
        for k in K:
            for s in S:
                time_hr = time_min[(i, j)] / 60.0
                GC[(i, j, k, s)] = ticket[(i, j)] + time_hr * VOT[s]

    # -------------------------
    # 2) Build model (ILP)
    # -------------------------
    m = gp.Model("MunichAir_StochasticVOT")

    # Decision variables
    # y_i = 1 if vertiport i is opened
    y = m.addVars(N, vtype=GRB.BINARY, name="y")

    # x_ijk = 1 if flow k is routed via link i->j
    x = m.addVars(L, K, vtype=GRB.BINARY, name="x")  # only feasible links

    # -------------------------
    # 3) Constraints (from model.txt)
    # -------------------------
    # (1) Hub constraint: open exactly P vertiports
    m.addConstr(gp.quicksum(y[i] for i in N) == P, name="open_exactly_P")

    # (2) Routing logic: can use i->j only if both endpoints are open
    for (i, j) in L:
        for k in K:
            m.addConstr(x[i, j, k] <= y[i], name=f"open_origin[{i},{j},{k}]")
            m.addConstr(x[i, j, k] <= y[j], name=f"open_dest[{i},{j},{k}]")

    # (3) Flow assignment: each flow chooses exactly one route
    for k in K:
        m.addConstr(gp.quicksum(x[i, j, k] for (i, j) in L) == 1, name=f"assign_one[{k}]")

    # -------------------------
    # 4) Objective: minimize expected total generalized cost
    # min Σ_s Prob_s Σ_k Σ_(i,j) Flow_k * GC_ijks * x_ijk
    # -------------------------
    obj = gp.quicksum(
        Prob[s] * gp.quicksum(
            Flow[k] * gp.quicksum(GC[(i, j, k, s)] * x[i, j, k] for (i, j) in L)
            for k in K
        )
        for s in S
    )
    m.setObjective(obj, GRB.MINIMIZE)

    # -------------------------
    # 5) Solve
    # -------------------------
    m.optimize()

    # -------------------------
    # 6) Output
    # -------------------------
    if m.status == GRB.OPTIMAL:
        open_ports = [i for i in N if y[i].X > 0.5]
        chosen = []
        for (i, j) in L:
            for k in K:
                if x[i, j, k].X > 0.5:
                    chosen.append((k, i, j))

        print("\n==============================")
        print("Optimal solution found")
        print("==============================")
        print(f"Objective (Expected Total GC) = {m.objVal:.6f}")
        print(f"Opened vertiports (P={P}): {open_ports}")
        print("Chosen routes:")
        for (k, i, j) in chosen:
            print(f"  {k}: {i} -> {j}")

        print("\n--- Scenario cost details (per passenger) ---")
        for s in S:
            for (k, i, j) in chosen:
                print(f"  {s}, {k}, {i}->{j}: GC = {GC[(i, j, k, s)]:.6f} $/pax")

        print("\n--- Scenario total cost (all passengers) ---")
        for s in S:
            total_s = sum(Flow[k] * GC[(i, j, k, s)] for (k, i, j) in chosen)
            print(f"  {s}: total = {total_s:.6f}, weighted = {Prob[s]*total_s:.6f}")

    else:
        print(f"No optimal solution. Status = {m.status}")

if __name__ == "__main__":
    solve_munichair_stochastic_vot()
