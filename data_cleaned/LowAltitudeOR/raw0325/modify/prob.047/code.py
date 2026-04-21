import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_uam_network_design():
    # --- Parameter Setup ---
    w       = 0.1   # cost-benefit valuation factor
    budget  = 4.0   # total construction budget (reduced from 6 to 4)
    p_dis   = 0.8   # network disturbance probability
    M_big   = 1e4   # big-M for demand upper bounds

    # --- Original Network (Example 1 from paper, Fig. 3) ---
    # Nodes: v1=0, v2=1, v3=2, v4=3  (0-indexed)
    # Links: e1(0->1), e2(0->2), e3(1->3), e4(2->3)  (0-indexed)
    n_v = 4; n_e = 4; n_s = 3
    C_v0 = [10.0, 15.0, 15.0, 10.0]
    C_e0 = [8.0,  4.0,  4.0,  8.0]
    # Incidence matrix: E[v][e] = -1 if e leaves v, +1 if e enters v, 0 otherwise
    E = [[-1,-1,0,0],[1,0,-1,0],[0,1,0,-1],[0,0,1,1]]
    E_plus = [[abs(E[v][e]) for e in range(n_e)] for v in range(n_v)]
    # Delta1[v][s] = 1 if v is destination of demand s
    Delta1 = [[0,0,0],[1,0,0],[0,0,0],[0,1,1]]
    # Delta2[v][s] = -1 if v is origin of demand s
    Delta2 = [[-1,-1,0],[0,0,0],[0,0,-1],[0,0,0]]

    # --- Backup Node v5 ---
    n_b = 1; n_Z = 3
    C_b_all = [[0.0, 1.0, 2.0]]
    F_cost  = [[0.0, 4.0, 6.0]]
    Delta_adj = [[0],[0],[0],[1]]  # v5 is adjacent to v4 (index 3)

    # --- Disruption Scenarios ---
    all_scenarios = [
        ("node",0,5.0,0.05),("node",0,0.0,0.05),
        ("node",1,10.0,0.10),("node",1,5.0,0.05),
        ("node",2,10.0,0.10),("node",2,5.0,0.05),
        ("node",3,5.0,0.05),("node",3,0.0,0.05),
        ("link",0,4.0,0.05),("link",0,0.0,0.05),
        ("link",1,2.0,0.05),("link",2,2.0,0.05),
        ("link",3,4.0,0.05),("link",3,0.0,0.05),
    ]

    def solve_tp_lp(Ce, Cv, ne, nv, ns):
        """Solve throughput LP for given capacities"""
        m = gp.Model(); m.setParam("OutputFlag",0)
        X  = m.addVars(ne, ns, lb=0, name="X")
        D1 = m.addVars(nv, ns, lb=0, name="D1")
        D2 = m.addVars(nv, ns, lb=0, name="D2")
        
        # Maximize total fulfilled demand at destinations
        m.setObjective(gp.quicksum(D1[v,s] for v in range(nv) for s in range(ns)), GRB.MAXIMIZE)
        
        # Flow conservation: inflow - outflow = demand fulfilled - demand generated
        for v in range(nv):
            for s in range(ns):
                m.addConstr(gp.quicksum(E[v][e]*X[e,s] for e in range(ne)) == D1[v,s] - D2[v,s])
        
        # Link capacity constraints
        for e in range(ne):
            m.addConstr(gp.quicksum(X[e,s] for s in range(ns)) <= Ce[e])
        
        # Node capacity constraints (total flow through node)
        for v in range(nv):
            m.addConstr(gp.quicksum(E_plus[v][e]*X[e,s] for e in range(ne) for s in range(ns)) <= Cv[v])
        
        # Total demand fulfilled = total demand generated (flow balance)
        for s in range(ns):
            m.addConstr(gp.quicksum(D1[v,s] for v in range(nv)) == gp.quicksum(D2[v,s] for v in range(nv)))
        
        # Demand can only be fulfilled at destinations
        for v in range(nv):
            for s in range(ns):
                m.addConstr(D1[v,s] <= Delta1[v][s]*M_big)
        
        # Demand can only be generated at origins
        for v in range(nv):
            for s in range(ns):
                m.addConstr(D2[v,s] <= -Delta2[v][s]*M_big)
        
        m.optimize()
        return m.ObjVal if m.Status==GRB.OPTIMAL else 0.0

    S_und = solve_tp_lp(C_e0, C_v0, n_e, n_v, n_s)

    print("="*55); print("Preprocessing Info"); print("="*55)
    print(f"\nUndisturbed throughput S*(N(0,0)) = {S_und:.2f}")
    print(f"Disruption scenarios: {len(all_scenarios)}")
    print(f"Backup v5: capacities={C_b_all[0]}, costs={F_cost[0]}")
    print(f"Budget={budget}, w={w}")

    print("\n"+"="*55); print("Solving MILP for Backup Vertiport Design"); print("="*55)

    model = gp.Model("UAM_NetworkDesign")
    model.setParam("OutputFlag", 0)

    # Binary variables for backup capacity selection
    Z = model.addVars(n_b, n_Z, vtype=GRB.BINARY, name="Z")
    cap_b = model.addVars(n_b, lb=0, name="cap_b")
    for b in range(n_b):
        model.addConstr(cap_b[b] == gp.quicksum(C_b_all[b][m]*Z[b,m] for m in range(n_Z)))

    # Variables for each disruption scenario
    X_sc = {}; D1_sc = {}; D2_sc = {}
    for idx, sc in enumerate(all_scenarios):
        sc_type, sc_idx, sc_cap, _ = sc
        X_sc[idx]  = model.addVars(n_e, n_s, lb=0, name=f"X_{idx}")
        D1_sc[idx] = model.addVars(n_v, n_s, lb=0, name=f"D1_{idx}")
        D2_sc[idx] = model.addVars(n_v, n_s, lb=0, name=f"D2_{idx}")
        
        # Set disrupted capacities
        Ce_sc = list(C_e0); Cv_sc = list(C_v0)
        if sc_type=="node": 
            Cv_sc[sc_idx]=sc_cap
        else: 
            Ce_sc[sc_idx]=sc_cap
        
        # Flow conservation
        for v in range(n_v):
            for s in range(n_s):
                model.addConstr(gp.quicksum(E[v][e]*X_sc[idx][e,s] for e in range(n_e)) == D1_sc[idx][v,s] - D2_sc[idx][v,s])
        
        # Link capacity constraints
        for e in range(n_e):
            model.addConstr(gp.quicksum(X_sc[idx][e,s] for s in range(n_s)) <= Ce_sc[e])
        
        # Node capacity constraints with backup
        for v in range(n_v):
            backup_add = gp.quicksum(Delta_adj[v][b]*cap_b[b] for b in range(n_b))
            if sc_type=="node" and sc_idx==v:
                # Disrupted node gets backup capacity
                model.addConstr(gp.quicksum(E_plus[v][e]*X_sc[idx][e,s] for e in range(n_e) for s in range(n_s)) <= sc_cap + backup_add)
            else:
                model.addConstr(gp.quicksum(E_plus[v][e]*X_sc[idx][e,s] for e in range(n_e) for s in range(n_s)) <= Cv_sc[v])
        
        # Flow balance
        for s in range(n_s):
            model.addConstr(gp.quicksum(D1_sc[idx][v,s] for v in range(n_v)) == gp.quicksum(D2_sc[idx][v,s] for v in range(n_v)))
        
        # Demand fulfillment constraints
        for v in range(n_v):
            for s in range(n_s):
                model.addConstr(D1_sc[idx][v,s] <= Delta1[v][s]*M_big)
                model.addConstr(D2_sc[idx][v,s] <= -Delta2[v][s]*M_big)

    # Capacity selection constraints
    for b in range(n_b):
        model.addConstr(gp.quicksum(Z[b,m] for m in range(n_Z))==1)
    
    # Budget constraint (tightened: f=4 instead of 6)
    model.addConstr(gp.quicksum(F_cost[b][m]*Z[b,m] for b in range(n_b) for m in range(n_Z)) <= budget)

    # Mandatory backup deployment: "not built" option (Z[b,0]=1) is prohibited
    for b in range(n_b):
        model.addConstr(Z[b,0] == 0, name=f"mandatory_backup_{b}")

    # Objective: maximize expected throughput minus construction cost
    exp_tp = (1-p_dis)*S_und + gp.quicksum(
        all_scenarios[idx][3]*gp.quicksum(D1_sc[idx][v,s] for v in range(n_v) for s in range(n_s))
        for idx in range(len(all_scenarios)))
    cost_expr = gp.quicksum(F_cost[b][m]*Z[b,m] for b in range(n_b) for m in range(n_Z))
    model.setObjective(exp_tp - w*cost_expr, GRB.MAXIMIZE)
    
    model.optimize()

    print("\n"+"="*55); print("Solution Results"); print("="*55)
    if model.status == GRB.OPTIMAL:
        Z_val = [[Z[b,m].X for m in range(n_Z)] for b in range(n_b)]
        sel_caps  = [C_b_all[b][int(np.argmax(Z_val[b]))] for b in range(n_b)]
        sel_costs = [F_cost[b][int(np.argmax(Z_val[b]))]  for b in range(n_b)]
        total_cost = sum(sel_costs)
        E_dis = sum(all_scenarios[idx][3]*sum(D1_sc[idx][v,s].X for v in range(n_v) for s in range(n_s)) for idx in range(len(all_scenarios)))
        E_total = (1-p_dis)*S_und + E_dis
        
        print(f"\nOptimal objective value: {model.ObjVal:.4f}")
        print(f"\nBackup vertiport design:")
        for b in range(n_b):
            print(f"  v{4+b+1}: capacity={sel_caps[b]:.0f},  cost={sel_costs[b]:.0f}")
        print(f"  Total construction cost={total_cost:.0f}  (budget={budget:.0f})")
        print(f"\n  Undisturbed contribution: {(1-p_dis)*S_und:.4f}")
        print(f"  Disturbed contribution:   {E_dis:.4f}")
        print(f"  Total expected throughput:{E_total:.4f}")
        print(f"  Cost penalty (w*cost):    {w*total_cost:.4f}")
        print("\n  Per-scenario throughputs:")
        for idx,sc in enumerate(all_scenarios):
            tp_val=sum(D1_sc[idx][v,s].X for v in range(n_v) for s in range(n_s))
            print(f"    {sc[0]}[{sc[1]}]={sc[2]:.0f} (p={sc[3]}): S*={tp_val:.2f}")
        return {"status":"optimal","obj":round(model.ObjVal,4)}
    else:
        print("No optimal solution found"); 
        return {"status":"infeasible"}

if __name__ == "__main__":
    result = solve_uam_network_design()
    print("\n"+"="*55)
    print(f"Final result: {result}")
