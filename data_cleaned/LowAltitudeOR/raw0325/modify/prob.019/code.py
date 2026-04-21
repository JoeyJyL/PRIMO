import gurobipy as gp
from gurobipy import GRB

def solve_bilevel_network():
    # --- 1. Data Input ---
    # Nodes: 0=S, 1=D1, 2=D2
    nodes = [0, 1, 2]
    demands = {1: 10, 2: 10} 
    
    # Candidate Edges: (u, v) -> dist
    edges = {
        (0, 1): 10.0,    # S->D1
        (0, 2): 14.14,   # S->D2
        (1, 2): 10.0     # D1->D2
    }
    
    # Parameters
    # Increased alpha to emphasize operational efficiency
    alpha_ops = 0.5
    BigM = 1000
    
    # --- 2. Model ---
    model = gp.Model("BiLevel_UAV")
    
    # Variables
    y = model.addVars(edges.keys(), vtype=GRB.BINARY, name="OpenLink")
    x = model.addVars(edges.keys(), demands.keys(), vtype=GRB.CONTINUOUS, name="Flow")
    
    # Objective
    obj_len = gp.quicksum(edges[e] * y[e] for e in edges)
    obj_ops = gp.quicksum(edges[(u,v)] * x[u,v,k] for (u,v) in edges for k in demands)
    
    model.setObjective(obj_len + alpha_ops * obj_ops, GRB.MINIMIZE)
    
    # Constraints
    
    # 1. Topology (Big-M)
    for (u,v) in edges:
        model.addConstr(gp.quicksum(x[u,v,k] for k in demands) <= BigM * y[u,v], name=f"Build_{u}_{v}")
        
    # 2. Flow Conservation
    for k in demands:
        for n in nodes:
            flow_out = gp.quicksum(x[n,j,k] for j in nodes if (n,j) in edges)
            flow_in  = gp.quicksum(x[j,n,k] for j in nodes if (j,n) in edges)
            
            if n == 0: # Source
                rhs = demands[k]
            elif n == k: # Sink
                rhs = -demands[k]
            else: # Transit
                rhs = 0
            model.addConstr(flow_out - flow_in == rhs, name=f"FlowBal_{n}_{k}")
            
    # --- 3. Solve ---
    model.optimize()
    
    # --- 4. Output ---
    if model.status == GRB.OPTIMAL:
        open_links = [str(e) for e in edges if y[e].X > 0.5]
        
        # Calculate real values
        net_len = sum(edges[e] for e in edges if y[e].X > 0.5)
        ops_cost = sum(edges[(u,v)] * x[u,v,k].X for (u,v) in edges for k in demands)
        
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "built_links": open_links,
            "details": {
                "network_length": net_len,
                "operational_cost": ops_cost
            }
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_bilevel_network())
