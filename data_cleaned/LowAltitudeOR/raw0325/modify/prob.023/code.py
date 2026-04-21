import gurobipy as gp
from gurobipy import GRB
import math

def solve_latency_lrp():
    # --- 1. Data Input ---
    # Nodes: 0=FC1, 1=FC2, 2=C1, 3=C2, 4=C3, 5=C4
    # FC1(0,0), FC2(10,0)
    # C1(1,0), C2(2,0), C3(9,0), C4(8,0)
    
    depots = {
        0: {'loc': (0, 0), 'fixed': 20},
        1: {'loc': (10, 0), 'fixed': 20}
    }
    
    customers = {
        2: {'loc': (1, 0)},
        3: {'loc': (2, 0)},
        4: {'loc': (9, 0)},
        5: {'loc': (8, 0)}
    }
    
    # Combined Nodes
    all_nodes = {**{k: v['loc'] for k,v in depots.items()}, 
                 **{k: v['loc'] for k,v in customers.items()}}
    node_ids = list(all_nodes.keys())
    depot_ids = list(depots.keys())
    cust_ids = list(customers.keys())
    
    # Worst-Case Travel Time (Dist / Speed=1)
    def get_time(i, j):
        p1, p2 = all_nodes[i], all_nodes[j]
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
    times = {(i,j): get_time(i,j) for i in node_ids for j in node_ids}
    
    BigM = 1000
    
    # --- 2. Model ---
    model = gp.Model("Latency_LRP")
    
    # Variables
    y = model.addVars(depot_ids, vtype=GRB.BINARY, name="OpenDepot")
    x = model.addVars(node_ids, node_ids, vtype=GRB.BINARY, name="Route")
    t = model.addVars(node_ids, vtype=GRB.CONTINUOUS, name="ArrivalTime")
    
    # Objective: Fixed Cost + Sum of Arrival Times (Latency)
    obj_fixed = gp.quicksum(depots[d]['fixed'] * y[d] for d in depot_ids)
    obj_latency = gp.quicksum(t[c] for c in cust_ids)
    
    model.setObjective(obj_fixed + obj_latency, GRB.MINIMIZE)
    
    # Constraints
    
    # 1. Degree Constraints (Single visit per customer)
    for c in cust_ids:
        model.addConstr(gp.quicksum(x[i, c] for i in node_ids if i != c) == 1, name=f"In_{c}")
        model.addConstr(gp.quicksum(x[c, j] for j in node_ids if j != c) == 1, name=f"Out_{c}")
        
    # 2. Depot Logic
    # Drones can only leave opened depots
    for d in depot_ids:
        model.addConstr(gp.quicksum(x[d, j] for j in cust_ids) <= len(cust_ids) * y[d], name=f"DepotOpen_{d}")
        # Flow balance at depot (Out = In)
        model.addConstr(gp.quicksum(x[d, j] for j in cust_ids) == gp.quicksum(x[j, d] for j in cust_ids), name=f"DepotBal_{d}")

    # 3. Time Propagation (MTZ-like but for Latency)
    # t[j] >= t[i] + time[i,j] if x[i,j]=1
    # For Depot start:
    for d in depot_ids:
        model.addConstr(t[d] == 0, name=f"StartTime_{d}")
        
    for i in node_ids:
        for j in node_ids:
            if i != j and j not in depot_ids:
                model.addConstr(t[j] >= t[i] + times[i, j] - BigM * (1 - x[i, j]), name=f"TimeProp_{i}_{j}")

    # 4. Emergency Response Guarantee: max latency per customer <= 5
    for c in cust_ids:
        model.addConstr(t[c] <= 5, name=f"MaxLatency_{c}")
                
    # 5. No Inter-Depot
    for d1 in depot_ids:
        for d2 in depot_ids:
            model.addConstr(x[d1, d2] == 0)
            
    # --- 3. Solve ---
    model.optimize()
    
    # --- 4. Output ---
    if model.status == GRB.OPTIMAL:
        return {
            "status": "optimal",
            "obj": round(model.ObjVal, 2),
            "open_fc": [d for d in depot_ids if y[d].X > 0.5],
            "latencies": {c: round(t[c].X, 2) for c in cust_ids}
        }
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    print(solve_latency_lrp())