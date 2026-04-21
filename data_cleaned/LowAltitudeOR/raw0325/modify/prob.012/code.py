import gurobipy as gp
from gurobipy import GRB
import math
import json

def solve_drone_network_design():
    """
    Solve the Drone Network Design Problem for Opioid Overdose Response.
    Minimize average response time considering queueing delays and flight times.
    """
    
    # --- 1. Data Input ---
    
    # Candidate drone base locations: ID -> (x, y)
    bases = {
        'DB1': (10, 20),
        'DB2': (30, 15),
        'DB3': (25, 40),
        'DB4': (50, 25)
    }
    
    # OTR locations: ID -> (x, y, arrival_rate)
    otrs = {
        'O1': (12, 18, 0.8),
        'O2': (28, 20, 1.2),
        'O3': (35, 12, 0.6),
        'O4': (22, 38, 0.9),
        'O5': (48, 30, 0.7),
        'O6': (52, 22, 1.0)
    }
    
    # Parameters
    p = 5           # Total drones available
    q = 4           # Maximum drone bases to open
    M = 2           # Maximum drones per base
    v = 30.0        # Drone speed (units/hour)
    r = 25.0        # Drone coverage radius
    E_xi = 0.4      # Expected non-travel service time (hours)
    epsilon = 0.01  # Small value for stability constraint
    
    # Sets
    J = list(bases.keys())  # Drone bases
    I = list(otrs.keys())   # OTR locations
    
    # --- 2. Pre-processing ---
    
    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    # Calculate distances and coverage
    d = {}  # d[i,j] = distance from base j to OTR i
    E_S = {}  # E_S[i,j] = expected service time
    
    # Coverage sets
    I_j = {j: [] for j in J}  # OTRs within coverage of base j
    J_i = {i: [] for i in I}  # Bases that can cover OTR i
    
    for j in J:
        for i in I:
            dist = euclidean_distance(bases[j], (otrs[i][0], otrs[i][1]))
            d[i, j] = dist
            E_S[i, j] = dist / v + E_xi
            if dist <= r:
                I_j[j].append(i)
                J_i[i].append(j)
    
    # Arrival rates
    lam = {i: otrs[i][2] for i in I}
    total_lambda = sum(lam.values())
    
    # --- 3. Optimization Model ---
    model = gp.Model("Drone_Network_Design")
    model.setParam('TimeLimit', 300)
    model.setParam('NonConvex', 2)
    
    # Decision Variables
    
    # x[j]: 1 if base j is opened
    x = model.addVars(J, vtype=GRB.BINARY, name="x")
    
    # y[i,j]: 1 if OTR i is assigned to base j
    y = {}
    for i in I:
        for j in J_i[i]:
            y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
    
    # gamma[j,m]: 1 if exactly m drones are placed at base j
    gamma = {}
    for j in J:
        for m in range(1, M + 1):
            gamma[j, m] = model.addVar(vtype=GRB.BINARY, name=f"gamma_{j}_{m}")
    
    # Auxiliary variables for linearization
    # eta[j]: arrival rate at base j
    eta = model.addVars(J, vtype=GRB.CONTINUOUS, lb=0, name="eta")
    
    # weighted_service[j]: sum of lambda_i * y_ij * E[S_ij]
    ws = model.addVars(J, vtype=GRB.CONTINUOUS, lb=0, name="ws")
    
    # --- 4. Constraints ---
    
    # 1. Assignment constraint: each OTR assigned to exactly one base
    for i in I:
        if J_i[i]:
            model.addConstr(
                gp.quicksum(y[i, j] for j in J_i[i]) == 1,
                name=f"assign_{i}"
            )
        else:
            print(f"Warning: OTR {i} has no coverage!")
    
    # 2. Assignment only to opened bases
    for i in I:
        for j in J_i[i]:
            model.addConstr(y[i, j] <= x[j], name=f"open_{i}_{j}")
    
    # 3. Maximum bases constraint
    model.addConstr(gp.quicksum(x[j] for j in J) <= q, name="max_bases")
    
    # 3b. Airspace corridor conflict: DB1 and DB4 cannot both be open
    model.addConstr(x['DB1'] + x['DB4'] <= 1, name="airspace_conflict")
    
    # 4. Drone deployment constraint (using gamma)
    for j in J:
        # If base is open, exactly one gamma must be 1
        model.addConstr(
            gp.quicksum(gamma[j, m] for m in range(1, M + 1)) == x[j],
            name=f"drone_deploy_{j}"
        )
    
    # 5. Total drones constraint
    model.addConstr(
        gp.quicksum(m * gamma[j, m] for j in J for m in range(1, M + 1)) == p,
        name="total_drones"
    )
    
    # 6. Arrival rate definition
    for j in J:
        model.addConstr(
            eta[j] == gp.quicksum(lam[i] * y[i, j] for i in I_j[j]),
            name=f"eta_{j}"
        )
    
    # 7. Weighted service time definition
    for j in J:
        model.addConstr(
            ws[j] == gp.quicksum(lam[i] * y[i, j] * E_S[i, j] for i in I_j[j]),
            name=f"ws_{j}"
        )
    
    # 8. Stability constraint: ws[j] < K_j (number of drones at j)
    #    Only active when base is open (x[j]=1)
    for j in J:
        K_j = gp.quicksum(m * gamma[j, m] for m in range(1, M + 1))
        model.addConstr(ws[j] <= K_j - epsilon * x[j], name=f"stability_{j}")
    
    # --- 5. Objective Function ---
    # Minimize average response time
    # Flight time component
    flight_time = gp.quicksum(
        lam[i] * d[i, j] * y[i, j] / v / total_lambda
        for i in I for j in J_i[i]
    )
    
    # Queueing delay approximation
    queueing_penalty = 0.5 * gp.quicksum(ws[j] for j in J) / total_lambda
    
    model.setObjective(flight_time + queueing_penalty, GRB.MINIMIZE)
    
    # --- 6. Solve ---
    model.optimize()
    
    # --- 7. Output ---
    if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
        # Extract solution
        opened_bases = [j for j in J if x[j].X > 0.5]
        
        drones_at_base = {}
        for j in J:
            for m in range(1, M + 1):
                if gamma[j, m].X > 0.5:
                    drones_at_base[j] = m
        
        assignments = {}
        for i in I:
            for j in J_i[i]:
                if y[i, j].X > 0.5:
                    assignments[i] = j
        
        # Calculate actual response times
        total_response = 0
        for i in I:
            j = assignments[i]
            flight_t = d[i, j] / v
            total_response += lam[i] * flight_t
        avg_flight_time = total_response / total_lambda
        
        result = {
            "status": "optimal" if model.status == GRB.OPTIMAL else "feasible",
            "obj": round(model.ObjVal, 4),
            "avg_flight_time_hours": round(avg_flight_time, 4),
            "opened_bases": opened_bases,
            "drones_at_base": drones_at_base,
            "assignments": assignments
        }
        return result
    else:
        return {"status": "infeasible"}

if __name__ == "__main__":
    result = solve_drone_network_design()
    print(json.dumps(result, indent=4))
