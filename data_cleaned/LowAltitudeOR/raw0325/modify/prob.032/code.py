import gurobipy as gp
from gurobipy import GRB
import math
import json
 
def solve_skyport_location():
    """
    Solves a simplified Air Taxi Skyport Location Problem with mutual exclusion.
    Skyports at locations 2 and 5 cannot both be opened (airspace conflict).
    """
 
    # ================================================================
    # 1. DATA INPUT
    # ================================================================
 
    # --- Sets ---
    origins = [1, 2, 3, 4, 5]          # L: origin zones
    airports = ['A', 'B']               # J: destination airports
    candidates = [1, 2, 3, 4, 5]        # Candidate skyport locations (same as origins)
    p = 2                                # Number of skyports to open
 
    # --- Demand D_ij: monthly trips from origin i to airport j ---
    demand = {
        (1, 'A'): 150,  (1, 'B'): 100,
        (2, 'A'): 200,  (2, 'B'): 180,
        (3, 'A'): 120,  (3, 'B'): 160,
        (4, 'A'): 180,  (4, 'B'): 130,
        (5, 'A'): 100,  (5, 'B'): 220,
    }
 
    # --- Logit model parameters (from the paper) ---
    beta_TT_ground = 0.0313      # trip time coeff for ground taxi
    beta_CO_ground = -0.0125     # trip cost coeff for ground taxi
    beta_TL_air    = 0.018       # trip distance coeff for air taxi
    beta_CO_air    = -0.0213     # trip cost coeff for air taxi
 
    # --- Ground taxi direct: trip time c_ij (min), distance d_ij (miles) ---
    ground_direct_time = {
        (1, 'A'): 40, (1, 'B'): 55,
        (2, 'A'): 50, (2, 'B'): 35,
        (3, 'A'): 45, (3, 'B'): 60,
        (4, 'A'): 30, (4, 'B'): 50,
        (5, 'A'): 65, (5, 'B'): 25,
    }
    ground_direct_dist = {
        (1, 'A'): 12, (1, 'B'): 18,
        (2, 'A'): 15, (2, 'B'): 10,
        (3, 'A'): 14, (3, 'B'): 20,
        (4, 'A'):  9, (4, 'B'): 16,
        (5, 'A'): 22, (5, 'B'):  8,
    }
 
    # --- Ground access: trip time c_ik (min), distance d_ik (miles) ---
    ground_access_time = {
        (1,1): 0,  (1,2): 15, (1,3): 20, (1,4): 10, (1,5): 25,
        (2,1): 15, (2,2): 0,  (2,3): 12, (2,4): 18, (2,5): 22,
        (3,1): 20, (3,2): 12, (3,3): 0,  (3,4): 15, (3,5): 18,
        (4,1): 10, (4,2): 18, (4,3): 15, (4,4): 0,  (4,5): 20,
        (5,1): 25, (5,2): 22, (5,3): 18, (5,4): 20, (5,5): 0,
    }
    ground_access_dist = {
        (1,1): 0.0, (1,2): 5.0, (1,3): 7.0, (1,4): 3.0, (1,5): 9.0,
        (2,1): 5.0, (2,2): 0.0, (2,3): 4.0, (2,4): 6.0, (2,5): 8.0,
        (3,1): 7.0, (3,2): 4.0, (3,3): 0.0, (3,4): 5.0, (3,5): 6.0,
        (4,1): 3.0, (4,2): 6.0, (4,3): 5.0, (4,4): 0.0, (4,5): 7.0,
        (5,1): 9.0, (5,2): 8.0, (5,3): 6.0, (5,4): 7.0, (5,5): 0.0,
    }
 
    # --- Aerial distance d_kj (air miles) from skyport k to airport j ---
    aerial_dist = {
        (1, 'A'): 8.0,  (1, 'B'): 12.0,
        (2, 'A'): 10.0, (2, 'B'): 7.0,
        (3, 'A'): 9.0,  (3, 'B'): 13.0,
        (4, 'A'): 6.0,  (4, 'B'): 11.0,
        (5, 'A'): 14.0, (5, 'B'): 5.0,
    }
 
    # --- Fare parameters ---
    BaseFee = 3.0       # USD
    R_mile  = 1.5       # USD per mile
    R_minute = 0.3      # USD per minute
    R_airmile = 5.73    # USD per air mile (Short Term scenario)
    tt = 15              # total transfer time (minutes)
 
    # ================================================================
    # 2. PRE-PROCESSING: Compute choice probabilities & fares
    # ================================================================
 
    # Pre-compute ground taxi fare f_ij
    fare_ground = {}
    for (i, j) in ground_direct_time:
        fare_ground[(i, j)] = BaseFee + R_mile * ground_direct_dist[(i, j)] + R_minute * ground_direct_time[(i, j)]
 
    # Pre-compute access fare f_ik, air fare f_kj, total air fare f_ikj
    fare_access = {}
    for (i, k) in ground_access_time:
        fare_access[(i, k)] = BaseFee + R_mile * ground_access_dist[(i, k)] + R_minute * ground_access_time[(i, k)]
 
    fare_flight = {}
    for (k, j) in aerial_dist:
        fare_flight[(k, j)] = R_airmile * aerial_dist[(k, j)]
 
    # Total air taxi fare f_ikj = f_ik + t_k + f_kj
    transfer_cost = R_minute * tt
 
    fare_air_total = {}
    for i in origins:
        for k in candidates:
            for j in airports:
                fare_air_total[(i, k, j)] = fare_access[(i, k)] + transfer_cost + fare_flight[(k, j)]
 
    # Pre-compute choice probabilities theta_ikj using binary logit
    theta = {}
    for i in origins:
        for k in candidates:
            for j in airports:
                V_ground = beta_TT_ground * ground_direct_time[(i, j)] + beta_CO_ground * fare_ground[(i, j)]
                V_air = beta_TL_air * aerial_dist[(k, j)] + beta_CO_air * fare_air_total[(i, k, j)]
                exp_air = math.exp(V_air)
                exp_gnd = math.exp(V_ground)
                theta[(i, k, j)] = exp_air / (exp_air + exp_gnd)
 
    # Revenue coefficient for REV model
    revenue_coeff = {}
    for i in origins:
        for k in candidates:
            for j in airports:
                total_fare_ikj = fare_access[(i, k)] + fare_flight[(k, j)]
                revenue_coeff[(i, k, j)] = total_fare_ikj * theta[(i, k, j)] * demand[(i, j)]
 
    # ================================================================
    # 3. OPTIMIZATION MODEL (REV: Revenue Maximization)
    # ================================================================
 
    model = gp.Model("Skyport_Location_REV")
    model.setParam('OutputFlag', 0)
 
    # Decision variables
    y = model.addVars(candidates, vtype=GRB.BINARY, name="y")
    x = model.addVars(
        [(i, k, j) for i in origins for k in candidates for j in airports],
        vtype=GRB.BINARY, name="x"
    )
 
    # Objective: Maximize total air taxi revenue
    model.setObjective(
        gp.quicksum(revenue_coeff[(i, k, j)] * x[i, k, j]
                     for i in origins for k in candidates for j in airports),
        GRB.MAXIMIZE
    )
 
    # Constraint 1: Single allocation
    for i in origins:
        for j in airports:
            model.addConstr(
                gp.quicksum(x[i, k, j] for k in candidates) == 1,
                name=f"SingleAlloc_{i}_{j}"
            )
 
    # Constraint 2: Route activation
    for i in origins:
        for k in candidates:
            for j in airports:
                model.addConstr(
                    x[i, k, j] <= y[k],
                    name=f"LinkHub_{i}_{k}_{j}"
                )
 
    # Constraint 3: Budget — exactly p skyports
    model.addConstr(
        gp.quicksum(y[k] for k in candidates) == p,
        name="Budget"
    )
 
    # Constraint 4: Mutual Exclusion — skyports 1 and 2 cannot both be opened
    model.addConstr(
        y[1] + y[2] <= 1,
        name="MutualExclusion_1_2"
    )
 
    # ================================================================
    # 4. SOLVE
    # ================================================================
    model.optimize()
 
    # ================================================================
    # 5. OUTPUT
    # ================================================================
    if model.status == GRB.OPTIMAL:
        result = {
            "status": "optimal",
            "obj": round(model.ObjVal, 2),
        }
 
        print(f"Optimal Objective (Revenue): ${result['obj']:.2f}")
        print(f"\nOpen Skyports:")
        for k in candidates:
            if y[k].X > 0.5:
                print(f"  Skyport at location {k}: OPEN")
 
        print(f"\nDemand Allocations (x_ikj = 1):")
        for i in origins:
            for j in airports:
                for k in candidates:
                    if x[i, k, j].X > 0.5:
                        rid = theta[(i, k, j)] * demand[(i, j)]
                        rev = revenue_coeff[(i, k, j)]
                        print(f"  Origin {i} -> Skyport {k} -> Airport {j}: "
                              f"Riders={rid:.1f}, Revenue=${rev:.2f}, "
                              f"P(air)={theta[(i,k,j)]:.4f}")
 
        return result
    else:
        return {"status": "infeasible"}
 
 
if __name__ == "__main__":
    result = solve_skyport_location()
    print(f"\n{json.dumps(result, indent=4)}")
