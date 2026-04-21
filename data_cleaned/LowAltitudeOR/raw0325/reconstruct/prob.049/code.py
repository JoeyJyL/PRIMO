import gurobipy as gp
from gurobipy import GRB
import math
import itertools

def solve_multi_visit_drone_routing():
    """
    Multi-visit Drone Routing Problem (k-MVDRP)
    Indicator constraint formulation: explicit time variables + set partitioning.
    """
    
    # ==================== Data Initialization ====================
    
    depot = (0, 0)
    
    V = {
        0: (0, 0),
        1: (10, 5),
        2: (15, 10)
    }
    
    customers = {
        1: {'loc': (8, 3), 'weight': 0.5},
        2: {'loc': (12, 7), 'weight': 1.0},
        3: {'loc': (18, 12), 'weight': 0.8},
        4: {'loc': (6, 8), 'weight': 1.2},
        5: {'loc': (14, 4), 'weight': 0.6}
    }
    
    k = 2
    truck_speed = 10.0
    drone_speed = 10.0
    EMAX = 540.0
    launch_penalty = 30.0
    HOV = 31.0
    
    def energy_rate(weight):
        return 31.0 + 5.6 * weight
    
    def dist(loc1, loc2):
        return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    def truck_time(v1, v2):
        return dist(V[v1], V[v2]) / truck_speed
    
    def drone_time(loc1, loc2):
        return dist(loc1, loc2) / drone_speed
    
    print("=" * 70)
    print("k-MVDRP (Indicator Constraint + Set Partitioning Formulation)")
    print("=" * 70)
    print(f"\nProblem Parameters:")
    print(f"  Drones: {k}, Customers: {len(customers)}, Locations: {len(V)}")
    print(f"  EMAX: {EMAX} J, Truck: {truck_speed} m/s, Drone: {drone_speed} m/s")
    
    # ==================== Generate Feasible Operations ====================
    
    print("\n" + "=" * 70)
    print("Generating feasible operations...")
    print("=" * 70)
    
    C_list = list(customers.keys())
    operations = []
    op_id = 0
    
    for v_launch in V.keys():
        for v_land in V.keys():
            for num_customers in range(1, len(C_list) + 1):
                for customer_subset in itertools.combinations(C_list, num_customers):
                    for customer_order in itertools.permutations(customer_subset):
                        customers_per_drone = len(customer_order) // k
                        remainder = len(customer_order) % k
                        
                        drone_assignments = []
                        idx = 0
                        for d in range(k):
                            num_for_this_drone = customers_per_drone + (1 if d < remainder else 0)
                            drone_assignments.append(list(customer_order[idx:idx + num_for_this_drone]))
                            idx += num_for_this_drone
                        
                        feasible = True
                        drone_times = []
                        
                        for d_idx, assigned_customers in enumerate(drone_assignments):
                            if len(assigned_customers) == 0:
                                drone_times.append(0)
                                continue
                            
                            energy_used = 0
                            time_used = 0
                            current_loc = V[v_launch]
                            current_weight = sum(customers[c]['weight'] for c in assigned_customers)
                            
                            first_customer = assigned_customers[0]
                            d_to_first = dist(current_loc, customers[first_customer]['loc'])
                            energy_used += d_to_first * energy_rate(current_weight)
                            time_used += d_to_first / drone_speed
                            current_loc = customers[first_customer]['loc']
                            current_weight -= customers[first_customer]['weight']
                            
                            for i in range(1, len(assigned_customers)):
                                next_customer = assigned_customers[i]
                                d_to_next = dist(current_loc, customers[next_customer]['loc'])
                                energy_used += d_to_next * energy_rate(current_weight)
                                time_used += d_to_next / drone_speed
                                current_loc = customers[next_customer]['loc']
                                current_weight -= customers[next_customer]['weight']
                            
                            d_to_land = dist(current_loc, V[v_land])
                            energy_used += d_to_land * energy_rate(0)
                            time_used += d_to_land / drone_speed
                            
                            truck_travel_time = truck_time(v_launch, v_land)
                            if time_used < truck_travel_time:
                                hovering_time = truck_travel_time - time_used
                                energy_used += HOV * hovering_time
                            
                            if energy_used > EMAX:
                                feasible = False
                                break
                            
                            drone_times.append(time_used)
                        
                        if feasible:
                            tt = truck_time(v_launch, v_land)
                            has_drones = any(len(a) > 0 for a in drone_assignments)
                            
                            # Store component times (NOT precomputed max)
                            operations.append({
                                'id': op_id,
                                'launch': v_launch,
                                'land': v_land,
                                'customers': set(customer_order),
                                'truck_time': tt,
                                'drone_times': list(drone_times),
                                'has_drones': has_drones,
                                'drone_assignments': drone_assignments
                            })
                            op_id += 1
    
    print(f"\nGenerated {len(operations)} feasible operations")
    
    # ==================== Build Optimization Model ====================
    
    print("\n" + "=" * 70)
    print("Building optimization model (Indicator Constraints + Set Partitioning)")
    print("=" * 70)
    
    model = gp.Model("k-MVDRP_Indicator")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 300)
    
    n_ops = len(operations)
    
    # Decision variables
    x = model.addVars(n_ops, vtype=GRB.BINARY, name="x")
    # KEY CHANGE: explicit time variables (not precomputed parameters)
    T = model.addVars(n_ops, lb=0.0, vtype=GRB.CONTINUOUS, name="T")
    
    # Objective: minimize total time via explicit variables
    model.setObjective(
        gp.quicksum(T[o] for o in range(n_ops)),
        GRB.MINIMIZE
    )
    
    # C1: Indicator constraints for operation time
    for o in range(n_ops):
        op = operations[o]
        lp = launch_penalty if op['has_drones'] else 0.0
        
        # Truck time bound
        model.addGenConstrIndicator(
            x[o], True, T[o] >= op['truck_time'] + lp,
            name=f"ind_truck_{o}"
        )
        
        # Drone time bounds (per drone)
        for d_idx, dt_d in enumerate(op['drone_times']):
            if dt_d > 0:
                model.addGenConstrIndicator(
                    x[o], True, T[o] >= dt_d + lp,
                    name=f"ind_drone_{o}_{d_idx}"
                )
    
    # C2: Set partitioning — each customer served by EXACTLY one operation (== 1)
    for c in C_list:
        model.addConstr(
            gp.quicksum(x[o] for o in range(n_ops) if c in operations[o]['customers']) == 1,
            name=f"partition_{c}"
        )
    
    # C3: Flow conservation at each location
    for v in V.keys():
        model.addConstr(
            gp.quicksum(x[o] for o in range(n_ops) if operations[o]['launch'] == v) ==
            gp.quicksum(x[o] for o in range(n_ops) if operations[o]['land'] == v),
            name=f"flow_{v}"
        )
    
    # C4: Start at depot
    model.addConstr(
        gp.quicksum(x[o] for o in range(n_ops) if operations[o]['launch'] == 0) >= 1,
        name="start_depot"
    )
    
    # ==================== Solve ====================
    
    print("\nSolving...")
    model.optimize()
    
    # ==================== Results ====================
    
    print("\n" + "=" * 70)
    print("Optimization Results")
    print("=" * 70)
    
    if model.status == GRB.OPTIMAL:
        print(f"\nStatus: OPTIMAL")
        print(f"Total Route Completion Time: {model.ObjVal:.2f} seconds")
        
        selected_ops = [o for o in range(n_ops) if x[o].X > 0.5]
        
        print(f"\nSelected Operations ({len(selected_ops)}):")
        for op_idx in selected_ops:
            op = operations[op_idx]
            precomputed = max(op['truck_time'], max(op['drone_times']) if op['drone_times'] else 0)
            if op['has_drones']:
                precomputed += launch_penalty
            print(f"\n  Operation {op['id']}:")
            print(f"    Launch: V{op['launch']} {V[op['launch']]}")
            print(f"    Land: V{op['land']} {V[op['land']]}")
            print(f"    Customers: {sorted(op['customers'])}")
            print(f"    T[o] (indicator var): {T[op_idx].X:.2f} s")
            print(f"    Truck time: {op['truck_time']:.2f} s, Drone times: {[f'{d:.2f}' for d in op['drone_times']]}")
            print(f"    Precomputed time (verification): {precomputed:.2f} s")
            for d_idx, assigned in enumerate(op['drone_assignments']):
                if assigned:
                    print(f"      Drone {d_idx + 1}: {assigned}")
        
        return {
            "status": "optimal",
            "obj": round(model.ObjVal, 2)
        }
    
    elif model.status == GRB.TIME_LIMIT:
        print(f"\nStatus: TIME LIMIT")
        print(f"Best: {model.ObjVal:.2f} s, Gap: {model.MIPGap * 100:.2f}%")
        return {"status": "time_limit", "obj": round(model.ObjVal, 2)}
    
    else:
        print(f"\nStatus: {model.status}")
        return {"status": "infeasible"}


if __name__ == "__main__":
    result = solve_multi_visit_drone_routing()
    print("\n" + "=" * 70)
    print(f"Final Result: {result}")
    print("=" * 70)
