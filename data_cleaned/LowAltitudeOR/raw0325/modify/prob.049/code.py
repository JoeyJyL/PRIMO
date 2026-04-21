import gurobipy as gp
from gurobipy import GRB
import math
import itertools

def solve_multi_visit_drone_routing():
    """
    Multi-visit Drone Routing Problem (k-MVDRP)
    Minimize total route completion time for truck-and-drone delivery system
    """
    
    # ==================== Data Initialization ====================
    
    # Depot location
    depot = (0, 0)
    
    # Launch/landing locations (including depot)
    V = {
        0: (0, 0),   # Depot
        1: (10, 5),
        2: (15, 10)
    }
    
    # Customer locations and package weights
    customers = {
        1: {'loc': (8, 3), 'weight': 0.5},
        2: {'loc': (12, 7), 'weight': 1.0},
        3: {'loc': (18, 12), 'weight': 0.8},
        4: {'loc': (6, 8), 'weight': 1.2},
        5: {'loc': (14, 4), 'weight': 0.6}
    }
    
    # Parameters
    k = 2  # Number of drones
    truck_speed = 10.0  # m/s
    drone_speed = 10.0  # m/s
    EMAX = 400.0  # Joules (reduced from 540 due to cold-weather battery degradation)
    launch_penalty = 30.0  # seconds
    HOV = 31.0  # Hovering energy rate (J/s)
    
    # Energy function: e(W) = 31.0 + 5.6 * W (J/m)
    def energy_rate(weight):
        return 31.0 + 5.6 * weight
    
    # Distance calculation
    def dist(loc1, loc2):
        return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    # Travel times
    def truck_time(v1, v2):
        return dist(V[v1], V[v2]) / truck_speed
    
    def drone_time(loc1, loc2):
        return dist(loc1, loc2) / drone_speed
    
    print("=" * 70)
    print("Multi-Visit Drone Routing Problem (k-MVDRP)")
    print("=" * 70)
    print(f"\nProblem Parameters:")
    print(f"  Number of drones: {k}")
    print(f"  Number of customers: {len(customers)}")
    print(f"  Number of launch/landing locations: {len(V)}")
    print(f"  Drone battery capacity: {EMAX} J (reduced from 540)")
    print(f"  Truck speed: {truck_speed} m/s")
    print(f"  Drone speed: {drone_speed} m/s")
    
    # ==================== Generate Feasible Operations ====================
    
    print("\n" + "=" * 70)
    print("Generating feasible operations...")
    print("=" * 70)
    
    C_list = list(customers.keys())
    operations = []
    op_id = 0
    
    # Generate operations: for each pair of launch/landing locations
    for v_launch in V.keys():
        for v_land in V.keys():
            # Try different customer subsets
            for num_customers in range(1, len(C_list) + 1):
                for customer_subset in itertools.combinations(C_list, num_customers):
                    # Try different permutations of customer visit order
                    for customer_order in itertools.permutations(customer_subset):
                        # Assign customers to drones (block partitioning)
                        customers_per_drone = len(customer_order) // k
                        remainder = len(customer_order) % k
                        
                        drone_assignments = []
                        idx = 0
                        for d in range(k):
                            num_for_this_drone = customers_per_drone + (1 if d < remainder else 0)
                            drone_assignments.append(list(customer_order[idx:idx + num_for_this_drone]))
                            idx += num_for_this_drone
                        
                        # Check energy feasibility for each drone
                        feasible = True
                        drone_times = []
                        
                        for d_idx, assigned_customers in enumerate(drone_assignments):
                            if len(assigned_customers) == 0:
                                drone_times.append(0)
                                continue
                            
                            # Calculate energy and time for this drone's route
                            energy_used = 0
                            time_used = 0
                            current_loc = V[v_launch]
                            current_weight = sum(customers[c]['weight'] for c in assigned_customers)
                            
                            # Fly to first customer
                            first_customer = assigned_customers[0]
                            d_to_first = dist(current_loc, customers[first_customer]['loc'])
                            energy_used += d_to_first * energy_rate(current_weight)
                            time_used += d_to_first / drone_speed
                            current_loc = customers[first_customer]['loc']
                            current_weight -= customers[first_customer]['weight']
                            
                            # Visit remaining customers
                            for i in range(1, len(assigned_customers)):
                                next_customer = assigned_customers[i]
                                d_to_next = dist(current_loc, customers[next_customer]['loc'])
                                energy_used += d_to_next * energy_rate(current_weight)
                                time_used += d_to_next / drone_speed
                                current_loc = customers[next_customer]['loc']
                                current_weight -= customers[next_customer]['weight']
                            
                            # Return to landing location
                            d_to_land = dist(current_loc, V[v_land])
                            energy_used += d_to_land * energy_rate(0)
                            time_used += d_to_land / drone_speed
                            
                            # Add hovering energy if drone arrives before truck
                            truck_travel_time = truck_time(v_launch, v_land)
                            if time_used < truck_travel_time:
                                hovering_time = truck_travel_time - time_used
                                energy_used += HOV * hovering_time
                            
                            if energy_used > EMAX:
                                feasible = False
                                break
                            
                            drone_times.append(time_used)
                        
                        if feasible:
                            # Calculate operation time
                            truck_travel_time = truck_time(v_launch, v_land)
                            max_drone_time = max(drone_times) if drone_times else 0
                            op_time = max(truck_travel_time, max_drone_time)
                            
                            # Add launch penalty if any drone launches
                            if any(len(a) > 0 for a in drone_assignments):
                                op_time += launch_penalty
                            
                            operations.append({
                                'id': op_id,
                                'launch': v_launch,
                                'land': v_land,
                                'customers': set(customer_order),
                                'time': op_time,
                                'drone_assignments': drone_assignments
                            })
                            op_id += 1
    
    print(f"\nGenerated {len(operations)} feasible operations")
    
    # ==================== Build Optimization Model ====================
    
    print("\n" + "=" * 70)
    print("Building optimization model...")
    print("=" * 70)
    
    model = gp.Model("k-MVDRP")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 300)  # 5 minute time limit
    
    # Decision variables
    x = model.addVars(len(operations), vtype=GRB.BINARY, name="x")
    
    # Objective: minimize total route completion time
    model.setObjective(
        gp.quicksum(operations[o]['time'] * x[o] for o in range(len(operations))),
        GRB.MINIMIZE
    )
    
    # Constraint: Each customer must be covered
    for c in C_list:
        model.addConstr(
            gp.quicksum(x[o] for o in range(len(operations)) if c in operations[o]['customers']) >= 1,
            name=f"cover_customer_{c}"
        )
    
    # Constraint: Flow conservation at each location
    for v in V.keys():
        model.addConstr(
            gp.quicksum(x[o] for o in range(len(operations)) if operations[o]['launch'] == v) ==
            gp.quicksum(x[o] for o in range(len(operations)) if operations[o]['land'] == v),
            name=f"flow_{v}"
        )
    
    # Constraint: Start and end at depot
    model.addConstr(
        gp.quicksum(x[o] for o in range(len(operations)) if operations[o]['launch'] == 0) >= 1,
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
        
        selected_ops = [o for o in range(len(operations)) if x[o].X > 0.5]
        
        print(f"\nSelected Operations ({len(selected_ops)}):")
        for op_idx in selected_ops:
            op = operations[op_idx]
            print(f"\n  Operation {op['id']}:")
            print(f"    Launch from: V{op['launch']} {V[op['launch']]}")
            print(f"    Land at: V{op['land']} {V[op['land']]}")
            print(f"    Customers served: {sorted(op['customers'])}")
            print(f"    Time: {op['time']:.2f} seconds")
            for d_idx, assigned in enumerate(op['drone_assignments']):
                if assigned:
                    print(f"      Drone {d_idx + 1}: {assigned}")
        
        return {
            "status": "optimal",
            "obj": round(model.ObjVal,2)
        }
    
    elif model.status == GRB.TIME_LIMIT:
        print(f"\nStatus: TIME LIMIT REACHED")
        print(f"Best solution found: {model.ObjVal:.2f} seconds")
        print(f"Gap: {model.MIPGap * 100:.2f}%")
        
        return {
            "status": "time_limit",
            "obj": model.ObjVal
        }
    
    else:
        print(f"\nStatus: {model.status}")
        return {"status": "infeasible"}


if __name__ == "__main__":
    result = solve_multi_visit_drone_routing()
    print("\n" + "=" * 70)
    print(f"Final Result: {result}")
    print("=" * 70)
