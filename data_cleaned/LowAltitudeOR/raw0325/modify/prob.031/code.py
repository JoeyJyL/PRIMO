"""
TD-MTDRP Solver: K=2 Trucks with Drones
=========================================
Solves the Time-Dependent Multiple Truck-Drone Routing Problem
via exhaustive enumeration for the small-scale instance.

Instance: 6 customers, 2 trucks, 2 drones each
Constraints: capacity, battery, payload, time windows, co-assignment (C2+C3)
Objective: minimize total duration (sum of both truck route durations)
"""

import itertools
import math

# ============================================================
# Problem Data
# ============================================================

depot = (50, 50)

customers = {
    'C1': {'pos': (20, 80), 'demand': 5,  'weight': 4,  'st': 2, 'sd': 1,   'tw': (0, 80)},
    'C2': {'pos': (80, 80), 'demand': 8,  'weight': 12, 'st': 3, 'sd': 1.5, 'tw': (0, 200)},
    'C3': {'pos': (20, 20), 'demand': 6,  'weight': 5,  'st': 2, 'sd': 1,   'tw': (0, 200)},
    'C4': {'pos': (80, 20), 'demand': 7,  'weight': 9,  'st': 3, 'sd': 1.5, 'tw': (60, 180)},
    'C5': {'pos': (50, 80), 'demand': 4,  'weight': 3,  'st': 2, 'sd': 1,   'tw': (0, 60)},
    'C6': {'pos': (50, 20), 'demand': 5,  'weight': 6,  'st': 2, 'sd': 1,   'tw': (80, 200)},
}

K = 2       # number of trucks
m = 2       # max drones per truck
Q = 20      # truck capacity (max demand per truck)
D = 10      # drone payload limit
B = 40      # drone battery (max round-trip flight time)
drone_speed = 2.0
truck_speed = 1.0

all_customers = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
co_assign = ('C2', 'C3')  # must be on the same truck


# ============================================================
# Helper Functions
# ============================================================

def euclidean(a, b):
    """Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def get_pos(node):
    """Get position of a node (depot or customer)."""
    return depot if node == 'depot' else customers[node]['pos']


def truck_travel_time(a, b):
    """Time-independent truck travel time between two nodes."""
    return euclidean(get_pos(a), get_pos(b)) / truck_speed


def drone_travel_time(a, b):
    """Drone one-way travel time between two nodes."""
    return euclidean(get_pos(a), get_pos(b)) / drone_speed


def is_drone_eligible(c):
    """Check if customer c can be served by drone (weight constraint)."""
    return customers[c]['weight'] <= D


def is_drone_battery_feasible(truck_stop, drone_cust):
    """Check if drone round-trip from truck_stop to drone_cust is within battery."""
    round_trip = 2 * drone_travel_time(truck_stop, drone_cust)
    return round_trip <= B


# ============================================================
# Route Simulation
# ============================================================

def simulate_route(truck_order, drone_assign):
    """
    Simulate a single truck-drone route.
    
    Parameters:
        truck_order: list of customer names visited by the truck, in order
        drone_assign: dict {truck_stop: [drone_customers]} 
                      mapping each truck stop to customers served by drones from that stop
    
    Returns:
        duration (float) if feasible, None if infeasible (time window violation)
    """
    t = 0.0  # current time, depart depot at t=0
    prev = 'depot'

    for stop in truck_order:
        # --- Truck travels to next stop ---
        travel = truck_travel_time(prev, stop)
        arrive = t + travel

        # --- Time window check for truck stop ---
        e, l = customers[stop]['tw']
        svc_start = max(arrive, e)  # wait for earliest start if arrived early
        if svc_start > l:
            return None  # missed the time window

        # --- Truck service ---
        truck_done = svc_start + customers[stop]['st']

        # --- Drone operations from this stop ---
        drones = drone_assign.get(stop, [])
        max_drone_return = 0.0

        for dc in drones:
            # Drone flies out
            fly_time = drone_travel_time(stop, dc)
            drone_arrive = svc_start + fly_time

            # Time window check for drone customer
            dc_e, dc_l = customers[dc]['tw']
            drone_svc_start = max(drone_arrive, dc_e)
            if drone_svc_start > dc_l:
                return None  # drone missed customer's time window

            # Drone serves customer and flies back
            drone_done = drone_svc_start + customers[dc]['sd']
            drone_return = drone_done + fly_time  # symmetric return flight

            max_drone_return = max(max_drone_return, drone_return)

        # --- Truck departs when both truck service and all drones are done ---
        t = max(truck_done, max_drone_return) if drones else truck_done
        prev = stop

    # --- Return to depot ---
    t += truck_travel_time(prev, 'depot')
    return t


# ============================================================
# Route Configuration Generator
# ============================================================

def generate_route_configs(customer_set):
    """
    Generate all feasible (truck_order, drone_assign) configurations
    for a given set of customers assigned to one truck.
    
    Enumerates:
      - All subsets of drone-eligible customers to serve by drone
      - All assignments of drone customers to truck stops
      - All permutations of truck stop ordering
    
    Yields:
        (truck_order, drone_assign) tuples
    """
    customer_list = list(customer_set)
    drone_candidates = [c for c in customer_list if is_drone_eligible(c)]

    for n_drones in range(len(drone_candidates) + 1):
        for drone_served in itertools.combinations(drone_candidates, n_drones):
            drone_set = set(drone_served)
            truck_stops = [c for c in customer_list if c not in drone_set]

            # Must have at least one truck stop
            if not truck_stops:
                continue

            drone_list = list(drone_served)

            # --- Generate all valid drone-to-truck-stop assignments ---
            if not drone_list:
                valid_assignments = [{}]
            else:
                valid_assignments = []
                # Each drone customer chooses one truck stop
                for combo in itertools.product(range(len(truck_stops)), repeat=len(drone_list)):
                    assign = {}
                    feasible = True

                    for i, ts_idx in enumerate(combo):
                        ts = truck_stops[ts_idx]
                        dc = drone_list[i]

                        # Battery feasibility
                        if not is_drone_battery_feasible(ts, dc):
                            feasible = False
                            break

                        # Build assignment, check max drones per stop
                        if ts not in assign:
                            assign[ts] = []
                        assign[ts] = assign[ts] + [dc]
                        if len(assign[ts]) > m:
                            feasible = False
                            break

                    if feasible:
                        valid_assignments.append(assign)

            # --- Try all orderings of truck stops ---
            for assign in valid_assignments:
                for perm in itertools.permutations(truck_stops):
                    yield (list(perm), {k: list(v) for k, v in assign.items()})


# ============================================================
# Main Solver: Enumerate All Two-Truck Partitions
# ============================================================

def solve():
    """
    Exhaustive enumeration solver for K=2 trucks.
    
    1. Enumerate all ways to partition 6 customers into two groups
    2. Enforce co-assignment: C2 and C3 must be in the same group
    3. Enforce capacity: each group's total demand <= Q
    4. For each partition, find the best route for each truck
    5. Return the partition with minimum total duration
    """
    print("=" * 70)
    print("TD-MTDRP SOLVER (K=2, Exhaustive Enumeration)")
    print("=" * 70)

    # --- Print instance summary ---
    print(f"\nInstance:")
    print(f"  Trucks: {K}, Drones/truck: {m}, Capacity: {Q}")
    print(f"  Drone battery: {B}, Drone payload: {D}")
    print(f"  Co-assignment: {co_assign[0]} and {co_assign[1]} must share a truck")
    total_demand = sum(c['demand'] for c in customers.values())
    print(f"  Total demand: {total_demand} (requires both trucks since Q={Q})")
    print(f"\n  Customer details:")
    for c in all_customers:
        info = customers[c]
        drone_tag = "✓" if is_drone_eligible(c) else "✗ (too heavy)"
        print(f"    {c}: pos={info['pos']}, demand={info['demand']}, "
              f"weight={info['weight']}, TW={info['tw']}, drone={drone_tag}")

    # --- Enumerate partitions ---
    # Truck A must contain both C2 and C3 (co-assignment)
    # Choose which of the remaining customers also go to Truck A
    others = [c for c in all_customers if c not in co_assign]
    
    best_total = float('inf')
    best_solution = None
    total_configs = 0
    feasible_partitions = 0

    print(f"\n{'=' * 70}")
    print("Enumerating partitions...")
    print(f"{'=' * 70}")

    for r in range(len(others) + 1):
        for extra_A in itertools.combinations(others, r):
            set_A = set(co_assign) | set(extra_A)
            set_B = set(all_customers) - set_A

            # --- Capacity check ---
            demand_A = sum(customers[c]['demand'] for c in set_A)
            demand_B = sum(customers[c]['demand'] for c in set_B)
            if demand_A > Q or (set_B and demand_B > Q):
                continue
            # Truck B must have at least one customer
            if not set_B:
                continue

            # --- Find best route for each truck ---
            best_dur_A = float('inf')
            best_route_A = None

            for truck_order, drone_assign in generate_route_configs(set_A):
                dur = simulate_route(truck_order, drone_assign)
                total_configs += 1
                if dur is not None and dur < best_dur_A:
                    best_dur_A = dur
                    best_route_A = (truck_order, drone_assign)

            if best_route_A is None:
                continue  # no feasible route for Truck A

            best_dur_B = float('inf')
            best_route_B = None

            for truck_order, drone_assign in generate_route_configs(set_B):
                dur = simulate_route(truck_order, drone_assign)
                total_configs += 1
                if dur is not None and dur < best_dur_B:
                    best_dur_B = dur
                    best_route_B = (truck_order, drone_assign)

            if best_route_B is None:
                continue  # no feasible route for Truck B

            feasible_partitions += 1
            total_dur = best_dur_A + best_dur_B

            print(f"\n  Partition: A={sorted(set_A)}, B={sorted(set_B)}")
            print(f"    Demand: A={demand_A}, B={demand_B}")
            print(f"    Best: A={best_dur_A:.4f}, B={best_dur_B:.4f}, Total={total_dur:.4f}")

            if total_dur < best_total:
                best_total = total_dur
                best_solution = {
                    'truck_A': {
                        'customers': set_A,
                        'route': best_route_A[0],
                        'drones': best_route_A[1],
                        'duration': best_dur_A,
                    },
                    'truck_B': {
                        'customers': set_B,
                        'route': best_route_B[0],
                        'drones': best_route_B[1],
                        'duration': best_dur_B,
                    },
                }

    print(f"\n{'=' * 70}")
    print(f"Total route configurations evaluated: {total_configs}")
    print(f"Feasible partitions: {feasible_partitions}")
    print(f"{'=' * 70}")

    return best_solution, best_total


def print_solution(solution, total_duration):
    """Print the optimal solution with detailed timing."""
    if solution is None:
        print("\nNo feasible solution found!")
        return

    print(f"\n{'=' * 70}")
    print(f"★ OPTIMAL SOLUTION")
    print(f"{'=' * 70}")
    print(f"\n  Total Duration: {total_duration:.4f}")

    for name, truck in [("Truck A", solution['truck_A']),
                        ("Truck B", solution['truck_B'])]:
        route = truck['route']
        drones = truck['drones']
        dur = truck['duration']
        custs = sorted(truck['customers'])

        # Collect drone-served customers
        drone_served = []
        for stop in route:
            drone_served.extend(drones.get(stop, []))

        print(f"\n  {name} (duration = {dur:.4f}):")
        print(f"    Customers: {custs}")
        print(f"    Truck route: Depot → {' → '.join(route)} → Depot")
        print(f"    Truck-delivered: {route}")
        if drone_served:
            print(f"    Drone-delivered: {drone_served}")
            for stop in route:
                dcs = drones.get(stop, [])
                if dcs:
                    for dc in dcs:
                        rt = 2 * drone_travel_time(stop, dc)
                        print(f"      At {stop}: drone → {dc} "
                              f"(round-trip flight = {rt:.2f} ≤ B={B})")

    # --- Detailed timing for each truck ---
    for name, truck in [("Truck A", solution['truck_A']),
                        ("Truck B", solution['truck_B'])]:
        route = truck['route']
        drones = truck['drones']

        print(f"\n  {'─' * 50}")
        print(f"  {name} — Detailed Timing")
        print(f"  {'─' * 50}")

        t = 0.0
        prev = 'depot'
        print(f"  t={t:.2f}: Depart Depot {depot}")

        for stop in route:
            travel = truck_travel_time(prev, stop)
            arrive = t + travel
            e, l = customers[stop]['tw']
            svc_start = max(arrive, e)
            truck_done = svc_start + customers[stop]['st']

            print(f"\n  ┌─ Travel to {stop} ({customers[stop]['pos']}): "
                  f"{travel:.2f} time units")
            print(f"  │  t={arrive:.2f}: Arrive {stop} "
                  f"(TW=[{e}, {l}])")
            if svc_start > arrive:
                print(f"  │  ⏳ Wait until t={svc_start:.2f} "
                      f"(earliest start = {e})")
            print(f"  │  t={svc_start:.2f}: Begin truck service "
                  f"(duration = {customers[stop]['st']})")
            print(f"  │  t={truck_done:.2f}: Truck service complete")

            drone_custs = drones.get(stop, [])
            max_dr = 0.0
            for dc in drone_custs:
                fly = drone_travel_time(stop, dc)
                d_arrive = svc_start + fly
                dc_e, dc_l = customers[dc]['tw']
                d_svc_start = max(d_arrive, dc_e)
                d_done = d_svc_start + customers[dc]['sd']
                d_return = d_done + fly
                print(f"  │  ✈ Drone → {dc} ({customers[dc]['pos']})")
                print(f"  │    Fly out: {fly:.2f}, "
                      f"arrive t={d_arrive:.2f} (TW=[{dc_e},{dc_l}])")
                print(f"  │    Service: {customers[dc]['sd']} units, "
                      f"done t={d_done:.2f}")
                print(f"  │    Fly back: {fly:.2f}, "
                      f"return t={d_return:.2f}")
                max_dr = max(max_dr, d_return)

            if drone_custs:
                wait = max(0, max_dr - truck_done)
                if wait > 0:
                    print(f"  │  ⏳ Truck waits {wait:.2f} for drones")
            t = max(truck_done, max_dr) if drone_custs else truck_done
            print(f"  └─ t={t:.2f}: Depart {stop}")
            prev = stop

        travel_back = truck_travel_time(prev, 'depot')
        t += travel_back
        print(f"\n  ┌─ Travel to Depot: {travel_back:.2f} time units")
        print(f"  └─ t={t:.2f}: Arrive Depot")

    # --- Constraint verification ---
    print(f"\n  {'=' * 50}")
    print(f"  CONSTRAINT VERIFICATION")
    print(f"  {'=' * 50}")

    # Capacity
    for name, truck in [("A", solution['truck_A']), ("B", solution['truck_B'])]:
        dem = sum(customers[c]['demand'] for c in truck['customers'])
        ok = dem <= Q
        print(f"  Capacity Truck {name}: {dem} ≤ {Q}  {'✓' if ok else '✗'}")

    # Co-assignment
    a_custs = solution['truck_A']['customers']
    co_ok = (co_assign[0] in a_custs and co_assign[1] in a_custs)
    print(f"  Co-assignment {co_assign}: both on Truck A  {'✓' if co_ok else '✗'}")

    # Fleet size
    print(f"  Fleet size: 2 ≤ K={K}  ✓")

    # All customers served exactly once
    all_served = solution['truck_A']['customers'] | solution['truck_B']['customers']
    drone_A = []
    for s in solution['truck_A']['route']:
        drone_A.extend(solution['truck_A']['drones'].get(s, []))
    drone_B = []
    for s in solution['truck_B']['route']:
        drone_B.extend(solution['truck_B']['drones'].get(s, []))
    all_in_routes = (set(solution['truck_A']['route']) | set(drone_A) |
                     set(solution['truck_B']['route']) | set(drone_B))
    cover_ok = all_in_routes == set(all_customers)
    print(f"  All customers served: {sorted(all_in_routes)} = {sorted(all_customers)}  "
          f"{'✓' if cover_ok else '✗'}")

    # Battery
    battery_ok = True
    for name, truck in [("A", solution['truck_A']), ("B", solution['truck_B'])]:
        for stop in truck['route']:
            for dc in truck['drones'].get(stop, []):
                rt = 2 * drone_travel_time(stop, dc)
                if rt > B:
                    print(f"  Battery VIOLATION: {stop}→{dc} rt={rt:.2f} > {B}")
                    battery_ok = False
    print(f"  Drone battery: all round-trips ≤ {B}  {'✓' if battery_ok else '✗'}")

    # Drone payload
    payload_ok = True
    for name, truck in [("A", solution['truck_A']), ("B", solution['truck_B'])]:
        for stop in truck['route']:
            for dc in truck['drones'].get(stop, []):
                if customers[dc]['weight'] > D:
                    print(f"  Payload VIOLATION: {dc} weight={customers[dc]['weight']} > {D}")
                    payload_ok = False
    print(f"  Drone payload: all weights ≤ {D}  {'✓' if payload_ok else '✗'}")

    print(f"\n  ★ ALL CONSTRAINTS SATISFIED ✓" if all([
        co_ok, cover_ok, battery_ok, payload_ok
    ]) else "\n  ✗ SOME CONSTRAINTS VIOLATED")


# ============================================================
# Distance Matrix
# ============================================================

def print_distance_matrix():
    """Print the distance matrix for reference."""
    nodes = ['depot'] + all_customers
    print(f"\n{'=' * 70}")
    print("DISTANCE MATRIX (Euclidean)")
    print(f"{'=' * 70}")
    print(f"{'':>7}", end="")
    for n in nodes:
        print(f"{n:>8}", end="")
    print()
    for a in nodes:
        print(f"{a:>7}", end="")
        for b in nodes:
            d = euclidean(get_pos(a), get_pos(b))
            print(f"{d:8.2f}", end="")
        print()


# ============================================================
# Entry Point
# ============================================================

if __name__ == '__main__':
    print_distance_matrix()
    solution, total_duration = solve()
    print_solution(solution, total_duration)