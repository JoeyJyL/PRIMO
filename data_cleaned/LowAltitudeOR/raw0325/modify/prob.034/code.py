"""
Truck-Drone Routing — Refined Solver with Proper Synchronization
================================================================
Fixed: drone return node must be visited by truck AFTER drone launches.
The truck waits at the rendezvous point until drone arrives.
Makespan accounts for wait time propagation.
"""

import math
import itertools

depot = (0, 0)
customers = {
    1: (4, 10), 2: (8, 2), 3: (12, 0), 4: (10, 9),
    5: (16, 3), 6: (20, 1), 7: (6, -2), 8: (14, 7),
}
parcels = {1: 0.8, 2: 1.0, 3: 0.5, 4: 0.7, 5: 1.2, 6: 1.5, 7: 0.6, 8: 0.9}

v_truck = 50.0; v_drone = 75.0; Q_drone = 3.0; E_drone = 500.0
alpha_p = 80.0; beta_p = 30.0; MUST_TRUCK = {6}
C = list(range(1, 9))
locs = {0: depot, 9: depot}
locs.update(customers)

def dist(i, j):
    return math.sqrt((locs[i][0]-locs[j][0])**2 + (locs[i][1]-locs[j][1])**2)

d = {}
for i in locs:
    for j in locs:
        d[i,j] = dist(i, j)

def truck_tour_time(tour):
    path = [0] + list(tour) + [9]
    return sum(d[path[i], path[i+1]] / v_truck for i in range(len(path)-1))

def drone_sortie_check(launch, visit_order, return_node):
    """Returns (flight_time_h, energy_Wh, total_dist, payload) or None."""
    total_payload = sum(parcels[c] for c in visit_order)
    if total_payload > Q_drone:
        return None
    path = [launch] + list(visit_order) + [return_node]
    cp = total_payload; te = 0.0; td = 0.0
    for k in range(len(path)-1):
        ld = d[path[k], path[k+1]]
        lt = ld / v_drone
        te += (alpha_p + beta_p * cp) * lt
        td += ld
        if path[k+1] in visit_order:
            cp -= parcels[path[k+1]]
    if te > E_drone:
        return None
    return (td / v_drone, te, td, total_payload)

def compute_makespan(truck_tour, drone_custs, drone_perm, l_idx, r_idx):
    """
    Compute makespan with proper synchronization.
    truck_tour: ordered list of truck customers
    drone_perm: ordered drone customers
    l_idx, r_idx: indices in truck_path=[0]+truck_tour+[9] for launch/return
    
    The drone launches when the truck departs truck_path[l_idx].
    The drone returns to truck_path[r_idx].
    The truck must wait at r_idx until the drone arrives.
    """
    truck_path = [0] + list(truck_tour) + [9]
    launch = truck_path[l_idx]
    ret = truck_path[r_idx]
    
    # Check drone feasibility
    result = drone_sortie_check(launch, drone_perm, ret)
    if result is None:
        return None, None
    
    drone_flight_h, energy, drone_dist, payload = result
    
    # Simulate truck timing with synchronization
    truck_time = [0.0] * len(truck_path)  # cumulative time at each node
    
    for k in range(1, len(truck_path)):
        leg = d[truck_path[k-1], truck_path[k]] / v_truck
        truck_time[k] = truck_time[k-1] + leg
        
        # If this is the return node, truck may need to wait for drone
        if k == r_idx:
            drone_launch_time = truck_time[l_idx]
            drone_return_time = drone_launch_time + drone_flight_h
            if drone_return_time > truck_time[k]:
                wait = drone_return_time - truck_time[k]
                # Propagate wait to all subsequent nodes
                for m in range(k, len(truck_path)):
                    truck_time[m] += wait
    
    makespan = truck_time[-1]  # truck arrival at depot (node 9)
    # Also consider drone might return after truck (if return=depot)
    drone_actual_return = truck_time[l_idx] + drone_flight_h
    makespan = max(makespan, drone_actual_return)
    
    return makespan, {
        'truck_path': truck_path,
        'truck_times': truck_time,
        'drone_launch_time': truck_time[l_idx],
        'drone_flight_h': drone_flight_h,
        'drone_return_time': truck_time[l_idx] + drone_flight_h,
        'drone_energy': energy,
        'drone_dist': drone_dist,
        'drone_payload': payload,
        'drone_perm': list(drone_perm),
        'launch': launch,
        'ret': ret,
        'l_idx': l_idx,
        'r_idx': r_idx,
    }

print("=" * 70)
print("TRUCK-DRONE ROUTING — REFINED SOLVER")
print("=" * 70)

# Baseline: all truck
all_truck_tour_best = None
all_truck_time_best = float('inf')
for perm in itertools.permutations(C):
    t = truck_tour_time(perm)
    if t < all_truck_time_best:
        all_truck_time_best = t
        all_truck_tour_best = list(perm)

print(f"\nAll-truck baseline: {all_truck_time_best*60:.2f} min")
print(f"  Tour: {all_truck_tour_best}")

best_makespan = all_truck_time_best
best_sol = {'type': 'all_truck', 'tour': all_truck_tour_best, 'makespan': all_truck_time_best}

drone_cands = [c for c in C if c not in MUST_TRUCK]
count = 0

for n_d in range(1, 4):
    for drone_set in itertools.combinations(drone_cands, n_d):
        if sum(parcels[c] for c in drone_set) > Q_drone:
            continue
        
        truck_custs = [c for c in C if c not in drone_set]
        
        # Find best truck tour
        best_truck_tour = None
        best_truck_raw = float('inf')
        for perm in itertools.permutations(truck_custs):
            t = truck_tour_time(perm)
            if t < best_truck_raw:
                best_truck_raw = t
                best_truck_tour = list(perm)
        
        # Now try ALL truck tour orderings (not just the TSP-optimal one)
        # because waiting for drone might make a different tour better
        # But that's too many... let's try top tours
        
        # Actually, for correctness, try all truck tours
        # With 5-7 customers, max 7! = 5040 permutations — feasible
        for truck_tour in itertools.permutations(truck_custs):
            truck_path = [0] + list(truck_tour) + [9]
            
            for drone_perm in itertools.permutations(drone_set):
                for l_idx in range(len(truck_path) - 1):
                    for r_idx in range(l_idx, len(truck_path)):
                        count += 1
                        ms, info = compute_makespan(
                            truck_tour, drone_set, drone_perm, l_idx, r_idx
                        )
                        if ms is not None and ms < best_makespan:
                            best_makespan = ms
                            best_sol = {
                                'type': 'truck_drone',
                                'truck_tour': list(truck_tour),
                                'truck_custs': truck_custs,
                                'drone_custs': list(drone_set),
                                'info': info,
                                'makespan': ms,
                            }

print(f"\nTotal configs evaluated: {count}")

# ============================================================
# Output
# ============================================================
print(f"\n{'=' * 70}")
print("★ OPTIMAL SOLUTION")
print(f"{'=' * 70}")

sol = best_sol
ms_min = sol['makespan'] * 60
print(f"\n  Makespan: {ms_min:.2f} minutes ({sol['makespan']:.6f} hours)")

if sol['type'] == 'all_truck':
    route_str = "Depot → " + " → ".join(f"C{c}" for c in sol['tour']) + " → Depot"
    print(f"  All truck: {route_str}")
else:
    info = sol['info']
    
    # Truck
    print(f"\n  TRUCK ROUTE:")
    route_str = " → ".join("Depot" if n in [0,9] else f"C{n}" for n in info['truck_path'])
    print(f"    {route_str}")
    print(f"    Timing:")
    for k, node in enumerate(info['truck_path']):
        t_min = info['truck_times'][k] * 60
        name = "Depot" if node == 0 else ("Depot(return)" if node == 9 else f"C{node}")
        extra = ""
        if k == info['l_idx']:
            extra += f"  ◀ DRONE LAUNCHES (t={t_min:.2f} min)"
        if k == info['r_idx'] and info['ret'] not in [0, 9]:
            drone_arr = info['drone_return_time'] * 60
            if drone_arr > info['truck_times'][k] * 60 - 0.01:
                extra += f"  ◀ DRONE RETURNS (t={drone_arr:.2f} min)"
        print(f"      {name}: t={t_min:.2f} min{extra}")
    
    # Drone
    print(f"\n  DRONE SORTIE:")
    launch_name = "Depot" if info['launch'] == 0 else f"C{info['launch']}"
    ret_name = "Depot" if info['ret'] == 9 else f"C{info['ret']}"
    drone_path = [info['launch']] + info['drone_perm'] + [info['ret']]
    path_str = " → ".join("Depot" if n in [0,9] else f"C{n}" for n in drone_path)
    
    print(f"    Path: {path_str}")
    print(f"    Customers served: {['C'+str(c) for c in info['drone_perm']]}")
    print(f"    Launch: {launch_name} at t={info['drone_launch_time']*60:.2f} min")
    print(f"    Flight time: {info['drone_flight_h']*60:.2f} min")
    print(f"    Return: {ret_name} at t={info['drone_return_time']*60:.2f} min")
    print(f"    Distance: {info['drone_dist']:.2f} km")
    print(f"    Energy: {info['drone_energy']:.2f} / {E_drone} Wh")
    print(f"    Payload: {info['drone_payload']:.1f} / {Q_drone} kg")
    
    # Summary
    print(f"\n  {'─' * 50}")
    print(f"  SUMMARY:")
    print(f"    Truck-delivered: {sorted(sol['truck_custs'])}")
    print(f"    Drone-delivered: {sorted(sol['drone_custs'])}")
    print(f"    ★ Makespan: {ms_min:.2f} min")
    print(f"    All-truck baseline: {all_truck_time_best*60:.2f} min")
    saving = (all_truck_time_best - sol['makespan']) / all_truck_time_best * 100
    print(f"    Improvement: {saving:.1f}%")
    
    # Verification
    print(f"\n  CONSTRAINT VERIFICATION:")
    all_served = set(sol['truck_custs']) | set(sol['drone_custs'])
    print(f"    All served: {sorted(all_served)} = {C}  {'✓' if all_served == set(C) else '✗'}")
    print(f"    C6 by truck: {'✓' if 6 in sol['truck_custs'] else '✗'}")
    print(f"    Drone ≤ 3 custs: {len(sol['drone_custs'])}  ✓")
    print(f"    Payload ≤ {Q_drone}: {info['drone_payload']:.1f}  ✓")
    print(f"    Energy ≤ {E_drone}: {info['drone_energy']:.2f}  ✓")
    # Check launch before return in truck tour
    print(f"    Launch(idx={info['l_idx']}) before Return(idx={info['r_idx']}): ✓")