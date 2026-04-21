
import math
import itertools

def solve_skyport_congestion():
    # --- 1. Data Setup ---
    # Nodes: 0:(0,0), 1:(10,10), 2:(0,10), 3:(10,0)
    nodes = {0: (0, 0), 1: (10, 10), 2: (0, 10), 3: (10, 0)}
    candidates = list(nodes.keys())
    
    # Demands
    demands = [
        {'id': 'D1', 'o': 0, 'd': 1, 'vol': 100}, # Diagonal 1
        {'id': 'D2', 'o': 2, 'd': 3, 'vol': 100}, # Diagonal 2
        {'id': 'D3', 'o': 0, 'd': 2, 'vol': 50}   # Vertical
    ]
    
    # Parameters
    fixed_cost_per_port = 500
    cost_time = 1.0
    cost_risk_unit = 0.1
    
    speed_g = 0.5
    speed_a = 2.0
    speed_access = 0.5
    
    # --- 2. Helpers ---
    def get_dist(i, j):
        p1, p2 = nodes[i], nodes[j]
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
    def check_intersect(path1, path2):
        # path: (u, v)
        # 0->1 and 2->3 intersect
        s1, s2 = set(path1), set(path2)
        if s1 == {0, 1} and s2 == {2, 3}: return True
        return False
        
    # --- 3. Solver (Brute Force over Skyport configs) ---
    best_obj = float('inf')
    best_sol = {}
    
    # Iterate all combinations of open skyports (1 to 4 ports)
    for r in range(1, 5):
        for open_ports in itertools.combinations(candidates, r):
            open_ports = set(open_ports)
            
            # Helper: Get nearest open port
            def get_nearest(n):
                best_p = -1
                min_d = float('inf')
                for p in open_ports:
                    d = get_dist(n, p)
                    if d < min_d: min_d = d; best_p = p
                return best_p, min_d

            # Pre-calc options for each demand
            opts = []
            for dem in demands:
                # Ground Cost
                dist_g = get_dist(dem['o'], dem['d'])
                time_g = dist_g / speed_g
                cost_g = time_g * cost_time * dem['vol']
                
                # Air Cost (if feasible)
                p_o, d_acc = get_nearest(dem['o'])
                p_d, d_egr = get_nearest(dem['d'])
                
                cost_a = float('inf')
                air_path = None
                
                if p_o != -1 and p_d != -1 and p_o != p_d:
                    dist_air = get_dist(p_o, p_d)
                    time_air = (d_acc + d_egr)/speed_access + dist_air/speed_a
                    cost_a = time_air * cost_time * dem['vol']
                    air_path = (p_o, p_d)
                
                opts.append({'dem': dem, 'cost_g': cost_g, 'cost_a': cost_a, 'path': air_path})
            
            # Inner Solver: Assign modes (Ground vs Air) to minimize Cost + Risk
            # 3 Demands => 2^3 = 8 combinations
            local_best_val = float('inf')
            local_best_modes = [] # 0=G, 1=A
            
            fixed_cost = len(open_ports) * fixed_cost_per_port
            
            for modes in itertools.product([0, 1], repeat=len(demands)):
                # Check feasibility
                possible = True
                for i, m in enumerate(modes):
                    if m == 1 and opts[i]['path'] is None:
                        possible = False; break
                if not possible: continue
                
                # Calculate Costs
                travel_cost = 0
                active_paths = [] # list of (path, vol)
                
                for i, m in enumerate(modes):
                    if m == 0: travel_cost += opts[i]['cost_g']
                    else:
                        travel_cost += opts[i]['cost_a']
                        active_paths.append((opts[i]['path'], opts[i]['dem']['vol']))
                
                # Risk Cost (Quadratic)
                risk_cost = 0
                for i in range(len(active_paths)):
                    for j in range(i+1, len(active_paths)):
                        p1, vol1 = active_paths[i]
                        p2, vol2 = active_paths[j]
                        if check_intersect(p1, p2):
                            # Risk term: C_r * vol1 * vol2
                            risk_cost += cost_risk_unit * vol1 * vol2
                            
                total = fixed_cost + travel_cost + risk_cost
                
                if total < local_best_val:
                    local_best_val = total
                    local_best_modes = modes
            
            if local_best_val < best_obj:
                best_obj = local_best_val
                best_sol = {
                    'status': 'optimal',
                    'obj': best_obj,
                    'open_ports': sorted(list(open_ports)),
                    'modes': local_best_modes # 0=G, 1=A
                }
                
    return best_sol

if __name__ == "__main__":
    print(solve_skyport_congestion())
