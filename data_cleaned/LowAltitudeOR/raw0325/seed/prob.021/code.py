
import math
import heapq

def solve_drone_charging_network():
    # 1. Data
    # Nodes: 0=Source, 1=Dest, 2,3,4=Candidate CS
    nodes = {
        0: (0, 0),
        1: (22, 0),
        2: (11, 0),    # CS1 (Direct)
        3: (11, 5),    # CS2 (Detour North)
        4: (11, -5)    # CS3 (Detour South)
    }
    
    # Costs
    fixed_costs = {
        0: 0, 1: 0,    # Source/Dest free
        2: 500.0,      # Premium
        3: 100.0,      # Cheap
        4: 100.0
    }
    
    # Parameters
    max_range = 15.0 
    travel_cost_per_km = 1.0
    
    # 2. Build Graph (Adjacency List)
    adj = {i: [] for i in nodes}
    
    def get_dist(i, j):
        p1, p2 = nodes[i], nodes[j]
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
    for i in nodes:
        for j in nodes:
            if i == j: continue
            d = get_dist(i, j)
            if d <= max_range:
                adj[i].append((j, d))

    # 3. Solver: Dijkstra with Node Costs
    # Cost state: cumulative travel + cumulative fixed costs
    min_costs = {i: float('inf') for i in nodes}
    min_costs[0] = 0
    previous = {i: None for i in nodes}
    
    # Priority Queue: (cost, u)
    pq = [(0, 0)]
    
    while pq:
        curr_cost, u = heapq.heappop(pq)
        
        if curr_cost > min_costs[u]:
            continue
        
        if u == 1: # Reached Dest
            continue
            
        for v, dist_uv in adj[u]:
            travel = dist_uv * travel_cost_per_km
            node_cost = fixed_costs[v]
            
            # Note: In standard Dijkstra, node cost is added upon arrival
            # Be careful not to double count if revisiting (but this is DAG-like here)
            # Better: Cost to REACH v = Cost(u) + Edge(u,v) + Fixed(v)
            
            new_cost = curr_cost + travel + node_cost
            
            if new_cost < min_costs[v]:
                min_costs[v] = new_cost
                previous[v] = u
                heapq.heappush(pq, (new_cost, v))
                
    # 4. Reconstruct Path
    path = []
    curr = 1
    if min_costs[1] == float('inf'):
        return {"status": "infeasible"}
        
    while curr is not None:
        path.append(curr)
        curr = previous[curr]
    path.reverse()
    
    # 5. Output
    open_stations = [n for n in path if n in [2,3,4]]
    total_dist = sum(get_dist(path[i], path[i+1]) for i in range(len(path)-1))
    
    return {
        "status": "optimal",
        "obj": min_costs[1],
        "path": path,
        "open_stations": open_stations,
        "details": {
            "travel_distance": total_dist,
            "fixed_cost": sum(fixed_costs[n] for n in open_stations)
        }
    }

if __name__ == "__main__":
    print(solve_drone_charging_network())

