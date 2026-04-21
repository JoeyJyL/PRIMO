# -*- coding: utf-8 -*-
"""
TD-MTDRP - Simplified Direct MILP
1 truck, 2 drones, 6 customers
"""

import gurobipy as gp
from gurobipy import GRB
import math

def solve():
    depot = (50, 50)
    cust = {
        1: {'pos':(20,80),'dem':5,'wt':4,'st':2,'sd':1},
        2: {'pos':(80,80),'dem':8,'wt':12,'st':3,'sd':1.5},
        3: {'pos':(20,20),'dem':6,'wt':5,'st':2,'sd':1},
        4: {'pos':(80,20),'dem':7,'wt':9,'st':3,'sd':1.5},
        5: {'pos':(50,80),'dem':4,'wt':3,'st':2,'sd':1},
        6: {'pos':(50,20),'dem':5,'wt':6,'st':2,'sd':1},
    }
    N = [1,2,3,4,5,6]
    m = 2; Q = 30; D = 10; B = 20

    def dist(a,b): return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
    def pos(i): return depot if i in [0,7] else cust[i]['pos']
    def tau(i,j): return dist(pos(i),pos(j))
    def td(i,j): return dist(pos(i),pos(j))/2.0

    # Drone eligible + feasible pairs
    drone_elig = [i for i in N if cust[i]['wt'] <= D]  # [1,3,4,5,6]
    # Feasible drone arcs: one-way flight time <= B
    da = [(i,j) for i in [0]+N for j in drone_elig if i!=j and td(i,j)<=B]
    print(f"Drone arcs: {[(i,j) for i,j in da]}")

    model = gp.Model("TDMTDRP"); model.setParam("OutputFlag",1)

    # Truck arcs (including depot self-loop NOT allowed)
    ta = [(i,j) for i in [0]+N for j in N+[7] if i!=j]
    x = model.addVars(ta, vtype=GRB.BINARY, name="x")
    y = model.addVars(da, vtype=GRB.BINARY, name="y")
    
    # z[j]=1 if j is served by truck (delivery at j)
    z = model.addVars(N, vtype=GRB.BINARY, name="z")
    
    # Timing
    arr = model.addVars([0]+N+[7], lb=0, name="arr")
    dep = model.addVars([0]+N+[7], lb=0, name="dep")
    u = model.addVars([0]+N+[7], lb=0, ub=8, name="u")

    M = 500

    # Objective: minimize return time
    model.setObjective(arr[7], GRB.MINIMIZE)

    # (1) Each customer served exactly once
    for j in N:
        ds = gp.quicksum(y[i,j] for i in [0]+N if (i,j) in da)
        model.addConstr(z[j] + ds == 1, f"serve_{j}")

    # (2) C2 must be truck (weight > D)
    model.addConstr(z[2] == 1, "c2_truck")

    # (3) Truck departs depot once, returns once
    model.addConstr(gp.quicksum(x[0,j] for j in N if (0,j) in ta)==1, "depart")
    model.addConstr(gp.quicksum(x[i,7] for i in N if (i,7) in ta)==1, "return")

    # (4) Flow: truck visits node j => inflow = outflow
    # Truck visits j if: (a) serves j, OR (b) launches drone from j
    # Let v[j] = 1 if truck visits j
    v = model.addVars(N, vtype=GRB.BINARY, name="v")
    
    for j in N:
        inf = gp.quicksum(x[i,j] for i in [0]+N if (i,j) in ta)
        outf = gp.quicksum(x[j,k] for k in N+[7] if (j,k) in ta)
        model.addConstr(inf == v[j], f"inf_{j}")
        model.addConstr(outf == v[j], f"outf_{j}")

    # Truck must visit j if it serves j
    for j in N:
        model.addConstr(z[j] <= v[j], f"visit_if_serve_{j}")

    # Drone can only launch from visited node
    for (i,j) in da:
        if i == 0: continue
        model.addConstr(y[i,j] <= v[i], f"launch_{i}_{j}")

    # Max m drones per stop
    for i in [0]+N:
        drones_from_i = gp.quicksum(y[i,k] for k in drone_elig if (i,k) in da)
        model.addConstr(drones_from_i <= m, f"maxd_{i}")

    # Capacity (truck-carried parcels only)
    model.addConstr(gp.quicksum(cust[j]['dem']*z[j] for j in N) <= Q, "cap")

    # MTZ
    model.addConstr(u[0]==0,"u0")
    for (i,j) in ta:
        if j not in [0,7]:
            model.addConstr(u[j] >= u[i]+1 - 8*(1-x[i,j]), f"mtz_{i}_{j}")

    # Timing: depot
    model.addConstr(arr[0]==0, "arr0")
    # dep[0] >= 0, and >= drone round trip if launched from depot
    for (i,j) in da:
        if i == 0:
            rt = td(0,j) + cust[j]['sd'] + td(j,0)
            model.addConstr(dep[0] >= rt * y[0,j], f"depot_drone_{j}")

    # Truck travel
    for (i,j) in ta:
        model.addConstr(arr[j] >= dep[i] + tau(i,j) - M*(1-x[i,j]), f"tt_{i}_{j}")

    # Departure from customer nodes
    for j in N:
        # Service time (only if truck serves)
        model.addConstr(dep[j] >= arr[j] + cust[j]['st']*z[j] - M*(1-v[j]), f"svc_{j}")
        # At minimum, dep >= arr if visited
        model.addConstr(dep[j] >= arr[j] - M*(1-v[j]), f"dep_arr_{j}")
        # Wait for drones
        for k in drone_elig:
            if (j,k) in da:
                rt = td(j,k) + cust[k]['sd'] + td(k,j)
                model.addConstr(dep[j] >= arr[j] + rt - M*(1-y[j,k]), f"dret_{j}_{k}")
    
    # If not visited, arr/dep are free (no constraint needed, they'll be 0 or unconstrained)

    model.optimize()

    if model.SolCount > 0:
        print(f"\n{'='*60}")
        print(f"OPTIMAL SOLUTION: Total duration = {model.ObjVal:.2f}")
        print(f"{'='*60}")
        
        # Route
        route = [0]
        cur = 0
        vis = set()
        while cur != 7:
            nxt = None
            for j in N+[7]:
                if (cur,j) in ta and x[cur,j].X > 0.5 and j not in vis:
                    nxt = j; break
            if nxt is None: break
            route.append(nxt); vis.add(nxt); cur = nxt
        rstr = ' -> '.join(['Depot' if n in [0,7] else f'C{n}' for n in route])
        print(f"\nTruck route: {rstr}")
        
        print(f"\nTiming:")
        for n in route:
            nm = 'Depot_start' if n==0 else ('Depot_end' if n==7 else f'C{n}')
            print(f"  {nm}: arr={arr[n].X:.2f}, dep={dep[n].X:.2f}")
        
        print(f"\nAssignments:")
        for j in N:
            if z[j].X > 0.5:
                print(f"  C{j}: TRUCK (demand={cust[j]['dem']})")
            else:
                for i in [0]+N:
                    if (i,j) in da and y[i,j].X > 0.5:
                        src = 'Depot' if i==0 else f'C{i}'
                        rt = td(i,j)+cust[j]['sd']+td(j,i)
                        print(f"  C{j}: DRONE from {src} (wt={cust[j]['wt']}, round_trip={rt:.2f})")
        
        print(f"\nDrone sorties:")
        for i in [0]+N:
            for k in drone_elig:
                if (i,k) in da and y[i,k].X > 0.5:
                    src = 'Depot' if i==0 else f'C{i}'
                    print(f"  {src}->C{k}: fly={td(i,k):.1f} + svc={cust[k]['sd']} + fly={td(k,i):.1f} = {td(i,k)+cust[k]['sd']+td(k,i):.1f}")
        
        td_sum = sum(cust[j]['dem']*z[j].X for j in N)
        dd_sum = sum(cust[j]['dem']*(1-z[j].X) for j in N)
        print(f"\nDemand: truck={td_sum:.0f}, drone={dd_sum:.0f}, total={td_sum+dd_sum:.0f}")
    else:
        print(f"Status: {model.status}")
        if model.status == GRB.INFEASIBLE:
            model.computeIIS()
            print("IIS constraints:")
            for c in model.getConstrs():
                if c.IISConstr: print(f"  {c.ConstrName}")

solve()