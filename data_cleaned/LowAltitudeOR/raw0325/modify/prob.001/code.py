import gurobipy as gp
from gurobipy import GRB
import math

def solve_drone_location():
    # --- 参数设置 ---
    fp = 10.0  # 最大载荷航程 (站点间)
    fd = 6.0   # 配送范围 (站点到客户)
    p_stations = 3  # 最大站点数 (1仓库 + 2新站点)
    
    # 站点坐标
    stations = {
        0: (2, 10),   # 仓库
        1: (8, 14),   # S1
        2: (8, 6),    # S2
        3: (14, 14),  # S3
        4: (14, 6),   # S4
        5: (18, 10)   # S5
    }
    
    # 客户需求点
    customers = {
        6: {'loc': (4, 12), 'dem': 10},   # C1
        7: {'loc': (4, 8),  'dem': 15},   # C2
        8: {'loc': (10, 16), 'dem': 20},  # C3
        9: {'loc': (10, 4),  'dem': 25},  # C4 (医疗物资转运点，强制覆盖)
        10: {'loc': (16, 12), 'dem': 30}, # C5
        11: {'loc': (16, 8),  'dem': 20}  # C6
    }
    
    warehouse_indices = [0]
    candidate_indices = list(stations.keys())
    customer_indices = list(customers.keys())

    # 强制覆盖的客户 (C4 = index 9)
    mandatory_customers = [9]

    # 障碍物定义: x=[11, 13], y=[0, 15]
    obs_x_min, obs_x_max = 11, 13
    obs_y_max = 15

    def get_dist(p1, p2):
        """计算考虑障碍物的ESP距离（修正版：保证对称性）"""
        x1, y1 = p1
        x2, y2 = p2
        
        left, right = min(x1, x2), max(x1, x2)
        
        # 检查是否需要绕行
        intersects = False
        if left < obs_x_min and right > obs_x_max:
            # 计算线段在障碍物边界处的y值
            if abs(x2 - x1) > 1e-9:
                slope = (y2 - y1) / (x2 - x1)
                y_at_11 = y1 + slope * (11 - x1)
                y_at_13 = y1 + slope * (13 - x1)
                # 如果线段在障碍物x范围内的y值低于障碍物顶部，则需要绕行
                if y_at_11 < obs_y_max or y_at_13 < obs_y_max:
                    intersects = True

        if intersects:
            # 绕行路径：经过障碍物顶部的两个角点 (11,15) 和 (13,15)
            # 修正：根据起点终点的x坐标决定绕行顺序，保证距离对称
            corner_left = (obs_x_min, obs_y_max)   # (11, 15)
            corner_right = (obs_x_max, obs_y_max)  # (13, 15)
            
            # 计算经过两个角点的路径距离
            # 路径: p1 -> 近角点 -> 远角点 -> p2
            if x1 < x2:  # 从左到右
                d1 = math.sqrt((corner_left[0] - x1)**2 + (corner_left[1] - y1)**2)
                d2 = math.sqrt((x2 - corner_right[0])**2 + (y2 - corner_right[1])**2)
            else:  # 从右到左
                d1 = math.sqrt((corner_right[0] - x1)**2 + (corner_right[1] - y1)**2)
                d2 = math.sqrt((x2 - corner_left[0])**2 + (y2 - corner_left[1])**2)
            
            d_top = obs_x_max - obs_x_min  # 顶部边缘距离 = 2.0
            return d1 + d_top + d2
        else:
            return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    # 预计算覆盖集合 N_k
    N = {k: [] for k in customer_indices}
    for k in customer_indices:
        for j in candidate_indices:
            if get_dist(customers[k]['loc'], stations[j]) <= fd:
                N[k].append(j)

    # 预计算连通集合 Omega_i
    Omega = {i: [] for i in candidate_indices}
    for i in candidate_indices:
        for j in candidate_indices:
            if i == j:
                continue
            if get_dist(stations[i], stations[j]) <= fp:
                Omega[i].append(j)

    M = len(candidate_indices)  # Big M

    # --- 输出预处理信息 ---
    print("=" * 50)
    print("预处理信息")
    print("=" * 50)
    
    print("\n站点间ESP距离矩阵:")
    print("      ", end="")
    for j in candidate_indices:
        name = "W1" if j == 0 else f"S{j}"
        print(f"{name:>7}", end="")
    print()
    for i in candidate_indices:
        name = "W1" if i == 0 else f"S{i}"
        print(f"{name:>5} ", end="")
        for j in candidate_indices:
            if i == j:
                print(f"{'--':>7}", end="")
            else:
                d = get_dist(stations[i], stations[j])
                mark = "✓" if d <= fp else ""
                print(f"{d:>6.2f}{mark}", end="")
        print()
    
    print("\n连通集合 Omega (距离 ≤ fp):")
    for i in candidate_indices:
        name = "W1" if i == 0 else f"S{i}"
        neighbors = ["W1" if s == 0 else f"S{s}" for s in Omega[i]]
        print(f"  {name}: {neighbors}")
    
    print("\n覆盖集合 N (距离 ≤ fd):")
    for k in customer_indices:
        name = f"C{k-5}"
        sites = ["W1" if s == 0 else f"S{s}" for s in N[k]]
        print(f"  {name} (需求={customers[k]['dem']}): {sites}")

    print(f"\n强制覆盖客户: C4 (医疗物资转运点)")

    # --- 建立模型 ---
    print("\n" + "=" * 50)
    print("求解优化模型")
    print("=" * 50)
    
    model = gp.Model("DroneNetwork")
    model.setParam('OutputFlag', 0)  # 关闭详细输出

    # 决策变量
    X = model.addVars(candidate_indices, vtype=GRB.BINARY, name="X")
    Y = model.addVars(customer_indices, vtype=GRB.BINARY, name="Y")
    Z = model.addVars(candidate_indices, candidate_indices, vtype=GRB.INTEGER, lb=0, name="Z")

    # 目标函数：最大化覆盖需求
    model.setObjective(
        gp.quicksum(customers[k]['dem'] * Y[k] for k in customer_indices), 
        GRB.MAXIMIZE
    )

    # 约束1：覆盖定义
    for k in customer_indices:
        model.addConstr(
            gp.quicksum(X[j] for j in N[k]) >= Y[k], 
            name=f"Cover_{k}"
        )

    # 约束2：站点预算
    model.addConstr(
        gp.quicksum(X[i] for i in candidate_indices) == p_stations, 
        name="Budget"
    )
    
    # 约束3：仓库必须开放
    model.addConstr(X[0] == 1, name="Warehouse_Open")

    # 约束4：流容量
    for i in candidate_indices:
        out_flow = gp.quicksum(Z[i, j] for j in Omega[i])
        model.addConstr(out_flow <= (M - 1) * X[i], name=f"FlowCap_{i}")

    # 约束5&6：流守恒（向根节点汇聚）
    for i in candidate_indices:
        flow_out = gp.quicksum(Z[i, j] for j in Omega[i])
        flow_in = gp.quicksum(Z[j, i] for j in candidate_indices if i in Omega[j])
        
        if i in warehouse_indices:
            # 仓库作为汇点
            model.addConstr(flow_out - flow_in <= X[i] - M, name=f"FlowRoot_{i}")
        else:
            # 非仓库节点：若激活则必须有净流出
            model.addConstr(flow_out - flow_in >= X[i], name=f"FlowCons_{i}")

    # 约束7：强制覆盖 (C4 医疗物资转运点)
    for k in mandatory_customers:
        model.addConstr(Y[k] == 1, name=f"Mandatory_{k}")

    # 求解
    model.optimize()

    # --- 输出结果 ---
    print("\n" + "=" * 50)
    print("求解结果")
    print("=" * 50)
    
    if model.status == GRB.OPTIMAL:
        sol_stations = [i for i in candidate_indices if X[i].X > 0.5]
        covered_cust = [k for k in customer_indices if Y[k].X > 0.5]
        
        print(f"\n最优目标值: {model.ObjVal:.0f}")
        
        print("\n选中的站点:")
        for i in sol_stations:
            name = "W1 (仓库)" if i == 0 else f"S{i}"
            print(f"  {name}: {stations[i]}")
        
        print("\n客户覆盖情况:")
        total_demand = sum(customers[k]['dem'] for k in customer_indices)
        covered_demand = 0
        for k in customer_indices:
            name = f"C{k-5}"
            dem = customers[k]['dem']
            mandatory_mark = " [强制]" if k in mandatory_customers else ""
            if k in covered_cust:
                covered_demand += dem
                covering = [("W1" if s == 0 else f"S{s}") for s in N[k] if s in sol_stations]
                print(f"  {name} (需求={dem}){mandatory_mark}: ✓ 被 {covering} 覆盖")
            else:
                print(f"  {name} (需求={dem}){mandatory_mark}: ✗ 未覆盖")
        
        print(f"\n总覆盖率: {covered_demand}/{total_demand} ({100*covered_demand/total_demand:.1f}%)")
        
        print("\n网络连接流:")
        for i in candidate_indices:
            for j in Omega[i]:
                if Z[i, j].X > 0.5:
                    from_name = "W1" if i == 0 else f"S{i}"
                    to_name = "W1" if j == 0 else f"S{j}"
                    print(f"  {from_name} -> {to_name}: {Z[i,j].X:.0f}")
        
        return {
            "status": "optimal",
            "obj": model.ObjVal,
            "stations_selected": sol_stations,
            "customers_covered": covered_cust
        }
    else:
        print("模型不可行!")
        return {"status": "infeasible"}


if __name__ == "__main__":
    result = solve_drone_location()
    print("\n" + "=" * 50)
    print(f"最终结果: {result}")
