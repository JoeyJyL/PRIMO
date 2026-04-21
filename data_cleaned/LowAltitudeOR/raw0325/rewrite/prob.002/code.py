import gurobipy as gp
from gurobipy import GRB

def solve_robust_facility_location():
    # ==========================================
    # 1. 问题数据定义
    # ==========================================
    
    # 集合定义
    HUBS = ['H1', 'H2', 'H3']
    POINTS = ['C1', 'C2', 'C3', 'C4']
    SCENARIOS = ['Scenario_A', 'Scenario_B']
    
    # 建设成本 (Fixed Cost)
    build_cost = {
        'H1': 5000,
        'H2': 4000,
        'H3': 7000
    }
    
    # 距离矩阵 (km)
    # 999 表示距离 > 10km (不可达)
    # 结构: distance[hub][point]
    distance = {
        'H1': {'C1': 4,   'C2': 8,   'C3': 999, 'C4': 5},
        'H2': {'C1': 6,   'C2': 3,   'C3': 9,   'C4': 999},
        'H3': {'C1': 5,   'C2': 999, 'C3': 4,   'C4': 2}
    }
    
    # 需求量矩阵 (单位物资)
    # 结构: demand[point][scenario]
    demand = {
        'C1': {'Scenario_A': 10, 'Scenario_B': 50},
        'C2': {'Scenario_A': 20, 'Scenario_B': 20},
        'C3': {'Scenario_A': 15, 'Scenario_B': 80},
        'C4': {'Scenario_A': 10, 'Scenario_B': 10}
    }
    
    # 单位运输成本 ($/unit/km)
    UNIT_TRANS_COST = 1.0
    
    # ==========================================
    # 2. 模型构建
    # ==========================================
    
    try:
        # 创建模型
        m = gp.Model("Robust_Drone_Hub_Location")
        
        # --- 决策变量 ---
        
        # Stage 1: 建设决策 y[j] (0/1)
        y = m.addVars(HUBS, vtype=GRB.BINARY, name="Build")
        
        # Stage 2: 分配决策 x[j, i, s] (0/1)
        # 表示在情景 s 下，点 i 是否由枢纽 j 服务
        x = m.addVars(HUBS, POINTS, SCENARIOS, vtype=GRB.BINARY, name="Assign")
        
        # 辅助变量 Z: 代表最坏情景下的运营成本 (Continuous)
        Z = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="WorstCaseCost")
        
        # --- 目标函数 ---
        # Min (总建设成本 + 最坏情景运营成本 Z)
        total_fixed_cost = gp.quicksum(build_cost[j] * y[j] for j in HUBS)
        m.setObjective(total_fixed_cost + Z, GRB.MINIMIZE)
        
        # --- 约束条件 ---
        
        # 约束 1: 鲁棒性约束 (Min-Max 线性化)
        # Z >= 任意情景 s 的总运营成本
        for s in SCENARIOS:
            scenario_op_cost = gp.quicksum(
                demand[i][s] * distance[j][i] * UNIT_TRANS_COST * x[j, i, s]
                for j in HUBS for i in POINTS
                if distance[j][i] < 999 # 只计算有效路径
            )
            m.addConstr(Z >= scenario_op_cost, name=f"Robust_Z_{s}")
            
        # 约束 2: 强制覆盖约束
        # 在任意情景 s 下，每个需求点 i 必须被分配给恰好一个有效的枢纽
        for s in SCENARIOS:
            for i in POINTS:
                # 仅针对距离 <= 10km 的枢纽求和
                valid_hubs = [j for j in HUBS if distance[j][i] < 999]
                m.addConstr(
                    gp.quicksum(x[j, i, s] for j in valid_hubs) == 1,
                    name=f"Cover_{i}_{s}"
                )
        
        # 约束 3: 建设关联约束
        # 只有当枢纽 j 建设了(y[j]=1)，才能在任意情景下分配任务给它
        for s in SCENARIOS:
            for j in HUBS:
                for i in POINTS:
                    if distance[j][i] < 999:
                        m.addConstr(x[j, i, s] <= y[j], name=f"Link_{j}_{i}_{s}")

        # ==========================================
        # 3. 求解与结果输出
        # ==========================================
        
        print("\nStarting Optimization...")
        m.optimize()
        
        if m.status == GRB.OPTIMAL:
            print("\n" + "="*40)
            print(f"✅ Optimal Solution Found")
            print("="*40)
            print(f"Total Robust Cost: ${m.ObjVal:,.2f}")
            print(f"  - Construction Cost: ${total_fixed_cost.getValue():,.2f}")
            print(f"  - Worst-Case Op Cost (Z): ${Z.X:,.2f}")
            
            # 输出建设方案
            print("\n🏗️  Hub Construction Plan (Stage 1):")
            built_hubs = [j for j in HUBS if y[j].X > 0.5]
            for j in built_hubs:
                print(f"  - Build Hub {j} (Cost: ${build_cost[j]})")
                
            # 输出各情景下的运营详情
            print("\n🚚 Operational Plan (Stage 2):")
            for s in SCENARIOS:
                print(f"\n  [ {s} ]")
                current_op_cost = 0
                print(f"    {'Point':<6} {'Demand':<8} {'Served By':<10} {'Dist':<6} {'Cost'}")
                print(f"    {'-'*6} {'-'*8} {'-'*10} {'-'*6} {'-'*6}")
                
                for i in POINTS:
                    for j in HUBS:
                        if x[j, i, s].X > 0.5:
                            cost = demand[i][s] * distance[j][i] * UNIT_TRANS_COST
                            current_op_cost += cost
                            print(f"    {i:<6} {demand[i][s]:<8} {j:<10} {distance[j][i]:<6} ${cost:.1f}")
                            
                print(f"    >> Total Op Cost for {s}: ${current_op_cost:,.2f}")
                if abs(current_op_cost - Z.X) < 1e-4:
                    print(f"    ⚠️  (This is the Worst-Case Scenario determining Z)")
                    
        else:
            print("No optimal solution found.")
            
    except gp.GurobiError as e:
        print(f"Gurobi Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    solve_robust_facility_location()