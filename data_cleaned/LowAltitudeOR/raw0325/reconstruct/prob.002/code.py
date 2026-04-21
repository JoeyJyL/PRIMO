import gurobipy as gp
from gurobipy import GRB

def solve_robust_facility_location():
    # ==========================================
    # 1. 问题数据定义
    # ==========================================
    
    HUBS = ['H1', 'H2', 'H3']
    POINTS = ['C1', 'C2', 'C3', 'C4']
    SCENARIOS = ['Scenario_A', 'Scenario_B']
    
    build_cost = {
        'H1': 5000,
        'H2': 4000,
        'H3': 7000
    }
    
    distance = {
        'H1': {'C1': 4,   'C2': 8,   'C3': 999, 'C4': 5},
        'H2': {'C1': 6,   'C2': 3,   'C3': 9,   'C4': 999},
        'H3': {'C1': 5,   'C2': 999, 'C3': 4,   'C4': 2}
    }
    
    demand = {
        'C1': {'Scenario_A': 10, 'Scenario_B': 50},
        'C2': {'Scenario_A': 20, 'Scenario_B': 20},
        'C3': {'Scenario_A': 15, 'Scenario_B': 80},
        'C4': {'Scenario_A': 10, 'Scenario_B': 10}
    }
    
    UNIT_TRANS_COST = 1.0
    
    # 预计算可达集合
    # C_j: 从 hub j 可达的需求点集合
    C_j = {j: [i for i in POINTS if distance[j][i] < 999] for j in HUBS}
    # H_i: 可服务需求点 i 的 hub 集合
    H_i = {i: [j for j in HUBS if distance[j][i] < 999] for i in POINTS}
    # M_j: Big-M 值 = 可达需求点数量
    M_j = {j: len(C_j[j]) for j in HUBS}

    # ==========================================
    # 2. 模型构建 (Big-M 聚合容量 + 辅助成本变量)
    # ==========================================
    
    try:
        m = gp.Model("Robust_Hub_BigM")
        
        # --- 决策变量 ---
        y = m.addVars(HUBS, vtype=GRB.BINARY, name="Build")
        
        x = m.addVars(
            [(j, i, s) for j in HUBS for i in POINTS for s in SCENARIOS
             if distance[j][i] < 999],
            vtype=GRB.BINARY, name="Assign"
        )
        
        # 辅助变量: 每个情景的运营成本
        phi = m.addVars(SCENARIOS, vtype=GRB.CONTINUOUS, lb=0, name="ScenCost")
        
        # 辅助变量: 最坏情景运营成本
        Z = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="WorstCaseCost")
        
        # --- 目标函数 ---
        total_fixed_cost = gp.quicksum(build_cost[j] * y[j] for j in HUBS)
        m.setObjective(total_fixed_cost + Z, GRB.MINIMIZE)
        
        # --- 约束条件 ---
        
        # 约束1: 情景成本定义 (辅助变量)
        for s in SCENARIOS:
            m.addConstr(
                phi[s] == gp.quicksum(
                    demand[i][s] * distance[j][i] * UNIT_TRANS_COST * x[j, i, s]
                    for j in HUBS for i in POINTS
                    if distance[j][i] < 999
                ),
                name=f"ScenCostDef_{s}"
            )
        
        # 约束2: 鲁棒性约束 (Z >= 每个情景成本)
        for s in SCENARIOS:
            m.addConstr(Z >= phi[s], name=f"Robust_Z_{s}")
            
        # 约束3: 强制覆盖约束
        for s in SCENARIOS:
            for i in POINTS:
                valid_hubs = H_i[i]
                m.addConstr(
                    gp.quicksum(x[j, i, s] for j in valid_hubs) == 1,
                    name=f"Cover_{i}_{s}"
                )
        
        # 约束4: Big-M 聚合容量约束 (替代单独的 x <= y 约束)
        for s in SCENARIOS:
            for j in HUBS:
                reachable = C_j[j]
                if reachable:
                    m.addConstr(
                        gp.quicksum(x[j, i, s] for i in reachable) <= M_j[j] * y[j],
                        name=f"BigM_Cap_{j}_{s}"
                    )

        # ==========================================
        # 3. 求解与结果输出
        # ==========================================
        
        print("\nStarting Optimization...")
        m.optimize()
        
        if m.status == GRB.OPTIMAL:
            print("\n" + "="*40)
            print(f"Optimal Solution Found")
            print("="*40)
            print(f"Total Robust Cost: ${m.ObjVal:,.2f}")
            print(f"  - Construction Cost: ${total_fixed_cost.getValue():,.2f}")
            print(f"  - Worst-Case Op Cost (Z): ${Z.X:,.2f}")
            
            for s in SCENARIOS:
                print(f"  - {s} Op Cost (phi): ${phi[s].X:,.2f}")
            
            print("\nHub Construction Plan (Stage 1):")
            built_hubs = [j for j in HUBS if y[j].X > 0.5]
            for j in built_hubs:
                print(f"  - Build Hub {j} (Cost: ${build_cost[j]}, BigM={M_j[j]})")
                
            print("\nOperational Plan (Stage 2):")
            for s in SCENARIOS:
                print(f"\n  [ {s} ]")
                current_op_cost = 0
                print(f"    {'Point':<6} {'Demand':<8} {'Served By':<10} {'Dist':<6} {'Cost'}")
                print(f"    {'-'*6} {'-'*8} {'-'*10} {'-'*6} {'-'*6}")
                
                for i in POINTS:
                    for j in HUBS:
                        if distance[j][i] < 999 and x[j, i, s].X > 0.5:
                            cost = demand[i][s] * distance[j][i] * UNIT_TRANS_COST
                            current_op_cost += cost
                            print(f"    {i:<6} {demand[i][s]:<8} {j:<10} {distance[j][i]:<6} ${cost:.1f}")
                            
                print(f"    >> Total Op Cost for {s}: ${current_op_cost:,.2f}")
                if abs(current_op_cost - Z.X) < 1e-4:
                    print(f"    (This is the Worst-Case Scenario determining Z)")
            
            return {
                "status": "optimal",
                "obj": m.ObjVal
            }
        else:
            print("No optimal solution found.")
            return {"status": "infeasible"}
            
    except gp.GurobiError as e:
        print(f"Gurobi Error: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    result = solve_robust_facility_location()
    print(f"\nFinal Result: {result}")
