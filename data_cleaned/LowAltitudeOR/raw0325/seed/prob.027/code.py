"""
SkyCity Transport Authority - UAM垂直起降场选址与需求路由优化
问题类型: 整数线性规划 (ILP)
目标: 最大化总利润 (收入 - 运营成本 - 固定成本)
"""

import gurobipy as gp
from gurobipy import GRB


def solve_uam_vertiport():
    # ========== 数据输入 ==========
    # 候选垂直起降场集合 M
    M = [1, 2, 3, 4, 5]
    vertiport_names = {
        1: "Downtown",
        2: "Tech Park", 
        3: "Airport",
        4: "University",
        5: "Shopping Mall"
    }
    
    # 固定成本 f_i ($/day)
    f = {1: 5000, 2: 4500, 3: 6000, 4: 4000, 5: 5500}
    
    # OD需求对集合 TR
    TR = [1, 2, 3, 4, 5, 6]
    od_info = {
        1: ("Zone A", "Zone B"),
        2: ("Zone A", "Zone C"),
        3: ("Zone B", "Zone C"),
        4: ("Zone C", "Zone D"),
        5: ("Zone A", "Zone D"),
        6: ("Zone B", "Zone D")
    }
    
    # 需求量 d^tr (pax/day)
    d = {1: 120, 2: 80, 3: 60, 4: 100, 5: 50, 6: 40}
    
    # 票价收入 R^tr ($/pax)
    R = {1: 150, 2: 100, 3: 80, 4: 120, 5: 180, 6: 140}
    
    # 运营成本矩阵 c_km ($/pax)
    c = {}
    for k in M:
        for m in M:
            if k != m:
                c[k, m] = 20 + abs(k - m) * 10
    
    # 需建设的垂直起降场数量
    p = 3
    
    # ========== 输出问题信息 ==========
    print("=" * 60)
    print("SkyCity UAM Network - 垂直起降场选址优化")
    print("=" * 60)
    
    print("\n【候选垂直起降场】")
    for i in M:
        print(f"  {i}. {vertiport_names[i]}: 固定成本 ${f[i]}/天")
    
    print(f"\n【参数】需建设站点数 p = {p}")
    
    print("\n【OD需求对】")
    for tr in TR:
        print(f"  OD{tr}: {od_info[tr][0]} → {od_info[tr][1]}, "
              f"需求={d[tr]}人, 票价=${R[tr]}/人")
    
    print("\n【运营成本矩阵 c_km ($/人)】")
    header = "      " + "".join([f"{m:>6}" for m in M])
    print(header)
    for k in M:
        row = f"  {k}:  "
        for m in M:
            if k == m:
                row += f"{'--':>6}"
            else:
                row += f"{c[k,m]:>6}"
        print(row)
    
    # ========== 构建模型 ==========
    print("\n" + "=" * 60)
    print("构建优化模型")
    print("=" * 60)
    
    model = gp.Model("UAM_Vertiport_Siting")
    model.setParam('OutputFlag', 0)
    
    # 决策变量 x_i: 站点i是否建设
    x = model.addVars(M, vtype=GRB.BINARY, name="x")
    
    # 决策变量 y_km^tr: OD对tr是否通过(k,m)路由
    y = {}
    for tr in TR:
        for k in M:
            for m in M:
                if k != m:
                    y[tr, k, m] = model.addVar(vtype=GRB.BINARY, 
                                                name=f"y_{tr}_{k}_{m}")
    
    # 目标函数: max Σ d^tr(R^tr - c_km)y_km^tr - Σ f_i x_i
    profit = gp.quicksum(
        d[tr] * (R[tr] - c[k, m]) * y[tr, k, m]
        for tr in TR for k in M for m in M if k != m
    )
    fixed_cost = gp.quicksum(f[i] * x[i] for i in M)
    
    model.setObjective(profit - fixed_cost, GRB.MAXIMIZE)
    
    # 约束1: 恰好选择p个站点
    model.addConstr(gp.quicksum(x[i] for i in M) == p, name="SitingBudget")
    
    # 约束2: 路由有效性 - 起点站必须开放
    for tr in TR:
        for k in M:
            model.addConstr(
                gp.quicksum(y[tr, k, m] for m in M if m != k) <= x[k],
                name=f"OriginValid_{tr}_{k}"
            )
    
    # 约束3: 路由有效性 - 终点站必须开放
    for tr in TR:
        for m in M:
            model.addConstr(
                gp.quicksum(y[tr, k, m] for k in M if k != m) <= x[m],
                name=f"DestValid_{tr}_{m}"
            )
    
    # 约束4: 单一路由 - 每个OD对最多服务一次
    for tr in TR:
        model.addConstr(
            gp.quicksum(y[tr, k, m] for k in M for m in M if k != m) <= 1,
            name=f"SingleRoute_{tr}"
        )
    
    print("\n模型构建完成:")
    print(f"  决策变量: x_i (i=1..5), y_km^tr (tr=1..6, k,m=1..5, k≠m)")
    print(f"  目标: 最大化利润")
    print(f"  约束: 站点预算、路由有效性、单一路由")
    
    # ========== 求解 ==========
    print("\n【求解中...】")
    model.optimize()
    
    # ========== 输出结果 ==========
    print("\n" + "=" * 60)
    print("求解结果")
    print("=" * 60)
    
    if model.status == GRB.OPTIMAL:
        print(f"\n✓ 最优目标值 (总利润): ${model.ObjVal:.2f}/天")
        
        # 选中的站点
        selected = [i for i in M if x[i].X > 0.5]
        total_fixed = sum(f[i] for i in selected)
        
        print(f"\n【选中的垂直起降场】({len(selected)}个)")
        for i in selected:
            print(f"  V{i} ({vertiport_names[i]}): 固定成本 ${f[i]}/天")
        print(f"  总固定成本: ${total_fixed}/天")
        
        # 服务的OD对
        print("\n【需求服务情况】")
        total_revenue = 0
        total_op_cost = 0
        served_count = 0
        
        for tr in TR:
            served = False
            for k in M:
                for m in M:
                    if k != m and y[tr, k, m].X > 0.5:
                        revenue = d[tr] * R[tr]
                        op_cost = d[tr] * c[k, m]
                        net = d[tr] * (R[tr] - c[k, m])
                        total_revenue += revenue
                        total_op_cost += op_cost
                        served_count += 1
                        print(f"  OD{tr} ({od_info[tr][0]}→{od_info[tr][1]}): "
                              f"路由 V{k}→V{m}, 需求={d[tr]}, "
                              f"收入=${revenue}, 运营成本=${op_cost}, 净利润=${net}")
                        served = True
                        break
                if served:
                    break
            if not served:
                print(f"  OD{tr} ({od_info[tr][0]}→{od_info[tr][1]}): 未服务")
        
        print(f"\n【利润汇总】")
        print(f"  服务OD对数: {served_count}/{len(TR)}")
        print(f"  总收入: ${total_revenue}")
        print(f"  总运营成本: ${total_op_cost}")
        print(f"  总固定成本: ${total_fixed}")
        print(f"  净利润: ${total_revenue - total_op_cost - total_fixed}")
        
        return {
            "status": "optimal",
            "objective": model.ObjVal,
            "selected_vertiports": selected,
            "served_od_pairs": served_count
        }
    else:
        print("✗ 求解失败!")
        return {"status": "failed"}


if __name__ == "__main__":
    result = solve_uam_vertiport()
    print("\n" + "=" * 60)
    print(f"最终结果: {result}")