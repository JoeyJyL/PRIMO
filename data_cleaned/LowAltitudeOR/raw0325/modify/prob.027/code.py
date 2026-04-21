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
    
    # 需建设的垂直起降场数量 (reduced from 3 to 2)
    p = 2
    
    # ========== 构建模型 ==========
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
    
    # ========== 求解 ==========
    model.optimize()
    
    # ========== 输出结果 ==========
    if model.status == GRB.OPTIMAL:
        selected = [i for i in M if x[i].X > 0.5]
        total_fixed = sum(f[i] for i in selected)
        
        total_revenue = 0
        total_op_cost = 0
        served_count = 0
        
        for tr in TR:
            for k in M:
                for m in M:
                    if k != m and y[tr, k, m].X > 0.5:
                        revenue = d[tr] * R[tr]
                        op_cost = d[tr] * c[k, m]
                        total_revenue += revenue
                        total_op_cost += op_cost
                        served_count += 1
        
        return {
            "status": "optimal",
            "obj": round(model.ObjVal, 2),
            "selected_vertiports": selected,
            "served_od_pairs": served_count,
            "total_revenue": total_revenue,
            "total_op_cost": total_op_cost,
            "total_fixed_cost": total_fixed
        }
    else:
        return {"status": "failed"}


if __name__ == "__main__":
    result = solve_uam_vertiport()
    print(result)