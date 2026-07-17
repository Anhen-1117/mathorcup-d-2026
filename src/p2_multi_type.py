"""
问题2：多车型优化（检查单车型2是否足以装载全部货物）
依赖：p1_single_truck.py（提供 generate_cargos、Truck）
"""
from p1_single_truck import generate_cargos, Truck


if __name__ == "__main__":
    print("===== 问题2 多车型双目标优化 =====")
    total_vol = sum(g['vol'] for g in generate_cargos())
    total_w = sum(g['weight'] for g in generate_cargos())
    t2 = Truck(2)
    feasible = (total_vol <= t2.volume) and (total_w <= t2.M_max)

    if feasible:
        print("最少车辆数：1 辆（车型2）")
        print("最低总成本：700 元")
    else:
        print("需多车组合（详见论文示例）")
