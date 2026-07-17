"""
问题1–2：车型2 最少车辆数
依赖：p1_single_truck.py（提供 IGA、generate_cargos）
"""
from p1_single_truck import generate_cargos, IGA


if __name__ == "__main__":
    print("===== 问题1–2 车型2 最少车辆数 =====")
    cargos = generate_cargos()
    remaining = cargos.copy()
    trucks_used = 0

    while remaining:
        iga = IGA(2, remaining)
        bf, plan = iga.run()
        ids = {g['id'] for g in plan}
        remaining = [g for g in remaining if g['id'] not in ids]
        trucks_used += 1

    print(f"最少需要车型2：{trucks_used} 辆")
    print("可装完全部300件")
