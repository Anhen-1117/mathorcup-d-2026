"""
车型1+2 双车型优化配送方案

问题描述：
- 同时可选用车型1、车型2两种车辆
- 目标1：总运输车辆最少
- 目标2：总运输成本最低
- 输出两种目标的对比结果

车辆参数：
- 车型1：420×210×220 cm，额定载重6000kg，单次成本450元
- 车型2：680×245×250 cm，额定载重10000kg，单次成本700元

货物信息：
- G1：80件，60×40×30 cm，12kg
- G2：100件，50×35×25 cm，8kg
- G3：30件，70×50×40 cm，15kg（易碎件，只能放地面）
- G4：40件，80×60×50 cm，25kg
- G5：50件，40×40×60 cm，18kg
"""

import numpy as np
import random
import time
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

# ===================== 车辆预计算数据 =====================
TRUCK_PARAMS = {
    1: {'L': 420, 'W': 210, 'H': 220, 'M_max': 6000, 'cost': 450},
    2: {'L': 680, 'W': 245, 'H': 250, 'M_max': 10000, 'cost': 700}
}
for t in [1, 2]:
    p = TRUCK_PARAMS[t]
    p['volume'] = p['L'] * p['W'] * p['H'] / 1e6


def generate_cargos():
    """生成货物数据"""
    return (
        [{'id': f'G1_{i+1}', 'type': 1, 'l': 60, 'w': 40, 'h': 30, 'weight': 12,
          'vol': 0.072, 'area': 2400} for i in range(80)] +
        [{'id': f'G2_{i+1}', 'type': 1, 'l': 50, 'w': 35, 'h': 25, 'weight': 8,
          'vol': 0.04375, 'area': 1750} for i in range(100)] +
        [{'id': f'G3_{i+1}', 'type': 2, 'l': 70, 'w': 50, 'h': 40, 'weight': 15,
          'vol': 0.14, 'area': 3500} for i in range(30)] +
        [{'id': f'G4_{i+1}', 'type': 3, 'l': 80, 'w': 60, 'h': 50, 'weight': 25,
          'vol': 0.24, 'area': 4800} for i in range(40)] +
        [{'id': f'G5_{i+1}', 'type': 3, 'l': 40, 'w': 40, 'h': 60, 'weight': 18,
          'vol': 0.096, 'area': 1600} for i in range(50)]
    )


def blf_decode(seq, truck_params):
    """BLF解码算法"""
    L, W, H = truck_params['L'], truck_params['W'], truck_params['H']
    M_max = truck_params['M_max']
    H_limit = H - 3

    placed = []
    tw, tv = 0, 0

    for g in seq:
        ok = False

        # 标准件6种姿态，定向件和易碎件1种姿态
        if g['type'] == 1:
            poses = [(g['l'], g['w'], g['h']), (g['l'], g['h'], g['w']),
                     (g['w'], g['l'], g['h']), (g['w'], g['h'], g['l']),
                     (g['h'], g['l'], g['w']), (g['h'], g['w'], g['l'])]
        else:
            poses = [(g['l'], g['w'], g['h'])]

        # 易碎件只能放地面
        if g['type'] == 2:
            z_list = [0]
        else:
            z_set = {0}
            for p in placed:
                z_set.add(p['z'] + p['h'])
            z_list = sorted(z_set)

        for cl, cw, ch in poses:
            if ch > H_limit:
                continue

            valid_z = [z for z in z_list if z + ch <= H_limit]
            if not valid_z:
                continue

            # 候选点
            candidates = set()
            if cl <= L and cw <= W:
                candidates.add((0, 0))
                candidates.add((L - cl, 0))
                candidates.add((0, W - cw))
                candidates.add((L - cl, W - cw))

            for p in placed:
                x1, y1 = p['x'] + p['l'], p['y']
                if x1 + cl <= L and cw <= W:
                    candidates.add((x1, y1))
                x2, y2 = p['x'], p['y'] + p['w']
                if cl <= L and y2 + cw <= W:
                    candidates.add((x2, y2))

            for z in valid_z:
                # 分离同层和支撑层货物
                same_z = []
                support = []
                for p in placed:
                    if p['z'] <= z < p['z'] + p['h']:
                        same_z.append(p)
                    elif p['z'] + p['h'] == z:
                        support.append(p)

                g_weight = g['weight']
                g_area = cl * cw

                for x, y in candidates:
                    if x < 0 or y < 0 or x + cl > L or y + cw > W:
                        continue

                    # 碰撞检测
                    collide = False
                    for p in same_z:
                        if not (x + cl <= p['x'] or p['x'] + p['l'] <= x or
                                y + cw <= p['y'] or p['y'] + p['w'] <= y):
                            collide = True
                            break
                    if collide:
                        continue

                    # 支撑检查
                    if z > 0 and support:
                        support_area = 0
                        for p in support:
                            sx = max(x, p['x'])
                            ex = min(x + cl, p['x'] + p['l'])
                            sy = max(y, p['y'])
                            ey = min(y + cw, p['y'] + p['w'])
                            if sx < ex and sy < ey:
                                support_area += (ex - sx) * (ey - sy)
                        if support_area < g_area:
                            continue

                    # 承压检查
                    if g_area > 0 and g_weight / (g_area / 1e4) > 500:
                        continue

                    # 载重检查
                    if tw + g_weight > M_max:
                        continue

                    # 放置成功
                    placed.append({
                        'id': g['id'], 'type': g['type'],
                        'x': x, 'y': y, 'z': z,
                        'l': cl, 'w': cw, 'h': ch,
                        'weight': g_weight, 'vol': g['vol']
                    })
                    tw += g_weight
                    tv += g['vol']
                    ok = True
                    break
                if ok:
                    break
            if ok:
                break

    return min(tv / truck_params['volume'], tw / M_max), placed


def iga_evolve(cargos, truck_params, pop=150, iterations=150):
    """IGA遗传算法求解装箱问题
    pop=150, iterations=150 → 每车型 22,500 次评估
    预估每辆车耗时约 8-10 分钟，总计约 2-3 小时
    """
    n = len(cargos)
    if n == 0:
        return 0, []

    # 初始化种群
    base = sorted(cargos, key=lambda x: (x['type'], -x['weight'] / x['vol']))
    pop_list = [base.copy() for _ in range(pop)]
    for ind in pop_list:
        random.shuffle(ind[:n // 2])

    best_fit = 0
    best_plan = []

    # 进度条：显示迭代进度
    with tqdm(total=iterations, desc=f"车型{truck_params['cost']}元进化", unit="代", leave=False) as pbar:
        for gen in range(iterations):
            # 评估
            results = [(blf_decode(ind, truck_params)[0], ind) for ind in pop_list]
            results.sort(key=lambda x: -x[0])

            if results[0][0] > best_fit:
                best_fit, best_plan = results[0]

            # 精英保留
            new_pop = [results[0][1].copy()]

            # 生成新个体
            while len(new_pop) < pop:
                p1 = pop_list[random.randint(0, pop - 1)]
                p2 = pop_list[random.randint(0, pop - 1)]

                if len(p1) < 2:
                    child = p1.copy()
                else:
                    a, b = sorted(random.sample(range(len(p1)), 2))
                    child = p1[:a] + p2[a:b] + p1[b:]
                    # 去重
                    seen = set()
                    child = [c for c in child if c['id'] not in seen and not seen.add(c['id'])]

                if len(child) >= 2 and random.random() < 0.1:
                    i, j = random.sample(range(len(child)), 2)
                    child[i], child[j] = child[j], child[i]

                new_pop.append(child)

            pop_list = new_pop[:pop]
            pbar.update(1)
            pbar.set_postfix({'最优满载率': f'{best_fit:.1%}'})

    return best_fit, best_plan


def solve_objective1(cargos):
    """目标1：车辆最少 - 每步选择装载货物最多的车型"""
    remaining = cargos.copy()
    trucks = []
    total = len(cargos)
    start_t = time.time()
    last_log = start_t

    print("\n===== 目标1：计算最少车辆 =====")
    print(f"总货物数: {total} 件")

    pbar = tqdm(total=total, desc="目标1装载", unit="件", leave=False)

    while remaining:
        best_res = None
        best_type = 1

        for t in [1, 2]:
            fit, plan = iga_evolve(remaining, TRUCK_PARAMS[t])
            if not best_res or len(plan) > len(best_res[1]):
                best_res = (fit, plan)
                best_type = t

        if not best_res or len(best_res[1]) == 0:
            break

        fit, plan = best_res
        p = TRUCK_PARAMS[best_type]
        v = sum(x['vol'] for x in plan)
        w = sum(x['weight'] for x in plan)

        trucks.append({
            'type': best_type,
            'fitness': fit,
            'volume_util': v / p['volume'],
            'weight_util': w / p['M_max'],
            'count': len(plan),
            'cost': p['cost'],
            'plan': plan
        })

        loaded_ids = {x['id'] for x in plan}
        remaining = [x for x in remaining if x['id'] not in loaded_ids]
        pbar.update(len(plan))

        if time.time() - last_log >= 60:
            elapsed = time.time() - start_t
            loaded = total - len(remaining)
            print(f"\n[{elapsed:.0f}秒] 已装载 {loaded}/{total} 件, 车辆数: {len(trucks)}")
            last_log = time.time()

    pbar.close()
    elapsed = time.time() - start_t
    print(f"\n目标1完成！总车辆数: {len(trucks)}, 耗时: {elapsed:.1f}秒")
    return trucks


def solve_objective2(cargos):
    """目标2：成本最低 - 每步选择单件成本最低的车型"""
    remaining = cargos.copy()
    trucks = []
    total = len(cargos)
    start_t = time.time()
    last_log = start_t

    print("\n===== 目标2：计算最低成本 =====")
    print(f"总货物数: {total} 件")

    pbar = tqdm(total=total, desc="目标2装载", unit="件", leave=False)

    while remaining:
        best_res = None
        best_type = 1
        best_cost_per_cargo = float('inf')

        for t in [1, 2]:
            fit, plan = iga_evolve(remaining, TRUCK_PARAMS[t])
            if len(plan) > 0:
                cost_per = TRUCK_PARAMS[t]['cost'] / len(plan)
                if cost_per < best_cost_per_cargo:
                    best_cost_per_cargo = cost_per
                    best_res = (fit, plan)
                    best_type = t

        if not best_res or len(best_res[1]) == 0:
            break

        fit, plan = best_res
        p = TRUCK_PARAMS[best_type]
        v = sum(x['vol'] for x in plan)
        w = sum(x['weight'] for x in plan)

        trucks.append({
            'type': best_type,
            'fitness': fit,
            'volume_util': v / p['volume'],
            'weight_util': w / p['M_max'],
            'count': len(plan),
            'cost': p['cost'],
            'plan': plan
        })

        loaded_ids = {x['id'] for x in plan}
        remaining = [x for x in remaining if x['id'] not in loaded_ids]
        pbar.update(len(plan))

        if time.time() - last_log >= 60:
            elapsed = time.time() - start_t
            loaded = total - len(remaining)
            print(f"\n[{elapsed:.0f}秒] 已装载 {loaded}/{total} 件, 车辆数: {len(trucks)}")
            last_log = time.time()

    pbar.close()
    elapsed = time.time() - start_t
    print(f"\n目标2完成！总车辆数: {len(trucks)}, 耗时: {elapsed:.1f}秒")
    return trucks


def print_results(r1, r2):
    """打印对比结果"""
    print("\n" + "=" * 70)
    print("==================== 两种目标方案对比 ====================")
    print("=" * 70)

    v1, c1 = len(r1), sum(t['cost'] for t in r1)
    v2, c2 = len(r2), sum(t['cost'] for t in r2)

    print(f"\n{'指标':<20} {'目标1(车辆最少)':<20} {'目标2(成本最低)':<20}")
    print("-" * 60)
    print(f"{'总车辆数':<20} {v1:<20} {v2:<20}")
    print(f"{'总成本(元)':<20} {c1:<20} {c2:<20}")
    print(f"{'装载货物数':<20} {sum(t['count'] for t in r1):<20} {sum(t['count'] for t in r2):<20}")
    print(f"{'车型1数量':<20} {sum(1 for t in r1 if t['type']==1):<20} {sum(1 for t in r2 if t['type']==1):<20}")
    print(f"{'车型2数量':<20} {sum(1 for t in r1 if t['type']==2):<20} {sum(1 for t in r2 if t['type']==2):<20}")

    print("\n" + "=" * 70)
    print("==================== 各车辆装载情况 ====================")
    print("=" * 70)

    for title, res in [("目标1(车辆最少)", r1), ("目标2(成本最低)", r2)]:
        print(f"\n----- {title} -----")
        print(f"{'ID':<5} {'车型':<6} {'满载率':<10} {'空间率':<10} {'载重率':<10} {'件数':<6} {'成本':<6}")
        print("-" * 55)
        for i, t in enumerate(res, 1):
            print(f"{i:<5} {t['type']:<6} {t['fitness']:.1%}     {t['volume_util']:.1%}     {t['weight_util']:.1%}     {t['count']:<6} {t['cost']:<6}")

    print("\n" + "=" * 70)
    print("==================== 分析结论 ====================")
    print("=" * 70)

    diff_v = v1 - v2
    diff_c = c1 - c2

    if diff_v < 0:
        print(f"\n目标1车辆数更少，少{diff_v}辆")
    elif diff_v > 0:
        print(f"\n目标2车辆数更少，少{diff_v}辆")
    else:
        print(f"\n两种目标车辆数相同，均为{v1}辆")

    if diff_c < 0:
        print(f"目标1成本更低，节省{diff_c}元")
    elif diff_c > 0:
        print(f"目标2成本更低，节省{diff_c}元")
    else:
        print(f"两种目标成本相同，均为{c1}元")

    print("\n说明：目标1追求车辆数少，可能使用更多大型车；")
    print("      目标2追求成本低，倾向于选择性价比高的配置")
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("========== 车型1+2 双车型优化配送方案 ==========")
    print("========== 精确版（种群150, 迭代150）==========")
    print("=" * 70)
    print("\n车型1: 420×210×220 cm, 载重6000kg, 成本450元")
    print("车型2: 680×245×250 cm, 载重10000kg, 成本700元")
    print("\nIGA参数: 种群150, 迭代150")
    print("预估总运行时间: 约2-3小时")

    cargos = generate_cargos()
    print(f"\n货物总数: {len(cargos)} 件")

    print("\n" + "=" * 70)
    print("【目标1】总运输车辆最少")
    print("=" * 70)
    result1 = solve_objective1(cargos)

    print("\n" + "=" * 70)
    print("【目标2】总运输成本最低")
    print("=" * 70)
    result2 = solve_objective2(cargos)

    print_results(result1, result2)
