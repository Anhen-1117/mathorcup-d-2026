import numpy as np
import random
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

# ===================== 车辆定义（论文参数） =====================
class Truck:
    def __init__(self, t):
        # 车型1（轻型厢式货车）
        self.L, self.W, self.H = 420, 210, 220
        self.M_max = 6000
        self.cost = 450
        self.volume = self.L * self.W * self.H / 1e6

# ===================== 货物定义（论文300件） =====================
def generate_cargos():
    cargos = []
    # G1 80件 标准件 60×40×30 12
    cargos.extend([{'id': f'G1_{i+1}', 'type':1, 'l':60,'w':40,'h':30,'weight':12, 'vol':60*40*30/1e6} for i in range(80)])
    # G2 100件 标准件 50×35×25 8
    cargos.extend([{'id': f'G2_{i+1}', 'type':1, 'l':50,'w':35,'h':25,'weight':8, 'vol':50*35*25/1e6} for i in range(100)])
    # G3 30件 易碎件 70×50×40 15
    cargos.extend([{'id': f'G3_{i+1}', 'type':2, 'l':70,'w':50,'h':40,'weight':15, 'vol':70*50*40/1e6} for i in range(30)])
    # G4 40件 定向件 80×60×50 25
    cargos.extend([{'id': f'G4_{i+1}', 'type':3, 'l':80,'w':60,'h':50,'weight':25, 'vol':80*60*50/1e6} for i in range(40)])
    # G5 50件 定向件 40×40×60 18
    cargos.extend([{'id': f'G5_{i+1}', 'type':3, 'l':40,'w':40,'h':60,'weight':18, 'vol':40*40*60/1e6} for i in range(50)])
    return cargos

# ===================== BLF 解码（论文约束全满足） =====================
def blf_decode(seq, truck_type):
    truck = Truck(truck_type)
    placed = []
    tw, tv = 0, 0
    
    # 预计算车辆尺寸，避免重复访问属性
    truck_L, truck_W, truck_H = truck.L, truck.W, truck.H
    truck_M_max = truck.M_max

    for i, g in enumerate(seq):
        ok = False

        # 姿态：标准件6种，其余1种
        if g['type'] == 1:
            poses = [(g['l'],g['w'],g['h']),(g['l'],g['h'],g['w']),(g['w'],g['l'],g['h']),
                     (g['w'],g['h'],g['l']),(g['h'],g['l'],g['w']),(g['h'],g['w'],g['l'])]
        else:
            poses = [(g['l'],g['w'],g['h'])]

        # 预计算z层
        if g['type'] == 2:
            z_list = [0]
        else:
            z_set = {0}
            for p in placed:
                z_set.add(p['z'] + p['h'])
            z_list = sorted(z_set)

        for cl, cw, ch in poses:
            if ch > truck_H - 3:
                continue

            # 过滤有效z层
            valid_z = [z for z in z_list if z + ch <= truck_H - 3]
            if not valid_z:
                continue

            # 预计算候选点
            candidates = set()
            # 角落候选点
            if cl <= truck_L and cw <= truck_W:
                candidates.add((0, 0))
                candidates.add((truck_L - cl, 0))
                candidates.add((0, truck_W - cw))
                candidates.add((truck_L - cl, truck_W - cw))
            # 已放货物边缘候选点
            for p in placed:
                x1, y1 = p['x'] + p['l'], p['y']
                if x1 + cl <= truck_L and cw <= truck_W:
                    candidates.add((x1, y1))
                x2, y2 = p['x'], p['y'] + p['w']
                if cl <= truck_L and y2 + cw <= truck_W:
                    candidates.add((x2, y2))

            # 快速碰撞检测
            for z in valid_z:
                # 按z层过滤已放置货物
                same_z_placed = []
                support_layer = []
                for p in placed:
                    pz = p['z']
                    pzh = pz + p['h']
                    if pz <= z < pzh:
                        same_z_placed.append(p)
                    elif pzh == z:
                        support_layer.append(p)
                
                # 向量化碰撞检测
                same_z_count = len(same_z_placed)
                if same_z_count > 0:
                    # 提取坐标和尺寸（一次遍历完成）
                    px = np.empty(same_z_count, dtype=np.float64)
                    py = np.empty(same_z_count, dtype=np.float64)
                    pl = np.empty(same_z_count, dtype=np.float64)
                    pw = np.empty(same_z_count, dtype=np.float64)
                    for i_p, p in enumerate(same_z_placed):
                        px[i_p] = p['x']
                        py[i_p] = p['y']
                        pl[i_p] = p['l']
                        pw[i_p] = p['w']
                
                # 预计算货物重量和体积
                g_weight = g['weight']
                g_vol = g['vol']
                
                # 预计算货物底面积
                cargo_area = cl * cw
                
                for x, y in candidates:
                    # 边界检查
                    if x < 0 or y < 0 or x + cl > truck_L or y + cw > truck_W:
                        continue

                    # 碰撞检测（只检查同一z层）
                    collide = False
                    if same_z_count > 0:
                        # 向量化碰撞检测
                        x_end = x + cl
                        y_end = y + cw
                        # 优化碰撞检测逻辑，减少嵌套逻辑运算
                        no_collision = (x_end <= px) | (px + pl <= x) | (y_end <= py) | (py + pw <= y)
                        collide = not np.all(no_collision)
                    if collide:
                        continue

                    # 支撑面积计算（只检查下方货物）
                    if z > 0 and support_layer:
                        support_area = 0
                        x_end = x + cl
                        y_end = y + cw
                        for p in support_layer:
                            px_start = max(x, p['x'])
                            px_end = min(x_end, p['x'] + p['l'])
                            py_start = max(y, p['y'])
                            py_end = min(y_end, p['y'] + p['w'])
                            if px_start < px_end and py_start < py_end:
                                support_area += (px_end - px_start) * (py_end - py_start)
                        if support_area < cargo_area:
                            continue

                    # 承重检查
                    if cargo_area > 0 and g_weight / (cargo_area / 1e4) > 500:
                        continue

                    # 载重约束
                    if tw + g_weight > truck_M_max:
                        continue

                    # 放置货物
                    placed.append({
                        'id': g['id'], 'type': g['type'], 'x': x, 'y': y, 'z': z,
                        'l': cl, 'w': cw, 'h': ch, 'weight': g_weight, 'vol': g_vol,
                        'pose': (cl, cw, ch), 'pos': (x, y, z)
                    })
                    tw += g_weight
                    tv += g_vol
                    ok = True
                    break
                if ok:
                    break
            if ok:
                break

    util_v = tv / truck.volume
    util_w = tw / truck_M_max
    return min(util_v, util_w), placed, util_v, util_w

# ===================== IGA 改进遗传算法（论文参数） =====================
class IGA:
    def __init__(self, truck_type, cargos, pop=150, iter=300):
        self.t = truck_type
        self.cargos = cargos
        self.pop = pop
        self.iter = iter
        self.best_fit = 0
        self.best_plan = []

    def fit(self, ind):
        fit, plan, _, _ = blf_decode(ind, self.t)
        return fit, plan

    def run(self):
        # 初始种群：重泡货优先
        base = sorted(self.cargos, key=lambda x:(x['type'], -x['weight']/x['vol']))
        pop = [base.copy() for _ in range(self.pop)]
        for ind in pop:
            random.shuffle(ind[:len(ind)//2])

        # 使用tqdm显示进度条
        from tqdm import tqdm
        
        print(f"车型{self.t} - 开始进化，种群大小: {self.pop}，迭代次数: {self.iter}")
        
        # 预计算种群大小
        pop_size = self.pop
        
        for gen in tqdm(range(self.iter), desc=f"车型{self.t} 进化中", unit="代"):
            # 计算适应度和放置方案
            fit_plan_pairs = []
            for ind in pop:
                fit, plan = self.fit(ind)
                fit_plan_pairs.append((fit, plan))
            
            # 分离适应度和方案
            fits = [fp[0] for fp in fit_plan_pairs]
            plans = [fp[1] for fp in fit_plan_pairs]
            
            # 找到最佳个体
            best_idx = np.argmax(fits)
            if fits[best_idx] > self.best_fit:
                self.best_fit = fits[best_idx]
                self.best_plan = plans[best_idx]

            # 精英保留
            new_pop = [pop[best_idx]]
            
            # 预计算种群长度
            cargo_len = len(self.cargos)
            
            while len(new_pop) < pop_size:
                # 选择父母
                p1 = pop[np.random.randint(pop_size)]
                p2 = pop[np.random.randint(pop_size)]
                
                # 处理边界情况
                p1_len = len(p1)
                p2_len = len(p2)
                
                if p1_len < 2 or p2_len < 2:
                    # 选择较长的父代
                    child = p1.copy() if p1_len >= p2_len else p2.copy()
                else:
                    # 交叉操作
                    a, b = sorted(random.sample(range(p1_len), 2))
                    child = p1[:a] + p2[a:b] + p1[b:]
                    
                    # 去重
                    seen = set()
                    unique_child = []
                    for c in child:
                        c_id = c['id']
                        if c_id not in seen:
                            seen.add(c_id)
                            unique_child.append(c)
                    child = unique_child
                
                # 变异操作
                child_len = len(child)
                if child_len >= 2 and random.random() < 0.1:
                    i, j = random.sample(range(child_len), 2)
                    child[i], child[j] = child[j], child[i]
                
                new_pop.append(child)
            
            # 更新种群
            pop = new_pop[:pop_size]
        
        print(f"车型{self.t} - 进化完成，最佳适应度: {self.best_fit:.2%}")
        return self.best_fit, self.best_plan

# ===================== 最少用车计算 =====================
def calculate_min_trucks(cargos):
    """计算装载所有货物所需的最少车辆数"""
    remaining_cargos = cargos.copy()
    trucks = []
    truck_id = 1
    
    print("\n===== 开始计算最少用车方案 =====")
    
    # 估算最大可能的车辆数，用于进度条
    max_trucks = len(cargos) // 10 + 1  # 假设每辆车至少装10件货物
    
    from tqdm import tqdm
    with tqdm(total=max_trucks, desc="处理车辆", unit="辆") as pbar:
        while remaining_cargos:
            # 使用IGA算法优化当前车辆的装载方案
            iga = IGA(1, remaining_cargos, pop=50, iter=50)  # 适度增加种群大小和迭代次数，平衡速度和效果
            best_fit, best_plan = iga.run()
            
            if not best_plan:
                print("警告：剩余货物无法装载，可能存在尺寸或重量过大的货物")
                break
            
            # 计算当前车辆的装载信息
            v = sum([g['vol'] for g in best_plan])
            w = sum([g['weight'] for g in best_plan])
            truck = Truck(1)
            volume_util = v/truck.volume
            weight_util = w/truck.M_max
            
            print(f"\n车辆 {truck_id} 装载情况：")
            print(f"  综合满载率: {best_fit:.2%}")
            print(f"  空间利用率: {volume_util:.2%}")
            print(f"  载重利用率: {weight_util:.2%}")
            print(f"  装载件数: {len(best_plan)} 件")
            
            # 输出详细的装箱方案（右后下角为原点）
            print(f"\n  车辆 {truck_id} 详细装箱方案：")
            print(f"  {'货物ID':<10} {'类型':<10} {'姿态':<15} {'放置坐标(右后下原点)':<30} {'尺寸(cm)':<15} {'重量(kg)':<10}")
            print(f"  {'-' * 90}")
            
            for i, g in enumerate(best_plan, 1):
                # 提取姿态信息
                pose = (g.get('l', 0), g.get('w', 0), g.get('h', 0))
                # 原坐标（左下角为原点）
                old_pos = g.get('pos', (0, 0, 0))
                # 转换为右后下角为原点的坐标
                # 新x = 车辆长度 - 原x - 货物长度
                # 新y = 车辆宽度 - 原y - 货物宽度
                # 新z = 原z（保持不变）
                new_x = truck.L - old_pos[0] - pose[0]
                new_y = truck.W - old_pos[1] - pose[1]
                new_z = old_pos[2]
                new_pos = (new_x, new_y, new_z)
                # 货物类型名称
                type_name = {1: '标准件', 2: '易碎件', 3: '定向件'}.get(g['type'], '未知')
                
                print(f"  {g['id']:<10} {type_name:<10} {str(pose):<15} {str(new_pos):<30} {f'{pose[0]}×{pose[1]}×{pose[2]}':<15} {g['weight']:<10}")
            
            # 记录当前车辆的装载方案
            trucks.append({
                'id': truck_id,
                'plan': best_plan,
                'fitness': best_fit,
                'volume_util': volume_util,
                'weight_util': weight_util,
                'count': len(best_plan)
            })
            
            # 从剩余货物中移除已装载的货物
            loaded_ids = {g['id'] for g in best_plan}
            remaining_cargos = [g for g in remaining_cargos if g['id'] not in loaded_ids]
            
            print(f"  剩余货物: {len(remaining_cargos)} 件")
            truck_id += 1
            pbar.update(1)
            
            # 如果剩余货物很少，调整进度条总数
            if len(remaining_cargos) < 10 and pbar.total > truck_id:
                pbar.total = truck_id
                pbar.refresh()
    
    return trucks

# ===================== 主程序：问题1–1 车型1 =====================
if __name__ == "__main__":
    # 生成货物
    cargos = generate_cargos()
    
    # 1. 单车满载率最大化
    print("\n===== 问题1–1 车型1 单车满载率最大化 =====")
    iga = IGA(1, cargos, pop=50, iter=50)  # 适度增加种群大小和迭代次数，平衡速度和效果
    best_fit, best_plan = iga.run()
    v = sum([g['vol'] for g in best_plan])
    w = sum([g['weight'] for g in best_plan])
    truck = Truck(1)
    print(f"综合满载率: {best_fit:.2%}")
    print(f"空间利用率: {v/truck.volume:.2%}")
    print(f"载重利用率: {w/truck.M_max:.2%}")
    print(f"装载件数: {len(best_plan)}/{len(cargos)}")
    
    # 输出详细的装箱方案
    print("\n===== 详细装箱方案 =====")
    print(f"{'货物ID':<10} {'类型':<10} {'姿态':<15} {'放置坐标':<20} {'尺寸(cm)':<15} {'重量(kg)':<10}")
    print("-" * 90)
    
    for i, g in enumerate(best_plan, 1):
        # 提取姿态信息
        pose = (g.get('l', 0), g.get('w', 0), g.get('h', 0))
        # 提取坐标信息
        pos = g.get('pos', (0, 0, 0))
        # 货物类型名称
        type_name = {1: '标准件', 2: '易碎件', 3: '定向件'}.get(g['type'], '未知')
        
        print(f"{g['id']:<10} {type_name:<10} {str(pose):<15} {str(pos):<20} {f'{pose[0]}×{pose[1]}×{pose[2]}':<15} {g['weight']:<10}")
    
    # 输出货物统计信息
    print("\n===== 货物统计信息 =====")
    type_count = {1: 0, 2: 0, 3: 0}
    for g in best_plan:
        type_count[g['type']] += 1
    
    print(f"标准件: {type_count[1]} 件")
    print(f"易碎件: {type_count[2]} 件")
    print(f"定向件: {type_count[3]} 件")
    
    # 2. 最少用车计算（独立运行）
    print("\n" + "="*70)
    print("===== 最少用车方案计算 =====")
    print("="*70)
    trucks = calculate_min_trucks(cargos)
    
    # 输出最少用车结果
    print("\n===== 最少用车方案结果 =====")
    print(f"总车辆数: {len(trucks)} 辆")
    print("各车辆装载情况:")
    print("-" * 80)
    print(f"{'车辆ID':<10} {'满载率':<10} {'空间利用率':<15} {'载重利用率':<15} {'装载件数':<10}")
    print("-" * 80)
    
    total_cargos = 0
    for truck_info in trucks:
        print(f"{truck_info['id']:<10} {truck_info['fitness']:.2%}    {truck_info['volume_util']:.2%}         {truck_info['weight_util']:.2%}         {truck_info['count']:<10}")
        total_cargos += truck_info['count']
    
    print("-" * 80)
    print(f"总计装载: {total_cargos}/{len(cargos)} 件货物")
