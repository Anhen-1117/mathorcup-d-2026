"""
MathorCup D题 — 三维装箱 3D 可视化
读取装箱结果CSV，绘制每个车辆的3D装箱效果图
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import os

# 中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "..", "results", "database")
OUTPUT = os.path.join(BASE, "..", "results", "viz")
os.makedirs(OUTPUT, exist_ok=True)

# 车型尺寸 (cm)
VEHICLE_DIMS = {
    1: {"name": "7.6m 厢式车", "length": 760, "width": 230, "height": 230},
    2: {"name": "9.6m 厢式车", "length": 960, "width": 230, "height": 270},
}

# 货物类型颜色映射
ITEM_COLORS = {
    "标准件": "#3498db",
    "大件": "#e74c3c",
    "易碎品": "#f39c12",
}


def hex_to_rgba(hex_color, alpha=0.8):
    """十六进制颜色转RGBA"""
    h = hex_color.lstrip("#")
    rgb = tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return (*rgb, alpha)


def draw_cuboid(ax, x, y, z, dx, dy, dz, color, edge_color="#2c3e50", alpha=0.8):
    """在3D轴上绘制一个长方体"""
    # 定义8个顶点
    vertices = np.array([
        [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
        [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]
    ])
    # 6个面
    faces = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]
    ]
    rgba = hex_to_rgba(color, alpha)
    for face in faces:
        poly = vertices[face]
        ax.add_collection3d(
            Poly3DCollection([poly], color=rgba, edgecolor=edge_color, linewidth=0.3, alpha=alpha)
        )


def draw_vehicle_frame(ax, length, width, height, color="#34495e"):
    """绘制车辆轮廓线框"""
    vertices = np.array([
        [0, 0, 0], [length, 0, 0], [length, width, 0], [0, width, 0],
        [0, 0, height], [length, 0, height], [length, width, height], [0, width, height]
    ])
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    for edge in edges:
        pts = vertices[edge]
        ax.plot3D(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=1.5, linestyle="--", alpha=0.4)


def parse_csv(csv_path):
    """解析装箱结果CSV — 不同CSV列数不同，按位置智能映射"""
    try:
        df = pd.read_csv(csv_path, encoding="gb18030", header=0)
    except:
        try:
            df = pd.read_csv(csv_path, encoding="gbk", header=0)
        except:
            try:
                df = pd.read_csv(csv_path, encoding="utf-8", header=0)
            except:
                return None

    n_cols = len(df.columns)
    col_names = list(df.columns)
    result = pd.DataFrame()

    # 通用：按列位置映射
    if n_cols >= 11:  # 多车型：区域, 车辆, 货物ID, 类型, x, y, z, dx, dy, dz, 重量
        result["vehicle"] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        result["id"] = df.iloc[:, 2]
        result["type"] = df.iloc[:, 3].astype(str)
        result["x"] = pd.to_numeric(df.iloc[:, 4], errors="coerce")
        result["y"] = pd.to_numeric(df.iloc[:, 5], errors="coerce")
        result["z"] = pd.to_numeric(df.iloc[:, 6], errors="coerce")
        result["dx"] = pd.to_numeric(df.iloc[:, 7], errors="coerce")
        result["dy"] = pd.to_numeric(df.iloc[:, 8], errors="coerce")
        result["dz"] = pd.to_numeric(df.iloc[:, 9], errors="coerce")
    elif n_cols == 10:  # 少车型：区域, 货物ID, 类型, x, y, z, dx, dy, dz, 重量
        result["vehicle"] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        result["id"] = df.iloc[:, 1]
        result["type"] = df.iloc[:, 2].astype(str)
        result["x"] = pd.to_numeric(df.iloc[:, 3], errors="coerce")
        result["y"] = pd.to_numeric(df.iloc[:, 4], errors="coerce")
        result["z"] = pd.to_numeric(df.iloc[:, 5], errors="coerce")
        result["dx"] = pd.to_numeric(df.iloc[:, 6], errors="coerce")
        result["dy"] = pd.to_numeric(df.iloc[:, 7], errors="coerce")
        result["dz"] = pd.to_numeric(df.iloc[:, 8], errors="coerce")
    elif n_cols == 9:  # 单车型：货物ID, 类型, x, y, z, dx, dy, dz, 重量
        result["vehicle"] = 1
        result["id"] = df.iloc[:, 0]
        result["type"] = df.iloc[:, 1].astype(str)
        result["x"] = pd.to_numeric(df.iloc[:, 2], errors="coerce")
        result["y"] = pd.to_numeric(df.iloc[:, 3], errors="coerce")
        result["z"] = pd.to_numeric(df.iloc[:, 4], errors="coerce")
        result["dx"] = pd.to_numeric(df.iloc[:, 5], errors="coerce")
        result["dy"] = pd.to_numeric(df.iloc[:, 6], errors="coerce")
        result["dz"] = pd.to_numeric(df.iloc[:, 7], errors="coerce")
    else:
        return None

    # 按ID分配颜色（每个货物独立颜色）
    palette = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#1f77b4"]
    def pick_color(i):
        try:
            return palette[int(float(i)) % len(palette)]
        except:
            return "#95a5a6"
    result["color"] = result["id"].apply(pick_color)
    return result


def plot_vehicle(ax, df_vehicle, vehicle_id, vehicle_dim):
    """绘制单个车辆的所有货物"""
    length, width, height = vehicle_dim["length"], vehicle_dim["width"], vehicle_dim["height"]
    draw_vehicle_frame(ax, length, width, height)

    for _, row in df_vehicle.iterrows():
        x = float(row["x"])
        y = float(row["y"])
        z = float(row["z"])
        dx = float(row["dx"])
        dy = float(row["dy"])
        dz = float(row["dz"])
        color = row.get("color", "#3498db")
        draw_cuboid(ax, x, y, z, dx, dy, dz, color)

    ax.set_xlim(0, length)
    ax.set_ylim(0, width)
    ax.set_zlim(0, height)
    ax.set_xlabel("长度 (cm)", fontsize=10)
    ax.set_ylabel("宽度 (cm)", fontsize=10)
    ax.set_zlabel("高度 (cm)", fontsize=10)
    ax.set_title(f"{vehicle_dim['name']} (ID: {vehicle_id})", fontsize=12)
    ax.view_init(elev=25, azim=-60)


def main():
    print("生成三维装箱可视化...")
    # 处理所有CSV文件
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and f != ".gitkeep"]
    csv_files.sort()

    for csv_file in csv_files:
        csv_path = os.path.join(DATA_DIR, csv_file)
        df = parse_csv(csv_path)
        if df is None:
            print(f"  跳过: {csv_file} (解析失败)")
            continue

        vehicles = df["vehicle"].unique()
        n_vehicles = len(vehicles)
        cols = min(3, n_vehicles)
        rows = (n_vehicles + cols - 1) // cols

        fig = plt.figure(figsize=(6 * cols, 5 * rows))

        for idx, v_id in enumerate(vehicles):
            df_v = df[df["vehicle"] == v_id]
            # 判断车型（根据数据中的车辆ID或尺寸推断）
            v_dim = VEHICLE_DIMS[1]  # 默认7.6m
            if n_vehicles == 1:
                ax = fig.add_subplot(1, 1, 1, projection="3d")
            else:
                ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
            plot_vehicle(ax, df_v, v_id, v_dim)

        plt.suptitle(f"三维装箱结果 — {csv_file}", fontsize=14, y=1.02)
        fig.tight_layout()
        out_name = csv_file.replace(".csv", "_3d.png")
        fig.savefig(os.path.join(OUTPUT, out_name), dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  + {out_name} ({len(vehicles)}辆车)")

    # 总览对比图
    if csv_files:
        fig, axes = plt.subplots(1, min(2, len(csv_files)), figsize=(14, 5))
        if len(csv_files) == 1:
            axes = [axes]
        for idx, csv_file in enumerate(csv_files[:2]):
            csv_path = os.path.join(DATA_DIR, csv_file)
            df = parse_csv(csv_path)
            if df is None:
                continue
            ax = axes[idx]
            vehicle_counts = df["vehicle"].value_counts().sort_index()
            n_types = df.groupby("vehicle")["type"].apply(lambda x: x.nunique())
            ax.bar(vehicle_counts.index, vehicle_counts.values, color="#3498db", alpha=0.7)
            ax.set_xlabel("车辆ID")
            ax.set_ylabel("货物数量")
            ax.set_title(f"{csv_file.replace('.csv','')}")
            for i, v in enumerate(vehicle_counts.values):
                ax.text(list(vehicle_counts.index)[i], v + 0.5, str(v), ha="center", fontsize=9)
        fig.suptitle("各车辆装载货物数量对比", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT, "summary_装载对比.png"), dpi=200, bbox_inches="tight")
        plt.close()
        print("  + summary_装载对比.png")

    print(f"完成! 输出目录: {OUTPUT}")


if __name__ == "__main__":
    main()
