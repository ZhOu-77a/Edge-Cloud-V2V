import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import os
import colorsys

# === 配置 ===
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
CSV_PATH = "/home/zhoujh/Edge-Cloud-diffusion/MyCogVideo-v2v/experiment_results_prompt_diff.csv"
OUTPUT_DIR = "plots_diff_prompt/18"
# 实验默认值 (用于控制变量)
DEFAULT_FPS = 8
DEFAULT_STEPS = 30
DEFAULT_CFG = 1.0

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def get_distinct_colors(n):
    """
    生成 n 个区分度极高的颜色。
    优先使用手动定义的强对比色列表，不够时使用 HSV 均匀分布。
    """
    # 1. 手动定义的高对比度颜色表
    base_colors = [
        '#E6194B', # Red
        '#3CB44B', # Green
        '#FFE119', # Yellow
        '#4363D8', # Blue
        '#F58231', # Orange
        '#911EB4', # Purple
        '#42D4F4', # Cyan
        '#F032E6', # Magenta
        '#BFEF45', # Lime
        '#FABEBE', # Pink
        '#469990', # Teal
        '#DCBEFF', # Lavender
        '#9A6324', # Brown
        '#FFFAC8', # Beige
        '#800000', # Maroon
        '#AAFFC3', # Mint
        '#808000', # Olive
        '#FFD8B1', # Apricot
        '#000075', # Navy
        '#A9A9A9', # Grey
    ]
    
    if n <= len(base_colors):
        return base_colors[:n]
    
    # 2. 如果数量超过预定义，使用 Golden Angle 生成 HSV 颜色
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + 0.3 * (i % 2) # 饱和度交替
        value = 0.8 + 0.2 * (i % 2)      # 亮度交替
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    return colors

def get_marker(seed_idx):
    """不同的 Seed 使用不同的点标记"""
    # 0:圆, 1:三角, 2:星号(类似雪花), 3:方块, 4:菱形
    markers = ['o', '^', '*', 's', 'D', 'x']
    return markers[seed_idx % len(markers)]

def get_linestyle(seed_idx):
    """不同的 Seed 使用不同的线型"""
    styles = ['-', '--', '-.', ':']
    return styles[seed_idx % len(styles)]

def plot_prompt_comparison(df):
    print(f"📊 Analyzing {len(df)} rows from {CSV_PATH}...")
    
    # 数据清洗
    cols = ['fps', 'steps', 'cfg', 'clip_score', 'warp_error']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
    
    unique_videos = df['video'].unique()
    unique_prompts = sorted(df['prompt_name'].unique()) # 排序保证颜色固定
    unique_seeds = sorted(df['seed'].unique())
    
    print(f"   Videos: {len(unique_videos)} | Prompts: {len(unique_prompts)} | Seeds: {len(unique_seeds)}")

    # 1. 分配高对比度基础颜色 (每个 Prompt 一个颜色)
    distinct_colors = get_distinct_colors(len(unique_prompts))
    prompt_base_colors = {name: distinct_colors[i] for i, name in enumerate(unique_prompts)}

    # === 针对每个 Video 画图 ===
    for video in unique_videos:
        print(f"   👉 Plotting for Video: {video}")
        df_vid = df[df['video'] == video]
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle(f"Prompt Difficulty Analysis | Video: {video}\n(Color=Prompt, Shape/Style=Seed)", fontsize=20, weight='bold')
        
        # 定义子图配置
        plot_configs = [
            # (行, 列, X轴, Y轴, 标题)
            (0, 0, 'steps', 'clip_score', 'Steps vs Semantic (CLIP)'),
            (0, 1, 'steps', 'warp_error', 'Steps vs Structure (Warp)'),
            (1, 0, 'cfg',   'clip_score', 'CFG vs Semantic (CLIP)'),
            (1, 1, 'cfg',   'warp_error', 'CFG vs Structure (Warp)')
        ]

        lines_for_legend = [] # 用于自定义 Prompt 图例
        labels_for_legend = []

        for row, col, x_param, y_param, title in plot_configs:
            ax = axes[row, col]
            
            # 筛选数据 (控制变量)
            if x_param == 'steps':
                # 固定 CFG 和 FPS
                data_subset = df_vid[
                    (np.isclose(df_vid['cfg'], DEFAULT_CFG)) & 
                    (df_vid['fps'] == DEFAULT_FPS)
                ]
            else: # x_param == 'cfg'
                # 固定 Steps 和 FPS
                data_subset = df_vid[
                    (df_vid['steps'] == DEFAULT_STEPS) & 
                    (df_vid['fps'] == DEFAULT_FPS)
                ]

            # 绘图循环
            for p_name in unique_prompts:
                base_c = prompt_base_colors[p_name]
                
                for s_idx, seed in enumerate(unique_seeds):
                    # 获取当前 Prompt + Seed 的数据
                    d = data_subset[
                        (data_subset['prompt_name'] == p_name) & 
                        (data_subset['seed'] == seed)
                    ].sort_values(x_param)
                    
                    if d.empty: continue
                    
                    # 获取样式
                    marker_style = get_marker(s_idx)
                    line_style = get_linestyle(s_idx)
                    
                    # 画线 (同一 Prompt 颜色完全相同，仅靠形状区分 Seed)
                    line, = ax.plot(
                        d[x_param], d[y_param], 
                        marker=marker_style, markersize=6, 
                        linestyle=line_style, linewidth=2,
                        color=base_c, # 不变色
                        alpha=0.8,
                        label=p_name if s_idx == len(unique_seeds)-1 else ""
                    )
                    
                    # 收集图例信息 (只在第一张子图收集一次)
                    if row == 0 and col == 0 and s_idx == 0:
                        # 这里收集的是基础颜色的线条，用于图例展示 Prompt
                        proxy_line = plt.Line2D([0], [0], color=base_c, lw=3)
                        lines_for_legend.append(proxy_line)
                        labels_for_legend.append(p_name)

            ax.set_title(title, fontsize=14, weight='bold')
            ax.set_xlabel(x_param.upper(), fontsize=12)
            ax.set_ylabel(y_param, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.4)

        # === 统一图例 (放在右侧) ===
        # 1. Prompt 图例 (颜色)
        legend1 = fig.legend(
            lines_for_legend, labels_for_legend, 
            loc='center right', title="Prompts (Color)", 
            bbox_to_anchor=(0.98, 0.6), fontsize=10, frameon=True
        )
        
        # 2. Seed 样式图例 (黑色，展示形状和线型)
        seed_lines = []
        seed_labels = []
        for s_idx, seed in enumerate(unique_seeds):
            # 使用黑色展示线型和点
            l = plt.Line2D([0], [0], color='black', 
                           marker=get_marker(s_idx), markersize=6,
                           linestyle=get_linestyle(s_idx), linewidth=1.5,
                           label=f"Seed {seed}")
            seed_lines.append(l)
            seed_labels.append(f"Seed {seed}")
            
        legend2 = fig.legend(
            seed_lines, seed_labels, 
            loc='center right', title="Seeds (Shape)", 
            bbox_to_anchor=(0.98, 0.25), fontsize=10, frameon=True
        )
        
        # 调整布局，留出右侧给图例
        plt.subplots_adjust(right=0.85, top=0.9, wspace=0.2, hspace=0.3)
        
        save_path = os.path.join(OUTPUT_DIR, f"prompt_analysis_{video}.png")
        plt.savefig(save_path, dpi=150) # 提高一点分辨率
        plt.close()
        print(f"   💾 Saved: {save_path}")

if __name__ == "__main__":
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        plot_prompt_comparison(df)
    else:
        print(f"❌ CSV not found: {CSV_PATH}")