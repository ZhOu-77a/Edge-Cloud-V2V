import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

# === 🛠️ 配置区域 ===
# 您的实验结果 CSV 路径
# CSV_PATH = "output/9_streetview4_seed2_each/experiment_results_batch_gpu34.csv" 
CSV_PATH = "experiment_results_prompt_diff.csv" 

# 图片保存的文件夹
OUTPUT_DIR = "plots_replot"

# 实验中的默认参数 (用于控制变量法)
DEFAULT_FPS = 8
DEFAULT_STEPS = 30
DEFAULT_CFG = 1.0

# ====================

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def plot_all_results(df_raw):
    try:
        print(f"\n📊 Starting Plotting from {CSV_PATH}...")
        print(f"   Total rows: {len(df_raw)}")
        
        # 1. 数据清洗：确保数值列是数字类型
        numeric_cols = ['fps', 'steps', 'cfg', 'latency', 'clip_score', 'clip_consistency', 'warp_error']
        for col in numeric_cols:
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        
        # 2. 准备参数
        unique_videos = df_raw['video'].unique()
        unique_seeds = df_raw['seed'].unique()
        
        print(f"   Videos found: {unique_videos}")
        print(f"   Seeds found: {unique_seeds}")

        # 辅助筛选函数：获取控制变量后的数据切片
        def get_slice(df_subset, param):
            # 筛选逻辑：当分析 param 时，其他两个变量保持 DEFAULT 值
            if param == 'fps': 
                return df_subset[
                    (df_subset['steps'] == DEFAULT_STEPS) & 
                    (np.isclose(df_subset['cfg'], DEFAULT_CFG))
                ].sort_values('fps')
            if param == 'steps':
                return df_subset[
                    (df_subset['fps'] == DEFAULT_FPS) & 
                    (np.isclose(df_subset['cfg'], DEFAULT_CFG))
                ].sort_values('steps')
            if param == 'cfg':
                return df_subset[
                    (df_subset['fps'] == DEFAULT_FPS) & 
                    (df_subset['steps'] == DEFAULT_STEPS)
                ].sort_values('cfg')

        # 绘图配置
        params = ['fps', 'steps', 'cfg'] # 行 (Rows)
        metrics = [                      # 列 (Cols)
            ('latency', 'Latency (s) ↓', 'k'),
            ('clip_score', 'CLIP Text Score (Quality) ↑', 'purple'), 
            ('warp_error', 'Warp Error (Structure) ↓', 'r'),
            ('clip_consistency', 'CLIP Consistency (Smoothness) ↑', 'g')
        ]
        
        # 配色方案
        # 为 Video 或 Seed 生成足够多的颜色
        color_map = cm.get_cmap('tab10') 

        # =========================================================
        # 场景 1: 按 Seed 分组 (一张图包含一个 Seed 下的所有 Video)
        # =========================================================
        print(f"\n👉 Generating Group 1: Comparison across Videos (Fixed Seed)...")
        for seed in unique_seeds:
            df_seed = df_raw[df_raw['seed'] == seed]
            if df_seed.empty: continue
            
            fig, axes = plt.subplots(3, 4, figsize=(24, 15))
            # [修改] 大标题字体加大
            fig.suptitle(f"Comparison across Videos (Seed={seed})", fontsize=24, weight='bold')
            
            for row, param in enumerate(params):
                for col, (metric, title, base_color) in enumerate(metrics):
                    ax = axes[row, col]
                    
                    # 遍历每个 Video 画线
                    for i, video in enumerate(unique_videos):
                        df_video_seed = df_seed[df_seed['video'] == video]
                        data = get_slice(df_video_seed, param)
                        
                        if not data.empty and metric in data.columns:
                            ax.plot(
                                data[param], data[metric], 
                                marker='o', markersize=8, linewidth=2.5, alpha=0.8,
                                label=f"{video}", 
                                color=color_map(i % 10)
                            )
                    
                    # [修改] 子标题、坐标轴标签、刻度字体加大
                    ax.set_title(f"{param.upper()} vs {title}", fontsize=16, weight='bold')
                    ax.set_xlabel(param.upper(), fontsize=14)
                    ax.tick_params(axis='both', which='major', labelsize=12) # 刻度字体
                    ax.grid(True, linestyle='--', alpha=0.5)
                    
                    # 为了美观，只在第一列显示图例
                    if col == 0: 
                        # [修改] 图例字体加大
                        ax.legend(fontsize=12, loc='best')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 留出顶部 Title 空间
            save_path = os.path.join(OUTPUT_DIR, f"plot_by_seed_{seed}.png")
            plt.savefig(save_path, dpi=100)
            plt.close(fig)
            print(f"   Saved: {save_path}")

        # =========================================================
        # 场景 2: 按 Video 分组 (一张图包含一个 Video 下的所有 Seed)
        # =========================================================
        print(f"\n👉 Generating Group 2: Comparison across Seeds (Fixed Video)...")
        for video in unique_videos:
            df_video = df_raw[df_raw['video'] == video]
            if df_video.empty: continue
            
            fig, axes = plt.subplots(3, 4, figsize=(24, 15))
            # [修改] 大标题字体加大
            fig.suptitle(f"Comparison across Seeds (Video={video})", fontsize=24, weight='bold')
            
            for row, param in enumerate(params):
                for col, (metric, title, base_color) in enumerate(metrics):
                    ax = axes[row, col]
                    
                    # 遍历每个 Seed 画线
                    for i, seed in enumerate(unique_seeds):
                        df_video_seed = df_video[df_video['seed'] == seed]
                        data = get_slice(df_video_seed, param)
                        
                        if not data.empty and metric in data.columns:
                            ax.plot(
                                data[param], data[metric], 
                                marker='^', markersize=8, linewidth=2.5, alpha=0.8,
                                label=f"Seed {seed}", 
                                color=color_map(i % 10)
                            )
                    
                    # [修改] 子标题、坐标轴标签、刻度字体加大
                    ax.set_title(f"{param.upper()} vs {title}", fontsize=16, weight='bold')
                    ax.set_xlabel(param.upper(), fontsize=14)
                    ax.tick_params(axis='both', which='major', labelsize=12) # 刻度字体
                    ax.grid(True, linestyle='--', alpha=0.5)
                    
                    if col == 0: 
                        # [修改] 图例字体加大
                        ax.legend(fontsize=12, loc='best')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = os.path.join(OUTPUT_DIR, f"plot_by_video_{video}.png")
            plt.savefig(save_path, dpi=100)
            plt.close(fig)
            print(f"   Saved: {save_path}")

        print(f"\n✅ All plots saved to directory: {OUTPUT_DIR}/")

    except Exception as e:
        print(f"❌ Plotting Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        plot_all_results(df)
    else:
        print(f"❌ File not found: {CSV_PATH}")
        print("Please check the path or run experiment first.")