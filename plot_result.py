import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import os
import shutil

# === 🛠️ 配置区域 ===
CSV_PATH = "experiment_results_prompt_diff.csv" 

# 输出根目录
OUTPUT_ROOT = "plots_replot_by_prompt"

# 实验中的默认参数 (用于控制变量法)
DEFAULT_FPS = 8
DEFAULT_STEPS = 30
DEFAULT_CFG = 1.0

# ====================

if not os.path.exists(OUTPUT_ROOT):
    os.makedirs(OUTPUT_ROOT)

def plot_subset(df_subset, output_dir, prompt_name):
    """
    针对特定的 Prompt 数据子集进行绘图。
    包含两组图：
    1. 按 Seed 分组 (对比不同 Video)
    2. 按 Video 分组 (对比不同 Seed)
    """
    unique_videos = df_subset['video'].unique()
    unique_seeds = df_subset['seed'].unique()
    
    # 配色方案 (用于区分 Video 或 Seed)
    color_map = cm.get_cmap('tab10') 

    # 绘图配置
    params = ['fps', 'steps', 'cfg'] # 行
    metrics = [                      # 列
        ('latency', 'Latency (s) ↓', 'k'),
        ('clip_score', 'CLIP Text Score ↑', 'purple'), 
        ('warp_error', 'Warp Error ↓', 'r'),
        ('clip_consistency', 'CLIP Consistency ↑', 'g')
    ]

    # 辅助筛选函数
    def get_slice(df, param):
        if param == 'fps': 
            return df[
                (df['steps'] == DEFAULT_STEPS) & 
                (np.isclose(df['cfg'], DEFAULT_CFG))
            ].sort_values('fps')
        if param == 'steps':
            return df[
                (df['fps'] == DEFAULT_FPS) & 
                (np.isclose(df['cfg'], DEFAULT_CFG))
            ].sort_values('steps')
        if param == 'cfg':
            return df[
                (df['fps'] == DEFAULT_FPS) & 
                (df['steps'] == DEFAULT_STEPS)
            ].sort_values('cfg')

    # =========================================================
    # Group 1: Fixed Seed -> Compare Videos
    # =========================================================
    for seed in unique_seeds:
        df_seed = df_subset[df_subset['seed'] == seed]
        if df_seed.empty: continue
        
        fig, axes = plt.subplots(3, 4, figsize=(24, 15))
        # 标题包含 Prompt 信息
        fig.suptitle(f"[{prompt_name}] Compare Videos (Seed={seed})", fontsize=24, weight='bold')
        
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
                
                # 样式设置
                ax.set_title(f"{param.upper()} vs {title}", fontsize=16, weight='bold')
                ax.set_xlabel(param.upper(), fontsize=14)
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.grid(True, linestyle='--', alpha=0.5)
                
                if col == 0: 
                    ax.legend(fontsize=12, loc='best')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, f"compare_videos_seed{seed}.png")
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        print(f"     --> Saved: compare_videos_seed{seed}.png")

    # =========================================================
    # Group 2: Fixed Video -> Compare Seeds
    # =========================================================
    for video in unique_videos:
        df_video = df_subset[df_subset['video'] == video]
        if df_video.empty: continue
        
        fig, axes = plt.subplots(3, 4, figsize=(24, 15))
        fig.suptitle(f"[{prompt_name}] Compare Seeds (Video={video})", fontsize=24, weight='bold')
        
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
                
                ax.set_title(f"{param.upper()} vs {title}", fontsize=16, weight='bold')
                ax.set_xlabel(param.upper(), fontsize=14)
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.grid(True, linestyle='--', alpha=0.5)
                
                if col == 0: 
                    ax.legend(fontsize=12, loc='best')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, f"compare_seeds_video_{video}.png")
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        print(f"     --> Saved: compare_seeds_video_{video}.png")

def plot_all_results_by_prompt(df_raw):
    try:
        print(f"\n📊 Starting Plotting from {CSV_PATH}...")
        print(f"   Total rows: {len(df_raw)}")
        
        # 1. 数据清洗
        numeric_cols = ['fps', 'steps', 'cfg', 'latency', 'clip_score', 'clip_consistency', 'warp_error']
        for col in numeric_cols:
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        
        # 检查是否包含 prompt 信息
        if 'prompt_name' not in df_raw.columns:
            # 兼容旧版 CSV：如果没有 prompt 列，给一个默认值
            print("⚠️ 'prompt_name' column not found. Assuming Single Prompt.")
            df_raw['prompt_name'] = 'Default_Prompt'

        unique_prompts = df_raw['prompt_name'].unique()
        print(f"   Found {len(unique_prompts)} Prompts: {unique_prompts}")

        # === 核心循环：按 Prompt 遍历 ===
        for prompt_name in unique_prompts:
            # 1. 创建 Prompt 专属文件夹
            # 清理文件名中的非法字符 (空格, 冒号等)
            safe_name = "".join([c if c.isalnum() else "_" for c in str(prompt_name)])
            prompt_dir = os.path.join(OUTPUT_ROOT, safe_name)
            
            if not os.path.exists(prompt_dir):
                os.makedirs(prompt_dir)
            
            print(f"\n📂 Processing Prompt: {prompt_name}")
            print(f"   Target Directory: {prompt_dir}")

            # 2. 筛选数据
            df_subset = df_raw[df_raw['prompt_name'] == prompt_name]
            
            # 3. 调用画图逻辑
            plot_subset(df_subset, prompt_dir, prompt_name)

        print(f"\n✅ All plots generated in {OUTPUT_ROOT}/")

    except Exception as e:
        print(f"❌ Plotting Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        plot_all_results_by_prompt(df)
    else:
        print(f"❌ File not found: {CSV_PATH}")
        print("Please check the path or run experiment first.")