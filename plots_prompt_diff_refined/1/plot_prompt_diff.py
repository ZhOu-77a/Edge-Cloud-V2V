import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

CSV_PATH = "experiment_results_prompt_diff.csv"
OUTPUT_DIR = "plots_prompt_diff"

DEFAULT_FPS = 8
DEFAULT_STEPS = 30
DEFAULT_CFG = 1.0

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def plot_prompt_comparison(df):
    print(f"📊 Analyzing {len(df)} rows...")
    
    unique_videos = df['video'].unique()
    unique_prompts = df['prompt_name'].unique() # 使用名字而不是ID，图例更清晰
    
    # 颜色映射：给每个 Prompt 分配一个固定颜色
    color_map = cm.get_cmap('tab20', len(unique_prompts))
    prompt_color_dict = {name: color_map(i) for i, name in enumerate(unique_prompts)}

    # === 针对每个 Video 画图 ===
    for video in unique_videos:
        print(f"   👉 Plotting for Video: {video}")
        df_vid = df[df['video'] == video]
        
        # 创建画布：2行2列 (Steps vs Metric, CFG vs Metric)
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f"Prompt Difficulty Analysis | Video: {video}", fontsize=20, weight='bold')
        
        # --- 子图 1: Steps vs CLIP Score (固定 CFG/FPS) ---
        ax = axes[0, 0]
        data_subset = df_vid[
            (np.isclose(df_vid['cfg'], DEFAULT_CFG)) & 
            (df_vid['fps'] == DEFAULT_FPS)
        ]
        
        for name in unique_prompts:
            d = data_subset[data_subset['prompt_name'] == name].sort_values('steps')
            if not d.empty:
                ax.plot(d['steps'], d['clip_score'], marker='o', label=name, color=prompt_color_dict[name])
        
        ax.set_title("Steps vs CLIP Text Score (Semantic)", fontsize=14)
        ax.set_xlabel("Steps")
        ax.set_ylabel("CLIP Score")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8, loc='best')

        # --- 子图 2: Steps vs Warp Error ---
        ax = axes[0, 1]
        for name in unique_prompts:
            d = data_subset[data_subset['prompt_name'] == name].sort_values('steps')
            if not d.empty:
                ax.plot(d['steps'], d['warp_error'], marker='x', linestyle='--', color=prompt_color_dict[name])
        ax.set_title("Steps vs Warp Error (Structure)", fontsize=14)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Warp Error")
        ax.grid(True, linestyle='--', alpha=0.5)

        # --- 子图 3: CFG vs CLIP Score (固定 Steps/FPS) ---
        ax = axes[1, 0]
        data_subset_cfg = df_vid[
            (df_vid['steps'] == DEFAULT_STEPS) & 
            (df_vid['fps'] == DEFAULT_FPS)
        ]
        
        for name in unique_prompts:
            d = data_subset_cfg[data_subset_cfg['prompt_name'] == name].sort_values('cfg')
            if not d.empty:
                ax.plot(d['cfg'], d['clip_score'], marker='s', label=name, color=prompt_color_dict[name])
        
        ax.set_title("CFG Ratio vs CLIP Text Score", fontsize=14)
        ax.set_xlabel("CFG Ratio")
        ax.set_ylabel("CLIP Score")
        ax.grid(True, linestyle='--', alpha=0.5)

        # --- 子图 4: CFG vs Warp Error ---
        ax = axes[1, 1]
        for name in unique_prompts:
            d = data_subset_cfg[data_subset_cfg['prompt_name'] == name].sort_values('cfg')
            if not d.empty:
                ax.plot(d['cfg'], d['warp_error'], marker='^', linestyle='--', color=prompt_color_dict[name])
        ax.set_title("CFG Ratio vs Warp Error", fontsize=14)
        ax.set_xlabel("CFG Ratio")
        ax.set_ylabel("Warp Error")
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(OUTPUT_DIR, f"prompt_analysis_{video}.png"))
        plt.close()

if __name__ == "__main__":
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        plot_prompt_comparison(df)
    else:
        print("CSV not found.")