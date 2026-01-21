import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 🛠️ 配置区域 ===
RESULTS_CSV = "/home/zhoujh/Edge-Cloud-diffusion/MyCogVideo-v2v/output/15_new5_4_6_CFG7/experiment_results_prompt_diff.csv"  
GAP_CSV = "/home/zhoujh/Edge-Cloud-diffusion/MyCogVideo-v2v/similarity_of_video_prompt/source_target_gap_15.csv"            
OUTPUT_DIR = "analysis_fitting/15"

# === 核心阈值设置 ===
# 95%就认为已经"饱和"
SATURATION_THRESHOLD = 0.95   

# 过滤: 只保留语义生成成功的案例
MIN_VALID_CLIP_SCORE = 0.27   

# === 控制变量基准值 (需与实验一致) ===
FIXED_FPS = 8
FIXED_STEPS = 30
FIXED_CFG = 1.0 

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def find_knee_point_steps(group):
    """
    寻找 Steps 的拐点 (Cost-Effectiveness Point)。
    策略：在 CLIP Score 达到峰值的 95% 时，取最小的 Step。
    """
    # 1. 筛选: FPS=8, CFG=1.0 (观察 Steps 对分数的影响)
    subset = group[
        (group['fps'] == FIXED_FPS) & 
        (np.isclose(group['cfg'], FIXED_CFG))
    ].sort_values('steps')
    
    if subset.empty: return 30

    # 2. 找饱和点
    max_score = subset['clip_score'].max()
    target = max_score * SATURATION_THRESHOLD
    
    # 3. 找到第一个达到目标的 Step
    qualified = subset[subset['clip_score'] >= target]
    
    if not qualified.empty:
        return qualified['steps'].min()
    else:
        return subset['steps'].max()

def find_saturation_cfg(group):
    """
    找到能达到最高语义分数的 95% 的最小 CFG 值。
    """
    # 1. 筛选: FPS=8, Steps=30 (观察 CFG 对分数的影响)
    subset = group[
        (group['fps'] == FIXED_FPS) & 
        (group['steps'] == FIXED_STEPS)
    ].sort_values('cfg')
    
    if subset.empty: return 1.0

    # 2. 找饱和点
    max_score = subset['clip_score'].max()
    target = max_score * SATURATION_THRESHOLD
    
    # 3. 找到第一个达到目标的 CFG
    qualified = subset[subset['clip_score'] >= target]
    
    if not qualified.empty:
        return qualified['cfg'].min()
    else:
        # 如果都没达到(理论上max肯定能达到)，返回最高CFG
        return subset['cfg'].max()

def main():
    if not os.path.exists(RESULTS_CSV) or not os.path.exists(GAP_CSV):
        print("❌ CSVs not found.")
        return
        
    df_res = pd.read_csv(RESULTS_CSV)
    df_gap = pd.read_csv(GAP_CSV)
    
    # 1. 预处理字符串，去除空格防止匹配失败
    df_res['prompt_name'] = df_res['prompt_name'].astype(str).str.strip()
    df_gap['prompt_name'] = df_gap['prompt_name'].astype(str).str.strip()
    df_res['video'] = df_res['video'].astype(str).str.strip()
    df_gap['video'] = df_gap['video'].astype(str).str.strip()

    # 2. 合并数据
    gap_cols = ['video', 'prompt_name', 'initial_clip_score']
    gap_cols = [c for c in gap_cols if c in df_gap.columns]
    
    df_merged = pd.merge(df_res, df_gap[gap_cols], 
                         on=['video', 'prompt_name'], how='inner')
    
    print(f"🔹 Merged Data: {len(df_merged)} rows")
    if len(df_merged) == 0:
        print("❌ Merge failed. Please check prompt names in both CSVs.")
        # Debug info
        print("   Res Sample:", df_res['prompt_name'].unique()[:3])
        print("   Gap Sample:", df_gap['prompt_name'].unique()[:3])
        return

    optimal_data = []
    
    # 3. 分组计算 (Video + Prompt + Seed)
    # 修改：增加按 seed 分组，确保每个样本都是独立的，并且能保留 seed 信息
    grouped = df_merged.groupby(['video', 'prompt_name', 'seed'])
    print(f"🔹 Analyzing {len(grouped)} groups (Video + Prompt + Seed)...")

    skipped_count = 0
    
    for (vid, prompt, seed_val), group in grouped:
        # [过滤] 语义生成失败
        max_clip = group['clip_score'].max()
        if max_clip < MIN_VALID_CLIP_SCORE:
            # print(f"   ⚠️ Skipping {prompt[:10]}... (Max Score {max_clip:.2f} too low)")
            skipped_count += 1
            continue

        # === 计算最优参数 ===
        # X轴: 初始相似度
        init_sim = group['initial_clip_score'].mean()
        
        # Y轴: 最佳 Steps 和 CFG
        opt_steps = find_knee_point_steps(group)
        opt_cfg = find_saturation_cfg(group) # 使用新的逻辑
        
        optimal_data.append({
            "video": vid,               # 视频名称
            "prompt_name": prompt,      # 完整 Prompt 名称 (不截断)
            "seed": seed_val,           # 对应的随机种子
            "initial_sim": init_sim,
            "optimal_steps": opt_steps,
            "optimal_cfg": opt_cfg
        })
    
    print(f"✅ Valid Data Points: {len(optimal_data)} (Skipped {skipped_count} outliers)")
    
    if not optimal_data:
        print("❌ No valid data points found after filtering.")
        return

    df_opt = pd.DataFrame(optimal_data)
    df_opt.to_csv(os.path.join(OUTPUT_DIR, "optimal_params_clean.csv"), index=False)
    print(f"💾 Saved full details to {os.path.join(OUTPUT_DIR, 'optimal_params_clean.csv')}")
    
    # 4. 绘图与拟合
    sns.set_theme(style="whitegrid")
    
    # --- Plot A: Initial Sim vs Optimal Steps ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_opt, x="initial_sim", y="optimal_steps", s=100, color='royalblue', alpha=0.6)
    
    if len(df_opt) > 1:
        # 线性拟合
        z = np.polyfit(df_opt["initial_sim"], df_opt["optimal_steps"], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df_opt["initial_sim"].min(), df_opt["initial_sim"].max(), 100)
        plt.plot(x_range, p(x_range), "r--", linewidth=2.5, label=f"Fit: Steps = {z[0]:.2f} * Sim + {z[1]:.2f}")
        print(f"\n📉 [Formula] Steps = {z[0]:.4f} * Initial_Sim + {z[1]:.4f}")
    
    plt.title("Relationship: Initial Similarity vs. Optimal Steps (Target: 95% Quality)", fontsize=14)
    plt.xlabel("Initial CLIP Similarity", fontsize=12)
    plt.ylabel("Optimal Steps", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "relationship_steps_no_warp.png"))
    print(f"✅ Saved steps plot.")

    # --- Plot B: Initial Sim vs Optimal CFG ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_opt, x="initial_sim", y="optimal_cfg", s=100, color='forestgreen', alpha=0.6)
    
    if len(df_opt) > 1:
        z_cfg = np.polyfit(df_opt["initial_sim"], df_opt["optimal_cfg"], 1)
        p_cfg = np.poly1d(z_cfg)
        plt.plot(x_range, p_cfg(x_range), "orange", linestyle='--', linewidth=2.5, label=f"Fit: CFG = {z_cfg[0]:.2f} * Sim + {z_cfg[1]:.2f}")
        print(f"📉 [Formula] CFG Ratio = {z_cfg[0]:.4f} * Initial_Sim + {z_cfg[1]:.4f}")

    plt.title("Relationship: Initial Similarity vs. Optimal CFG (Target: 95% Quality)", fontsize=14)
    plt.xlabel("Initial CLIP Similarity", fontsize=12)
    plt.ylabel("Optimal CFG Ratio", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "relationship_cfg_no_warp.png"))
    print(f"✅ Saved cfg plot.")

if __name__ == "__main__":
    main()