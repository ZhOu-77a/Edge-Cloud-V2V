import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


CSV_PATH = "experiment_results_prompt_diff_19.csv"
GAP_CSV_PATH = "similarity_of_video_prompt/source_target_gap_19.csv"
OUTPUT_ROOT = "plots_grouped_analysis_group_4"

# === 实验基准参数 ===
DEFAULT_FPS = 8
DEFAULT_STEPS = 30
DEFAULT_CFG = 1.0
SCORE_THRESHOLD = 0.25  # 最终达标阈值
GROUP_BY = 4

if not os.path.exists(OUTPUT_ROOT): os.makedirs(OUTPUT_ROOT)

def process_and_plot():
    # 1. 加载数据
    if not os.path.exists(CSV_PATH) or not os.path.exists(GAP_CSV_PATH):
        print(f"找不到输入文件 {CSV_PATH} 或 {GAP_CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    df_gap = pd.read_csv(GAP_CSV_PATH)
    
    # 2. 预筛选：剔除最终效果不达标的对 ：在 FPS=8, CFG=1.0, Steps=最大的情况下，Clip Score 的均值（跨seed）是否 >= 0.25
    max_step_available = df['steps'].max()
    final_perf = df[
        (df['steps'] == max_step_available) & 
        (np.isclose(df['cfg'], DEFAULT_CFG)) & 
        (df['fps'] == DEFAULT_FPS)
    ].groupby(['video', 'prompt_name'])['clip_score'].mean().reset_index()
    
    valid_keys = final_perf[final_perf['clip_score'] >= SCORE_THRESHOLD][['video', 'prompt_name']]
    removed_keys = final_perf[final_perf['clip_score'] < SCORE_THRESHOLD][['video', 'prompt_name', 'clip_score']]
    
    # --- 打印剔除报表 ---
    print("\n" + "="*60)
    print(f"🚩 被剔除的 Video-Prompt 对 (最终 CLIP Score < {SCORE_THRESHOLD}):")
    if removed_keys.empty:
        print(" (无)")
    else:
        print(removed_keys.sort_values('clip_score').to_string(index=False))
    print("="*60)

    # 3. 数据过滤：
    df_filtered = pd.merge(df, valid_keys, on=['video', 'prompt_name'], how='inner')
    df_gap_filtered = pd.merge(df_gap, valid_keys, on=['video', 'prompt_name'], how='inner')

    # 4. 分组 (根据 initial_clip_score)
    df_gap_filtered['group'] = pd.qcut(df_gap_filtered['initial_clip_score'], GROUP_BY, labels=[f"Group_{i+1}" for i in range(GROUP_BY)])
    
    # --- 打印分组明细表 ---
    print("\n📊 分组详细清单 (按初始分数排序):")
    sorted_gap = df_gap_filtered.sort_values(['group', 'initial_clip_score'])
    print(sorted_gap[['group', 'video', 'prompt_name', 'initial_clip_score']].to_string(index=False))
    
    # 5. 遍历 Seed 生成图表
    unique_seeds = sorted(df_filtered['seed'].unique())
    group_names = [f"Group_{i+1}" for i in range(GROUP_BY)]
    colors = ['#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231']

    for seed in unique_seeds:
        seed_dir = os.path.join(OUTPUT_ROOT, f"seed_{seed}")
        if not os.path.exists(seed_dir): os.makedirs(seed_dir)
        
        df_seed = df_filtered[df_filtered['seed'] == seed].copy()

        # 定义绘图任务
        tasks = [
            ('steps', {'cfg': DEFAULT_CFG, 'fps': DEFAULT_FPS}, "Steps vs CLIP", "steps"),
            ('cfg', {'steps': DEFAULT_STEPS, 'fps': DEFAULT_FPS}, "CFG vs CLIP", "cfg")
        ]

        for x_var, filters, title_suffix, file_suffix in tasks:
            all_groups_avg = []

            for idx, g_name in enumerate(group_names):
                # 获取该组的 video-prompt 对及其初始分数
                g_pairs = df_gap_filtered[df_gap_filtered['group'] == g_name]
                df_g = pd.merge(df_seed, g_pairs[['video', 'prompt_name', 'initial_clip_score']], on=['video', 'prompt_name'])

                # 严格控制变量过滤
                mask = (df_g['fps'] == filters.get('fps', DEFAULT_FPS))
                if 'cfg' in filters: mask &= np.isclose(df_g['cfg'], filters['cfg'])
                if 'steps' in filters: mask &= (df_g['steps'] == filters['steps'])
                
                plot_data = df_g[mask].sort_values(x_var)
                if plot_data.empty: continue

                # --- 核心改进：为 Steps 图表插入 Step 0 ---
                if x_var == 'steps':
                    processed_data_list = []
                    for (v, p), sub_df in plot_data.groupby(['video', 'prompt_name']):
                        # 获取该对的初始分数
                        init_score = g_pairs[(g_pairs['video']==v) & (g_pairs['prompt_name']==p)]['initial_clip_score'].values[0]
                        # 构建 step 0 节点
                        step0_row = pd.DataFrame({'steps': [0], 'clip_score': [init_score], 'video': [v], 'prompt_name': [p]})
                        # 合并
                        combined = pd.concat([step0_row, sub_df[['steps', 'clip_score', 'video', 'prompt_name']]]).sort_values('steps')
                        processed_data_list.append(combined)
                    
                    final_plot_data = pd.concat(processed_data_list)
                else:
                    final_plot_data = plot_data

                # 计算组平均曲线
                g_avg = final_plot_data.groupby(x_var)['clip_score'].mean().reset_index()
                all_groups_avg.append((g_name, g_avg))

                # 组内详细曲线图
                fig, ax = plt.subplots(figsize=(10, 6))
                for (v, p), sub_df in final_plot_data.groupby(['video', 'prompt_name']):
                    ax.plot(sub_df[x_var], sub_df['clip_score'], alpha=0.15, linewidth=1)
                ax.plot(g_avg[x_var], g_avg['clip_score'], color='red', linewidth=3, label=f'{g_name} Mean')
                ax.set_title(f"{g_name} | {title_suffix} (Seed {seed})")
                ax.set_xlabel(x_var.capitalize())
                ax.set_ylabel("CLIP Score")
                ax.grid(True, ls='--', alpha=0.5)
                ax.legend()
                fig.savefig(os.path.join(seed_dir, f"{g_name}_{file_suffix}_detail.png"))
                plt.close(fig)

            # --- 绘制 GROUP_BY 组对比总图 ---
            plt.figure(figsize=(12, 7))
            for idx, (g_name, avg_df) in enumerate(all_groups_avg):
                plt.plot(avg_df[x_var], avg_df['clip_score'], marker='o', label=g_name, color=colors[idx], linewidth=2.5, markersize=4)
            
            plt.title(f"Group Comparison: {title_suffix} (Seed {seed})", fontsize=15, fontweight='bold')
            plt.xlabel(x_var.upper(), fontsize=12)
            plt.ylabel("Average CLIP Score", fontsize=12)
            plt.legend(title="Groups (Initial Similarity)", loc='lower right')
            plt.grid(True, linestyle=':', alpha=0.6)
            
            # 在 Steps 图中额外标注 0.25 阈值线
            if x_var == 'steps':
                plt.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Threshold 0.25')
            
            plt.savefig(os.path.join(seed_dir, f"TOTAL_COMPARE_{file_suffix}.png"), dpi=200)
            plt.close()

    print(f"\n✨ 处理完成！所有图表已保存在目录: {OUTPUT_ROOT}")

if __name__ == "__main__":
    process_and_plot()