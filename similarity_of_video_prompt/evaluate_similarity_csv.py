import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === 配置区域 ===
# 根据evaluate_similarity.py生成的.csv文件直接画图，便于调整画图修改（因为求相似度需要花时间）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# PROMPT_CONFIG_FILE = os.path.join(parent_dir, "prompts_config.json")
PROMPT_CONFIG_FILE = "/home/zhoujh/Edge-Cloud-diffusion/MyCogVideo-v2v/output/15_new5_4_6_CFG7/prompts_config.json"
INPUT_CSV = "source_target_gap_15.csv"
OUTPUT_PLOT_DIR = "plots_similarity_15"

if not os.path.exists(OUTPUT_PLOT_DIR):
    os.makedirs(OUTPUT_PLOT_DIR)

def main():
    # 1. 读取 Prompts JSON 用于排序
    if not os.path.exists(PROMPT_CONFIG_FILE):
        print(f"❌ Config file not found: {PROMPT_CONFIG_FILE}")
        return
    
    with open(PROMPT_CONFIG_FILE, 'r') as f:
        prompts_data = json.load(f)
    
    # 按照 ID 从小到大排序 (1, 2, 3...)
    prompts_data.sort(key=lambda x: x['id']) 
    prompt_order_list = [p['name'] for p in prompts_data]
    print(f"📖 Loaded {len(prompts_data)} prompts order definition.")

    # 2. 读取 CSV
    if not os.path.exists(INPUT_CSV):
        print(f"❌ CSV file not found: {INPUT_CSV}")
        return
    
    df = pd.read_csv(INPUT_CSV)
    print(f"📂 Loaded data from {INPUT_CSV}, rows: {len(df)}")

    # 3. 绘图
    plot_analysis(df, prompt_order_list)

def plot_analysis(df, prompt_order):
    print("📊 Generating plots...")
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(16, 10))
    
    # === 关键修正步骤 ===
    # 你的要求：左侧(Y轴)是 Video，下方(X轴)是 Prompt
    # 所以 pivot 必须这样写：
    # index="video"       -> 决定了 Y轴 是视频
    # columns="prompt_name" -> 决定了 X轴 是Prompt
    pivot_table = df.pivot(index="video", columns="prompt_name", values="initial_clip_score")
    
    # 强制让列（Prompt）按照 ID 顺序排列
    pivot_table = pivot_table.reindex(columns=prompt_order)
    
    # 绘制热力图
    # annot_kws: 设置格子内数字的样式
    ax = sns.heatmap(
        pivot_table, 
        annot=True, 
        fmt=".3f", 
        cmap="RdYlBu_r", 
        linewidths=.5,
        annot_kws={"size": 11, "weight": "bold"}
    )
    
    # === 样式与标签修正 ===
    # 标题
    plt.title("Initial Semantic Similarity (Source Video vs Target Prompt)", fontsize=20, fontweight='bold', pad=20)
    
    # 坐标轴大标题
    # 确保 Label 和数据轴对应：
    # pivot index 是 video -> Y Label 设为 Source Video
    # pivot columns 是 prompt -> X Label 设为 Target Prompt
    plt.xlabel("Target Prompt", fontsize=16, fontweight='bold', labelpad=15)
    plt.ylabel("Source Video", fontsize=16, fontweight='bold', labelpad=15)
    
    # === 刻度文字居中修正 ===
    # rotation=0: 文字不旋转，水平摆放（这样最居中）
    # ha='center': 文字中心对齐刻度线
    # 如果你的 Prompt 名字特别长导致重叠，可以把 rotation 改成 30
    plt.xticks(rotation=0, ha='center', fontsize=12, fontweight='bold')
    
    # Y轴刻度文字 (视频文件名)
    plt.yticks(rotation=0, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_PLOT_DIR, "heatmap_similarity_centered_15.png")
    plt.savefig(save_path)
    print(f" 👉 Heatmap saved to {save_path}")

if __name__ == "__main__":
    main()