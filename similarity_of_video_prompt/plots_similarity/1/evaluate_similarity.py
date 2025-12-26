import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# === 配置区域 ===
# 应该改成取前俩帧出来看看即可
INPUT_VIDEO_DIR = os.path.join(parent_dir, "asset/batch_videos_6s_5")
PROMPT_CONFIG_FILE = os.path.join(parent_dir, "prompts_config.json")

OUTPUT_CSV = "source_target_gap.csv"
OUTPUT_PLOT_DIR = "plots_similarity"

# 显卡设置
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from stream_metrics import calc_clip_score
if not os.path.exists(OUTPUT_PLOT_DIR):
    os.makedirs(OUTPUT_PLOT_DIR)

def main():
    # 1. 读取 Prompts
    if not os.path.exists(PROMPT_CONFIG_FILE):
        print(f"❌ Config file not found: {PROMPT_CONFIG_FILE}")
        return
    with open(PROMPT_CONFIG_FILE, 'r') as f:
        prompts_data = json.load(f)
    print(f"📖 Loaded {len(prompts_data)} prompts.")

    # 2. 读取 Videos
    if not os.path.exists(INPUT_VIDEO_DIR):
        print(f"❌ Video dir not found: {INPUT_VIDEO_DIR}")
        return
    video_files = [f for f in os.listdir(INPUT_VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
    video_files.sort()
    print(f"📂 Found {len(video_files)} videos.")

    results = []

    print("\n🚀 Starting Similarity Evaluation (Source Video vs Target Prompt)...")
    
    # 3. 双重循环计算
    # 这里的 CLIP Score 代表：原视频画面 与 目标文字描述 的相似程度
    # 分数越低 = 语义差距越大 = 理论上生成难度越大
    for video_file in tqdm(video_files, desc="Videos"):
        video_path = os.path.join(INPUT_VIDEO_DIR, video_file)
        video_name = os.path.splitext(video_file)[0]
        
        for p_item in prompts_data:
            p_id = p_item['id']
            p_text = p_item['prompt']
            # p_diff = p_item['difficulty']
            
            # 计算 CLIP Score (只取 text_score, 忽略 consistency)
            # 注意：这里我们是用 "原视频" 去测 "目标Prompt"
            score, _ = calc_clip_score(video_path, p_text)
            
            results.append({
                "video": video_name,
                "prompt_id": p_id,
                "prompt_name": p_item['name'],
                # "difficulty_label": p_diff,
                "initial_clip_score": score
            })

    # 4. 保存数据
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Similarity scores saved to {OUTPUT_CSV}")

    # 5. 可视化
    plot_analysis(df)

def plot_analysis(df):
    print("📊 Generating plots...")
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    
    # --- 图 1: 热力图 (Videos vs Prompts) ---
    # 这是一个矩阵，颜色越冷(蓝/紫)代表相似度越低(Gap越大)，颜色越暖(红)代表越相似
    plt.figure(figsize=(16, 10))
    pivot_table = df.pivot(index="prompt_name", columns="video", values="initial_clip_score")
    
    # 使用 'coolwarm' 色图，因为我们需要区分 高(相似) 和 低(不相似)
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="RdYlBu_r", linewidths=.5)
    
    plt.title("Initial Semantic Similarity (Source Video vs Target Prompt)", fontsize=16)
    plt.xlabel("Source Video", fontsize=12)
    plt.ylabel("Target Prompt", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "heatmap_similarity.png"))
    print("   👉 Heatmap saved.")

    # # --- 图 2: 难度标签验证 (Boxplot/Strip Plot) ---
    # # 验证您的 "Easy/Medium/Hard/Extreme" 标签是否真的对应了 CLIP Score 的下降
    # plt.figure(figsize=(12, 8))
    
    # # 定义顺序
    # order = ["Easy", "Medium", "Hard", "Extreme"]
    
    # # 箱线图展示分布
    # sns.boxplot(x="difficulty_label", y="initial_clip_score", data=df, order=order, palette="Set2", linewidth=1.5)
    # # 散点图展示具体点 (Jitter)
    # sns.stripplot(x="difficulty_label", y="initial_clip_score", data=df, order=order, color=".25", size=4, alpha=0.6)
    
    # plt.title("Verification of Difficulty Labels: Lower Score = Harder Task", fontsize=16)
    # plt.xlabel("Difficulty Label (Defined in JSON)", fontsize=12)
    # plt.ylabel("Initial CLIP Score (Source vs Prompt)", fontsize=12)
    
    # # 添加平均值连线，看趋势
    # means = df.groupby("difficulty_label")["initial_clip_score"].mean().reindex(order)
    # plt.plot(range(len(order)), means, marker='o', color='red', linewidth=2, linestyle='--', label="Mean Trend")
    # plt.legend()
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "difficulty_validation.png"))
    # print("   👉 Difficulty validation plot saved.")

    # # --- 图 3: 散点图 (ID vs Score) ---
    # # 直观展示每个 Prompt ID 的平均得分为多少
    # plt.figure(figsize=(14, 6))
    
    # avg_scores = df.groupby("prompt_id")["initial_clip_score"].mean().reset_index()
    # # 映射颜色
    # # 为了让不同的难度显示不同颜色，我们需要 merge 回去
    # diff_map = df[["prompt_id", "difficulty_label"]].drop_duplicates()
    # avg_scores = avg_scores.merge(diff_map, on="prompt_id")
    
    # sns.scatterplot(x="prompt_id", y="initial_clip_score", hue="difficulty_label", 
    #                 hue_order=order, data=avg_scores, s=100, palette="deep")
    
    # # 画线连接
    # plt.plot(avg_scores["prompt_id"], avg_scores["initial_clip_score"], color='gray', alpha=0.3)
    
    # plt.title("Average Initial Similarity per Prompt ID", fontsize=16)
    # plt.xlabel("Prompt ID", fontsize=12)
    # plt.ylabel("Average CLIP Score", fontsize=12)
    # plt.xticks(avg_scores["prompt_id"]) # 确保显示所有 ID
    # plt.grid(True, linestyle='--', alpha=0.6)
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(OUTPUT_PLOT_DIR, "scatter_id_score.png"))
    # print("   👉 Scatter plot saved.")

if __name__ == "__main__":
    main()