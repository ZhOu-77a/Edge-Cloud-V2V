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
INPUT_VIDEO_DIR = os.path.join(parent_dir, "asset/batch_videos_6s_5_")
PROMPT_CONFIG_FILE = os.path.join(parent_dir, "prompts_config.json")

OUTPUT_CSV = "source_target_gap.csv"
OUTPUT_PLOT_DIR = "plots_similarity"

# 显卡设置
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 尝试导入 calc_clip_score，防止报错
try:
    from stream_metrics import calc_clip_score
except ImportError:
    print("⚠️ Warning: 'stream_metrics' module not found. Make sure it is in the parent directory.")
    # 定义一个 dummy 函数防止代码运行崩溃（调试用）
    def calc_clip_score(video_path, text):
        return np.random.random(), 0.0

if not os.path.exists(OUTPUT_PLOT_DIR):
    os.makedirs(OUTPUT_PLOT_DIR)

def main():
    # 1. 读取 Prompts
    if not os.path.exists(PROMPT_CONFIG_FILE):
        print(f"❌ Config file not found: {PROMPT_CONFIG_FILE}")
        return
    with open(PROMPT_CONFIG_FILE, 'r') as f:
        prompts_data = json.load(f)
    
    # 【新增】确保按照 ID 排序，并提取名称列表用于后续画图排序
    # key=lambda x: x['id'] 保证了 prompt 列表是按 id 1, 2, 3... 顺序排列的
    prompts_data.sort(key=lambda x: x['id']) 
    prompt_order_list = [p['name'] for p in prompts_data]
    
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
    # 【修改】将排序好的 prompt 名字列表传进去
    plot_analysis(df, prompt_order_list)

def plot_analysis(df, prompt_order):
    print("📊 Generating plots...")
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    
    # --- 图 1: 热力图 (Videos vs Prompts) ---
    # 这是一个矩阵，颜色越冷(蓝/紫)代表相似度越低(Gap越大)，颜色越暖(红)代表越相似
    plt.figure(figsize=(16, 10))
    
    # 【修改 1】横纵坐标调换
    # index (Y轴) = video
    # columns (X轴) = prompt_name
    pivot_table = df.pivot(index="video", columns="prompt_name", values="initial_clip_score")
    
    # 【修改 2】强制按照 JSON ID 的顺序排列列(Columns)
    # reindex 会根据列表里的名字顺序重新排列列
    pivot_table = pivot_table.reindex(columns=prompt_order)
    
    # 使用 'coolwarm' 或 'RdYlBu_r' 色图，因为我们需要区分 高(相似) 和 低(不相似)
    # annot_kws={"size": 11, "weight": "bold"} 让格子里的数字更清晰
    ax = sns.heatmap(
        pivot_table, 
        annot=True, 
        fmt=".3f", 
        cmap="RdYlBu_r", 
        linewidths=.5,
        annot_kws={"size": 11}
    )
    
    # 【修改 3】字体变大、加粗
    plt.title("Initial Semantic Similarity (Source Video vs Target Prompt)", fontsize=20, fontweight='bold', pad=20)
    
    # 坐标轴标签设置
    plt.xlabel("Target Prompt", fontsize=16, fontweight='bold', labelpad=15)
    plt.ylabel("Source Video", fontsize=16, fontweight='bold', labelpad=15)
    
    # 刻度标签设置 
    # rotation=30: X轴文字倾斜30度，防止重叠
    plt.xticks( ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_PLOT_DIR, "heatmap_similarity.png")
    plt.savefig(save_path)
    print(f" 👉 Heatmap saved to {save_path}")

if __name__ == "__main__":
    main()