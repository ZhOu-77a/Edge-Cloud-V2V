# import os
# import sys
# import json
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
# from stream_metrics import calc_clip_score

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

# # 显卡设置
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# # === 配置区域 ===
# # 应该改成取前俩帧出来看看即可
# INPUT_VIDEO_DIR = "/home/zhoujh/Edge-Cloud-diffusion/Dataset/video_2s_1"
# PROMPT_CONFIG_FILE = "/home/zhoujh/Edge-Cloud-diffusion/MyCogVideo-v2v/prompts_config.json"


# OUTPUT_CSV = "source_target_gap_16.csv"
# OUTPUT_PLOT_DIR = "plots_similarity_16"




# if not os.path.exists(OUTPUT_PLOT_DIR):
#     os.makedirs(OUTPUT_PLOT_DIR)

# def main():
#     # 1. 读取 Prompts
#     if not os.path.exists(PROMPT_CONFIG_FILE):
#         print(f"❌ Config file not found: {PROMPT_CONFIG_FILE}")
#         return
#     with open(PROMPT_CONFIG_FILE, 'r') as f:
#         prompts_data = json.load(f)
    
#     # 【新增】确保按照 ID 排序，并提取名称列表用于后续画图排序
#     # key=lambda x: x['id'] 保证了 prompt 列表是按 id 1, 2, 3... 顺序排列的
#     prompts_data.sort(key=lambda x: x['id']) 
#     prompt_order_list = [p['name'] for p in prompts_data]
    
#     print(f"📖 Loaded {len(prompts_data)} prompts.")

#     # 2. 读取 Videos
#     if not os.path.exists(INPUT_VIDEO_DIR):
#         print(f"❌ Video dir not found: {INPUT_VIDEO_DIR}")
#         return
#     video_files = [f for f in os.listdir(INPUT_VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
#     video_files.sort()
#     print(f"📂 Found {len(video_files)} videos.")

#     results = []

#     print("\n🚀 Starting Similarity Evaluation (Source Video vs Target Prompt)...")
    
#     # 3. 双重循环计算
#     # 这里的 CLIP Score 代表：原视频画面 与 目标文字描述 的相似程度
#     # 分数越低 = 语义差距越大 = 理论上生成难度越大
#     for video_file in tqdm(video_files, desc="Videos"):
#         video_path = os.path.join(INPUT_VIDEO_DIR, video_file)
#         video_name = os.path.splitext(video_file)[0]
        
#         for p_item in prompts_data:
#             p_id = p_item['id']
#             p_text = p_item['prompt']
#             # p_diff = p_item['difficulty']
            
#             # 计算 CLIP Score (只取 text_score, 忽略 consistency)
#             # 注意：这里我们是用 "原视频" 去测 "目标Prompt"
#             score, _ = calc_clip_score(video_path, p_text)
            
#             results.append({
#                 "video": video_name,
#                 "prompt_id": p_id,
#                 "prompt_name": p_item['name'],
#                 # "difficulty_label": p_diff,
#                 "initial_clip_score": score
#             })

#     # 4. 保存数据
#     df = pd.DataFrame(results)
#     df.to_csv(OUTPUT_CSV, index=False)
#     print(f"\n💾 Similarity scores saved to {OUTPUT_CSV}")

#     # 5. 可视化
#     # 将排序好的 prompt 名字列表传进去
#     plot_analysis(df, prompt_order_list)

# def plot_analysis(df, prompt_order):
#     print("📊 Generating plots...")
    
#     # 设置绘图风格
#     sns.set_theme(style="whitegrid")
    
#     # --- 图 1: 热力图 (Videos vs Prompts) ---
#     # 颜色越冷(蓝/紫)代表相似度越低(Gap越大)，颜色越暖(红)代表越相似
#     plt.figure(figsize=(16, 10))
    
#     # 横纵坐标调换
#     # index (Y轴) = video
#     # columns (X轴) = prompt_name
#     pivot_table = df.pivot(index="video", columns="prompt_name", values="initial_clip_score")
    
#     # 按照 JSON ID 的顺序排列列(Columns)
#     # reindex 会根据列表里的名字顺序重新排列列
#     pivot_table = pivot_table.reindex(columns=prompt_order)
    
#     # 使用 'coolwarm' 或 'RdYlBu_r' 色图，因为我们需要区分 高(相似) 和 低(不相似)
#     # annot_kws={"size": 11, "weight": "bold"} 让格子里的数字更清晰
#     ax = sns.heatmap(
#         pivot_table, 
#         annot=True, 
#         fmt=".3f", 
#         cmap="RdYlBu_r", 
#         linewidths=.5,
#         annot_kws={"size": 11}
#     )
    
#     # 字体变大、加粗
#     plt.title("Initial Semantic Similarity (Source Video vs Target Prompt)", fontsize=20, fontweight='bold', pad=20)
    
#     # 坐标轴标签设置
#     plt.xlabel("Target Prompt", fontsize=16, fontweight='bold', labelpad=15)
#     plt.ylabel("Source Video", fontsize=16, fontweight='bold', labelpad=15)
    
#     # 刻度标签设置 
#     # rotation=30: X轴文字倾斜30度，防止重叠
#     plt.xticks( ha='right', fontsize=12)
#     plt.yticks(fontsize=12)
    
#     plt.tight_layout()
#     save_path = os.path.join(OUTPUT_PLOT_DIR, "heatmap_similarity.png")
#     plt.savefig(save_path)
#     print(f" 👉 Heatmap saved to {save_path}")

# if __name__ == "__main__":
#     main()

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 显卡设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
device = "cuda" if torch.cuda.is_available() else "cpu"

# === 配置区域 ===
# INPUT_VIDEO_DIR = os.path.join(parent_dir, "asset/batch_videos_new")
# PROMPT_CONFIG_FILE = os.path.join(parent_dir, "prompts_config.json")
INPUT_VIDEO_DIR = "/home/zhoujh/Edge-Cloud-diffusion/Dataset/video_2s_21"
PROMPT_CONFIG_FILE = "/home/zhoujh/Edge-Cloud-diffusion/MyCogVideo-v2v/prompts_config.json"


OUTPUT_CSV = "source_target_gap_19.csv"
OUTPUT_PLOT_DIR = "plots_similarity_19"

if not os.path.exists(OUTPUT_PLOT_DIR):
    os.makedirs(OUTPUT_PLOT_DIR)

# ===  Semantic Clip Score (3帧采样) ===
def calc_clip_score(video_path, prompt, model, processor):
    try:
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        video_embs = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return 0.0
        
        # 1. 获取总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return 0.0
            
        # 2. 均匀采样 3 帧索引
        target_indices = [0, total_frames // 2, max(0, total_frames - 1)]
        # # 获取所有帧的索引
        # target_indices = list(range(total_frames))
        target_indices = sorted(list(set(target_indices))) # 去重排序

        text_embeds = None

        # 3. 循环读取这 3 帧
        for frame_idx in target_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                
                with torch.no_grad():
                    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    
                    # 收集 Image Embedding
                    video_embs.append(outputs.image_embeds)
                    text_embeds = outputs.text_embeds
        
        cap.release()

        if not video_embs: 
            return 0.0

        # 4. 计算分数
        video_embs = torch.cat(video_embs, dim=0) # Shape: [3, 512]
        
        text_score = cos(text_embeds, video_embs).mean().cpu().item()

        return text_score

    except Exception as e:
        print(f"Error in CLIP Score: {e}")
        return 0.0

def main():
    # 1. 读取 Prompts
    if not os.path.exists(PROMPT_CONFIG_FILE):
        print(f"❌ Config file not found: {PROMPT_CONFIG_FILE}")
        return
    with open(PROMPT_CONFIG_FILE, 'r') as f:
        prompts_data = json.load(f)
    
    # 排序
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

    # === 初始化模型 ===
    print("⏳ Loading CLIP model (openai/clip-vit-base-patch32)...")
    try:
        model_id = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Failed to load CLIP model: {e}")
        return

    results = []
    print("\n🚀 Starting Similarity Evaluation (Text Score Only)...")
    
    # 3. 双重循环计算
    for video_file in tqdm(video_files, desc="Videos"):
        video_path = os.path.join(INPUT_VIDEO_DIR, video_file)
        video_name = os.path.splitext(video_file)[0]
        
        for p_item in prompts_data:
            p_id = p_item['id']
            p_text = p_item['prompt']
            
            # === 修改：只接收一个返回值 ===
            score = calc_clip_score(video_path, p_text, model, processor)
            
            results.append({
                "video": video_name,
                "prompt_id": p_id,
                "prompt_name": p_item['name'],
                "initial_clip_score": score
            })

    # 4. 保存数据
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 Similarity scores saved to {OUTPUT_CSV}")

    # 5. 可视化
    plot_analysis(df, prompt_order_list)

def plot_analysis(df, prompt_order):
    print("📊 Generating plots...")
    
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(16, 10))
    
    pivot_table = df.pivot(index="video", columns="prompt_name", values="initial_clip_score")
    pivot_table = pivot_table.reindex(columns=prompt_order)
    
    ax = sns.heatmap(
        pivot_table, 
        annot=True, 
        fmt=".3f", 
        cmap="RdYlBu_r", 
        linewidths=.5,
        annot_kws={"size": 11}
    )
    
    plt.title("Initial Semantic Similarity (Source Video vs Target Prompt)", fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("Target Prompt", fontsize=16, fontweight='bold', labelpad=15)
    plt.ylabel("Source Video", fontsize=16, fontweight='bold', labelpad=15)
    
    plt.xticks(ha='right', fontsize=12, rotation=30)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_PLOT_DIR, "heatmap_similarity.png")
    plt.savefig(save_path)
    print(f" 👉 Heatmap saved to {save_path}")

if __name__ == "__main__":
    main()