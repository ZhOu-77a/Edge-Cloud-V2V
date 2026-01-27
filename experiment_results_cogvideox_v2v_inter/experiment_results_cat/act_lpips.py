import os
import pandas as pd
import torch
import cv2
import lpips
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# === 路径配置 ===
CSV_PATH = "experiment_results/experiment_report.csv"
VIDEO_DIR = "experiment_results/"
INPUT_VIDEO_PATH = "output_debug/debug_cfg_1.0.mp4" # 作为 Ground Truth 参考
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_video_frames(video_path, target_size=None):
    """读取视频所有帧并预处理为 LPIPS 需要的格式 [-1, 1]"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
        
        # 预处理
        trans_list = []
        if target_size:
            trans_list.append(transforms.Resize(target_size))
        trans_list.append(transforms.ToTensor())
        # LPIPS 需要 normalization 到 [-1, 1]
        trans_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        
        trans = transforms.Compose(trans_list)
        frames.append(trans(pil_img).unsqueeze(0)) # [1, 3, H, W]
    
    cap.release()
    return frames

def run_lpips_eval():
    if not os.path.exists(CSV_PATH):
        print(f"❌ 找不到 CSV 文件: {CSV_PATH}")
        return

    print(f"⏳ Loading LPIPS model (AlexNet backbone)...")
    # AlexNet 比较轻量，适合做感知相似度度量；VGG 更重一些
    loss_fn = lpips.LPIPS(net='alex').to(DEVICE)

    # 1. 读取原始视频作为参考 (Reference)
    print(f"📼 Reading Reference Video: {INPUT_VIDEO_PATH}")
    if not os.path.exists(INPUT_VIDEO_PATH):
        print("❌ 找不到参考输入视频，无法计算失真度！")
        return
        
    ref_frames = load_video_frames(INPUT_VIDEO_PATH)
    # 获取尺寸，后续生成的视频最好 resize 到一样，虽然 LPIPS 支持不同尺寸但在 tensor 计算时需要对齐
    ref_h, ref_w = ref_frames[0].shape[2], ref_frames[0].shape[3]
    target_size = (ref_h, ref_w)

    df = pd.read_csv(CSV_PATH)
    lpips_scores = []

    print(f"🎬 Processing {len(df)} generated videos...")
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        video_name = f"{row['ID']}.mp4"
        video_path = os.path.join(VIDEO_DIR, video_name)
        
        if not os.path.exists(video_path):
            lpips_scores.append(None)
            continue
            
        # 读取生成视频
        gen_frames = load_video_frames(video_path, target_size=target_size)
        
        if len(gen_frames) == 0:
            lpips_scores.append(None)
            continue

        # 计算每一帧的 LPIPS 距离并取平均
        # 确保帧数对齐 (取最小值)
        n_frames = min(len(ref_frames), len(gen_frames))
        curr_score_sum = 0.0
        
        with torch.no_grad():
            for i in range(n_frames):
                # 输入都在 GPU 上
                ref = ref_frames[i].to(DEVICE)
                gen = gen_frames[i].to(DEVICE)
                dist = loss_fn(gen, ref)
                curr_score_sum += dist.item()
        
        avg_score = curr_score_sum / n_frames
        # LPIPS 是距离，越小越好。为了方便记录和归一化，我们保留原始值
        lpips_scores.append(round(avg_score, 5))

    # 更新 CSV
    df['LPIPS_Score'] = lpips_scores
    # 为了兼容以前的逻辑，可以生成一个 'Quality_Index'，比如 (1 - LPIPS) * 100，让它变成“越大越好”
    # 这里我们新增一列用于画图：Perceptual_Quality (越大越好)
    # 假设 LPIPS 范围通常在 0.0 ~ 0.7 之间
    valid_scores = [s for s in lpips_scores if s is not None]
    if valid_scores:
        max_lpips = max(valid_scores)
        # 简单的反转映射，仅供 3D 图颜色参考
        df['Quality_Index_InvLPIPS'] = df['LPIPS_Score'].apply(lambda x: (max_lpips - x) / max_lpips * 100 if x is not None else 0)

    df.to_csv(CSV_PATH, index=False)
    print(f"✅ LPIPS 评分完成，结果已保存至: {CSV_PATH}")
    print("👉 LPIPS 越低越好 (0 = 无失真，0.5+ = 严重失真/雪花)")

if __name__ == "__main__":
    run_lpips_eval()