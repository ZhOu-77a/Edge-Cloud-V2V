import os
import pandas as pd
import torch
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# === 路径配置 ===
# 根据你提供的截图，CSV就在当前目录下
CSV_PATH = "experiment_results/experiment_report.csv"
VIDEO_DIR = "experiment_results/"
PROMPT = "A cute cat."
MODEL_ID = "openai/clip-vit-base-patch32"

def run_clip_eval():
    if not os.path.exists(CSV_PATH):
        print(f"❌ 找不到 CSV 文件: {CSV_PATH}")
        return

    print("⏳ Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    df = pd.read_csv(CSV_PATH)
    scores = []

    print(f"🎬 Processing {len(df)} videos from {VIDEO_DIR}...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # 匹配文件名，如 N15_ratio0.3_m2.mp4
        video_name = f"{row['ID']}.mp4"
        video_path = os.path.join(VIDEO_DIR, video_name)
        
        if not os.path.exists(video_path):
            print(f"⚠️ 找不到视频: {video_name}")
            scores.append(0.0)
            continue
            
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 抽取最后一帧进行评估
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            scores.append(0.0)
            continue
            
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(text=[PROMPT], images=image, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            score = outputs.logits_per_image.item()
            
        scores.append(round(score, 4))

    # 更新 CSV 中的 Quality 字段
    df['Quality_Score'] = scores
    df.to_csv(CSV_PATH, index=False)
    print(f"✅ 已将 CLIP 分数更新至: {CSV_PATH}")

if __name__ == "__main__":
    run_clip_eval()