import os
import sys
import torch
import numpy as np

# === 🛠️ 配置区域 ===
GEN_VIDEO_PATH = "output_experiment_gpu34/front_forward_scene_001_front-forward/fps2_steps30_cfg1.0_seed43.mp4" 
SRC_VIDEO_PATH = "debug_inputs_check_gpu34/ref_front_forward_scene_001_front-forward_fps2.mp4"
PROMPT = "A video of streetview in cartoon style."

# 显卡设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Starting StreamV2V Evaluation on {DEVICE}...")

try:
    from stream_metrics import calc_clip_score, calc_warp_error
    print("📦 Metrics loaded.")
except ImportError as e:
    print(f"❌ Metrics Error: {e}")
    sys.exit(1)

def main():
    print("-" * 50)
    
    if not os.path.exists(GEN_VIDEO_PATH):
        print(f"❌ File not found: {GEN_VIDEO_PATH}")
        return

    # 1. CLIP Score (StreamV2V)
    # 返回: (Prompt一致性, 帧间一致性)
    print("\n[Metric 1] CLIP Score (StreamV2V)...")
    text_score, consist_score = calc_clip_score(GEN_VIDEO_PATH, PROMPT)
    print(f"   👉 Text Alignment Score:     {text_score:.4f} (Original 0-1)")
    print(f"   👉 Temporal Consistency:     {consist_score:.4f} (Original 0-1)")
    
    # 2. Warp Error (StreamV2V)
    print("\n[Metric 2] Warp Error (StreamV2V)...")
    if os.path.exists(SRC_VIDEO_PATH):
        warp_err = calc_warp_error(SRC_VIDEO_PATH, GEN_VIDEO_PATH)
        print(f"   👉 Warp Error:               {warp_err:.4f}")
    else:
        print("   ⚠️ Source video missing, skipping.")

    print("\n" + "-" * 50)

if __name__ == "__main__":
    main()