import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 根据实际情况修改
import sys
import torch
import requests
import base64
import numpy as np
import time
import cv2
from omegaconf import OmegaConf

# 引入 Wan2.2 依赖
current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path))]
for root in project_roots:
    sys.path.insert(0, root) if root not in sys.path else None

from videox_fun.models import AutoencoderKLWan
from videox_fun.utils.utils import get_video_to_video_latent, save_videos_grid

# ================= 配置 =================
MODEL_NAME = "models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP"
# Edge 端只需要 VAE 配置，可以单独加载 config 或指向同一个文件
CONFIG_PATH = "config/wan2.2/wan_civitai_i2v.yaml" 
DEVICE = "cuda"
WEIGHT_DTYPE = torch.bfloat16
CLOUD_URL = "http://127.0.0.1:12346/inference"
# CLOUD_URL = "http://172.16.33.142:12346/inference"

# 实验参数
INPUT_VIDEO_PATH = "asset/scene_021_left-forward.mp4" 
PROMPT = "A video of streetview in Minecraft voxel style, made of cube blocks, low poly, pixelated textures, blocky trees, high quality, detailed." 
NEGATIVE_PROMPT = "curves, round, high poly, low quality, blurry, distortion."
SAMPLE_SIZE = [480, 832]
FPS = 16
SEED = 43

TEST_STEPS = 50
STRENGTH = 0.5
GUIDANCE_SCALE = 6.0

print(f"🏠 [Edge] Initializing Wan2.2 Client...")

config = OmegaConf.load(CONFIG_PATH)

# 1. 加载 VAE
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(MODEL_NAME, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(DEVICE).to(WEIGHT_DTYPE)

# --- 辅助函数 ---
def encode_tensor(tensor):
    np_array = tensor.cpu().float().numpy().astype(np.float16)
    return base64.b64encode(np_array.tobytes()).decode('utf-8')

def decode_tensor(b64_str, shape):
    bytes_data = base64.b64decode(b64_str)
    np_array = np.frombuffer(bytes_data, dtype=np.float16)
    return torch.from_numpy(np_array.copy()).reshape(shape).to(DEVICE).to(WEIGHT_DTYPE)

def load_video_frames(video_path, frames_num, height, width):
    if not os.path.exists(video_path): raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < frames_num:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height))
        frame = (frame.astype(np.float32) / 127.5) - 1.0 
        frames.append(frame)
    cap.release()
    # 补齐
    if len(frames) > 0:
        while len(frames) < frames_num:
            frames.append(frames[-1])
    video_tensor = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0)
    return video_tensor.to(DEVICE).to(WEIGHT_DTYPE)

def main():
    print("🎬 Encoding Video...")
    
    # 计算帧数逻辑
    cap_temp = cv2.VideoCapture(INPUT_VIDEO_PATH)
    total_frames_raw = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_temp.release()
    # Wan2.2 VAE 压缩逻辑
    temporal_compression_ratio = 4
    VIDEO_LENGTH = ((total_frames_raw - 1) // 4) * 4 + 1
    if VIDEO_LENGTH < 5: VIDEO_LENGTH = 5
    
    # 调整 Sample Size (必须是16倍数)
    target_height = SAMPLE_SIZE[0]
    target_width  = SAMPLE_SIZE[1]

    # 加载并编码
    input_video_tensor = load_video_frames(INPUT_VIDEO_PATH, VIDEO_LENGTH, target_height, target_width)
    
    with torch.no_grad():
        # Wan2.2 V2V 推荐用 mode() 而非 sample() 以保持原视频特征
        init_latents = vae.encode(input_video_tensor).latent_dist.mode()
    
    # 2. 发送请求
    # Wan2.2 的 Latents 形状通常是 [B, 16, F_lat, H_lat, W_lat]
    print(f"   Latents Shape: {init_latents.shape}")
    
    payload = {
        "latents_b64": encode_tensor(init_latents),
        "shape": list(init_latents.shape),
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "steps": TEST_STEPS,
        "strength": STRENGTH,
        "guidance_scale": GUIDANCE_SCALE,
        "seed": SEED
    }

    print(f"🚀 Sending to Cloud (Steps={TEST_STEPS}, Strength={STRENGTH})...")
    
    try:
        t0 = time.time()
        session = requests.Session()
        resp = session.post(CLOUD_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        print(f"✅ Received Response. Cloud Time: {data['process_time']:.2f}s, Total: {time.time()-t0:.2f}s")
        
        # 3. 解码
        print("🏠 Decoding Latents...")
        result_b64 = data["result_b64"]
        latents_out = decode_tensor(result_b64, init_latents.shape)
        
        with torch.no_grad():
            frames = vae.decode(latents_out).sample
            frames = (frames / 2 + 0.5).clamp(0, 1).cpu().float()

        output_dir = "output_split_wan"
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        save_filename = os.path.join(output_dir, f"wan_edge_dynamic.mp4")
        
        save_videos_grid(frames, save_filename, fps=FPS)
        print(f"🎉 Video Saved: {save_filename}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()