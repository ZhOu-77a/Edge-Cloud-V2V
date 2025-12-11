import os
import sys
import torch
import requests
import base64
import numpy as np
import time
import imageio.v3 as iio
from PIL import Image


from videox_fun.models import AutoencoderKLCogVideoX
from videox_fun.utils.utils import get_video_to_video_latent, save_videos_grid

# === 调试函数 ===
def print_stats(step_name, tensor):
    if isinstance(tensor, torch.Tensor):
        t = tensor.float().cpu()
        print(f"🏠 [DEBUG-EDGE] {step_name:<25} | Mean: {t.mean():.6f} | Std: {t.std():.6f} | Min: {t.min():.4f} | Max: {t.max():.4f}")

# === 3. 配置 ===
CLOUD_URL = "http://127.0.0.1:12345/inference"
MODEL_NAME = "models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP" 
DEVICE = "cuda"
WEIGHT_DTYPE = torch.bfloat16

# INPUT_VIDEO = "asset/building.mp4" 
# PROMPT = "A building in cartoon style."
INPUT_VIDEO = "asset/inpaint_video.mp4" 
PROMPT = "A cute cat."
NEGATIVE_PROMPT = "The video is not of a high quality, low resolution, watermark, distortion."
SAMPLE_SIZE = [384, 672] 
VIDEO_LENGTH = 49        
FPS = 8
STRENGTH = 0.8
SEED = 4
print(f"🏠 [Edge] Initializing Client (Seed={SEED})...")

print("📦 Loading VAE...")
vae = AutoencoderKLCogVideoX.from_pretrained(MODEL_NAME, subfolder="vae").to(DEVICE).to(WEIGHT_DTYPE)

def encode_tensor(tensor):
    np_array = tensor.cpu().float().numpy().astype(np.float16)
    return base64.b64encode(np_array.tobytes()).decode('utf-8')

def decode_tensor(b64_str, shape):
    bytes_data = base64.b64decode(b64_str)
    np_array = np.frombuffer(bytes_data, dtype=np.float16)
    return torch.from_numpy(np_array.copy()).reshape(shape).to(DEVICE).to(WEIGHT_DTYPE)

def main():
    if not os.path.exists(INPUT_VIDEO):
        print(f"❌ Video not found: {INPUT_VIDEO}")
        return

    # 1. 预处理 (与 Benchmark 对齐)
    print(f"🔄 Preprocessing Video: {INPUT_VIDEO}")
    temporal_compression_ratio = vae.config.temporal_compression_ratio 
    target_video_length = int((VIDEO_LENGTH - 1) // temporal_compression_ratio * temporal_compression_ratio) + 1
    
    # 获取 [0, 1] 像素数据
    input_video, input_video_mask, _, _ = get_video_to_video_latent(
        INPUT_VIDEO, 
        video_length=target_video_length, 
        sample_size=SAMPLE_SIZE, 
        validation_video_mask=None, 
        fps=FPS
    )
    input_video = input_video.to(DEVICE).to(WEIGHT_DTYPE)
    print_stats("Input Video (Raw)", input_video)

    # 【核心修正 1】: 归一化到 [-1, 1]
    # 基准测试显示 Mean 应该是 0.003 左右，必须做这一步
    input_video = 2.0 * input_video - 1.0
    print_stats("Input Video (Norm)", input_video)

    # 2. VAE Encode
    print("🔄 VAE Encoding...")
    t0 = time.time()
    with torch.no_grad():
        # Encode -> Sample -> Scale (无 Shift)
        init_latents = vae.encode(input_video).latent_dist.sample()
        init_latents = init_latents * vae.config.scaling_factor

    print_stats("Encoded Latents", init_latents)
    print(f"⏱️  Encode Time: {time.time()-t0:.4f}s")

    # 3. Upload
    payload = {
        "latents_b64": encode_tensor(init_latents),
        "shape": list(init_latents.shape),
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "strength": STRENGTH,
        "steps": 50,
        "guidance_scale": 6.0,
        "seed": SEED  # 【新增】发送种子
    }

    print("🚀 Sending to Cloud...")
    try:
        t_start = time.time()
        # resp = requests.post(CLOUD_URL, json=payload)
        session = requests.Session()
        session.trust_env = False  # 禁止读取系统代理配置
        resp = session.post(CLOUD_URL, json=payload) # 使用 session.post
        resp.raise_for_status()
        print(f"⏱️  Cloud Time: {time.time()-t_start:.4f}s")
        data = resp.json()
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # 4. Decode
    print("🏠 [Edge] Decoding...")
    result_b64 = data["result_b64"]
    latents_out = decode_tensor(result_b64, init_latents.shape)
    
    with torch.no_grad():
        # 反 Scale
        latents_out = latents_out / vae.config.scaling_factor
        video_out = vae.decode(latents_out).sample

    # 【核心修正 2】: 反归一化 [-1, 1] -> [0, 1]
    # 如果不加这一步，颜色会异常 (Color Leakage)
    video_out = (video_out / 2.0 + 0.5).clamp(0, 1)
    
    # 5. Save
    print("💾 Saving Video...")
    # 定义输出目录
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    original_filename = os.path.basename(INPUT_VIDEO)
    save_path = os.path.abspath(os.path.join(output_dir, original_filename))
    
    # 转 float32 防止 numpy 报错
    video_out_cpu = video_out.to(dtype=torch.float32).cpu()
    save_videos_grid(video_out_cpu, save_path, fps=FPS)
    print(f"✅ Saved to {save_path}")

if __name__ == "__main__":
    main()