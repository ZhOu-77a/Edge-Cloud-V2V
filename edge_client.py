import os
import sys
import torch
import requests
import base64
import numpy as np
import time
import imageio.v3 as iio
from PIL import Image

# === 环境 ===
current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),                                       
    os.path.dirname(os.path.dirname(current_file_path)),                      
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))      
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.models import AutoencoderKLCogVideoX
from videox_fun.utils.utils import get_video_to_video_latent, save_videos_grid

# === 调试函数 ===
def print_debug(msg):
    print(f"🕵️ [DEBUG] {msg}")

# === 核心配置 ===
TEST_CFG_RATIO = 1 # 先测 1.0
TEST_FPS = 6
TEST_STEPS = 50

INPUT_VIDEO = "asset/inpaint_video.mp4" 
PROMPT = "A cute cat."
STRENGTH = 0.8
NEGATIVE_PROMPT = "The video is not of a high quality, low resolution, watermark, distortion."
SAMPLE_SIZE = [384, 672] 
VIDEO_LENGTH = 49 # 这里的设置仅作为上限参考
SEED = 43

CLOUD_URL = "http://127.0.0.1:12345/inference"
MODEL_NAME = "models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP"
DEVICE = "cuda"
WEIGHT_DTYPE = torch.bfloat16

if not os.path.exists(MODEL_NAME):
    ABS_MODEL_PATH = "/home/zhoujh/Edge-Cloud-diffusion/CogVideoX-Fun/" + MODEL_NAME
    if os.path.exists(ABS_MODEL_PATH): MODEL_NAME = ABS_MODEL_PATH

print(f"🏠 [Edge] Client Initializing (CFG={TEST_CFG_RATIO}, Strength={STRENGTH})...")

try:
    vae = AutoencoderKLCogVideoX.from_pretrained(MODEL_NAME, subfolder="vae").to(DEVICE).to(WEIGHT_DTYPE)
except Exception as e:
    print(f"❌ VAE Error: {e}")
    sys.exit(1)

def encode_tensor(tensor):
    np_array = tensor.cpu().float().numpy().astype(np.float16)
    return base64.b64encode(np_array.tobytes()).decode('utf-8')

def decode_tensor(b64_str, shape):
    bytes_data = base64.b64decode(b64_str)
    np_array = np.frombuffer(bytes_data, dtype=np.float16)
    return torch.from_numpy(np_array.copy()).reshape(shape).to(DEVICE).to(WEIGHT_DTYPE)

def main():
    # 1. 读取
    temporal_compression_ratio = vae.config.temporal_compression_ratio 
    target_video_length = int((VIDEO_LENGTH - 1) // temporal_compression_ratio * temporal_compression_ratio) + 1
    
    print_debug(f"Target Max Frames: {target_video_length}")
    
    input_video, input_video_mask, _, _ = get_video_to_video_latent(
        INPUT_VIDEO, 
        video_length=target_video_length, 
        sample_size=SAMPLE_SIZE, 
        validation_video_mask=None, 
        fps=TEST_FPS
    )
    
    real_frames = input_video.shape[2]
    print_debug(f"Actual Frames Read: {real_frames}")
    
    input_video = input_video.to(DEVICE).to(WEIGHT_DTYPE)
    input_video = 2.0 * input_video - 1.0

    print(f"🚀 Starting Pipeline...")
    t_start_total = time.time()

    # 2. Encode
    with torch.no_grad():
        init_latents = vae.encode(input_video).latent_dist.sample() * vae.config.scaling_factor
        if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
             init_latents = init_latents - vae.config.shift_factor
    
    # 3. Upload
    payload = {
        "latents_b64": encode_tensor(init_latents),
        "shape": list(init_latents.shape),
        "prompt": PROMPT, "negative_prompt": NEGATIVE_PROMPT,
        "strength": STRENGTH, "steps": TEST_STEPS,
        "guidance_scale": 6.0, "seed": SEED,
        "cfg_ratio": TEST_CFG_RATIO 
    }

    try:
        session = requests.Session(); session.trust_env = False 
        t_req_start = time.time()
        resp = session.post(CLOUD_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        print(f"⏱️  Cloud Time: {data.get('process_time', 0):.4f}s")
    except Exception as e:
        print(f"❌ Cloud Error: {e}")
        return

    # 4. Decode
    result_b64 = data["result_b64"]
    latents_out = decode_tensor(result_b64, init_latents.shape)
    
    with torch.no_grad():
        if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
             latents_out = latents_out + vae.config.shift_factor
        video_out = vae.decode(latents_out / vae.config.scaling_factor).sample
    
    total_latency = time.time() - t_start_total
    print(f"⏱️  Total Latency: {total_latency:.4f}s")

    # 5. Save
    output_dir = "output_debug"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    save_filename = os.path.join(output_dir, f"output_cfg_{TEST_CFG_RATIO}.mp4")
    
    video_out = (video_out / 2.0 + 0.5).clamp(0, 1)
    video_out = video_out.to(dtype=torch.float32).cpu()
    save_videos_grid(video_out, save_filename, fps=TEST_FPS)
    print(f"✅ Video saved to {save_filename}")

if __name__ == "__main__":
    main()