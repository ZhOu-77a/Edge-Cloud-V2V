import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import torch
import requests
import base64
import numpy as np
import time
from PIL import Image

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

# ================= 参数配置 =================
MODEL_NAME = "models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP"
DEVICE = "cuda"
WEIGHT_DTYPE = torch.bfloat16
CLOUD_URL = "http://127.0.0.1:12346/inference"

INPUT_VIDEO = "asset/inpaint_video.mp4" 
PROMPT = "A cute cat."
NEGATIVE_PROMPT = "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion."
SAMPLE_SIZE = [384, 672] 
VIDEO_LENGTH = 49 
SEED = 43

TEST_FPS = 8
TEST_STEPS = 30
STRENGTH = 0.8
GUIDANCE_SCALE = 6.0

print(f"🏠 [Edge] Initializing Client...")

vae = AutoencoderKLCogVideoX.from_pretrained(MODEL_NAME, subfolder="vae").to(DEVICE).to(WEIGHT_DTYPE)

def encode_tensor(tensor):
    np_array = tensor.cpu().float().numpy().astype(np.float16)
    return base64.b64encode(np_array.tobytes()).decode('utf-8')

def decode_tensor(b64_str, shape):
    bytes_data = base64.b64decode(b64_str)
    np_array = np.frombuffer(bytes_data, dtype=np.float16)
    return torch.from_numpy(np_array.copy()).reshape(shape).to(DEVICE).to(WEIGHT_DTYPE)

def main():
    print("🎬 Encoding Video...")
    temporal_compression_ratio = vae.config.temporal_compression_ratio 
    target_video_length = int((VIDEO_LENGTH - 1) // temporal_compression_ratio * temporal_compression_ratio) + 1
        
    input_video_tensor, _, _, _ = get_video_to_video_latent(
        INPUT_VIDEO, 
        video_length=target_video_length, 
        sample_size=SAMPLE_SIZE, 
        fps=TEST_FPS
    )
    
    input_video_tensor = input_video_tensor.to(DEVICE).to(WEIGHT_DTYPE)
    input_video_tensor = 2.0 * input_video_tensor - 1.0

    with torch.no_grad():
        init_latents = vae.encode(input_video_tensor).latent_dist.sample() * vae.config.scaling_factor
        if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
             init_latents = init_latents - vae.config.shift_factor
    
    latents_to_send = init_latents.permute(0, 2, 1, 3, 4)

    # Payload 只有最基础的生成参数
    payload = {
        "latents_b64": encode_tensor(latents_to_send),
        "shape": list(latents_to_send.shape),
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
        
        result_b64 = data["result_b64"]
        latents_out = decode_tensor(result_b64, latents_to_send.shape)
        
        print("🏠 Decoding Latents...")
        latents_to_decode = latents_out.permute(0, 2, 1, 3, 4)
        
        with torch.no_grad():
            if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
                 latents_to_decode = latents_to_decode + vae.config.shift_factor
            video_out = vae.decode(latents_to_decode / vae.config.scaling_factor).sample

        output_dir = "output_split"
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        save_filename = os.path.join(output_dir, f"cog_edge_dynamic_signal.mp4")
        
        video_out = (video_out / 2.0 + 0.5).clamp(0, 1).cpu().float()
        save_videos_grid(video_out, save_filename, fps=TEST_FPS)
        print(f"🎉 Video Saved: {save_filename}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()