import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import torch
import gc
import time
import numpy as np
from PIL import Image

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.models import CogVideoXTransformer3DModel, T5EncoderModel, T5Tokenizer, AutoencoderKLCogVideoX
from videox_fun.utils.fp8_optimization import convert_model_weight_to_float8, convert_weight_dtype_wrapper
from videox_fun.utils.utils import get_video_to_video_latent, save_videos_grid
from diffusers import DDIMScheduler

# ================= 1. 原始参数 =================
sampler_name     = "DDIM_Origin"
MODEL_NAME       = "models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP" 
DEVICE           = "cuda"
WEIGHT_DTYPE     = torch.bfloat16 
TEST_CFG_RATIO   = 1.0  
TEST_FPS         = 8
TEST_STEPS       = 50
STRENGTH         = 0.8
GUIDANCE_SCALE   = 6.0
INPUT_VIDEO      = "asset/inpaint_video.mp4" 
PROMPT           = "A cute cat."
NEGATIVE_PROMPT  = "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion. "
SAMPLE_SIZE      = [384, 672] 
VIDEO_LENGTH     = 49 
SEED             = 43



def flush():
    gc.collect()
    torch.cuda.empty_cache()

# ================= 2. 边缘侧逻辑：VAE 编码 =================
print("🏠 [Edge] Encoding...")
vae = AutoencoderKLCogVideoX.from_pretrained(MODEL_NAME, subfolder="vae").to(WEIGHT_DTYPE).to(DEVICE)
# vae.enable_tiling()

temporal_compression_ratio = vae.config.temporal_compression_ratio 
target_video_length = int((VIDEO_LENGTH - 1) // temporal_compression_ratio * temporal_compression_ratio) + 1

input_video, input_video_mask, _, _ = get_video_to_video_latent(
    INPUT_VIDEO, video_length=target_video_length, sample_size=SAMPLE_SIZE, fps=TEST_FPS
)
input_video = (2.0 * input_video - 1.0).to(DEVICE).to(WEIGHT_DTYPE)

with torch.no_grad():
    init_latents = vae.encode(input_video).latent_dist.sample()
    init_latents = init_latents * vae.config.scaling_factor
    if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
         init_latents = init_latents - vae.config.shift_factor

# 转换维度 (B C T H W -> B T C H W)
latents = init_latents.permute(0, 2, 1, 3, 4)
del vae
flush()

# ================= 3. 云端侧逻辑：文本编码 =================
print("☁️ [Cloud] Text Encoding...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(MODEL_NAME, subfolder="text_encoder", torch_dtype=WEIGHT_DTYPE).to(DEVICE)

with torch.no_grad():
    text_inputs = tokenizer(PROMPT, padding="max_length", max_length=226, truncation=True, add_special_tokens=True, return_tensors="pt")
    prompt_embeds = text_encoder(text_inputs.input_ids.to(DEVICE))[0]
    neg_inputs = tokenizer(NEGATIVE_PROMPT, padding="max_length", max_length=226, truncation=True, add_special_tokens=True, return_tensors="pt")
    negative_prompt_embeds = text_encoder(neg_inputs.input_ids.to(DEVICE))[0]
    
    prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds])
    prompt_embeds_single = prompt_embeds

del text_encoder
flush()

# ================= 4. 云端侧逻辑：Transformer 推理 =================
print("☁️ [Cloud] Transformer Denoising...")
transformer = CogVideoXTransformer3DModel.from_pretrained(MODEL_NAME, subfolder="transformer", torch_dtype=WEIGHT_DTYPE)
convert_model_weight_to_float8(transformer, exclude_module_name=[], device=DEVICE)
convert_weight_dtype_wrapper(transformer, WEIGHT_DTYPE)
transformer.to(DEVICE)

scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
scheduler.set_timesteps(TEST_STEPS, device=DEVICE)

init_timestep_idx = int(TEST_STEPS * STRENGTH)
init_timestep_idx = min(init_timestep_idx, TEST_STEPS - 1)
t_start = scheduler.timesteps[TEST_STEPS - init_timestep_idx]
timesteps = scheduler.timesteps[TEST_STEPS - init_timestep_idx:]

generator = torch.Generator(device=DEVICE).manual_seed(SEED)
noise = torch.randn(latents.shape, generator=generator, device=DEVICE, dtype=WEIGHT_DTYPE)
latents_noisy = scheduler.add_noise(latents, noise, torch.tensor([t_start]*latents.shape[0], device=DEVICE))

# Mask 逻辑
mask_base = torch.zeros_like(latents[:, :, :1, :, :]) 
ref_base = torch.zeros_like(latents)
inpaint_cfg = torch.cat([torch.cat([mask_base]*2), torch.cat([ref_base]*2)], dim=2)
inpaint_single = torch.cat([mask_base, ref_base], dim=2)

latents_curr = latents_noisy
cfg_stop_idx = int(len(timesteps) * TEST_CFG_RATIO) 

for i, t in enumerate(timesteps):
    
    do_cfg = (i < cfg_stop_idx) and (GUIDANCE_SCALE > 1.0)
    
    if do_cfg:
        latent_model_input = torch.cat([latents_curr] * 2)
        prompt_in = prompt_embeds_cfg
        inpaint_in = inpaint_cfg
    else:
        latent_model_input = latents_curr
        prompt_in = prompt_embeds_single
        inpaint_in = inpaint_single

    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
    
    # 动态匹配 3D 维度
    current_inpaint = inpaint_in.expand(latent_model_input.shape[0], -1, 17, -1, -1)
    t_tensor = t.expand(latent_model_input.shape[0])

    with torch.no_grad():
        noise_pred = transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_in,
            timestep=t_tensor,
            inpaint_latents=current_inpaint
        ).sample

    if do_cfg:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)

    latents_curr = scheduler.step(noise_pred, t, latents_curr).prev_sample
    
    if (i + 1) % 5 == 0:
        print(f"   Step {i+1}/{len(timesteps)} done.")

del transformer
flush()

# ================= 5. 边缘侧逻辑：VAE 解码 =================
print("🏠 [Edge] Decoding...")
vae = AutoencoderKLCogVideoX.from_pretrained(MODEL_NAME, subfolder="vae").to(WEIGHT_DTYPE).to(DEVICE)
# vae.enable_tiling()

latents_out = latents_curr.permute(0, 2, 1, 3, 4) # 还原为 B C T H W
with torch.no_grad():
    if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
         latents_out = latents_out + vae.config.shift_factor
    video_out = vae.decode(latents_out / vae.config.scaling_factor).sample

# 保存处理
output_dir = "output_debug"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, f"debug_cfg_{TEST_CFG_RATIO}.mp4")

video_out = (video_out / 2.0 + 0.5).clamp(0, 1).cpu().float()
save_videos_grid(video_out, save_path, fps=TEST_FPS)
print(f"✅ Video saved to: {save_path}")