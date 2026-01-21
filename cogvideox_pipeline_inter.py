import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import torch
import gc
import time
import numpy as np
from PIL import Image

# 路径处理
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

from Int_DDIMScheduler import Int_DDIMScheduler 

# ================= 1. 参数设置 =================
MODEL_NAME       = "models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP" 
DEVICE           = "cuda"
WEIGHT_DTYPE     = torch.bfloat16 

# 基础推理参数
TEST_CFG_RATIO   = 1.0  
TEST_FPS         = 8
TEST_STEPS       = 30   
STRENGTH         = 0.8  
GUIDANCE_SCALE   = 6.0
INPUT_VIDEO      = "asset/inpaint_video.mp4" 
PROMPT           = "A cute cat."
NEGATIVE_PROMPT  = "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion."
SAMPLE_SIZE      = [384, 672] 
VIDEO_LENGTH     = 49 
SEED             = 43


INTERRUPT_RATIO = 0.5
ACTUAL_STEPS = int(TEST_STEPS * STRENGTH)
INTERRUPT_AT_IDX = int(ACTUAL_STEPS * INTERRUPT_RATIO)
MIN_STEP = 3 

def check_external_interrupt_signal(current_loop_idx):
    if current_loop_idx == INTERRUPT_AT_IDX:
        return True, MIN_STEP 
    return False, 0

def flush():
    gc.collect()
    torch.cuda.empty_cache()

# ================= 2. 边缘侧：VAE 编码 =================
print("🏠 [Edge] Encoding...")
vae = AutoencoderKLCogVideoX.from_pretrained(MODEL_NAME, subfolder="vae").to(WEIGHT_DTYPE).to(DEVICE)
temporal_compression_ratio = vae.config.temporal_compression_ratio 
target_video_length = int((VIDEO_LENGTH - 1) // temporal_compression_ratio * temporal_compression_ratio) + 1
input_video, input_video_mask, _, _ = get_video_to_video_latent(
    INPUT_VIDEO, video_length=target_video_length, sample_size=SAMPLE_SIZE, fps=TEST_FPS
)
input_video = (2.0 * input_video - 1.0).to(DEVICE).to(WEIGHT_DTYPE)
with torch.no_grad():
    init_latents = vae.encode(input_video).latent_dist.sample() * vae.config.scaling_factor
    if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
         init_latents = init_latents - vae.config.shift_factor
latents = init_latents.permute(0, 2, 1, 3, 4)
del vae
flush()

# ================= 3. 云端侧：文本编码 =================
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

# ================= 4. 云端侧：Transformer 推理 =================
print("☁️ [Cloud] Transformer Denoising...")
transformer = CogVideoXTransformer3DModel.from_pretrained(MODEL_NAME, subfolder="transformer", torch_dtype=WEIGHT_DTYPE)
convert_model_weight_to_float8(transformer, exclude_module_name=[], device=DEVICE)
convert_weight_dtype_wrapper(transformer, WEIGHT_DTYPE)
transformer.to(DEVICE)


scheduler = Int_DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
scheduler.set_timesteps(TEST_STEPS, device=DEVICE)

init_timestep_idx = int(TEST_STEPS * STRENGTH)
init_timestep_idx = min(init_timestep_idx, TEST_STEPS - 1)
t_start = scheduler.timesteps[TEST_STEPS - init_timestep_idx] 
timesteps = scheduler.timesteps[TEST_STEPS - init_timestep_idx:].clone()

generator = torch.Generator(device=DEVICE).manual_seed(SEED)
noise = torch.randn(latents.shape, generator=generator, device=DEVICE, dtype=WEIGHT_DTYPE)
latents_curr = scheduler.add_noise(latents, noise, torch.tensor([t_start]*latents.shape[0], device=DEVICE))

mask_base = torch.zeros_like(latents[:, :, :1, :, :]) 
ref_base = torch.zeros_like(latents)
inpaint_cfg = torch.cat([torch.cat([mask_base]*2), torch.cat([ref_base]*2)], dim=2)
inpaint_single = torch.cat([mask_base, ref_base], dim=2)

cfg_stop_idx = int(len(timesteps) * TEST_CFG_RATIO) 
is_fast_mode = False
curr_idx = 0

print(f"🎬 Start Inference. Total Steps: {len(timesteps)}")

while curr_idx < len(timesteps):
    t = timesteps[curr_idx] 
    
    # --- [中断检测与重规划] ---
    if not is_fast_mode:  # 当 t =399进入中断
        interrupted, m_steps = check_external_interrupt_signal(curr_idx)
        if interrupted:
            print(f"\n🚨 [INTERRUPT] at step index {curr_idx} (t={t.item()}).")
            
            # 调用 Scheduler 重规划：根据当前 t 和 剩余 m 计算出新跑道
            # 这里 scheduler 内部会更新 self.timesteps
            # 返回的 new_timesteps 是 [next_t, ..., 0] (不含当前 t)
            new_steps_tensor = scheduler.replan_timesteps(t.item(), m_steps, device=DEVICE)  # new_steps_tensor = [266, 133]
            
            print(f"   -> New Schedule: {[t.item()] + new_steps_tensor.cpu().tolist()}")
            
            # 把当前 t 和剩下的拼起来，作为新的循环列表
            # 这样下一次循环就会取到 new_steps_tensor 的第一个元素
            timesteps = torch.cat([t.unsqueeze(0), new_steps_tensor])  # timesteps = [399, 266, 133]
            
            curr_idx = 0 
            is_fast_mode = True
            
            # t 保持不变，因为我们这轮循环还没跑完
            # 下一轮循环 curr_idx=1，就会取到 new_steps_tensor[0]，实现大跨步

    # CFG 逻辑
    do_cfg = (not is_fast_mode and curr_idx < cfg_stop_idx) and (GUIDANCE_SCALE > 1.0)
    
    # latent_model_input = torch.cat([latents_curr] * 2) if do_cfg else latents_curr
    
    if do_cfg:
        latent_model_input = torch.cat([latents_curr] * 2)
        prompt_in = prompt_embeds_cfg
        current_inpaint = inpaint_cfg
    else:
        latent_model_input = latents_curr
        prompt_in = prompt_embeds_single
        current_inpaint = inpaint_single
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
    
    # current_inpaint = inpaint_cfg if do_cfg else inpaint_single
    current_inpaint = current_inpaint.expand(latent_model_input.shape[0], -1, 17, -1, -1)
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

    # DDIM Step (scheduler 内部会根据 self.timesteps 找到正确的 prev_t)
    latents_curr = scheduler.step(noise_pred, t, latents_curr).prev_sample
    
    label = "FAST" if is_fast_mode else "NORMAL"
    print(f"   [{label}] Step {curr_idx+1}/{len(timesteps)} done (t={t.item()})")
    
    curr_idx += 1

del transformer
flush()

# ================= 5. VAE Decoding =================
print("🏠 [Edge] Decoding...")
vae = AutoencoderKLCogVideoX.from_pretrained(MODEL_NAME, subfolder="vae").to(WEIGHT_DTYPE).to(DEVICE)
latents_out = latents_curr.permute(0, 2, 1, 3, 4)
with torch.no_grad():
    if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
         latents_out = latents_out + vae.config.shift_factor
    video_out = vae.decode(latents_out / vae.config.scaling_factor).sample

os.makedirs("output_debug", exist_ok=True)
save_path = "output_debug/int_ddim_fixed.mp4"
video_out = (video_out / 2.0 + 0.5).clamp(0, 1).cpu().float()
save_videos_grid(video_out, save_path, fps=TEST_FPS)
print(f"✅ Finished! Video saved to: {save_path}")