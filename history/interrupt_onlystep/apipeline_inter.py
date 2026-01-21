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
from diffusers import DDIMScheduler # 继承然后改，step（）要改，根据传的带宽去动态决定剩下m步跑完

# ================= 1. 参数设置 =================
sampler_name     = "DDIM_Origin"
MODEL_NAME       = "models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP" 
DEVICE           = "cuda"
WEIGHT_DTYPE     = torch.bfloat16 

# 基础推理参数
TEST_CFG_RATIO   = 1.0  
TEST_FPS         = 8
TEST_STEPS       = 30   # 原始总步数 N
STRENGTH         = 0.8
GUIDANCE_SCALE   = 6.0
INPUT_VIDEO      = "asset/inpaint_video.mp4" 
PROMPT           = "A cute cat."
NEGATIVE_PROMPT  = "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion. "
SAMPLE_SIZE      = [384, 672] 
VIDEO_LENGTH     = 49 
SEED             = 43

INTERRUPT_RATIO = 0.5
ACTUAL_STEPS = int(TEST_STEPS * STRENGTH)
INTERRUPT_AT_STEP = int(ACTUAL_STEPS * INTERRUPT_RATIO)
MIN_STEP = 3 # 剩下的用3步跑完

def check_external_interrupt_signal(current_step):
    """
    模拟车联网带宽检测逻辑。
    实际应用中，这里应替换为读取传感器、网络状态或共享内存信号的逻辑。
    """
    # 模拟在第 15 步 (n=15) 时触发中断
    if current_step == INTERRUPT_AT_STEP:
        return True, MIN_STEP # 返回 (是否中断, 剩余目标步数m)
    return False, 0

def flush():
    gc.collect()
    torch.cuda.empty_cache()

# ================= 2. 边缘侧逻辑：VAE 编码 =================
print("🏠 [Edge] Encoding...")
vae = AutoencoderKLCogVideoX.from_pretrained(MODEL_NAME, subfolder="vae").to(WEIGHT_DTYPE).to(DEVICE)

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

# ================= 4. 云端侧逻辑：Transformer 推理 (包含中断模块) =================
print("☁️ [Cloud] Transformer Denoising...")
transformer = CogVideoXTransformer3DModel.from_pretrained(MODEL_NAME, subfolder="transformer", torch_dtype=WEIGHT_DTYPE)
convert_model_weight_to_float8(transformer, exclude_module_name=[], device=DEVICE)
convert_weight_dtype_wrapper(transformer, WEIGHT_DTYPE)
transformer.to(DEVICE)

scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
scheduler.set_timesteps(TEST_STEPS, device=DEVICE) # set one-time time_step

init_timestep_idx = int(TEST_STEPS * STRENGTH) # init_timestep_idx = 30*0.8 = 24
init_timestep_idx = min(init_timestep_idx, TEST_STEPS - 1)
t_start = scheduler.timesteps[TEST_STEPS - init_timestep_idx]   # t_start = 799
# timesteps = tensor([799, 766, 732, 699, 666, 632, 599, 566, 532, 499, 466, 432, 399, 366, 332, 299, 266, 232, 199, 166, 132,  99,  66,  32], device='cuda:0')
timesteps = scheduler.timesteps[TEST_STEPS - init_timestep_idx:].clone()

generator = torch.Generator(device=DEVICE).manual_seed(SEED)
noise = torch.randn(latents.shape, generator=generator, device=DEVICE, dtype=WEIGHT_DTYPE)
latents_noisy = scheduler.add_noise(latents, noise, torch.tensor([t_start]*latents.shape[0], device=DEVICE))

# Mask 准备
mask_base = torch.zeros_like(latents[:, :, :1, :, :]) 
ref_base = torch.zeros_like(latents)
inpaint_cfg = torch.cat([torch.cat([mask_base]*2), torch.cat([ref_base]*2)], dim=2)
inpaint_single = torch.cat([mask_base, ref_base], dim=2)

latents_curr = latents_noisy
cfg_stop_idx = int(len(timesteps) * TEST_CFG_RATIO) 

# --- [关键逻辑]：动态推理循环 ---
is_fast_mode = False  # 当中断开始后默认进入非cfg
curr_idx = 0

while curr_idx < len(timesteps):
    t = timesteps[curr_idx] 
    
    # ------------------ [动态中断检测模块] ------------------
    if not is_fast_mode:  # 中断后设置了is_fast_mode = True则将不在进入下面的中断设置初始化模块
        interrupted, m_steps = check_external_interrupt_signal(curr_idx)
        if interrupted:
            print(f"\n🚨 [INTERRUPT] Bandwidth drop! Re-scheduling {len(timesteps)-curr_idx} steps into {m_steps} steps.")
            
            # 时间步重映射：在当前索引到结束之间，均匀选取 m 个时间步
            if len(timesteps) - curr_idx > m_steps:
                # indices = np.linspace(curr_idx, len(timesteps) - 1, m_steps, dtype=int)  # 剩下的步数里面均匀的选 m steps, 获取新的indices
                indices = np.linspace(curr_idx-1, len(timesteps) - 1, m_steps+1, dtype=int)[1:]
                timesteps = timesteps[indices]   # 得到新的timesteps(对应DDPM的)
                curr_idx = 0 # 重置索引以开始跑新序列
                is_fast_mode = True
                t = timesteps[curr_idx]
            else:
                print("ℹ️ Remaining steps fewer than target m, continuing...")
    # ------------------------------------------------------

    # 决定是否使用 CFG (进入快速模式后可考虑关闭 CFG 以进一步提速)
    do_cfg = (not is_fast_mode and curr_idx < cfg_stop_idx) and (GUIDANCE_SCALE > 1.0)
    
    if do_cfg:
        latent_model_input = torch.cat([latents_curr] * 2)
        prompt_in = prompt_embeds_cfg
        inpaint_in = inpaint_cfg
    else:
        latent_model_input = latents_curr
        prompt_in = prompt_embeds_single
        inpaint_in = inpaint_single

    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
    
    # current_inpaint shape = torch.Size([2, 11, 17, 48, 84])  17在这应该是channel
    current_inpaint = inpaint_in.expand(latent_model_input.shape[0], -1, 17, -1, -1)
    
    t_tensor = t.expand(latent_model_input.shape[0]) # t_tensor = tensor([799, 799], device='cuda:0') # 当执行cfg的时候

    # 这一步就是预测噪声了，如果要改的话，从这里就得把对应的step给改了，这个的输出就是模型认为应该被减去的噪声量
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

    # DDIM Step  # 搞清楚他这里跟前面的noise_pred = transformer什么关系？是这个noise_pred = transformer就已经根据t生成了对应的噪声，还是
    # 跟这里的.step有关系，根据t和预测的噪声，以及prev去生成新的？到底哪里可以改变跨步的长度，就是要让他对应的一步去更多的噪声，看代码
    latents_curr = scheduler.step(noise_pred, t, latents_curr).prev_sample # 
    
    label = "FAST" if is_fast_mode else "NORMAL"
    print(f"   [{label}] Step {curr_idx+1}/{len(timesteps)} done (t={t.item()})")
    
    curr_idx += 1

del transformer
flush()

# ================= 5. 边缘侧逻辑：VAE 解码 =================
print("🏠 [Edge] Decoding...")
vae = AutoencoderKLCogVideoX.from_pretrained(MODEL_NAME, subfolder="vae").to(WEIGHT_DTYPE).to(DEVICE)

latents_out = latents_curr.permute(0, 2, 1, 3, 4)
with torch.no_grad():
    if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
         latents_out = latents_out + vae.config.shift_factor
    video_out = vae.decode(latents_out / vae.config.scaling_factor).sample

output_dir = "output_debug"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, f"dynamic_interrupt_res1.mp4")

video_out = (video_out / 2.0 + 0.5).clamp(0, 1).cpu().float()
save_videos_grid(video_out, save_path, fps=TEST_FPS)
print(f"✅ Finished! Video saved to: {save_path}")