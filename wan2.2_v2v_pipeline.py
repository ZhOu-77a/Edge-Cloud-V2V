# wan2.2_v2v_task, without interruption. Use cfg all the time!
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import torch
import gc
import math
import numpy as np
from PIL import Image
import cv2  
from omegaconf import OmegaConf

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
]
project_dir = os.path.dirname(os.path.abspath(__file__))
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.models import (AutoencoderKLWan, AutoTokenizer, WanT5EncoderModel, Wan2_2Transformer3DModel)
from videox_fun.utils.fp8_optimization import convert_model_weight_to_float8, convert_weight_dtype_wrapper
from videox_fun.utils.utils import get_video_to_video_latent,save_videos_grid, filter_kwargs
from diffusers import FlowMatchEulerDiscreteScheduler

# ================= 参数配置 =================
MODEL_NAME          = "models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP"
# CONFIG_PATH         = "config/wan2.2/wan_civitai_i2v.yaml"
CONFIG_PATH = os.path.join(project_dir, "config/wan2.2/wan_civitai_i2v.yaml")
DEVICE              = "cuda"
WEIGHT_DTYPE        = torch.bfloat16

# --- add/modify V2V para. ---
INPUT_VIDEO_PATH    = "asset/scene_021_left-forward.mp4" 
PROMPT              = "A video of streetview in Minecraft voxel style, made of cube blocks, low poly, pixelated textures, blocky trees, high quality, detailed." 
NEGATIVE_PROMPT     = "curves, round, high poly, low quality, blurry, distortion."
SAMPLE_SIZE         = [480, 832]
FPS                 = 16
SEED                = 43
GUIDANCE_SCALE      = 6.0
TEST_STEPS = 50
STRENGTH            = 0.5

config = OmegaConf.load(CONFIG_PATH)
temporal_compression_ratio = 4 
spatial_compression_ratio = 8 

def flush():
    gc.collect()
    torch.cuda.empty_cache()

# ================= 1 Edge: Preprocess load video and Vae encode=================
print("🏠 [Pre-Processing] Encoding Input Video...")

if not os.path.exists(INPUT_VIDEO_PATH):
    raise FileNotFoundError(f"Video not found: {INPUT_VIDEO_PATH}")

# get video frames length
cap_temp = cv2.VideoCapture(INPUT_VIDEO_PATH)
VIDEO_LENGTH_RAW = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
cap_temp.release()

VIDEO_LENGTH = ((VIDEO_LENGTH_RAW - 1) // temporal_compression_ratio) * temporal_compression_ratio + 1
print(f"🎬 Detected {VIDEO_LENGTH_RAW} frames. Using aligned VIDEO_LENGTH: {VIDEO_LENGTH}")
# ------------------------------------

# get_video_to_video_latent 返回的是 [1, 3, F, H, W]，数值范围 [0, 1]
input_video, _, _, _ = get_video_to_video_latent(
    INPUT_VIDEO_PATH, 
    video_length=VIDEO_LENGTH, 
    sample_size=SAMPLE_SIZE, 
    fps=FPS
)
input_video = (2.0 * input_video - 1.0).to(DEVICE).to(WEIGHT_DTYPE)

# load VAE
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(MODEL_NAME, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(DEVICE).to(WEIGHT_DTYPE)

with torch.no_grad():
    # VAE 编码
    init_latents = vae.encode(input_video).latent_dist.sample()
print(f"✅ Input video encoded. Latents shape: {init_latents.shape}")

del vae
flush()

# ================= 2. cloud：text encode =================
print("☁️ [Cloud] Text Encoding...")
tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_NAME, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')))
text_encoder = WanT5EncoderModel.from_pretrained(os.path.join(MODEL_NAME, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')), additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']), low_cpu_mem_usage=True, torch_dtype=WEIGHT_DTYPE).to(DEVICE).eval()
def get_prompt_embeds(prompt_str, max_len=512):
    text_inputs = tokenizer([prompt_str], padding="max_length", max_length=max_len, truncation=True, add_special_tokens=True, return_tensors="pt")
    text_input_ids = text_inputs.input_ids.to(DEVICE)
    prompt_attention_mask = text_inputs.attention_mask.to(DEVICE)
    embeds = text_encoder(text_input_ids, attention_mask=prompt_attention_mask)[0]
    seq_len = prompt_attention_mask.gt(0).sum(dim=1).long()[0]
    return embeds[0, :seq_len]
with torch.no_grad():
    context_prompt = get_prompt_embeds(PROMPT)
    context_neg = get_prompt_embeds(NEGATIVE_PROMPT)
    if GUIDANCE_SCALE > 1.0:
        context_input = [context_neg.cpu(), context_prompt.cpu()]
        context_input = [t.to(DEVICE) for t in context_input]
    else:
        context_input = [context_prompt.to(DEVICE)]
del tokenizer, text_encoder
flush()

print("☁️ [Cloud] Preparing Latents for V2V...")

target_latent_frames = init_latents.shape[2]
target_height = init_latents.shape[3]
target_width  = init_latents.shape[4]

scheduler = FlowMatchEulerDiscreteScheduler(
    **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)
scheduler.set_timesteps(TEST_STEPS, device=DEVICE)
timesteps = scheduler.timesteps

# 【核心 V2V 逻辑】：计算开始步数
init_timestep_idx = int((1.0 - STRENGTH) * TEST_STEPS) 
init_timestep_idx = max(0, min(init_timestep_idx, TEST_STEPS - 1))

start_timestep = timesteps[init_timestep_idx] # 给定50步,strength=0.6,实际跑30steps,start = 762.3315
timesteps = timesteps[init_timestep_idx:]
print(f"⚡ V2V Mode: Strength {STRENGTH}. Starting from step {init_timestep_idx+1} (t={start_timestep:.2f})")

generator = torch.Generator(device=DEVICE).manual_seed(SEED)
noise = torch.randn(init_latents.shape, generator=generator, device=DEVICE, dtype=WEIGHT_DTYPE)

# if timesteps[0] <= 1.0: 
#     t_val = start_timestep.cpu().item() # ddim
# else:
#     t_val = start_timestep.cpu().item() / 1000.0  # FlowMatch, t_val = 0.7623314819335938
# latents = (1 - t_val) * init_latents + t_val * noise  # flow matching走的是直线，他的正向过程就是这样的
t_tensor = start_timestep.unsqueeze(0).to(DEVICE)
latents = scheduler.scale_noise(sample=init_latents, timestep=t_tensor, noise=noise) # forward: add noise

# ================= 4. Transformer inference =================
with torch.no_grad():
    y_input = torch.zeros(
        (1, 20, target_latent_frames, target_height, target_width), 
        device=DEVICE, dtype=WEIGHT_DTYPE
    )
    y_model_input = torch.cat([y_input] * 2) if GUIDANCE_SCALE > 1.0 else y_input
    patch_size = (1, 2, 2)
    seq_len = math.ceil((target_height * target_width) / (patch_size[1] * patch_size[2]) * target_latent_frames)

def run_denoising_phase(phase_name, model_subpath_key, steps, current_latents):
    """
    通用去噪阶段处理函数：加载模型 -> FP8量化 -> 循环推理 -> 卸载模型
    """
    if len(steps) == 0:
        return current_latents

    print(f"🚀 [{phase_name}] Loading Transformer...")
    model = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(MODEL_NAME, config['transformer_additional_kwargs'].get(model_subpath_key, 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True, torch_dtype=WEIGHT_DTYPE,
    )
    
    # FP8 优化
    convert_model_weight_to_float8(model, exclude_module_name=["modulation",], device=DEVICE)
    convert_weight_dtype_wrapper(model, WEIGHT_DTYPE)
    model.freqs = model.freqs.to(DEVICE)
    model.to(DEVICE).eval()
    
    print(f"   -> Running inference for {len(steps)} steps...")
    for i, t in steps:
        latent_model_input = torch.cat([current_latents] * 2) if GUIDANCE_SCALE > 1.0 else current_latents
        timestep = t.expand(latent_model_input.shape[0])
        
        with torch.no_grad():
            noise_pred = model(x=latent_model_input, context=context_input, t=timestep, seq_len=seq_len, y=y_model_input)
        
        if GUIDANCE_SCALE > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
        
        current_latents = scheduler.step(noise_pred, t, current_latents, return_dict=False)[0]
        
        # 打印总体进度 (需要计算 offset)
        step_idx = init_timestep_idx + i + 1 if phase_name == "Phase 1" else init_timestep_idx + len(phase1_steps) + i + 1
        if step_idx % 5 == 0 or i == 0:
             print(f"      Step {step_idx}/{TEST_STEPS} done.")

    del model
    flush()
    print(f"✅ [{phase_name}] Finished & Offloaded.")
    return current_latents

phase1_steps = []
phase2_steps = []
boundary = config['transformer_additional_kwargs'].get('boundary', 0.900)
boundary_val = boundary * 1000 if timesteps[0] > 1.0 else boundary

for i, t in enumerate(timesteps):
    if t >= boundary_val:
        phase1_steps.append((i, t))
    else:
        phase2_steps.append((i, t))
print(f" -> Plan: {len(phase1_steps)} High-Noise steps, {len(phase2_steps)} Low-Noise steps.")

# Phase 1: High Noise
if len(phase1_steps) > 0:
    latents = run_denoising_phase("Phase 1", "transformer_high_noise_model_subpath", phase1_steps, latents)

# Phase 2: Low Noise
if len(phase2_steps) > 0:
    latents = run_denoising_phase("Phase 2", "transformer_low_noise_model_subpath", phase2_steps, latents)

# ================= 5. 解码 (重新加载 VAE) =================
print("🏠 [Edge] Decoding...")
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(MODEL_NAME, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(DEVICE).to(WEIGHT_DTYPE)

with torch.no_grad():
    frames = vae.decode(latents).sample
    frames = (frames / 2 + 0.5).clamp(0, 1)
    frames = frames.cpu().float()

save_path = "samples/output_v2v_streetview5.mp4"
save_videos_grid(frames, save_path, fps=FPS)
print(f"✅ Video saved to: {save_path}")