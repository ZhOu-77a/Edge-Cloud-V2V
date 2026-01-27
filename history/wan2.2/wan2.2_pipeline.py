import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import torch
import gc
import math
import numpy as np
from PIL import Image
import cv2  
from omegaconf import OmegaConf
# 这一版指定了输入视频的长度



current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.models import (AutoencoderKLWan, AutoTokenizer, WanT5EncoderModel, Wan2_2Transformer3DModel)
from videox_fun.utils.fp8_optimization import convert_model_weight_to_float8, convert_weight_dtype_wrapper
from videox_fun.utils.utils import save_videos_grid, filter_kwargs
from diffusers import FlowMatchEulerDiscreteScheduler

# ================= 1. 参数配置 =================
MODEL_NAME          = "models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP"
CONFIG_PATH         = "config/wan2.2/wan_civitai_i2v.yaml"
DEVICE              = "cuda"
WEIGHT_DTYPE        = torch.bfloat16

# --- 新增/修改 V2V 参数 ---
# INPUT_VIDEO_PATH    = "asset/inpaint_video.mp4" 
# PROMPT              = "一只白色的猫" 
# NEGATIVE_PROMPT     = "色调艳丽，畸形，模糊，扭曲的脸，不完整的手"
INPUT_VIDEO_PATH    = "asset/scene_021_left-forward.mp4" 
PROMPT              = "A video of streetview in Minecraft voxel style, made of cube blocks, low poly, pixelated textures, blocky trees, high quality, detailed." 
NEGATIVE_PROMPT     = "curves, round, high poly, low quality, blurry, distortion."
SAMPLE_SIZE         = [480, 832] # 最好和原视频比例一致
VIDEO_LENGTH        = 17         # 要生成的帧数
FPS                 = 16
SEED                = 43
GUIDANCE_SCALE      = 6.0
NUM_INFERENCE_STEPS = 50

STRENGTH            = 0.5

config = OmegaConf.load(CONFIG_PATH)

def flush():
    gc.collect()
    torch.cuda.empty_cache()

# ================= 1.5 预处理：加载视频并用 VAE 编码 =================
# V2V 需要先加载 VAE 来“看”原视频，这比 Text Encoding 更早
print("🏠 [Pre-Processing] Encoding Input Video...")

def load_video_frames(video_path, frames_num, height, width):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < frames_num:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height))
        # 归一化到 [-1, 1] 用于 VAE
        frame = (frame.astype(np.float32) / 127.5) - 1.0 
        frames.append(frame)
    cap.release()
    
    # 如果视频不够长，复制最后一帧（或者你可以做循环）
    while len(frames) < frames_num:
        frames.append(frames[-1])
        
    # [T, H, W, C] -> [C, T, H, W] -> Batch [1, C, T, H, W]
    video_tensor = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0)
    return video_tensor.to(DEVICE).to(WEIGHT_DTYPE)

# 加载 VAE
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(MODEL_NAME, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(DEVICE).to(WEIGHT_DTYPE)

# 编码原视频 -> Latents
with torch.no_grad():
    input_video_tensor = load_video_frames(INPUT_VIDEO_PATH, VIDEO_LENGTH, SAMPLE_SIZE[0], SAMPLE_SIZE[1])
    # VAE 编码
    init_latents = vae.encode(input_video_tensor).latent_dist.sample()
    # Wan VAE 缩放因子通常不需要手动乘，但在 Diffusers 中有时需要 map，这里假设 videox_fun 内部处理了
    # 如果生成的画面偏色或全黑，可能需要检查这里的 scaling factor

print(f"✅ Input video encoded. Latents shape: {init_latents.shape}")

del vae
flush()

# ================= 2. 云端侧逻辑：文本编码 (保持不变) =================
print("☁️ [Cloud] Text Encoding...")
# ... (此处代码与你原来的一致，省略以节省篇幅) ...
# 请把原来的 Text Encoding 代码段完整保留在这里
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

temporal_compression_ratio = 4 
spatial_compression_ratio = 8 


target_frames = init_latents.shape[2]
target_height = init_latents.shape[3]
target_width  = init_latents.shape[4]

scheduler = FlowMatchEulerDiscreteScheduler(
    **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)
scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=DEVICE)
timesteps = scheduler.timesteps

# 【核心 V2V 逻辑】：计算开始步数
# 假设 STRENGTH = 0.6，总步数 50。
# 我们需要跳过前 20 步 (保留 40% 原图信息)，从第 20 步开始去噪 (重绘 60%)。
# FlowMatch 的 timesteps 通常是从 1000 -> 0 或者是 0 -> 1000，取决于具体实现。
# Wan 使用 FlowMatchEuler，通常是 time=1.0 (纯噪) -> time=0.0 (纯图)。

# 找到切入的时间步
start_timestep_index = int((1.0 - STRENGTH) * NUM_INFERENCE_STEPS)
# 确保不越界
start_timestep_index = max(0, min(start_timestep_index, NUM_INFERENCE_STEPS - 1))

start_timestep = timesteps[start_timestep_index]
print(f"⚡ V2V Mode: Strength {STRENGTH}. Starting from step {start_timestep_index+1} (t={start_timestep:.2f})")

# 截取剩余的时间步
timesteps = timesteps[start_timestep_index:]

# 给原图 Latents 加噪
generator = torch.Generator(device=DEVICE).manual_seed(SEED)
noise = torch.randn(init_latents.shape, generator=generator, device=DEVICE, dtype=WEIGHT_DTYPE)

# FlowMatch 的加噪公式通常是线性的： x_t = (1 - t) * x_0 + t * noise (不同 scheduler 实现不同)
# 这里直接使用 scheduler 的 add_noise 方法最稳妥
# 注意：FlowMatchEulerDiscreteScheduler 的 add_noise 可能需要 sigmas，
# 简单起见，我们手动模拟 Flow Matching 的加噪（假设 t 也就是 sigma）：
# x_t = t * x_1 + (1 - t) * x_0  <-- Flow Matching 插值公式
# t 是当前时间步的值 (例如 0.6)
t_val = start_timestep.cpu().item() / 1000.0 # 假设 timestep 是 0-1000
# 如果 timestep 本身就是 0-1 (FlowMatch 常见)，则不需要除 1000
if timesteps[0] <= 1.0: 
    t_val = start_timestep.cpu().item()
else:
    t_val = start_timestep.cpu().item() / 1000.0

# 构造起始 Latents
# t=1 是全噪，t=0 是原图。我们要从 t=STRENGTH 开始。
latents = (1 - t_val) * init_latents + t_val * noise

# ================= 4. Transformer 推理 (保持大部分逻辑) =================
# Condition 依然全 0，因为我们是用 latents 本身带的信息，而不是 Inpainting Mask
with torch.no_grad():
    y_input = torch.zeros(
        (1, 20, target_frames, target_height, target_width), 
        device=DEVICE, dtype=WEIGHT_DTYPE
    )
    y_model_input = torch.cat([y_input] * 2) if GUIDANCE_SCALE > 1.0 else y_input
    patch_size = (1, 2, 2)
    seq_len = math.ceil((target_height * target_width) / (patch_size[1] * patch_size[2]) * target_frames)

# 分段逻辑重算 (因为 timesteps 变短了)
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

# --- Phase 1: High Noise Model ---
if len(phase1_steps) > 0:
    print("🚀 [Phase 1] Loading High Noise Transformer...")
    # ... (加载 Transformer 2 代码同原版) ...
    transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(MODEL_NAME, config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True, torch_dtype=WEIGHT_DTYPE,
    )
    convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=DEVICE)
    convert_weight_dtype_wrapper(transformer_2, WEIGHT_DTYPE)
    transformer_2.freqs = transformer_2.freqs.to(DEVICE)
    transformer_2.to(DEVICE).eval()
    
    for i, t in phase1_steps:
        # 这里的 loop 逻辑保持不变
        latent_model_input = torch.cat([latents] * 2) if GUIDANCE_SCALE > 1.0 else latents
        timestep = t.expand(latent_model_input.shape[0])
        with torch.no_grad():
            noise_pred = transformer_2(x=latent_model_input, context=context_input, t=timestep, seq_len=seq_len, y=y_model_input)
        if GUIDANCE_SCALE > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        print(f"Step {i+1} done.")
    
    del transformer_2
    flush()

# --- Phase 2: Low Noise Model ---
if len(phase2_steps) > 0:
    print("🚀 [Phase 2] Loading Low Noise Transformer...")
    # ... (加载 Transformer 1 代码同原版) ...
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(MODEL_NAME, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True, torch_dtype=WEIGHT_DTYPE,
    )
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=DEVICE)
    convert_weight_dtype_wrapper(transformer, WEIGHT_DTYPE)
    transformer.freqs = transformer.freqs.to(DEVICE)
    transformer.to(DEVICE).eval()

    for i, t in phase2_steps:
        latent_model_input = torch.cat([latents] * 2) if GUIDANCE_SCALE > 1.0 else latents
        timestep = t.expand(latent_model_input.shape[0])
        with torch.no_grad():
            noise_pred = transformer(x=latent_model_input, context=context_input, t=timestep, seq_len=seq_len, y=y_model_input)
        if GUIDANCE_SCALE > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        print(f"Step {i+1} done.")

    del transformer
    flush()

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

save_path = "samples/output_v2v_streetview3.mp4"
save_videos_grid(frames, save_path, fps=FPS)
print(f"✅ Video saved to: {save_path}")