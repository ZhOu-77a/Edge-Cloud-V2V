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
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.models import (AutoencoderKLWan, AutoTokenizer, WanT5EncoderModel, Wan2_2Transformer3DModel)
from videox_fun.utils.fp8_optimization import convert_model_weight_to_float8, convert_weight_dtype_wrapper
from videox_fun.utils.utils import get_video_to_video_latent,save_videos_grid, filter_kwargs
from diffusers import FlowMatchEulerDiscreteScheduler

# ================= 参数配置 =================
MODEL_NAME          = "models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP"
CONFIG_PATH         = "config/wan2.2/wan_civitai_i2v.yaml"
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
NUM_INFERENCE_STEPS = 50
STRENGTH            = 0.6

config = OmegaConf.load(CONFIG_PATH)

def flush():
    gc.collect()
    torch.cuda.empty_cache()

# ================= 1 Edge: Preprocess load video and Vae encode=================
print("🏠 [Pre-Processing] Encoding Input Video...")

# --- 【新增逻辑】动态计算视频帧数 ---
if not os.path.exists(INPUT_VIDEO_PATH):
    raise FileNotFoundError(f"Video not found: {INPUT_VIDEO_PATH}")

cap_temp = cv2.VideoCapture(INPUT_VIDEO_PATH)
total_frames_raw = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
cap_temp.release()

# 计算符合 Wan2.2 VAE 要求的帧数 (frames - 1) % 4 == 0
# 公式： ((总帧数 - 1) // 4) * 4 + 1
VIDEO_LENGTH = ((total_frames_raw - 1) // 4) * 4 + 1

# 防止视频过短导致计算出 1 或者更小
if VIDEO_LENGTH < 5:
    VIDEO_LENGTH = 1 if total_frames_raw == 1 else 5

print(f"🎬 Detected {total_frames_raw} frames. Using aligned VIDEO_LENGTH: {VIDEO_LENGTH}")
# ------------------------------------

def load_video_frames(video_path, frames_num, height, width):
    cap = cv2.VideoCapture(video_path)
    frames = []
    # 这里 frames_num 已经是我们计算好的动态长度了
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
    
    # 如果视频不够长（例如计算出17帧，但实际读取只有16帧），复制最后一帧补齐
    if len(frames) > 0:
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
    # 传入动态计算好的 VIDEO_LENGTH
    input_video_tensor = load_video_frames(INPUT_VIDEO_PATH, VIDEO_LENGTH, SAMPLE_SIZE[0], SAMPLE_SIZE[1])
    # VAE 编码
    init_latents = vae.encode(input_video_tensor).latent_dist.sample()

print(f"✅ Input video encoded. Latents shape: {init_latents.shape}")

del vae
flush()

# ================= 2. 云端侧逻辑：文本编码 (保持不变) =================
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
# init_timestep_idx = int((1.0 - STRENGTH) * TEST_STEPS) # 搞错了
init_timestep_idx = int(NUM_INFERENCE_STEPS * STRENGTH)
start_timestep_index = max(0, min(init_timestep_idx, NUM_INFERENCE_STEPS - 1))

start_timestep = timesteps[start_timestep_index]
print(f"⚡ V2V Mode: Strength {STRENGTH}. Starting from step {start_timestep_index+1} (t={start_timestep:.2f})")

timesteps = timesteps[start_timestep_index:]

generator = torch.Generator(device=DEVICE).manual_seed(SEED)
noise = torch.randn(init_latents.shape, generator=generator, device=DEVICE, dtype=WEIGHT_DTYPE)

if timesteps[0] <= 1.0: 
    t_val = start_timestep.cpu().item()
else:
    t_val = start_timestep.cpu().item() / 1000.0

latents = (1 - t_val) * init_latents + t_val * noise

# ================= 4. Transformer 推理 (保持大部分逻辑) =================
with torch.no_grad():
    y_input = torch.zeros(
        (1, 20, target_frames, target_height, target_width), 
        device=DEVICE, dtype=WEIGHT_DTYPE
    )
    y_model_input = torch.cat([y_input] * 2) if GUIDANCE_SCALE > 1.0 else y_input
    patch_size = (1, 2, 2)
    seq_len = math.ceil((target_height * target_width) / (patch_size[1] * patch_size[2]) * target_frames)

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

save_path = "samples/output_v2v_streetview4.mp4"
save_videos_grid(frames, save_path, fps=FPS)
print(f"✅ Video saved to: {save_path}")