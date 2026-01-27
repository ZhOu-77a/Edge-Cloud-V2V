import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import torch
import gc
import math
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

# ================= 0. 环境路径设置 =================
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

# 生成参数
PROMPT              = "一只棕色的狗摇着头，坐在舒适房间里的浅色沙发上。在狗的后面，架子上有一幅镶框的画，周围是粉红色的花朵。房间里柔和温暖的灯光营造出舒适的氛围。"
NEGATIVE_PROMPT     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
SAMPLE_SIZE         = [480, 832]
VIDEO_LENGTH        = 17
FPS                 = 16
SEED                = 43
GUIDANCE_SCALE      = 6.0
NUM_INFERENCE_STEPS = 16
# SHIFT               = 5.0

def flush():
    gc.collect()
    torch.cuda.empty_cache()

# 加载配置
config = OmegaConf.load(CONFIG_PATH)

# ================= 2. cloud：文本编码 (Text Encoding) =================
print("☁️ [Cloud] Text Encoding...")

tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(MODEL_NAME, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer'))
)
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(MODEL_NAME, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=WEIGHT_DTYPE,
).to(DEVICE).eval()

def get_prompt_embeds(prompt_str, max_len=512):
    text_inputs = tokenizer(
        [prompt_str],
        padding="max_length",
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(DEVICE)
    prompt_attention_mask = text_inputs.attention_mask.to(DEVICE)  
    seq_len = prompt_attention_mask.gt(0).sum(dim=1).long() 
    embeds = text_encoder(text_input_ids, attention_mask=prompt_attention_mask)[0]
    return embeds[0, :seq_len] # 去掉batch 维度，对应transformer输入为(Sequence_Length, Hidden_Dim)

with torch.no_grad():
    context_prompt = get_prompt_embeds(PROMPT)
    context_neg = get_prompt_embeds(NEGATIVE_PROMPT)
    
    if GUIDANCE_SCALE > 1.0:
        # 存入列表，Transformer 会自动处理 Batch 维度
        context_input = [context_neg.cpu(), context_prompt.cpu()]  # 防反了吗
        context_input = [t.to(DEVICE) for t in context_input]
    else:
        context_input = [context_prompt.to(DEVICE)]

del tokenizer, text_encoder
flush()
print("✅ Text encoded and Encoder offloaded.")

# ================= 3. latent and scheduler prepare =================
print("☁️ [Cloud] Preparing Latents & Scheduler...")
temporal_compression_ratio = 4 
spatial_compression_ratio = 8 
latent_channels = 16 

latents_frames = (VIDEO_LENGTH - 1) // temporal_compression_ratio + 1
target_height = SAMPLE_SIZE[0] // spatial_compression_ratio
target_width  = SAMPLE_SIZE[1] // spatial_compression_ratio

generator = torch.Generator(device=DEVICE).manual_seed(SEED)
latents = torch.randn(
    (1, latent_channels, latents_frames, target_height, target_width),
    generator=generator,
    device=DEVICE,
    dtype=WEIGHT_DTYPE
) # 1是batch_size,与pipeline一致

scheduler = FlowMatchEulerDiscreteScheduler(
    **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)
scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=DEVICE, mu=1) # Flow without shift, initialize timesteps
timesteps = scheduler.timesteps

# ================= 4. cloud：分段 Transformer 推理 =================
print("☁️ [Cloud] Starting Transformer Denoising (Sequential Mode)...")

boundary = config['transformer_additional_kwargs'].get('boundary', 0.900)
boundary_val = boundary * 1000

# 拆分时间步
phase1_steps = [] # High noise
phase2_steps = [] # Low noise

for i, t in enumerate(timesteps):
    if t >= boundary_val:
        phase1_steps.append((i, t))
    else:
        phase2_steps.append((i, t))

print(f"   -> Plan: {len(phase1_steps)} steps with High-Noise Model, {len(phase2_steps)} steps with Low-Noise Model.")


# Wan2.2 Inpaint 模型输入通道为 36 (16 Latent + 20 Condition (4 Mask + 16 Masked Image))
with torch.no_grad():
    y_input = torch.zeros((1, 20, latents_frames, target_height, target_width), device=DEVICE, dtype=WEIGHT_DTYPE)
    y_model_input = torch.cat([y_input] * 2) if GUIDANCE_SCALE > 1.0 else y_input
    # Wan2.2 Transformer Patch Size
    patch_size = (1, 2, 2)
    # seq_len用于 Transformer 内部计算 3D RoPE
    seq_len = math.ceil((target_height * target_width) / (patch_size[1] * patch_size[2]) * latents_frames)

# --- Phase 1: High Noise Model ---
if len(phase1_steps) > 0:
    print("🚀 [Phase 1] Loading High Noise Transformer (Transformer 2)...")
    transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(MODEL_NAME, config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=WEIGHT_DTYPE,
    )
    # "sequential_cpu_offload_and_qfloat8"
    convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=DEVICE)
    convert_weight_dtype_wrapper(transformer_2, WEIGHT_DTYPE)
    transformer_2.freqs = transformer_2.freqs.to(DEVICE) # freqs that RoPE needs
    transformer_2.to(DEVICE).eval() # set to inference mode
    
    print("   -> Running Phase 1 inference...")
    for i, t in phase1_steps:
        latent_model_input = torch.cat([latents] * 2) if GUIDANCE_SCALE > 1.0 else latents  # 后续改cfg_ratio可以在这改
        timestep = t.expand(latent_model_input.shape[0])
        
        with torch.no_grad():
            noise_pred = transformer_2(
                x=latent_model_input,
                context=context_input,
                t=timestep,
                seq_len=seq_len,
                y=y_model_input
            )

        if GUIDANCE_SCALE > 1.0:  # 改cfg ratio
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]  #更新latent

        if (i + 1) % 5 == 0 or i == 0:
            print(f"      Step {i+1}/{len(timesteps)} done.")

    del transformer_2
    flush()
    print("✅ Phase 1 finished & Offloaded.")


# --- Phase 2: Low Noise Model ---
if len(phase2_steps) > 0:
    print("🚀 [Phase 2] Loading Low Noise Transformer (Transformer 1)...")
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(MODEL_NAME, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=WEIGHT_DTYPE,
    )
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=DEVICE)
    convert_weight_dtype_wrapper(transformer, WEIGHT_DTYPE)
    transformer.freqs = transformer.freqs.to(DEVICE)
    transformer.to(DEVICE).eval()

    print("   -> Running Phase 2 inference...")
    for i, t in phase2_steps:
        latent_model_input = torch.cat([latents] * 2) if GUIDANCE_SCALE > 1.0 else latents
        timestep = t.expand(latent_model_input.shape[0])
        
        with torch.no_grad():
            noise_pred = transformer(
                x=latent_model_input,
                context=context_input,
                t=timestep,
                seq_len=seq_len,
                y=y_model_input
            )

        if GUIDANCE_SCALE > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        if (i + 1) % 5 == 0:
            print(f"      Step {i+1}/{len(timesteps)} done.")

    del transformer
    flush()
    print("✅ Phase 2 finished & Offloaded.")


# ================= 5. 边缘侧逻辑：VAE 解码 (Decoding) =================
print("🏠 [Edge] Decoding...")

vae = AutoencoderKLWan.from_pretrained(
    os.path.join(MODEL_NAME, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(DEVICE).to(WEIGHT_DTYPE)

with torch.no_grad():
    frames = vae.decode(latents).sample
    frames = (frames / 2 + 0.5).clamp(0, 1)
    frames = frames.cpu().float()

save_path = "samples/utput_wan2_2_sequential.mp4"
save_videos_grid(frames, save_path, fps=FPS)
print(f"✅ Video saved to: {save_path}")