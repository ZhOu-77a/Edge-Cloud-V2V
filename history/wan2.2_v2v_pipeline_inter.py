# wan2.2_v2v_task
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import torch
import gc
import math
import numpy as np
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
from videox_fun.utils.utils import get_video_to_video_latent, save_videos_grid, filter_kwargs
from utils1.Int_FlowDPMSolverMultistepScheduler import Int_FlowMatchEulerDiscreteScheduler

# ================= 参数配置 =================
MODEL_NAME          = "models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP"
# 确保使用绝对路径
CONFIG_PATH         = os.path.join(project_dir, "config/wan2.2/wan_civitai_i2v.yaml")
DEVICE              = "cuda"
WEIGHT_DTYPE        = torch.bfloat16

# --- V2V Params ---
INPUT_VIDEO_PATH    = "asset/scene_021_left-forward.mp4" 
PROMPT              = "A video of streetview in Minecraft voxel style, made of cube blocks, low poly, pixelated textures, blocky trees, high quality, detailed." 
NEGATIVE_PROMPT     = "curves, round, high poly, low quality, blurry, distortion."
SAMPLE_SIZE         = [480, 832]
FPS                 = 16
SEED                = 43
GUIDANCE_SCALE      = 6.0
TEST_STEPS          = 50
STRENGTH            = 0.5

# --- 中断参数 ---
INTERRUPT_RATIO     = 0.5
MIN_STEP            = 3    
ACTUAL_TOTAL_STEPS  = int(TEST_STEPS * STRENGTH) 
INTERRUPT_AT_INDEX  = int(ACTUAL_TOTAL_STEPS * INTERRUPT_RATIO)

config = OmegaConf.load(CONFIG_PATH)
temporal_compression_ratio = 4 
spatial_compression_ratio = 8 

# 全局状态控制
global_state = {
    "current_step_count": 0,  # 当前已跑步数计数器
    "is_interrupted": False,  # 是否已发生中断
}

def check_interrupt():
    """检测是否触发中断"""
    if not global_state["is_interrupted"] and global_state["current_step_count"] == INTERRUPT_AT_INDEX:
        print("Interrupt at :",INTERRUPT_AT_INDEX)
        return True
    return False

def flush():
    gc.collect()
    torch.cuda.empty_cache()

# ================= 1. Preprocess & VAE Encode =================
print("🏠 [Pre-Processing] Encoding Input Video...")
if not os.path.exists(INPUT_VIDEO_PATH):
    raise FileNotFoundError(f"Video not found: {INPUT_VIDEO_PATH}")

cap_temp = cv2.VideoCapture(INPUT_VIDEO_PATH)
VIDEO_LENGTH_RAW = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
cap_temp.release()

VIDEO_LENGTH = ((VIDEO_LENGTH_RAW - 1) // temporal_compression_ratio) * temporal_compression_ratio + 1
print(f"🎬 Detected {VIDEO_LENGTH_RAW} frames. Using aligned VIDEO_LENGTH: {VIDEO_LENGTH}")

# 加载视频并归一化
input_video, _, _, _ = get_video_to_video_latent(
    INPUT_VIDEO_PATH, video_length=VIDEO_LENGTH, sample_size=SAMPLE_SIZE, fps=FPS
)
input_video = (2.0 * input_video - 1.0).to(DEVICE).to(WEIGHT_DTYPE)

vae = AutoencoderKLWan.from_pretrained(
    os.path.join(MODEL_NAME, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(DEVICE).to(WEIGHT_DTYPE)

with torch.no_grad():
    init_latents = vae.encode(input_video).latent_dist.sample()
print(f"✅ Input video encoded. Latents shape: {init_latents.shape}")

del vae
flush()

# ================= 2. Text Encode =================
print("☁️ [Cloud] Text Encoding...")
tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_NAME, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')))
text_encoder = WanT5EncoderModel.from_pretrained(os.path.join(MODEL_NAME, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')), additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']), low_cpu_mem_usage=True, torch_dtype=WEIGHT_DTYPE).to(DEVICE).eval()

def get_prompt_embeds(prompt_str, max_len=512):
    text_inputs = tokenizer([prompt_str], padding="max_length", max_length=max_len, truncation=True, add_special_tokens=True, return_tensors="pt")
    embeds = text_encoder(text_inputs.input_ids.to(DEVICE), attention_mask=text_inputs.attention_mask.to(DEVICE))[0]
    seq_len = text_inputs.attention_mask.to(DEVICE).gt(0).sum(dim=1).long()[0]
    return embeds[0, :seq_len]

with torch.no_grad():
    context_prompt = get_prompt_embeds(PROMPT)
    context_neg = get_prompt_embeds(NEGATIVE_PROMPT)
    context_input = [context_neg.cpu(), context_prompt.cpu()] if GUIDANCE_SCALE > 1.0 else [context_prompt.cpu()]
    context_input = [t.to(DEVICE) for t in context_input]

del tokenizer, text_encoder
flush()

# ================= 3. Scheduler & Latents Prep =================
print("☁️ [Cloud] Preparing Latents for V2V...")

target_latent_frames = init_latents.shape[2]
target_height = init_latents.shape[3]
target_width  = init_latents.shape[4]

# 使用自定义调度器
scheduler = Int_FlowMatchEulerDiscreteScheduler(
    **filter_kwargs(Int_FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)
scheduler.set_timesteps(TEST_STEPS, device=DEVICE)
timesteps = scheduler.timesteps

# V2V Setup
steps_to_skip = int(TEST_STEPS * (1 - STRENGTH))
init_timestep_idx = max(0, min(steps_to_skip, TEST_STEPS - 1))

start_timestep = timesteps[init_timestep_idx]
timesteps = timesteps[init_timestep_idx:] 
print(f"⚡ V2V Mode: Strength {STRENGTH}. Steps: {len(timesteps)}. Start t={start_timestep:.2f}")

generator = torch.Generator(device=DEVICE).manual_seed(SEED)
noise = torch.randn(init_latents.shape, generator=generator, device=DEVICE, dtype=WEIGHT_DTYPE)

# Add Noise, using FlowMatching forward process
t_tensor = start_timestep.unsqueeze(0).to(DEVICE)
latents = scheduler.scale_noise(sample=init_latents, timestep=t_tensor, noise=noise)

# ================= 4. Transformer Logic (Model Manager) =================

with torch.no_grad():
    y_input = torch.zeros((1, 20, target_latent_frames, target_height, target_width), device=DEVICE, dtype=WEIGHT_DTYPE)
    y_model_input = torch.cat([y_input] * 2) if GUIDANCE_SCALE > 1.0 else y_input
    patch_size = (1, 2, 2)
    seq_len = math.ceil((target_height * target_width) / (patch_size[1] * patch_size[2]) * target_latent_frames)

class ModelManager:
    """管理模型的加载和卸载"""
    def __init__(self):
        self.current_model = None
        self.current_model_type = None # "high_noise" or "low_noise"

    def load_model(self, model_type):
        if self.current_model_type == model_type:
            print(f"⚡ [Optimization] '{model_type}' model is already loaded. Skipping load.")
            return self.current_model

        if self.current_model is not None:
            print(f"♻️ [Memory] Unloading '{self.current_model_type}' model...")
            del self.current_model
            self.current_model = None
            flush()

        print(f"🚀 [Loader] Loading '{model_type}' Transformer...")
        
        if model_type == "high_noise":
            subpath_key = 'transformer_high_noise_model_subpath'
        else:
            subpath_key = 'transformer_low_noise_model_subpath'

        subpath = config['transformer_additional_kwargs'].get(subpath_key, 'transformer')
        
        model = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(MODEL_NAME, subpath),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True, 
            torch_dtype=WEIGHT_DTYPE,
        )
        
        convert_model_weight_to_float8(model, exclude_module_name=["modulation",], device=DEVICE)
        convert_weight_dtype_wrapper(model, WEIGHT_DTYPE)
        model.freqs = model.freqs.to(DEVICE)
        model.to(DEVICE).eval()

        self.current_model = model
        self.current_model_type = model_type
        print(f"✅ [Loader] '{model_type}' loaded successfully.")
        
        return self.current_model

    def unload_current(self):
        if self.current_model is not None:
            print("🧹 [Final] Unloading remaining model...")
            del self.current_model
            self.current_model = None
            self.current_model_type = None
            flush()

def execute_steps(model, steps_list, current_latents):
    """纯推理函数，执行传入的 steps_list"""
    if len(steps_list) == 0:
        return current_latents, False, []

    print(f"   -> Executing {len(steps_list)} steps...")
    
    interrupted = False
    new_future_steps = []
    
    idx = 0
    while idx < len(steps_list):
        i, t = steps_list[idx]
        
        # --- 中断检测 ---
        if check_interrupt():
            print(f"\n🚨 [INTERRUPT] at global count {global_state['current_step_count']} (t={t.item():.4f}).")
            
            # Scheduler 重规划
            new_steps_tensor = scheduler.replan_timesteps(t.item(), MIN_STEP, device=DEVICE)
            print(f"   -> Re-Schedule Plan: {[t.item()] + new_steps_tensor.cpu().tolist()}")
            
            # 转换格式为 [(0, t), (0, t)...] 供主循环使用
            # 这里的 0 是假索引，不再重要，因为我们只关心 t
            new_future_steps = [(0, nt) for nt in new_steps_tensor]
            
            interrupted = True
            global_state["is_interrupted"] = True
            
            # 即使中断，当前步 t 仍需跑完
        
        # --- 推理 ---
        latent_model_input = torch.cat([current_latents] * 2) if GUIDANCE_SCALE > 1.0 else current_latents
        timestep = t.expand(latent_model_input.shape[0])
        
        with torch.no_grad():
            noise_pred = model(x=latent_model_input, context=context_input, t=timestep, seq_len=seq_len, y=y_model_input)
        
        if GUIDANCE_SCALE > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
        
        current_latents = scheduler.step(noise_pred, t, current_latents, return_dict=False)[0]
        
        global_state["current_step_count"] += 1
        print(f"      Step {global_state['current_step_count']} done (t={t.item():.4f}).")

        if interrupted:
            print("   -> Interrupted. Switching Plan.")
            break
            
        idx += 1

    return current_latents, interrupted, new_future_steps

# --- 主执行逻辑 ---
model_manager = ModelManager()

# 初始规划
boundary = config['transformer_additional_kwargs'].get('boundary', 0.900)
boundary_val = boundary * 1000 if timesteps[0] > 1.0 else boundary

current_steps_queue = [(i, t) for i, t in enumerate(timesteps)]

while len(current_steps_queue) > 0:
    # 1. 动态拆分
    phase1_queue = []
    phase2_queue = []
    for item in current_steps_queue:
        if item[1] >= boundary_val:
            phase1_queue.append(item)
        else:
            phase2_queue.append(item)
    
    print(f"\n📋 Plan Update: Phase 1 (High): {len(phase1_queue)}, Phase 2 (Low): {len(phase2_queue)}")
    
    # 2. 决策与加载
    if len(phase1_queue) > 0:
        target_phase = "high_noise"
        active_queue = phase1_queue
        next_iteration_queue = phase2_queue 
    else:
        target_phase = "low_noise"
        active_queue = phase2_queue
        next_iteration_queue = []
    current_model = None 
    flush()
    current_model = model_manager.load_model(target_phase) # 若模型没变，很快能完成

    # 3. 执行
    latents, is_interrupted, new_steps = execute_steps(current_model, active_queue, latents)

    # 4. 处理结果
    if is_interrupted:
        print("🔄 [Main Loop] Updating queue with Re-planned steps...")
        current_steps_queue = new_steps
    else:
        print(f"✅ [Main Loop] {target_phase} phase finished.")
        current_steps_queue = next_iteration_queue

print("✨ All Inference Steps Completed.")
model_manager.unload_current()

# ================= 5. Decode =================
print("🏠 [Edge] Decoding...")
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(MODEL_NAME, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(DEVICE).to(WEIGHT_DTYPE)

with torch.no_grad():
    frames = vae.decode(latents).sample
    frames = (frames / 2 + 0.5).clamp(0, 1).cpu().float()

save_path = "samples/inter/3.mp4"
save_videos_grid(frames, save_path, fps=FPS)
print(f"✅ Video saved to: {save_path}")