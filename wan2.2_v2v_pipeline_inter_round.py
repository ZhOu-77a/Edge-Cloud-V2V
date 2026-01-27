# wan2.2_v2v_task, use cfg all the time!
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import torch
import gc
import math
import numpy as np
import cv2  
from omegaconf import OmegaConf

# ================= 路径设置 =================
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

# ================= 基础配置 =================
MODEL_NAME          = "models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP"
CONFIG_PATH         = os.path.join(project_dir, "config/wan2.2/wan_civitai_i2v.yaml")
DEVICE              = "cuda"
WEIGHT_DTYPE        = torch.bfloat16
OUTPUT_DIR          = "experiment_results_wan2.2_v2v_inter" # 修改输出目录名以区分

# --- V2V Params ---
INPUT_VIDEO_PATH    = "asset/scene_021_left-forward.mp4" 
PROMPT              = "A video of streetview in Minecraft voxel style, made of cube blocks, low poly, pixelated textures, blocky trees, high quality, detailed." 
NEGATIVE_PROMPT     = "curves, round, high poly, low quality, blurry, distortion."
SAMPLE_SIZE         = [480, 832]
FPS                 = 16
SEED                = 43
GUIDANCE_SCALE      = 6.0
STRENGTH            = 0.5 # 保持固定

# --- 🧪 实验变量列表 (3个维度) ---
STEPS_LIST    = [20,30,40,50]           # 1. 总步数列表
RATIO_LIST    = [0.2, 0.3, 0.4, 0.5, 0.6]    # 2. 中断比例列表
MIN_STEP_LIST = [2,3,4,5]             # 3. 中断后最小剩余步数列表

config = OmegaConf.load(CONFIG_PATH)
temporal_compression_ratio = 4 
spatial_compression_ratio = 8 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def flush():
    gc.collect()
    torch.cuda.empty_cache()

# ================= 1. 公共预处理 (只运行一次) =================
print("🏠 [Pre-Processing] Encoding Input Video & Text (Once)...")

# 1.1 Video Encoding
if not os.path.exists(INPUT_VIDEO_PATH):
    raise FileNotFoundError(f"Video not found: {INPUT_VIDEO_PATH}")

cap_temp = cv2.VideoCapture(INPUT_VIDEO_PATH)
VIDEO_LENGTH_RAW = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
cap_temp.release()
VIDEO_LENGTH = ((VIDEO_LENGTH_RAW - 1) // temporal_compression_ratio) * temporal_compression_ratio + 1

input_video, _, _, _ = get_video_to_video_latent(
    INPUT_VIDEO_PATH, video_length=VIDEO_LENGTH, sample_size=SAMPLE_SIZE, fps=FPS
)
input_video = (2.0 * input_video - 1.0).to(DEVICE).to(WEIGHT_DTYPE)

vae = AutoencoderKLWan.from_pretrained(
    os.path.join(MODEL_NAME, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(DEVICE).to(WEIGHT_DTYPE)

with torch.no_grad():
    init_latents_base = vae.encode(input_video).latent_dist.sample()
print(f"✅ Init Latents Shape: {init_latents_base.shape}")

# 1.2 Text Encoding
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
    context_input_base = [context_neg.cpu(), context_prompt.cpu()] if GUIDANCE_SCALE > 1.0 else [context_prompt.cpu()]

del vae, tokenizer, text_encoder
flush()
print("✅ Pre-processing Done. Encoders offloaded.")

# ================= 2. Model Manager 类定义 =================
class ModelManager:
    def __init__(self):
        self.current_model = None
        self.current_model_type = None 

    def load_model(self, model_type):
        if self.current_model_type == model_type:
            return self.current_model

        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            flush()

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
        return self.current_model

    def unload_current(self):
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            self.current_model_type = None
            flush()

global_model_manager = ModelManager()

# ================= 3. 单次实验函数 =================
def run_experiment(steps_val, interrupt_ratio, min_step_val):
    print(f"\n🧪 >>> Experiment: Steps={steps_val}, Ratio={interrupt_ratio}, MinStep={min_step_val}")
    
    # --- 3.1 动态计算实验参数 ---
    # 实际要跑的步数 (V2V Strength 决定)
    ACTUAL_RUN_STEPS = int(steps_val * STRENGTH)
    # 中断发生的绝对位置 (相对于实际跑的步数)
    # 例如：steps=50, strength=0.5 -> actual=25. ratio=0.5 -> interrupt at index 12
    INTERRUPT_AT_INDEX = int(ACTUAL_RUN_STEPS * interrupt_ratio)
    
    global_state = {
        "current_step_count": 0,
        "is_interrupted": False,
        "interrupt_index": INTERRUPT_AT_INDEX, 
        "min_step": min_step_val
    }

    # --- 3.2 初始化 Scheduler (使用当前实验的 steps_val) ---
    scheduler = Int_FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(Int_FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )
    # 【关键】设置总步数
    scheduler.set_timesteps(steps_val, device=DEVICE)
    timesteps = scheduler.timesteps

    # V2V 截断 (根据 STRENGTH 和 当前步数 steps_val)
    steps_to_skip = int(steps_val * (1 - STRENGTH))
    init_timestep_idx = max(0, min(steps_to_skip, steps_val - 1))
    
    start_timestep = timesteps[init_timestep_idx]
    timesteps = timesteps[init_timestep_idx:] 
    
    # 生成初始 Latents
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    noise = torch.randn(init_latents_base.shape, generator=generator, device=DEVICE, dtype=WEIGHT_DTYPE)
    t_tensor = start_timestep.unsqueeze(0).to(DEVICE)
    latents = scheduler.scale_noise(sample=init_latents_base.clone(), timestep=t_tensor, noise=noise)

    context_input = [t.to(DEVICE) for t in context_input_base]

    # --- 3.3 Transformer 输入准备 ---
    target_latent_frames = init_latents_base.shape[2]
    target_height = init_latents_base.shape[3]
    target_width  = init_latents_base.shape[4]
    
    with torch.no_grad():
        y_input = torch.zeros((1, 20, target_latent_frames, target_height, target_width), device=DEVICE, dtype=WEIGHT_DTYPE)
        y_model_input = torch.cat([y_input] * 2) if GUIDANCE_SCALE > 1.0 else y_input
        patch_size = (1, 2, 2)
        seq_len = math.ceil((target_height * target_width) / (patch_size[1] * patch_size[2]) * target_latent_frames)

    # --- 3.4 内部执行函数 ---
    def check_interrupt_local():
        if not global_state["is_interrupted"] and global_state["current_step_count"] == global_state["interrupt_index"]:
            return True
        return False

    def execute_steps_local(model, steps_list, current_latents):
        if len(steps_list) == 0: return current_latents, False, []
        
        interrupted = False
        new_future_steps = []
        
        idx = 0
        while idx < len(steps_list):
            i, t = steps_list[idx]
            
            if check_interrupt_local():
                print(f"🚨 [INTERRUPT] at step count {global_state['current_step_count']}")
                new_steps_tensor = scheduler.replan_timesteps(t.item(), global_state["min_step"], device=DEVICE)
                print(f"   -> Re-Schedule Plan: {len(new_steps_tensor)} remaining steps.")
                new_future_steps = [(0, nt) for nt in new_steps_tensor]
                
                interrupted = True
                global_state["is_interrupted"] = True
            
            latent_model_input = torch.cat([current_latents] * 2) if GUIDANCE_SCALE > 1.0 else current_latents
            timestep = t.expand(latent_model_input.shape[0])
            
            with torch.no_grad():
                noise_pred = model(x=latent_model_input, context=context_input, t=timestep, seq_len=seq_len, y=y_model_input)
            
            if GUIDANCE_SCALE > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
            
            current_latents = scheduler.step(noise_pred, t, current_latents, return_dict=False)[0]
            
            global_state["current_step_count"] += 1
            
            if interrupted:
                break
            idx += 1
            
        return current_latents, interrupted, new_future_steps

    # --- 3.5 主循环 ---
    boundary = config['transformer_additional_kwargs'].get('boundary', 0.900)
    boundary_val = boundary * 1000 if timesteps[0] > 1.0 else boundary
    current_steps_queue = [(i, t) for i, t in enumerate(timesteps)]

    while len(current_steps_queue) > 0:
        phase1_queue = []
        phase2_queue = []
        for item in current_steps_queue:
            if item[1] >= boundary_val:
                phase1_queue.append(item)
            else:
                phase2_queue.append(item)
        
        if len(phase1_queue) > 0:
            target_phase = "high_noise"
            active_queue = phase1_queue
            next_iteration_queue = phase2_queue 
        else:
            target_phase = "low_noise"
            active_queue = phase2_queue
            next_iteration_queue = []
        
        current_model = global_model_manager.load_model(target_phase)
        latents, is_interrupted, new_steps = execute_steps_local(current_model, active_queue, latents)

        if is_interrupted:
            current_steps_queue = new_steps
        else:
            current_steps_queue = next_iteration_queue

    # --- 3.6 解码 ---
    print("🏠 [Edge] Decoding...")
    vae_local = AutoencoderKLWan.from_pretrained(
        os.path.join(MODEL_NAME, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(DEVICE).to(WEIGHT_DTYPE)

    with torch.no_grad():
        frames = vae_local.decode(latents).sample
        frames = (frames / 2 + 0.5).clamp(0, 1).cpu().float()
    
    del vae_local
    flush()

    # 文件名包含 steps
    save_filename = f"steps_{steps_val}_ratio_{interrupt_ratio}_min_{min_step_val}.mp4"
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    save_videos_grid(frames, save_path, fps=FPS)
    print(f"✅ Saved to: {save_path}")

# ================= 4. 启动三层循环 =================
print(f"🚀 Starting Experiment over: \n Steps={STEPS_LIST} \n Ratios={RATIO_LIST} \n MinSteps={MIN_STEP_LIST}")

total_exps = len(STEPS_LIST) * len(RATIO_LIST) * len(MIN_STEP_LIST)
curr_exp = 0

for s in STEPS_LIST:        # 1. 遍历步数
    for r in RATIO_LIST:    # 2. 遍历比例
        for m in MIN_STEP_LIST: # 3. 遍历最小步数
            curr_exp += 1
            print(f"\n========================================")
            print(f"Experiment {curr_exp}/{total_exps}: Steps={s}, Ratio={r}, MinStep={m}")
            print(f"========================================")
            run_experiment(s, r, m)

global_model_manager.unload_current()
print("\n✨ All Experiments Completed Successfully!")