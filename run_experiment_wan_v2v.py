import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import torch
import gc
import math
import json
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from omegaconf import OmegaConf
import lpips
from transformers import CLIPProcessor, CLIPModel

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

# ================= 1. 全局配置 =================
MODEL_NAME          = "models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP"
CONFIG_PATH         = "config/wan2.2/wan_civitai_i2v.yaml"
PROMPTS_JSON_PATH   = "/home/zhoujh/Edge-Cloud-diffusion/MyCogVideo-v2v/prompts_config.json"
INPUT_VIDEO_FOLDER  = "/home/zhoujh/Edge-Cloud-diffusion/Dataset/video_2s_22_low_wan" 
OUTPUT_ROOT         = "experiment_results_wan2.2_v2v"
CSV_PATH            = os.path.join(OUTPUT_ROOT, "experiment_metrics_v2v.csv")
DEVICE              = "cuda"
WEIGHT_DTYPE        = torch.bfloat16

# 生成参数
SAMPLE_SIZE         = [480, 832] # 输出分辨率，建议与输入比例一致
FPS                 = 16
SEED                = 43
GUIDANCE_SCALE      = 6.0
V2V_STRENGTH        = 0.6  # 0.5-0.8 之间，越大变化越大

config = OmegaConf.load(CONFIG_PATH)

def flush():
    gc.collect()
    torch.cuda.empty_cache()

# ================= 2. 指标计算类 (保持不变) =================
class MetricsCalculator:
    def __init__(self, device):
        self.device = device
        print("📊 Loading Metrics Models (LPIPS & CLIP)...")
        self.lpips_loss = lpips.LPIPS(net='alex').to(device)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def compute_clip_score(self, frames_tensor, prompt_text):
        # frames_tensor: [T, C, H, W]
        T = frames_tensor.shape[0]
        indices = [0, T//2, T-1]
        selected_frames = frames_tensor[indices]
        pil_images = [Image.fromarray((f.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) for f in selected_frames]
        inputs = self.clip_processor(text=[prompt_text], images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            score = outputs.logits_per_image.mean().item()
        return score

    def compute_lpips(self, current_frames, target_frames):
        curr = current_frames.to(self.device) * 2.0 - 1.0
        tgt = target_frames.to(self.device) * 2.0 - 1.0
        with torch.no_grad():
            dist = self.lpips_loss(curr, tgt)
            avg_dist = dist.mean().item()
        return avg_dist

# ================= 3. 辅助函数：读取视频 =================
def load_video_frames_tensor(video_path, height, width):
    """
    读取视频，计算合适帧数，返回 Tensor [1, C, T, H, W] 和 实际帧数
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算符合 Wan2.2 要求的帧数: (F-1)%4 == 0
    target_len = ((total_frames_raw - 1) // 4) * 4 + 1
    if target_len < 5: target_len = 1 if total_frames_raw == 1 else 5
    
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset
    while len(frames) < target_len:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height))
        frame = (frame.astype(np.float32) / 127.5) - 1.0 
        frames.append(frame)
    cap.release()

    # 补帧
    if len(frames) > 0:
        while len(frames) < target_len:
            frames.append(frames[-1])
            
    video_tensor = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0)
    return video_tensor.to(DEVICE).to(WEIGHT_DTYPE), target_len

# ================= 4. 核心 V2V 生成函数 =================
def generate_one_v2v(video_path, prompt, negative_prompt, steps, save_path):
    print(f"\n🎬 V2V Generating: Steps={steps} | Video={os.path.basename(video_path)} | Prompt={prompt[:20]}...")

    # --- 1. Load Video & Encode VAE (Pre-processing) ---
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(MODEL_NAME, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(DEVICE).to(WEIGHT_DTYPE)

    with torch.no_grad():
        input_video_tensor, video_len = load_video_frames_tensor(video_path, SAMPLE_SIZE[0], SAMPLE_SIZE[1])
        init_latents = vae.encode(input_video_tensor).latent_dist.sample() # [1, 16, T_lat, H_lat, W_lat]
    
    del vae, input_video_tensor
    flush()

    # --- 2. Text Encoding ---
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_NAME, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')))
    text_encoder = WanT5EncoderModel.from_pretrained(os.path.join(MODEL_NAME, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')), additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']), low_cpu_mem_usage=True, torch_dtype=WEIGHT_DTYPE).to(DEVICE).eval()
    
    def get_prompt_embeds(prompt_str):
        text_inputs = tokenizer([prompt_str], padding="max_length", max_length=512, truncation=True, add_special_tokens=True, return_tensors="pt")
        embeds = text_encoder(text_inputs.input_ids.to(DEVICE), attention_mask=text_inputs.attention_mask.to(DEVICE))[0]
        seq_len = text_inputs.attention_mask.gt(0).sum(dim=1).long()[0]
        return embeds[0, :seq_len]

    with torch.no_grad():
        context_prompt = get_prompt_embeds(prompt)
        context_neg = get_prompt_embeds(negative_prompt)
        context_input = [context_neg.cpu(), context_prompt.cpu()] if GUIDANCE_SCALE > 1.0 else [context_prompt.cpu()]
        context_input = [t.to(DEVICE) for t in context_input]

    del tokenizer, text_encoder
    flush()

    # --- 3. Latents & Scheduler (V2V Logic) ---
    target_frames = init_latents.shape[2]
    target_height = init_latents.shape[3]
    target_width  = init_latents.shape[4]

    scheduler = FlowMatchEulerDiscreteScheduler(**filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs'])))
    scheduler.set_timesteps(steps, device=DEVICE)
    timesteps = scheduler.timesteps

    # V2V: Calculate start index based on Strength
    start_timestep_index = int((1.0 - V2V_STRENGTH) * steps)
    start_timestep_index = max(0, min(start_timestep_index, steps - 1))
    start_timestep = timesteps[start_timestep_index]
    
    # Truncate timesteps
    actual_timesteps = timesteps[start_timestep_index:]

    # Add Noise
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    noise = torch.randn(init_latents.shape, generator=generator, device=DEVICE, dtype=WEIGHT_DTYPE)
    
    if timesteps[0] <= 1.0:
        t_val = start_timestep.cpu().item()
    else:
        t_val = start_timestep.cpu().item() / 1000.0
        
    latents = (1 - t_val) * init_latents + t_val * noise

    # --- 4. Split Steps ---
    boundary = config['transformer_additional_kwargs'].get('boundary', 0.900)
    boundary_val = boundary * 1000 if timesteps[0] > 1.0 else boundary
    
    phase1_steps = [(i, t) for i, t in enumerate(actual_timesteps) if t >= boundary_val]
    phase2_steps = [(i, t) for i, t in enumerate(actual_timesteps) if t < boundary_val]

    with torch.no_grad():
        y_input = torch.zeros((1, 20, target_frames, target_height, target_width), device=DEVICE, dtype=WEIGHT_DTYPE)
        y_model_input = torch.cat([y_input] * 2) if GUIDANCE_SCALE > 1.0 else y_input
        seq_len_trans = math.ceil((target_height * target_width) / (2 * 2) * target_frames)

    # --- 5. Transformer Inference ---
    # Phase 1
    if len(phase1_steps) > 0:
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
                noise_pred = transformer_2(x=latent_model_input, context=context_input, t=timestep, seq_len=seq_len_trans, y=y_model_input)
            if GUIDANCE_SCALE > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        del transformer_2
        flush()

    # Phase 2
    if len(phase2_steps) > 0:
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
                noise_pred = transformer(x=latent_model_input, context=context_input, t=timestep, seq_len=seq_len_trans, y=y_model_input)
            if GUIDANCE_SCALE > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        del transformer
        flush()

    # --- 6. Decode ---
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(MODEL_NAME, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(DEVICE).to(WEIGHT_DTYPE)

    with torch.no_grad():
        frames = vae.decode(latents).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        frames = frames.cpu().float() # [B, C, T, H, W]

    del vae
    flush()

    # Save
    save_videos_grid(frames, save_path, fps=FPS)
    
    # Return [T, C, H, W] for metrics
    return frames[0].permute(1, 0, 2, 3)

# ================= 5. 主程序 =================
if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        df = pd.DataFrame(columns=["video_name", "prompt_id", "prompt_name", "step", "strength", "clip_score", "lpips_score_vs_step50"])
        df.to_csv(CSV_PATH, index=False)
    
    # 1. 获取所有视频文件
    video_files = [f for f in os.listdir(INPUT_VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]
    video_files.sort()
    
    # 2. 获取所有 Prompt
    with open(PROMPTS_JSON_PATH, 'r') as f:
        prompts_data = json.load(f)

    # 3. 初始化 Metrics
    metrics_calc = MetricsCalculator(DEVICE)

    print(f"📂 Found {len(video_files)} videos and {len(prompts_data)} prompts.")

    # 4. 双重循环：视频 -> Prompt -> Step
    for vid_file in video_files:
        vid_path = os.path.join(INPUT_VIDEO_FOLDER, vid_file)
        vid_name_clean = os.path.splitext(vid_file)[0]
        
        for p_item in prompts_data:
            p_id = p_item['id']
            p_name = p_item['name']
            prompt = p_item['prompt']
            neg_prompt = p_item['negative_prompt']
            
            # 创建文件夹: 视频名_PromptID_PromptName
            safe_pname = "".join([c if c.isalnum() else "_" for c in p_name])
            folder_name = f"{vid_name_clean}_{p_id}_{safe_pname}"
            save_dir = os.path.join(OUTPUT_ROOT, folder_name)
            os.makedirs(save_dir, exist_ok=True)
            
            print(f"\n{'='*30}\nProcessing: Video={vid_file} | Prompt={p_name}\n{'='*30}")
            
            step_frames_cache = {} 
            
            # 遍历 Steps
            for step in range(1, 51):
                video_filename = f"step_{step:03d}.mp4"
                save_path = os.path.join(save_dir, video_filename)
                
                # A. Run V2V
                try:
                    frames = generate_one_v2v(vid_path, prompt, neg_prompt, step, save_path)
                    step_frames_cache[step] = frames
                    
                    # B. Clip Score
                    clip_s = metrics_calc.compute_clip_score(frames, prompt)
                    print(f" -> CLIP: {clip_s:.4f}")
                except Exception as e:
                    print(f"❌ Error at step {step}: {e}")
                    continue

            # C. LPIPS (Comparison with Step 50)
            if 50 in step_frames_cache:
                reference_frames = step_frames_cache[50]
                results_list = []
                
                for step in range(1, 51):
                    if step not in step_frames_cache: continue
                    
                    frames = step_frames_cache[step]
                    
                    if step == 50:
                        lpips_s = 0.0
                    else:
                        lpips_s = metrics_calc.compute_lpips(frames, reference_frames)
                    
                    # 重新计算/获取 CLIP (简单起见，这里假设上面打印了，这里存入CSV时需要数值，所以建议上面存起来，或者这里再调一次)
                    # 为了效率，我们直接复用刚刚算好的 CLIP，这里略作简化，实际写CSV时调用 Metrics
                    clip_s = metrics_calc.compute_clip_score(frames, prompt)

                    results_list.append({
                        "video_name": vid_file,
                        "prompt_id": p_id,
                        "prompt_name": p_name,
                        "step": step,
                        "strength": V2V_STRENGTH,
                        "clip_score": clip_s,
                        "lpips_score_vs_step50": lpips_s
                    })
                    print(f"Saved Metrics: Step {step} | LPIPS={lpips_s:.4f}")

                # Save CSV
                batch_df = pd.DataFrame(results_list)
                batch_df.to_csv(CSV_PATH, mode='a', header=False, index=False)
                print("💾 CSV Updated.")
            
            # Cleanup
            del step_frames_cache
            gc.collect()

    print("\n🎉 All V2V experiments finished!")