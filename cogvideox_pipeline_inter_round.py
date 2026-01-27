import os
# 优化显存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import torch
import gc
import time
import numpy as np
import pandas as pd

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

# [核心修改] 导入自定义调度器
from utils1.Int_DDIMScheduler import Int_DDIMScheduler

# 固定基础参数
MODEL_NAME      = "models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP" 
DEVICE          = "cuda"
WEIGHT_DTYPE    = torch.bfloat16 
TEST_FPS        = 8
STRENGTH        = 0.8
GUIDANCE_SCALE  = 6.0
# INPUT_VIDEO     = "asset/inpaint_video.mp4" 
# PROMPT          = "A cute cat."
# NEGATIVE_PROMPT = "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion. "
INPUT_VIDEO     = "../Dataset/video_2s_1_low/scene_021_left-forward.mp4" 
PROMPT          = "A video of streetview in Japanese anime style, Makoto Shinkai aesthetics, vibrant colors, beautiful clouds, 2d animation, high quality, detailed."
NEGATIVE_PROMPT = "photorealistic, 3d, real world, low quality, blurry, distortion."
SAMPLE_SIZE     = [272, 480] # [480, 270] 
# SAMPLE_SIZE     = [384, 672] 
VIDEO_LENGTH    = 49 
SEED            = 43
TEST_CFG_RATIO  = 1.0

# ================= 1. 实验矩阵配置 =================
N_LIST = [20, 25, 30, 35, 40, 45, 50] 
RATIO_LIST = [0.3, 0.4, 0.5, 0.6, 0.66, 0.7] 
M_LIST = [2, 3, 4, 5] 

# N_LIST = [20] 
# RATIO_LIST = [0.7] 
# M_LIST = [3, 4, 5] 

def flush():
    gc.collect()
    torch.cuda.empty_cache()

# ================= 2. 模型初始加载 =================
print("⏳ Loading Models...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(MODEL_NAME, subfolder="text_encoder", torch_dtype=WEIGHT_DTYPE)
transformer = CogVideoXTransformer3DModel.from_pretrained(MODEL_NAME, subfolder="transformer", torch_dtype=WEIGHT_DTYPE)
convert_model_weight_to_float8(transformer, exclude_module_name=[], device="cpu")
convert_weight_dtype_wrapper(transformer, WEIGHT_DTYPE)
vae = AutoencoderKLCogVideoX.from_pretrained(MODEL_NAME, subfolder="vae")

# 这里的 scheduler 只是占位，后面循环里会重新初始化
scheduler_proto = Int_DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

# --- 预处理文本特征 ---
text_encoder.to(DEVICE)
with torch.no_grad():
    text_inputs = tokenizer(PROMPT, padding="max_length", max_length=226, truncation=True, add_special_tokens=True, return_tensors="pt")
    prompt_embeds = text_encoder(text_inputs.input_ids.to(DEVICE))[0]
    neg_inputs = tokenizer(NEGATIVE_PROMPT, padding="max_length", max_length=226, truncation=True, add_special_tokens=True, return_tensors="pt")
    negative_prompt_embeds = text_encoder(neg_inputs.input_ids.to(DEVICE))[0]
    prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds])
    prompt_embeds_single = prompt_embeds
text_encoder.to("cpu")
flush()

# --- 预处理视频帧 ---
temporal_compression_ratio = vae.config.temporal_compression_ratio 
target_video_length = int((VIDEO_LENGTH - 1) // temporal_compression_ratio * temporal_compression_ratio) + 1
input_video_raw, _, _, _ = get_video_to_video_latent(INPUT_VIDEO, video_length=target_video_length, sample_size=SAMPLE_SIZE, fps=TEST_FPS)
input_video_tensor = (2.0 * input_video_raw - 1.0).to(WEIGHT_DTYPE)

print("✅ Ready. Starting Experiments...")
results_log = []
os.makedirs("experiment_results", exist_ok=True)

# ================= 3. 实验循环 =================
for N in N_LIST:
    # 基于 Strength 预先计算该 N 下的实际步数
    ACTUAL_STEPS = int(N * STRENGTH)
    
    for ratio in RATIO_LIST:
        # 基于实际步数计算中断点
        n_interrupt = int(ACTUAL_STEPS * ratio)
        
        for m in M_LIST:
            exp_id = f"N{N}_ratio{ratio}_m{m}"
            print(f"\n🚀 Running Exp: {exp_id} (Actual: {ACTUAL_STEPS}, Interrupt idx: {n_interrupt})")
            
            flush()
            exp_latency = 0
            
            with torch.no_grad():
                # --- A. 边缘侧：VAE Encode ---
                vae.to(DEVICE).to(dtype=WEIGHT_DTYPE)
                v_in = input_video_tensor.to(DEVICE).to(dtype=WEIGHT_DTYPE)
                t_enc_start = time.time()
                init_latents = vae.encode(v_in).latent_dist.sample() * vae.config.scaling_factor
                if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
                    init_latents = init_latents - vae.config.shift_factor
                latents = init_latents.permute(0, 2, 1, 3, 4)
                exp_latency += (time.time() - t_enc_start)
                vae.to("cpu")
                flush()

                # --- B. 云端侧：Denoising ---
                transformer.to(DEVICE)
                
                # [关键] 每次实验重新初始化 Scheduler，防止状态污染
                scheduler = Int_DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
                scheduler.set_timesteps(N, device=DEVICE)
                
                init_t_idx = min(ACTUAL_STEPS, N - 1)
                t_start_val = scheduler.timesteps[N - init_t_idx]
                timesteps = scheduler.timesteps[N - init_t_idx:].clone()
                
                generator = torch.Generator(device=DEVICE).manual_seed(SEED)
                noise = torch.randn(latents.shape, generator=generator, device=DEVICE, dtype=WEIGHT_DTYPE)
                latents_curr = scheduler.add_noise(latents, noise, torch.tensor([t_start_val]*latents.shape[0], device=DEVICE))
                
                mask_base = torch.zeros_like(latents[:, :, :1, :, :]) 
                ref_base = torch.zeros_like(latents)
                inpaint_cfg = torch.cat([torch.cat([mask_base]*2), torch.cat([ref_base]*2)], dim=2)
                inpaint_single = torch.cat([mask_base, ref_base], dim=2)

                t_denoise_start = time.time()
                is_fast_mode = False
                curr_idx = 0
                cfg_stop_idx = int(len(timesteps) * TEST_CFG_RATIO)

                while curr_idx < len(timesteps):
                    t = timesteps[curr_idx]
                    
                    # --- [动态中断逻辑] ---
                    if not is_fast_mode and curr_idx == n_interrupt:
                        print(f"   🚨 [INTERRUPT] Re-planning at step {curr_idx} (t={t.item()}) -> finish in {m} steps.")
                        
                        # 调用 Scheduler 的重规划
                        # 返回的是不包含当前 t 的未来时间步
                        new_steps_tensor = scheduler.replan_timesteps(t.item(), m, device=DEVICE)
                        
                        # 重构循环列表：当前 t + 未来 steps
                        timesteps = torch.cat([t.unsqueeze(0), new_steps_tensor])
                        
                        # 重置索引：下一轮循环 idx=1，正好取到 new_steps_tensor[0]
                        curr_idx = 0
                        is_fast_mode = True
                        
                        # 当前 t 保持不变，继续跑完这一步
                    
                    # CFG 逻辑
                    do_cfg = (not is_fast_mode and curr_idx < cfg_stop_idx) and (GUIDANCE_SCALE > 1.0)
                    
                    latent_model_input = torch.cat([latents_curr] * 2) if do_cfg else latents_curr
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                    
                    prompt_in = prompt_embeds_cfg if do_cfg else prompt_embeds_single
                    inpaint_in = inpaint_cfg if do_cfg else inpaint_single
                    current_inpaint = inpaint_in.expand(latent_model_input.shape[0], -1, 17, -1, -1)
                    
                    # 预测噪声
                    t_tensor = t.expand(latent_model_input.shape[0])
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
                    curr_idx += 1
                
                exp_latency += (time.time() - t_denoise_start)
                transformer.to("cpu")
                flush()

                # --- C. 边缘侧：VAE Decode ---
                vae.to(DEVICE).to(dtype=WEIGHT_DTYPE)
                t_dec_start = time.time()
                latents_out = latents_curr.permute(0, 2, 1, 3, 4)
                if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
                    latents_out = latents_out + vae.config.shift_factor
                video_out = vae.decode(latents_out / vae.config.scaling_factor).sample
                exp_latency += (time.time() - t_dec_start)
                
                # 保存视频结果
                video_save = (video_out / 2.0 + 0.5).clamp(0, 1).cpu().float()
                save_path = f"experiment_results/{exp_id}.mp4"
                save_videos_grid(video_save, save_path, fps=TEST_FPS)
                vae.to("cpu")
                flush()

            results_log.append({"ID": exp_id, "N": N, "Ratio": ratio, "m": m, "Latency(s)": round(exp_latency, 2)})
            print(f"✅ {exp_id} Done. Time: {exp_latency:.2f}s")

# 输出实验报告
df = pd.DataFrame(results_log)
print("\n" + "="*50 + "\n实验汇总报告\n" + "="*50)
print(df.to_string(index=False))
df.to_csv("experiment_results/experiment_report.csv", index=False)