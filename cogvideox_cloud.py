import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import torch
import base64
import uvicorn
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from videox_fun.models import CogVideoXTransformer3DModel, T5EncoderModel, T5Tokenizer
from videox_fun.utils.fp8_optimization import convert_model_weight_to_float8, convert_weight_dtype_wrapper
from utils1.Int_DDIMScheduler import Int_DDIMScheduler 

# ================= 配置 =================
MODEL_NAME = "models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP" 
DEVICE = "cuda"
WEIGHT_DTYPE = torch.bfloat16 
PORT = 12346

# --- 云端策略配置 ---
# 1. 静态策略：云端决定前 70% 步数用 CFG 
CLOUD_CFG_RATIO = 0.7 
# 2. 动态策略参数
CLOUD_MIN_STEP = 3

app = FastAPI()

print(f"☁️ [Cloud] Initializing Server on {DEVICE}...")

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(MODEL_NAME, subfolder="text_encoder", torch_dtype=WEIGHT_DTYPE).to(DEVICE)
transformer = CogVideoXTransformer3DModel.from_pretrained(MODEL_NAME, subfolder="transformer", torch_dtype=WEIGHT_DTYPE)

convert_model_weight_to_float8(transformer, exclude_module_name=[], device=DEVICE)
convert_weight_dtype_wrapper(transformer, WEIGHT_DTYPE)
transformer.to(DEVICE)

scheduler_base = Int_DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

print("✅ Server Models Loaded!")

def decode_tensor(b64_str, shape):
    bytes_data = base64.b64decode(b64_str)
    np_array = np.frombuffer(bytes_data, dtype=np.float16)
    tensor = torch.from_numpy(np_array.copy()).reshape(shape)
    return tensor.to(DEVICE).to(WEIGHT_DTYPE)

def encode_tensor(tensor):
    np_array = tensor.cpu().float().numpy().astype(np.float16)
    return base64.b64encode(np_array.tobytes()).decode('utf-8')

# --- 【模拟】实时带宽检测信号 ---
def check_realtime_bandwidth(current_step, total_steps):
    """
    实际场景：这里会读取 Redis/Shared Memory 或 监控 API 检查当前带宽。
    模拟逻辑：假设跑到 50% 进度时，模拟接收到了“带宽拥塞”信号。
    """
    # 模拟：在第 15 步左右触发中断 (假设总步数30)
    SIMULATE_BAD_NETWORK_AT_RATIO = 0.5
    
    if current_step == int(total_steps * SIMULATE_BAD_NETWORK_AT_RATIO):
        return True 
    return False

class FunRequest(BaseModel):
    latents_b64: str
    shape: list
    prompt: str
    negative_prompt: str
    steps: int = 50
    strength: float = 0.8
    guidance_scale: float = 6.0
    seed: int = 43
    # Edge 端只需要传基础参数，不需要传任何策略参数

@app.get("/health")
async def health_check():
    return {"status": "ready"}

@app.post("/inference")
async def inference(req: FunRequest):
    t_start = time.time()
    try:
        print(f"\n🔍 [Request] Seed={req.seed} | Strength={req.strength}")

        # 1. 还原 Latents
        latents = decode_tensor(req.latents_b64, req.shape)
        
        # 2. 文本编码
        with torch.no_grad():
            text_inputs = tokenizer(req.prompt, padding="max_length", max_length=226, truncation=True, add_special_tokens=True, return_tensors="pt")
            prompt_embeds = text_encoder(text_inputs.input_ids.to(DEVICE))[0]
            neg_inputs = tokenizer(req.negative_prompt, padding="max_length", max_length=226, truncation=True, add_special_tokens=True, return_tensors="pt")
            negative_prompt_embeds = text_encoder(neg_inputs.input_ids.to(DEVICE))[0]
            
            prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds])
            prompt_embeds_single = prompt_embeds

        # 3. Scheduler & V2V
        import copy
        scheduler = copy.deepcopy(scheduler_base)
        scheduler.set_timesteps(req.steps, device=DEVICE)

        init_timestep_idx = int(req.steps * req.strength)
        init_timestep_idx = min(init_timestep_idx, req.steps - 1)
        t_start_val = scheduler.timesteps[req.steps - init_timestep_idx]
        timesteps = scheduler.timesteps[req.steps - init_timestep_idx:].clone()

        generator = torch.Generator(device=DEVICE).manual_seed(req.seed)
        noise = torch.randn(latents.shape, generator=generator, device=DEVICE, dtype=WEIGHT_DTYPE)
        latents_curr = scheduler.add_noise(latents, noise, torch.tensor([t_start_val]*latents.shape[0], device=DEVICE))

        # 4. Condition
        mask_base = torch.zeros_like(latents[:, :, :1, :, :]) 
        ref_base = torch.zeros_like(latents)
        inpaint_cfg = torch.cat([torch.cat([mask_base]*2), torch.cat([ref_base]*2)], dim=2)
        inpaint_single = torch.cat([mask_base, ref_base], dim=2)

        # 5. 计算 CFG 截止点 (这是云端一开始就决定的静态策略)
        actual_run_steps = len(timesteps)
        cfg_stop_idx = int(actual_run_steps * CLOUD_CFG_RATIO)

        print(f"   Plan: Running {actual_run_steps} steps. Initial CFG Stop @ {cfg_stop_idx}")

        is_fast_mode = False
        curr_idx = 0

        # --- 推理循环 ---
        while curr_idx < len(timesteps):
            t = timesteps[curr_idx]

            # A. 实时中断检测
            # 只有在非快速模式下才检测，避免重复触发
            if not is_fast_mode:
                # 调用实时信号检测函数
                signal_bad_network = check_realtime_bandwidth(curr_idx, actual_run_steps)
                
                if signal_bad_network:
                    print(f"\n🚨 [REAL-TIME SIGNAL] Bandwidth Drop detected at step {curr_idx} (t={t.item()}).")
                    
                    # 立即执行重规划
                    new_steps_tensor = scheduler.replan_timesteps(t.item(), CLOUD_MIN_STEP, device=DEVICE)
                    print(f"   -> Re-Schedule Plan: {[t.item()] + new_steps_tensor.cpu().tolist()}")
                    
                    timesteps = torch.cat([t.unsqueeze(0), new_steps_tensor])
                    curr_idx = 0
                    is_fast_mode = True 
                    print("   -> 📉 Switching to Fast Mode (CFG Disabled).")

            # B. 动态 CFG 逻辑
            # 1. 只要发生中断 (is_fast_mode=True)，CFG 永久关闭
            # 2. 否则，根据云端预设的 ratio 决定
            do_cfg = (not is_fast_mode and curr_idx < cfg_stop_idx) and (req.guidance_scale > 1.0)

            # C. 输入准备
            if do_cfg:
                latent_model_input = torch.cat([latents_curr] * 2)
                prompt_in = prompt_embeds_cfg
                current_inpaint = inpaint_cfg
            else:
                latent_model_input = latents_curr
                prompt_in = prompt_embeds_single
                current_inpaint = inpaint_single
            
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            current_inpaint = current_inpaint.expand(latent_model_input.shape[0], -1, 17, -1, -1)
            t_tensor = t.expand(latent_model_input.shape[0])

            # D. 前向
            with torch.no_grad():
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_in,
                    timestep=t_tensor,
                    inpaint_latents=current_inpaint
                ).sample

            # E. CFG 计算
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + req.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # F. Step
            latents_curr = scheduler.step(noise_pred, t, latents_curr).prev_sample
            
            status = "CFG" if do_cfg else "Fast/No-CFG"
            print(f"   Step {curr_idx+1}/{len(timesteps)} done [{status}]")
            
            curr_idx += 1

        process_time = time.time() - t_start
        print(f"✅ Inference Done in {process_time:.2f}s")
        return {
            "result_b64": encode_tensor(latents_curr),
            "process_time": process_time
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)