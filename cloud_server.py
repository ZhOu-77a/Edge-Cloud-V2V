import os
import sys
import torch
import base64
import uvicorn
import gc
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel



from videox_fun.models import CogVideoXTransformer3DModel, T5EncoderModel, T5Tokenizer, AutoencoderKLCogVideoX
from videox_fun.utils.fp8_optimization import convert_model_weight_to_float8, convert_weight_dtype_wrapper
from diffusers import DDIMScheduler

# === 调试函数 ===
def print_stats(step_name, tensor):
    if isinstance(tensor, torch.Tensor):
        t = tensor.float().cpu()
        print(f"☁️ [DEBUG-CLOUD] {step_name:<25} | Mean: {t.mean():.6f} | Std: {t.std():.6f}")

# === 3. 配置区域 ===
MODEL_NAME = "models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP" 
DEVICE = "cuda"
WEIGHT_DTYPE = torch.bfloat16 
PORT = 12345

app = FastAPI()

print(f"☁️ [Cloud] Initializing VideoX-Fun Server on {DEVICE}...")

# === 4. 模型加载 ===
# 加载 VAE 仅用于占位和配置读取，实际不跑
vae = AutoencoderKLCogVideoX.from_pretrained(MODEL_NAME, subfolder="vae").to(DEVICE).to(WEIGHT_DTYPE)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(MODEL_NAME, subfolder="text_encoder", torch_dtype=WEIGHT_DTYPE).to(DEVICE)

print("📦 Loading Transformer...")
transformer = CogVideoXTransformer3DModel.from_pretrained(
    MODEL_NAME, 
    subfolder="transformer",
    low_cpu_mem_usage=True,
    torch_dtype=WEIGHT_DTYPE,
)

print("🚀 Applying FP8 Optimization...")
convert_model_weight_to_float8(transformer, exclude_module_name=[], device=DEVICE)
convert_weight_dtype_wrapper(transformer, WEIGHT_DTYPE)
transformer.to(DEVICE)

scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

print("✅ Server Models Loaded!")

# === 5. 辅助函数 ===
def decode_tensor(b64_str, shape):
    bytes_data = base64.b64decode(b64_str)
    np_array = np.frombuffer(bytes_data, dtype=np.float16)
    tensor = torch.from_numpy(np_array.copy()).reshape(shape)
    return tensor.to(DEVICE).to(WEIGHT_DTYPE)

def encode_tensor(tensor):
    np_array = tensor.cpu().float().numpy().astype(np.float16)
    return base64.b64encode(np_array.tobytes()).decode('utf-8')

class FunRequest(BaseModel):
    latents_b64: str
    prompt: str
    negative_prompt: str
    shape: list
    steps: int = 50
    strength: float = 0.7
    guidance_scale: float = 6.0
    seed: int = 43  # 【新增】接收种子

@app.post("/inference")
async def inference(req: FunRequest):
    try:
        print(f"☁️ Processing: {req.prompt[:20]}... Strength={req.strength} Seed={req.seed}")
        
        # 1. 还原 Latent
        init_latents = decode_tensor(req.latents_b64, req.shape)
        # [B, C, F, H, W] -> [B, F, C, H, W]
        latents = init_latents.permute(0, 2, 1, 3, 4)
        
        print_stats("Received Latents", latents)

        # 2. 文本编码
        text_inputs = tokenizer(req.prompt, padding="max_length", max_length=226, truncation=True, add_special_tokens=True, return_tensors="pt")
        prompt_embeds = text_encoder(text_inputs.input_ids.to(DEVICE))[0]
        neg_inputs = tokenizer(req.negative_prompt, padding="max_length", max_length=226, truncation=True, add_special_tokens=True, return_tensors="pt")
        negative_prompt_embeds = text_encoder(neg_inputs.input_ids.to(DEVICE))[0]
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 3. Scheduler
        scheduler.set_timesteps(req.steps, device=DEVICE)
        init_timestep_idx = int(req.steps * req.strength)
        init_timestep_idx = min(init_timestep_idx, req.steps - 1)
        t_start = scheduler.timesteps[req.steps - init_timestep_idx]
        timesteps = scheduler.timesteps[req.steps - init_timestep_idx:]

        #使用 Generator 生成确定性噪声
        generator = torch.Generator(device=DEVICE).manual_seed(req.seed)
        
        # 4. 加噪: 注意：randn 必须传入 generator 参数
        noise = torch.randn(latents.shape, generator=generator, device=DEVICE, dtype=WEIGHT_DTYPE)
        
        # noise = torch.randn_like(latents)
        t_start_expand = torch.tensor([t_start] * latents.shape[0], device=DEVICE)
        latents_noisy = scheduler.add_noise(latents, noise, t_start_expand)

        # === 【核心复刻】构造 Mask & Ref (全 0) ===
        # 根据 Pipeline 默认行为：Mask=0, Ref=0
        mask_base = torch.zeros_like(latents[:, :, :1, :, :]) 
        ref_input_base = torch.zeros_like(latents)
        
        mask_input_full = torch.cat([mask_base] * 2)
        ref_input_full = torch.cat([ref_input_base] * 2)
        
        # 拼接 17 通道 (1 Mask + 16 Ref)
        # 注意: 这里的维度是 Channel (dim=2)
        inpaint_latents_base = torch.cat([mask_input_full, ref_input_full], dim=2)

        # 5. Denoising Loop
        latents_curr = latents_noisy
        
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents_curr] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            # 动态对齐 shape
            current_shape = latent_model_input.shape
            inpaint_latents = inpaint_latents_base.expand(
                current_shape[0], current_shape[1], 17, current_shape[3], current_shape[4]
            )

            # 调用 Transformer (传入独立的 inpaint_latents)
            t_tensor = torch.tensor([t], device=DEVICE, dtype=torch.long)
            t_tensor = t_tensor.repeat(latent_model_input.shape[0])

            with torch.no_grad():
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=t_tensor,
                    inpaint_latents=inpaint_latents,
                    image_rotary_emb=None
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + req.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents_curr = scheduler.step(noise_pred, t, latents_curr).prev_sample

        # 6. 返回
        final_latents = latents_curr.permute(0, 2, 1, 3, 4)
        result_b64 = encode_tensor(final_latents)
        
        return {"result_b64": result_b64}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)