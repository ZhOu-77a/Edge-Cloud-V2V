import os
import sys
import torch
import base64
import uvicorn
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# === 1. 环境与路径设置 ===
from videox_fun.models import CogVideoXTransformer3DModel, T5EncoderModel, T5Tokenizer, AutoencoderKLCogVideoX
from videox_fun.utils.fp8_optimization import convert_model_weight_to_float8, convert_weight_dtype_wrapper
from diffusers import DDIMScheduler

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),                                       
    os.path.dirname(os.path.dirname(current_file_path)),                      
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))      
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

# === 3. 配置区域 ===
MODEL_NAME = "models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP" 
if not os.path.exists(MODEL_NAME):
    ABS_MODEL_PATH = "/home/zhoujh/Edge-Cloud-diffusion/CogVideoX-Fun/" + MODEL_NAME
    if os.path.exists(ABS_MODEL_PATH): MODEL_NAME = ABS_MODEL_PATH

DEVICE = "cuda"
WEIGHT_DTYPE = torch.bfloat16 
PORT = 12345

app = FastAPI()

print(f"☁️ [Cloud] Initializing DEBUG Server on {DEVICE}...")

# === 4. 模型加载 ===
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(MODEL_NAME, subfolder="text_encoder", torch_dtype=WEIGHT_DTYPE).to(DEVICE)
transformer = CogVideoXTransformer3DModel.from_pretrained(MODEL_NAME, subfolder="transformer", low_cpu_mem_usage=True, torch_dtype=WEIGHT_DTYPE)
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
    strength: float = 0.8
    guidance_scale: float = 6.0
    seed: int = 43
    cfg_ratio: float = 1.0 

@app.get("/health")
async def health_check():
    return {"status": "ready"}

@app.post("/inference")
async def inference(req: FunRequest):
    t_server_start = time.time()
    try:
        # === 🔍 调试打印 1: 输入信息 ===
        print(f"\n🔍 [DEBUG] New Request | Seed: {req.seed} | CFG Ratio: {req.cfg_ratio}")
        
        # 1. 还原 Latent
        init_latents = decode_tensor(req.latents_b64, req.shape)
        latents = init_latents.permute(0, 2, 1, 3, 4)
        print(f"   Shape Received: {list(latents.shape)} (Frames={latents.shape[2]})")

        # 2. 文本编码
        text_inputs = tokenizer(req.prompt, padding="max_length", max_length=226, truncation=True, add_special_tokens=True, return_tensors="pt")
        prompt_embeds = text_encoder(text_inputs.input_ids.to(DEVICE))[0]
        neg_inputs = tokenizer(req.negative_prompt, padding="max_length", max_length=226, truncation=True, add_special_tokens=True, return_tensors="pt")
        negative_prompt_embeds = text_encoder(neg_inputs.input_ids.to(DEVICE))[0]
        
        prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_embeds_single = prompt_embeds

        # 3. Scheduler
        scheduler.set_timesteps(req.steps, device=DEVICE)
        init_timestep_idx = int(req.steps * req.strength)
        init_timestep_idx = min(init_timestep_idx, req.steps - 1)
        t_start = scheduler.timesteps[req.steps - init_timestep_idx]
        timesteps = scheduler.timesteps[req.steps - init_timestep_idx:]
        
        print(f"   Denoising Steps: {len(timesteps)} (Total {req.steps} * Strength {req.strength})")

        generator = torch.Generator(device=DEVICE).manual_seed(req.seed)
        noise = torch.randn(latents.shape, generator=generator, device=DEVICE, dtype=WEIGHT_DTYPE)
        
        t_start_expand = torch.tensor([t_start] * latents.shape[0], device=DEVICE)
        latents_noisy = scheduler.add_noise(latents, noise, t_start_expand)

        # 4. Inpaint Mask
        mask_base = torch.zeros_like(latents[:, :, :1, :, :]) 
        ref_input_base = torch.zeros_like(latents)
        
        mask_input_cfg = torch.cat([mask_base] * 2)
        ref_input_cfg = torch.cat([ref_input_base] * 2)
        inpaint_latents_cfg = torch.cat([mask_input_cfg, ref_input_cfg], dim=2)
        
        mask_input_single = mask_base
        ref_input_single = ref_input_base
        inpaint_latents_single = torch.cat([mask_input_single, ref_input_single], dim=2)

        # 5. Denoising Loop
        latents_curr = latents_noisy
        
        # === 🔍 调试打印 2: CFG 截止点 ===
        cfg_stop_idx = int(len(timesteps) * req.cfg_ratio)
        print(f"   CFG Strategy: Steps 0-{cfg_stop_idx-1} [ON], Steps {cfg_stop_idx}-{len(timesteps)-1} [OFF]")

        for i, t in enumerate(timesteps):
            do_cfg = (i < cfg_stop_idx) and (req.guidance_scale > 1.0)
            
            # 只有第0步和切换点打印，避免刷屏
            if i == 0 or i == cfg_stop_idx or i == cfg_stop_idx - 1:
                status = "DOUBLE (CFG)" if do_cfg else "SINGLE (Fast)"
                print(f"   Step {i}/{len(timesteps)} | Mode: {status}")

            if do_cfg:
                latent_model_input = torch.cat([latents_curr] * 2)
                inpaint_in = inpaint_latents_cfg
                prompt_in = prompt_embeds_cfg
            else:
                latent_model_input = latents_curr
                inpaint_in = inpaint_latents_single
                prompt_in = prompt_embeds_single

            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            current_shape = latent_model_input.shape
            inpaint_latents = inpaint_in.expand(
                current_shape[0], current_shape[1], 17, current_shape[3], current_shape[4]
            )

            t_tensor = torch.tensor([t], device=DEVICE, dtype=torch.long)
            t_tensor = t_tensor.repeat(latent_model_input.shape[0])

            with torch.no_grad():
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_in,
                    timestep=t_tensor,
                    inpaint_latents=inpaint_latents,
                    image_rotary_emb=None
                ).sample

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + req.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents_curr = scheduler.step(noise_pred, t, latents_curr).prev_sample

        final_latents = latents_curr.permute(0, 2, 1, 3, 4)
        result_b64 = encode_tensor(final_latents)
        
        process_time = time.time() - t_server_start
        print(f"✅ Done. Time: {process_time:.4f}s")
        return {"result_b64": result_b64, "process_time": process_time}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)