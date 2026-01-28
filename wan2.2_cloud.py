import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import torch
import base64
import uvicorn
import time
import math
import gc
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from omegaconf import OmegaConf

# 引入 Wan2.2 依赖
current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path))]
for root in project_roots:
    sys.path.insert(0, root) if root not in sys.path else None

from videox_fun.models import (AutoTokenizer, WanT5EncoderModel, Wan2_2Transformer3DModel)
from videox_fun.utils.fp8_optimization import convert_model_weight_to_float8, convert_weight_dtype_wrapper
from videox_fun.utils.utils import filter_kwargs
from utils1.Int_FlowDPMSolverMultistepScheduler import Int_FlowMatchEulerDiscreteScheduler

# ================= 配置 =================
MODEL_NAME = "models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP"
CONFIG_PATH = "config/wan2.2/wan_civitai_i2v.yaml" 
DEVICE = "cuda"
WEIGHT_DTYPE = torch.bfloat16
PORT = 12346

# --- 策略配置 ---
# 这里对应你原始脚本里的逻辑：前 30% 步数用 CFG
CLOUD_CFG_RATIO = 0.3      
CLOUD_MIN_STEP = 3         

app = FastAPI()
config = OmegaConf.load(CONFIG_PATH)

def flush():
    gc.collect()
    torch.cuda.empty_cache()

print(f"☁️ [Cloud] Initializing Wan2.2 Server on {DEVICE}...")

# 1. 全局加载 Tokenizer (CPU)
tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_NAME, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')))

# 2. 全局加载 Text Encoder (但先放在 CPU 上不进显存！)
print("   Loading Text Encoder to CPU memory...")
text_encoder_cpu = WanT5EncoderModel.from_pretrained(
    os.path.join(MODEL_NAME, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=WEIGHT_DTYPE,
)
# 注意：这里没有 .to(DEVICE)，为了省显存

# --- Transformer 管理类 ---
class ModelManager:
    def __init__(self):
        self.current_model = None
        self.current_model_type = None 

    def load_model(self, model_type):
        if self.current_model_type == model_type and self.current_model is not None:
            return self.current_model

        # 必须先清理旧模型
        self.unload_all()

        print(f"🚀 [Loader] Loading Transformer '{model_type}'...")
        
        subpath_key = 'transformer_high_noise_model_subpath' if model_type == "high_noise" else 'transformer_low_noise_model_subpath'
        subpath = config['transformer_additional_kwargs'].get(subpath_key, 'transformer')
        
        model = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(MODEL_NAME, subpath),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True, 
            torch_dtype=WEIGHT_DTYPE,
        )
        
        # FP8 量化并移入 GPU
        convert_model_weight_to_float8(model, exclude_module_name=["modulation",], device=DEVICE)
        convert_weight_dtype_wrapper(model, WEIGHT_DTYPE)
        model.freqs = model.freqs.to(DEVICE)
        model.to(DEVICE).eval()

        self.current_model = model
        self.current_model_type = model_type
        print(f"✅ [Loader] Transformer '{model_type}' Loaded.")
        return self.current_model
    
    def unload_all(self):
        if self.current_model is not None:
            print("🧹 [Memory] Unloading Transformer to free VRAM for T5...")
            del self.current_model
            self.current_model = None
            self.current_model_type = None
            flush()

model_manager = ModelManager()
print("✅ Server Ready!")

# --- 辅助函数 ---
def decode_tensor(b64_str, shape):
    bytes_data = base64.b64decode(b64_str)
    np_array = np.frombuffer(bytes_data, dtype=np.float16)
    tensor = torch.from_numpy(np_array.copy()).reshape(shape)
    return tensor.to(DEVICE).to(WEIGHT_DTYPE)

def encode_tensor(tensor):
    np_array = tensor.cpu().float().numpy().astype(np.float16)
    return base64.b64encode(np_array.tobytes()).decode('utf-8')

# 获取 Embedding (包含显存搬运逻辑)
def get_prompt_embeds_on_demand(prompt_str, max_len=512):
    # 1. 确保 Text Encoder 在 GPU
    text_encoder_cpu.to(DEVICE)
    
    try:
        text_inputs = tokenizer([prompt_str], padding="max_length", max_length=max_len, truncation=True, add_special_tokens=True, return_tensors="pt")
        text_input_ids = text_inputs.input_ids.to(DEVICE)
        prompt_attention_mask = text_inputs.attention_mask.to(DEVICE)
        
        embeds = text_encoder_cpu(text_input_ids, attention_mask=prompt_attention_mask)[0]
        seq_len = prompt_attention_mask.gt(0).sum(dim=1).long()[0]
        result = embeds[0, :seq_len]
        return result
    finally:
        # 不论成功失败，暂时不挪回CPU，等所有Prompt处理完统一挪，或者依赖 inference 里的逻辑
        pass

# --- 模拟实时带宽检测 ---
def check_realtime_bandwidth(current_step, total_steps):
    SIMULATE_BAD_NETWORK_AT_RATIO = 0.5
    if current_step == int(total_steps * SIMULATE_BAD_NETWORK_AT_RATIO):
        return True
    return False

# --- 请求模型 ---
class WanRequest(BaseModel):
    latents_b64: str
    shape: list
    prompt: str
    negative_prompt: str
    steps: int = 50
    strength: float = 0.5
    guidance_scale: float = 6.0
    seed: int = 43

@app.post("/inference")
async def inference(req: WanRequest):
    t_start = time.time()
    try:
        print(f"\n🔍 [Request] Seed={req.seed} | Strength={req.strength}")

        # --- 步骤 1: 显存清理 & 文本编码 ---
        # 确保 Transformer 已卸载
        model_manager.unload_all()
        
        print("⚡ [Phase] Text Encoding (T5 on GPU)...")
        # 将 T5 搬入 GPU
        text_encoder_cpu.to(DEVICE)
        
        with torch.no_grad():
            context_prompt = get_prompt_embeds_on_demand(req.prompt)
            context_neg = get_prompt_embeds_on_demand(req.negative_prompt)
            
            # 将结果移回 CPU 保存，准备释放显存
            context_input_cfg = [context_neg.cpu(), context_prompt.cpu()]
            context_input_single = [context_prompt.cpu()]
        
        # 立即将 T5 搬回 CPU 并清理显存
        text_encoder_cpu.to("cpu")
        flush()
        print("✅ [Phase] Text Encoding Done. T5 offloaded.")

        # --- 步骤 2: 准备推理 ---
        init_latents = decode_tensor(req.latents_b64, req.shape)

        scheduler = Int_FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(Int_FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )
        scheduler.set_timesteps(req.steps, device=DEVICE)
        
        steps_to_skip = int(req.steps * (1 - req.strength))
        init_timestep_idx = max(0, min(steps_to_skip, req.steps - 1))
        start_timestep = scheduler.timesteps[init_timestep_idx]
        timesteps = scheduler.timesteps[init_timestep_idx:] 
        
        generator = torch.Generator(device=DEVICE).manual_seed(req.seed)
        noise = torch.randn(init_latents.shape, generator=generator, device=DEVICE, dtype=WEIGHT_DTYPE)
        t_tensor = start_timestep.unsqueeze(0).to(DEVICE)
        latents_curr = scheduler.scale_noise(sample=init_latents, timestep=t_tensor, noise=noise)

        # 准备 Condition (Mask)
        target_latent_frames = init_latents.shape[2]
        target_height = init_latents.shape[3]
        target_width  = init_latents.shape[4]
        
        with torch.no_grad():
            y_input = torch.zeros((1, 20, target_latent_frames, target_height, target_width), device=DEVICE, dtype=WEIGHT_DTYPE)
            y_model_input_cfg = torch.cat([y_input] * 2) 
            y_model_input_single = y_input 
            
            patch_size = (1, 2, 2)
            seq_len = math.ceil((target_height * target_width) / (patch_size[1] * patch_size[2]) * target_latent_frames)

        # 规划
        actual_run_steps = len(timesteps)
        cfg_stop_idx = int(actual_run_steps * CLOUD_CFG_RATIO)
        print(f"   Plan: Running {actual_run_steps} steps. Initial CFG Stop @ {cfg_stop_idx}")

        boundary = config['transformer_additional_kwargs'].get('boundary', 0.900)
        boundary_val = boundary * 1000 if timesteps[0] > 1.0 else boundary

        is_fast_mode = False
        current_steps_queue = [(i, t) for i, t in enumerate(timesteps)]
        global_step_count = 0

        # --- 步骤 3: 循环推理 (动态加载 Transformer) ---
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
            
            # 加载当前阶段需要的 Transformer
            # ModelManager 会自动卸载旧模型（如果类型不同）
            current_model = model_manager.load_model(target_phase)

            idx = 0
            interrupted_in_phase = False
            
            while idx < len(active_queue):
                i, t = active_queue[idx]
                
                # A. 实时中断检测
                if not is_fast_mode:
                    if check_realtime_bandwidth(global_step_count, actual_run_steps):
                        print(f"\n🚨 [INTERRUPT] Cloud triggered interrupt at step {global_step_count}")
                        new_steps_tensor = scheduler.replan_timesteps(t.item(), CLOUD_MIN_STEP, device=DEVICE)
                        print(f"   -> Re-Schedule Plan: {[t.item()] + new_steps_tensor.cpu().tolist()}")
                        new_future_steps = [(0, nt) for nt in new_steps_tensor]
                        next_iteration_queue = new_future_steps
                        interrupted_in_phase = True
                        is_fast_mode = True
                        print("   -> 📉 Switching to Fast Mode (CFG Disabled).")

                # B. 动态 CFG
                do_cfg = (not is_fast_mode and global_step_count < cfg_stop_idx) and (req.guidance_scale > 1.0)

                # C. 输入准备
                if do_cfg:
                    latent_in = torch.cat([latents_curr] * 2)
                    # 将 CPU 上的 Embedding 移到 GPU
                    context_in = [t.to(DEVICE) for t in context_input_cfg]
                    y_in = y_model_input_cfg
                else:
                    latent_in = latents_curr
                    context_in = [t.to(DEVICE) for t in context_input_single]
                    y_in = y_model_input_single
                
                t_emb = t.expand(latent_in.shape[0])

                # D. 推理
                with torch.no_grad():
                    noise_pred = current_model(
                        x=latent_in, context=context_in, t=t_emb, seq_len=seq_len, y=y_in
                    )

                # E. CFG
                if do_cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + req.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # F. Step
                latents_curr = scheduler.step(noise_pred, t, latents_curr, return_dict=False)[0]
                
                global_step_count += 1
                status = "CFG" if do_cfg else "Fast/No-CFG"
                print(f"      Step {global_step_count} done [{status}]")

                if interrupted_in_phase:
                    break
                idx += 1
            
            current_steps_queue = next_iteration_queue

        process_time = time.time() - t_start
        print(f"✅ Inference Done in {process_time:.2f}s")
        
        # --- 步骤 4: 任务结束，彻底清理 ---
        # 卸载 Transformer，为下一个请求的 T5 腾地
        model_manager.unload_all()

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