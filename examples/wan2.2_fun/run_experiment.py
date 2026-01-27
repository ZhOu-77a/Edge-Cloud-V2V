import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import torch
import gc
import math
import json
import pandas as pd
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
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

# ================= 1. 全局配置 & 模型路径 =================
MODEL_NAME          = "models/Diffusion_Transformer/Wan2.2-Fun-A14B-InP"
CONFIG_PATH         = "config/wan2.2/wan_civitai_i2v.yaml"
PROMPTS_JSON_PATH   = "/home/zhoujh/Edge-Cloud-diffusion/MyCogVideo-v2v/prompts_config.json"
OUTPUT_ROOT         = "experiment_results_wan2.2_t2v"
CSV_PATH            = os.path.join(OUTPUT_ROOT, "experiment_metrics.csv")
DEVICE              = "cuda"
WEIGHT_DTYPE        = torch.bfloat16

# 固定生成参数
SAMPLE_SIZE         = [480, 832]
VIDEO_LENGTH        = 17
FPS                 = 16
SEED                = 43
GUIDANCE_SCALE      = 6.0
SHIFT               = 5.0
# NUM_INFERENCE_STEPS 将在循环中动态改变

# 加载配置
config = OmegaConf.load(CONFIG_PATH)

def flush():
    gc.collect()
    torch.cuda.empty_cache()

# ================= 2. 指标计算类 =================
class MetricsCalculator:
    def __init__(self, device):
        self.device = device
        print("📊 Loading Metrics Models (LPIPS & CLIP)...")
        # LPIPS (AlexNet backbone)
        self.lpips_loss = lpips.LPIPS(net='alex').to(device)
        # CLIP
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def compute_clip_score(self, frames_tensor, prompt_text):
        """
        frames_tensor: [T, C, H, W] in range [0, 1] (CPU or GPU)
        """
        # 取中间帧和首尾帧求平均，或者每一帧都算
        # 这里为了效率，取 3 帧 (首、中、尾)
        T = frames_tensor.shape[0]
        indices = [0, T//2, T-1]
        selected_frames = frames_tensor[indices] # [3, C, H, W]
        
        # 转换 tensor [0,1] -> PIL Images list
        pil_images = [Image.fromarray((f.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) for f in selected_frames]
        
        inputs = self.clip_processor(text=[prompt_text], images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # [3, 1]
            score = logits_per_image.mean().item()
            
        return score

    def compute_lpips(self, current_frames, target_frames):
        """
        计算当前视频与目标视频（通常是Step 50）的感知距离
        frames: [T, C, H, W] in range [0, 1]
        """
        # LPIPS expects input in range [-1, 1]
        curr = current_frames.to(self.device) * 2.0 - 1.0
        tgt = target_frames.to(self.device) * 2.0 - 1.0
        
        with torch.no_grad():
            # 计算每一帧的距离然后求平均
            dist = self.lpips_loss(curr, tgt) # [T, 1, 1, 1]
            avg_dist = dist.mean().item()
            
        return avg_dist

# ================= 3. 核心生成函数 =================
def generate_one_video(prompt, negative_prompt, steps, save_path):
    """
    运行一次完整的生成流程并返回生成的帧 Tensor (CPU)
    """
    print(f"\n🎬 Generating: Steps={steps} | Prompt={prompt[:30]}...")
    
    # --- 1. Text Encoding ---
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
            [prompt_str], padding="max_length", max_length=max_len, truncation=True,
            add_special_tokens=True, return_tensors="pt",
        )
        embeds = text_encoder(text_inputs.input_ids.to(DEVICE), attention_mask=text_inputs.attention_mask.to(DEVICE))[0]
        seq_len = text_inputs.attention_mask.gt(0).sum(dim=1).long()[0]
        return embeds[0, :seq_len]

    with torch.no_grad():
        context_prompt = get_prompt_embeds(prompt)
        context_neg = get_prompt_embeds(negative_prompt)
        if GUIDANCE_SCALE > 1.0:
            context_input = [context_neg.cpu(), context_prompt.cpu()]
            context_input = [t.to(DEVICE) for t in context_input]
        else:
            context_input = [context_prompt.to(DEVICE)]

    del tokenizer, text_encoder
    flush()

    # --- 2. Latents & Scheduler ---
    temporal_compression_ratio = 4
    spatial_compression_ratio = 8
    latent_channels = 16
    target_frames = (VIDEO_LENGTH - 1) // temporal_compression_ratio + 1
    target_height = SAMPLE_SIZE[0] // spatial_compression_ratio
    target_width  = SAMPLE_SIZE[1] // spatial_compression_ratio

    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    latents = torch.randn(
        (1, latent_channels, target_frames, target_height, target_width),
        generator=generator, device=DEVICE, dtype=WEIGHT_DTYPE
    )

    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )
    scheduler.set_timesteps(steps, device=DEVICE)
    timesteps = scheduler.timesteps

    # --- 3. Split Steps (High/Low Noise) ---
    boundary = config['transformer_additional_kwargs'].get('boundary', 0.900)
    boundary_val = boundary * 1000
    phase1_steps = [(i, t) for i, t in enumerate(timesteps) if t >= boundary_val]
    phase2_steps = [(i, t) for i, t in enumerate(timesteps) if t < boundary_val]

    # Condition Y
    with torch.no_grad():
        y_input = torch.zeros((1, 20, target_frames, target_height, target_width), device=DEVICE, dtype=WEIGHT_DTYPE)
        y_model_input = torch.cat([y_input] * 2) if GUIDANCE_SCALE > 1.0 else y_input
        seq_len_trans = math.ceil((target_height * target_width) / (2 * 2) * target_frames)

    # --- 4. Phase 1 Inference ---
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

    # --- 5. Phase 2 Inference ---
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
        frames = frames.cpu().float() # 此时形状已经是 [B, C, T, H, W] 即 [1, 3, 17, 480, 832]

    del vae
    flush()

    save_videos_grid(frames, save_path, fps=FPS)
    
    # 返回格式调整为 [T, C, H, W] 以便后面计算 CLIP/LPIPS
    # frames[0] 取出 batch 维度 -> [3, 17, H, W]
    # .permute(1, 0, 2, 3) 调整维度 -> [17, 3, H, W]
    return frames[0].permute(1, 0, 2, 3)

# ================= 4. 主程序 =================
if __name__ == "__main__":
    # 1. 准备目录和CSV
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        df = pd.DataFrame(columns=["prompt_id", "prompt_name", "step", "clip_score", "lpips_score_vs_step50"])
        df.to_csv(CSV_PATH, index=False)
    
    # 2. 加载 Prompts
    with open(PROMPTS_JSON_PATH, 'r') as f:
        prompts_data = json.load(f)

    # 3. 初始化指标计算器
    # 注意：如果显存非常紧张（小于24G），此处初始化可能会占用显存导致后面生成 OOM。
    # 如果遇到 OOM，需要把 MetricsCalculator 的初始化放到生成循环内部（生成后加载，算完删除）
    metrics_calc = MetricsCalculator(DEVICE)

    # 4. 开始实验循环
    for item in prompts_data:
        p_id = item['id']
        p_name = item['name']
        prompt = item['prompt']
        neg_prompt = item['negative_prompt']
        
        # 创建 prompt 对应的文件夹
        safe_name = "".join([c if c.isalnum() else "_" for c in p_name])
        folder_name = f"{p_id}_{safe_name}"
        save_dir = os.path.join(OUTPUT_ROOT, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*20}\nProcessing Prompt ID: {p_id} ({p_name})\n{'='*20}")
        
        # 临时存储生成结果以便计算 LPIPS
        # key: step, value: frames_tensor (CPU)
        step_frames_cache = {} 
        
        # Step 循环 1 到 50
        for step in range(1, 51):
            video_filename = f"step_{step:03d}.mp4"
            save_path = os.path.join(save_dir, video_filename)
            
            # A. 生成视频
            # 如果文件已存在则跳过？(可选)
            # if os.path.exists(save_path): continue 
            
            frames = generate_one_video(prompt, neg_prompt, step, save_path)
            step_frames_cache[step] = frames
            
            # B. 计算 CLIP Score (不需要参考视频)
            clip_s = metrics_calc.compute_clip_score(frames, prompt)
            print(f" -> Step {step} CLIP Score: {clip_s:.4f}")
            
            # C. 记录数据 (LPIPS 稍后计算)
            # 先占位
            new_row = {
                "prompt_id": p_id,
                "prompt_name": p_name,
                "step": step,
                "clip_score": clip_s,
                "lpips_score_vs_step50": None # 待填
            }
            
            # 追加写入（为了防止程序中途崩掉，我们先写一行，lpips 后面 update）
            # 但为了方便，我们等 Step 50 跑完再一次性计算 LPIPS 并写入 CSV 比较整洁
            
        # D. Step 50 跑完后，计算 LPIPS 并写入 CSV
        print(f"✅ Finished 1-50 steps for prompt {p_id}. Computing LPIPS...")
        
        reference_frames = step_frames_cache[50] # 获取 Step 50 的结果作为 Ground Truth
        
        results_list = []
        for step in range(1, 51):
            frames = step_frames_cache[step]
            
            # 计算 LPIPS (当前 step vs step 50)
            if step == 50:
                lpips_s = 0.0
            else:
                lpips_s = metrics_calc.compute_lpips(frames, reference_frames)
            
            # 重新计算一遍 CLIP (或者从上面存下来，这里为了逻辑简单重新调一遍计算函数其实很快)
            # 实际上上面已经算过了，这里我们假设上面只是打印。
            # 为了代码整洁，我们在上面循环里其实应该存到一个 list 里。
            
            # 修正逻辑：我们用一个 list 存结果
            clip_s = metrics_calc.compute_clip_score(frames, prompt) # 也可以上面存dict里取
            
            results_list.append({
                "prompt_id": p_id,
                "prompt_name": p_name,
                "step": step,
                "clip_score": clip_s,
                "lpips_score_vs_step50": lpips_s
            })
            
            print(f"Step {step}: CLIP={clip_s:.4f}, LPIPS={lpips_s:.4f}")

        # E. 写入 CSV
        batch_df = pd.DataFrame(results_list)
        batch_df.to_csv(CSV_PATH, mode='a', header=False, index=False)
        print(f"💾 Metrics saved to {CSV_PATH}")
        
        # F. 清理内存，防止 prompt 之间内存泄漏
        del step_frames_cache
        gc.collect()

    print("\n🎉 All experiments finished!")