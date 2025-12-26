import os
import sys
import socket
import time
import requests
import subprocess
import torch
import numpy as np
import pandas as pd
import imageio.v3 as iio
from PIL import Image
import base64
import gc
import traceback
import logging
import json

# === 1. 核心配置 ===
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment_prompt_diff.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 导入指标
try:
    from stream_metrics import calc_clip_score, calc_warp_error
    logger.info("📦 Metrics loaded.")
except ImportError as e:
    logger.error(f"❌ Metrics Error: {e}")
    sys.exit(1)

# === 项目配置 ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT) if PROJECT_ROOT not in sys.path else None

from videox_fun.models import AutoencoderKLCogVideoX
from videox_fun.utils.utils import save_videos_grid

SERVER_SCRIPT = "cloud_server.py"
SERVER_PORT = 12346 
SERVER_LOG_FILE = "server_prompt.log"
BASE_URL = f"http://127.0.0.1:{SERVER_PORT}"
CLOUD_URL = f"{BASE_URL}/inference"
HEALTH_URL = f"{BASE_URL}/health"

# === 实验变量 ===
INPUT_VIDEO_DIR = "asset/batch_videos_6s_2"
PROMPT_CONFIG_FILE = "prompts_config.json"
CSV_RESULT_PATH = "experiment_results_prompt_diff.csv" # 结果文件路径

MODEL_NAME = "models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP"
if not os.path.exists(MODEL_NAME):
    ABS_MODEL_PATH = "/home/zhoujh/Edge-Cloud-diffusion/CogVideoX-Fun/" + MODEL_NAME
    if os.path.exists(ABS_MODEL_PATH): MODEL_NAME = ABS_MODEL_PATH
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DTYPE = torch.bfloat16

BANDWIDTH_MBPS = 5.0 
FIXED_DURATION = 6.0 
SAMPLE_SIZE = [384, 672] 
STRENGTH = 0.7

# === 优化变量范围 ===
FPS_LIST = [2, 3, 4, 5, 6, 7, 8]
STEPS_LIST = [2, 4, 6, 8, 10, 12, 14, 16, 20, 30]
CFG_LIST = [0.0, 0.2, 0.3, 0.5, 0.8, 1.0]

DEFAULT_FPS = 8
DEFAULT_STEPS = 30
DEFAULT_CFG = 1.0 

# 【修改 1】增加 Trial 次数
NUM_TRIALS = 3
START_SEED = 43 

OUTPUT_DIR = "output_experiment_prompt_diff"
DEBUG_INPUT_DIR = "debug_inputs_check_prompt"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
if not os.path.exists(DEBUG_INPUT_DIR): os.makedirs(DEBUG_INPUT_DIR)

vae = None

def load_vae():
    global vae
    if vae is None:
        try:
            logger.info("📦 Loading VAE...")
            vae = AutoencoderKLCogVideoX.from_pretrained(MODEL_NAME, subfolder="vae").to(DEVICE).to(WEIGHT_DTYPE)
        except Exception as e: 
            logger.error(f"VAE Load Error: {e}")
            raise e

def encode_tensor(tensor):
    np_array = tensor.cpu().float().numpy().astype(np.float16)
    return base64.b64encode(np_array.tobytes()).decode('utf-8')

def decode_tensor(b64_str, shape):
    bytes_data = base64.b64decode(b64_str)
    np_array = np.frombuffer(bytes_data, dtype=np.float16)
    return torch.from_numpy(np_array.copy()).reshape(shape).to(DEVICE).to(WEIGHT_DTYPE)

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def wait_for_server(max_retries=60):
    session = requests.Session(); session.trust_env = False
    for i in range(max_retries):
        try:
            if session.get(HEALTH_URL, timeout=2).status_code == 200:
                return True
        except: 
            time.sleep(2)
    return False

def simulate_latency(data_bytes, bandwidth_mbps):
    bits = data_bytes * 8
    bandwidth_bps = bandwidth_mbps * 1_000_000
    latency = bits / bandwidth_bps
    final_latency = max(0, latency * np.random.uniform(0.95, 1.05))
    time.sleep(final_latency)
    return final_latency

def load_and_process_video(video_path, target_frames, target_hw):
    raw_frames = iio.imread(video_path, plugin="pyav")
    total_frames = len(raw_frames)
    indices = np.linspace(0, total_frames - 1, target_frames).astype(int)
    
    processed_frames = []
    tg_h, tg_w = target_hw
    for idx in indices:
        img = Image.fromarray(raw_frames[idx])
        w, h = img.size
        # Center Crop
        target_aspect = tg_w / tg_h
        curr_aspect = w / h
        if curr_aspect > target_aspect:
            new_w = int(h * target_aspect)
            offset = (w - new_w) // 2
            img = img.crop((offset, 0, offset + new_w, h))
        else:
            new_h = int(w / target_aspect)
            offset = (h - new_h) // 2
            img = img.crop((0, offset, w, offset + new_h))
        img = img.resize((tg_w, tg_h), Image.BICUBIC)
        processed_frames.append(np.array(img))
        
    video_np = np.stack(processed_frames).astype(np.float32) / 255.0
    video_np = video_np.transpose(0, 3, 1, 2)
    tensor = torch.from_numpy(video_np).unsqueeze(0).permute(0, 2, 1, 3, 4)
    return tensor

# === 辅助函数：即时写入 CSV ===
def append_to_csv(result_dict, file_path):
    try:
        df = pd.DataFrame([result_dict])
        # 如果文件不存在，写入 header；如果存在，追加模式 (mode='a') 且不写 header
        need_header = not os.path.exists(file_path)
        df.to_csv(file_path, mode='a', header=need_header, index=False)
    except Exception as e:
        logger.error(f"❌ Failed to append to CSV: {e}")

# === 核心运行逻辑 ===
def run_single_trial(video_path, video_name, prompt_data, fps, steps, cfg, seed):
    # 解析 Prompt 数据
    p_id = prompt_data['id']
    p_name = prompt_data['name']
    prompt_text = prompt_data['prompt']
    neg_prompt = prompt_data.get('negative_prompt', "")

    trial_id = f"Vid={video_name} PromptID={p_id} FPS={fps} Steps={steps} CFG={cfg} Seed={seed}"
    logger.info(f"▶️ Start Trial: {trial_id}")
    
    gc.collect(); torch.cuda.empty_cache()
    load_vae()
    
    # 1. 视频处理
    raw_target_frames = int(FIXED_DURATION * fps)
    MAX_FRAMES_SUPPORTED = 49 
    if raw_target_frames > MAX_FRAMES_SUPPORTED: raw_target_frames = MAX_FRAMES_SUPPORTED
    
    compression = vae.config.temporal_compression_ratio 
    aligned_frames = int((raw_target_frames - 1) // compression * compression) + 1
    if aligned_frames < raw_target_frames: aligned_frames += compression
    
    input_video = load_and_process_video(video_path, aligned_frames, SAMPLE_SIZE)

    # 保存参考视频 (Ref)
    ref_video_path = os.path.join(DEBUG_INPUT_DIR, f"ref_{video_name}_fps{fps}.mp4")
    debug_vid = input_video.squeeze(0).permute(1, 2, 3, 0).cpu().numpy() 
    debug_vid = (debug_vid * 255).astype(np.uint8)
    # 如果文件已存在，可以选择不重复写入，但为了稳妥起见还是覆盖
    iio.imwrite(ref_video_path, debug_vid, fps=fps, codec='libx264')

    input_video = input_video.to(DEVICE).to(WEIGHT_DTYPE)
    input_video = 2.0 * input_video - 1.0
    
    t_start_total = time.time()
    
    # 2. Encode
    with torch.no_grad():
        init_latents = vae.encode(input_video).latent_dist.sample() * vae.config.scaling_factor
        if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
                init_latents = init_latents - vae.config.shift_factor
    
    # 3. Request (使用当前 Prompt)
    latents_b64 = encode_tensor(init_latents)
    payload = {
        "latents_b64": latents_b64, "shape": list(init_latents.shape),
        "prompt": prompt_text,          # 动态 Prompt
        "negative_prompt": neg_prompt,  # 动态 Negative
        "strength": STRENGTH, "steps": steps, "guidance_scale": 6.0, 
        "seed": seed, "cfg_ratio": cfg
    }
    
    t_up = simulate_latency(len(latents_b64), BANDWIDTH_MBPS)
    session = requests.Session(); session.trust_env = False 
    resp = session.post(CLOUD_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    t_down = simulate_latency(len(data["result_b64"]), BANDWIDTH_MBPS)
    
    # 4. Decode
    latents_out = decode_tensor(data["result_b64"], init_latents.shape)
    with torch.no_grad():
        if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
                latents_out = latents_out + vae.config.shift_factor
        video_out = vae.decode(latents_out / vae.config.scaling_factor).sample
    
    video_out = (video_out / 2.0 + 0.5).clamp(0, 1)
    total_latency = time.time() - t_start_total
    
    # 5. Save & Metrics
    # 按层级目录保存: OUTPUT_DIR / VideoName / PID / fps_steps_cfg_seed.mp4
    video_out_subdir = os.path.join(OUTPUT_DIR, video_name, f"pid{p_id}")
    if not os.path.exists(video_out_subdir):
        os.makedirs(video_out_subdir, exist_ok=True)
    
    # 文件名包含 seed
    save_filename = f"fps{fps}_steps{steps}_cfg{cfg}_seed{seed}.mp4"
    gen_video_path = os.path.join(video_out_subdir, save_filename)
    
    save_videos_grid(video_out.to(dtype=torch.float32).cpu(), gen_video_path, fps=fps)
    
    # Metrics
    clip_text_score, clip_consistency = calc_clip_score(gen_video_path, prompt_text)
    warp_error = calc_warp_error(ref_video_path, gen_video_path)
    
    # Optional: 清理生成视频 (若硬盘空间不足可开启)
    # try: os.remove(gen_video_path)
    # except: pass

    return {
        "video": video_name,
        "prompt_id": p_id,
        "prompt_name": p_name,
        "prompt_difficulty": prompt_data['difficulty'],
        "fps": fps, "steps": steps, "cfg": cfg, "seed": seed,
        "latency": total_latency, 
        "clip_score": clip_text_score,
        "clip_consistency": clip_consistency,
        "warp_error": warp_error
    }

def run_batch_for_prompt(video_path, video_name, prompt_data, param_name, param_list, fixed_params):
    logger.info(f"   👉 Varying {param_name} for Prompt: {prompt_data['name']}")
    
    for val in param_list:
        for i in range(NUM_TRIALS):
            seed = START_SEED + i
            args = fixed_params.copy()
            args[param_name] = val
            
            try:
                res = run_single_trial(
                    video_path, video_name, prompt_data,
                    args['fps'], args['steps'], args['cfg'], seed
                )
                if res:
                    # 即时写入 CSV
                    append_to_csv(res, CSV_RESULT_PATH)
                    logger.info(f"   💾 Result saved for {param_name}={val} (Seed {seed})")
            except Exception as e:
                logger.error(f"❌ Failed: {prompt_data['name']} | {param_name}={val} | Seed {seed}")
                logger.error(f"   Reason: {str(e).splitlines()[-1]}")

def main():
    server_process = None
    server_started_by_me = False
    
    if not is_port_in_use(SERVER_PORT):
        logger.info(f"☁️ Server not running. Starting local {SERVER_SCRIPT} on GPU 3...")
        server_env = os.environ.copy()
        server_env["CUDA_VISIBLE_DEVICES"] = "3"
        server_env["PORT"] = str(SERVER_PORT)
        try:
            log_file = open(SERVER_LOG_FILE, "w")
            server_process = subprocess.Popen(
                [sys.executable, SERVER_SCRIPT],
                env=server_env,
                stdout=log_file,
                stderr=log_file
            )
            server_started_by_me = True
            logger.info(f"   Server started with PID {server_process.pid}")
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return

    try:
        if not wait_for_server(): 
            logger.error("❌ Server start failed or not reachable.")
            if server_started_by_me and server_process:
                server_process.terminate()
            return
        
        # 1. 准备 CSV
        if os.path.exists(CSV_RESULT_PATH):
            os.remove(CSV_RESULT_PATH)
            logger.info(f"🗑️ Cleaned up old results CSV: {CSV_RESULT_PATH}")

        # 2. 读取 Prompts
        if not os.path.exists(PROMPT_CONFIG_FILE):
            logger.error(f"❌ Config file not found: {PROMPT_CONFIG_FILE}")
            return
        with open(PROMPT_CONFIG_FILE, 'r') as f:
            prompts_data = json.load(f)
        logger.info(f"📖 Loaded {len(prompts_data)} prompts.")

        # 3. 读取 Videos
        video_files = []
        if os.path.exists(INPUT_VIDEO_DIR):
            for f in os.listdir(INPUT_VIDEO_DIR):
                if f.endswith('.mp4'): video_files.append(os.path.join(INPUT_VIDEO_DIR, f))
        
        fixed = {'fps': DEFAULT_FPS, 'steps': DEFAULT_STEPS, 'cfg': DEFAULT_CFG}

        # === 实验循环 ===
        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            logger.info(f"\n🎬 Processing Video: {video_name}")
            
            for prompt_item in prompts_data:
                logger.info(f" 📝 Using Prompt ID {prompt_item['id']}: {prompt_item['name']}")
                
                # 实验 1: 变化 Steps
                run_batch_for_prompt(video_path, video_name, prompt_item, 'steps', STEPS_LIST, fixed)
                
                # 实验 2: 变化 CFG
                run_batch_for_prompt(video_path, video_name, prompt_item, 'cfg', CFG_LIST, fixed)
                
                # 实验 3: 变化 FPS
                run_batch_for_prompt(video_path, video_name, prompt_item, 'fps', FPS_LIST, fixed)

        logger.info(f"\n✅ All experiments finished. Data in {CSV_RESULT_PATH}")

    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user.")
    finally:
        if server_started_by_me and server_process:
            logger.info("🛑 Stopping background server...")
            server_process.terminate()
            server_process.wait()
            try: log_file.close()
            except: pass

if __name__ == "__main__":
    main()