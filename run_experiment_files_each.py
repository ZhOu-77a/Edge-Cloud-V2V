import os
import sys
import socket
import time
import requests
import subprocess
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio.v3 as iio
from PIL import Image
import base64
import gc
import traceback
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment_debug_gpu34.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


from stream_metrics import (
    calc_clip_score, # 返回 (text_score, consistency_score)
    calc_warp_error  # 返回 warp_error
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT) if PROJECT_ROOT not in sys.path else None

from videox_fun.models import AutoencoderKLCogVideoX
from videox_fun.utils.utils import save_videos_grid

SERVER_SCRIPT = "cloud_server.py"
# 端口配置
SERVER_PORT = 12346 
SERVER_LOG_FILE = "server_gpu34.log"
BASE_URL = f"http://127.0.0.1:{SERVER_PORT}"
CLOUD_URL = f"{BASE_URL}/inference"
HEALTH_URL = f"{BASE_URL}/health"

# === 实验变量 ===
INPUT_VIDEO_DIR = "asset/batch_videos_6s_4"

PROMPT = "A video of streetview in cartoon style."
NEGATIVE_PROMPT = "The video is not of a high quality, low resolution, watermark, distortion."

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
STEPS_LIST = [2, 4, 6, 8, 10, 12, 14, 16, 20, 30, 40, 50]
CFG_LIST = [0.0, 0.2, 0.3, 0.5, 0.8, 1.0]

DEFAULT_FPS = 8
DEFAULT_STEPS = 30
DEFAULT_CFG = 1.0 

NUM_TRIALS = 2
START_SEED = 43 

OUTPUT_DIR = "output_experiment_gpu34"
DEBUG_INPUT_DIR = "debug_inputs_check_gpu34" # 用于存放Metric计算所需的参考视频
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
    logger.info(f"⏳ Waiting for server at {HEALTH_URL} (max {max_retries*2}s)...")
    for i in range(max_retries):
        try:
            if session.get(HEALTH_URL, timeout=2).status_code == 200:
                logger.info("✅ Server Connected and Ready.")
                return True
        except: 
            if i % 5 == 0:
                logger.warning(f"   Waiting for server... ({i+1}/{max_retries})")
            time.sleep(2)
    return False

def simulate_latency(data_bytes, bandwidth_mbps):
    bits = data_bytes * 8
    bandwidth_bps = bandwidth_mbps * 1_000_000
    latency = bits / bandwidth_bps
    final_latency = max(0, latency * np.random.uniform(0.95, 1.05))
    time.sleep(final_latency)
    return final_latency

# === 视频读取函数 ===
def load_and_process_video(video_path, target_frames, target_hw):
    raw_frames = iio.imread(video_path, plugin="pyav")
    total_frames = len(raw_frames)
    indices = np.linspace(0, total_frames - 1, target_frames).astype(int)
    
    logger.info(f"   🎞️ [Video Load] Total: {total_frames} -> Target: {target_frames}")
    logger.info(f"   📍 [Video Load] Sample Indices: {indices.tolist()}")

    processed_frames = []
    tg_h, tg_w = target_hw
    
    for idx in indices:
        img = Image.fromarray(raw_frames[idx])
        w, h = img.size
        
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

# === 单次实验逻辑 ===
def run_single_trial(video_path, video_name, fps, steps, cfg, seed, save_video=False):
    trial_id = f"Vid={video_name} FPS={fps} Steps={steps} CFG={cfg} Seed={seed}"
    logger.info(f"▶️ Start Trial: {trial_id}")
    
    gc.collect(); torch.cuda.empty_cache()
    load_vae()
    
    raw_target_frames = int(FIXED_DURATION * fps)
    
    # 【限制】防止超过模型上限 (49帧)
    MAX_FRAMES_SUPPORTED = 49 
    if raw_target_frames > MAX_FRAMES_SUPPORTED:
        logger.warning(f"⚠️ FPS={fps} results in {raw_target_frames} frames, capping to {MAX_FRAMES_SUPPORTED}!")
        raw_target_frames = MAX_FRAMES_SUPPORTED

    compression = vae.config.temporal_compression_ratio 
    aligned_frames = int((raw_target_frames - 1) // compression * compression) + 1
    if aligned_frames < raw_target_frames: aligned_frames += compression
    
    # 这里会打印 indices
    input_video = load_and_process_video(video_path, aligned_frames, SAMPLE_SIZE)

    # 保存参考视频 (用于计算 Warp Error)
    ref_video_path = os.path.join(DEBUG_INPUT_DIR, f"ref_{video_name}_fps{fps}.mp4")
    debug_vid = input_video.squeeze(0).permute(1, 2, 3, 0).cpu().numpy() 
    debug_vid = (debug_vid * 255).astype(np.uint8)
    iio.imwrite(ref_video_path, debug_vid, fps=fps, codec='libx264')

    input_video = input_video.to(DEVICE).to(WEIGHT_DTYPE)
    input_video = 2.0 * input_video - 1.0
    
    t_start_total = time.time()
    
    # Encode
    with torch.no_grad():
        init_latents = vae.encode(input_video).latent_dist.sample() * vae.config.scaling_factor
        if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
                init_latents = init_latents - vae.config.shift_factor
    
    # Request
    latents_b64 = encode_tensor(init_latents)
    payload = {
        "latents_b64": latents_b64, "shape": list(init_latents.shape),
        "prompt": PROMPT, "negative_prompt": NEGATIVE_PROMPT,
        "strength": STRENGTH, "steps": steps, "guidance_scale": 6.0, 
        "seed": seed, "cfg_ratio": cfg
    }
    
    t_up = simulate_latency(len(latents_b64), BANDWIDTH_MBPS)
    
    session = requests.Session(); session.trust_env = False 
    t_req_start = time.time()
    resp = session.post(CLOUD_URL, json=payload, timeout=600)
    resp.raise_for_status()
    t_req_dur = time.time() - t_req_start
    data = resp.json()

    cloud_proc = data.get("process_time", 0.0)
    result_b64 = data["result_b64"]
    t_down = simulate_latency(len(result_b64), BANDWIDTH_MBPS)
    
    # Decode
    t_dec_start = time.time()
    latents_out = decode_tensor(result_b64, init_latents.shape)
    with torch.no_grad():
        if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
                latents_out = latents_out + vae.config.shift_factor
        video_out = vae.decode(latents_out / vae.config.scaling_factor).sample
    
    video_out = (video_out / 2.0 + 0.5).clamp(0, 1)
    t_decode = time.time() - t_dec_start
    
    total_latency = time.time() - t_start_total
    
    # Save Generated Video
    video_out_dir = os.path.join(OUTPUT_DIR, video_name)
    if not os.path.exists(video_out_dir): os.makedirs(video_out_dir)
    
    save_filename = f"fps{fps}_steps{steps}_cfg{cfg}_seed{seed}.mp4"
    gen_video_path = os.path.join(video_out_dir, save_filename)
    
    save_videos_grid(video_out.to(dtype=torch.float32).cpu(), gen_video_path, fps=fps)
    if save_video:
        logger.info(f"   💾 Saved: {gen_video_path}") 
    
    # === 7. Metrics Calculation ===
    # Metric 1: CLIP Score (Text & Consistency)
    # calc_clip_score 返回 (text_score, consistency_score)
    clip_text_score, clip_consistency = calc_clip_score(gen_video_path, PROMPT)
    
    # Metric 2: Warp Error (Smoothness)
    warp_error = calc_warp_error(ref_video_path, gen_video_path)
    
    if not save_video:
        try: os.remove(gen_video_path)
        except: pass

    return {
        "video": video_name,
        "fps": fps, "steps": steps, "cfg": cfg, "seed": seed,
        "latency": total_latency, 
        "clip_score": clip_text_score,       # Quality (Semantic)
        "clip_consistency": clip_consistency, # Smoothness (Temporal)
        "warp_error": warp_error             # Smoothness (Motion/Structure)
    }

def run_experiment_batch(video_path, video_name, param_name, param_list, fixed_params):
    raw_results = []        
    logger.info(f"\n=== Experiment: Video={video_name} | Varying {param_name.upper()} ===")
    
    for val in param_list:
        logger.info(f"👉 Testing {param_name}={val}...")
        for i in range(NUM_TRIALS):
            # 保存所有视频
            save_flag = True       
            
            current_seed = START_SEED + i
            args = fixed_params.copy()
            args[param_name] = val
            
            try:
                res = run_single_trial(
                    video_path, video_name, 
                    args['fps'], args['steps'], args['cfg'], 
                    current_seed, save_video=save_flag
                )
                if res: raw_results.append(res)
            except Exception as e:
                logger.error(f"❌ Failed at {param_name}={val}, seed={current_seed}")
                logger.error(f"   Reason: {str(e).splitlines()[-1]}") 
                
    return raw_results

def plot_all_results(df_raw):
    """
    修改后的绘图逻辑：
    1. 按 Seed 分组：每个 Seed 画一张图，对比不同 Video。
    2. 按 Video 分组：每个 Video 画一张图，对比不同 Seed。
    """
    try:
        print("\n📊 Starting Detailed Plotting...")
        
        # 准备工作：确保所有列是数值型
        plot_cols = ['fps', 'steps', 'cfg', 'latency', 'clip_score', 'clip_consistency', 'warp_error']
        for col in plot_cols:
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

        # 辅助筛选函数
        def get_slice(df_subset, param):
            if param == 'fps': 
                return df_subset[(df_subset['steps']==DEFAULT_STEPS) & (np.isclose(df_subset['cfg'], DEFAULT_CFG))].sort_values('fps')
            if param == 'steps':
                return df_subset[(df_subset['fps']==DEFAULT_FPS) & (np.isclose(df_subset['cfg'], DEFAULT_CFG))].sort_values('steps')
            if param == 'cfg':
                return df_subset[(df_subset['fps']==DEFAULT_FPS) & (df_subset['steps']==DEFAULT_STEPS)].sort_values('cfg')

        # 图表参数
        params = ['fps', 'steps', 'cfg']
        metrics = [
            ('latency', 'Latency (s) ↓', 'k'),
            ('clip_score', 'CLIP Text Score ↑', 'purple'), 
            ('warp_error', 'Warp Error ↓', 'r'),
            ('clip_consistency', 'CLIP Consistency ↑', 'g')
        ]
        
        unique_videos = df_raw['video'].unique()
        unique_seeds = df_raw['seed'].unique()
        
        # 配色方案 (用于区分多条线)
        colors = cm.get_cmap('tab10', max(len(unique_videos), len(unique_seeds)) + 1)

        # === 场景 1: 按 Seed 分组 (对比 Video) ===
        print(f"👉 Plotting Group 1: By Seed (Comparing Videos)... Total {len(unique_seeds)} plots.")
        for seed in unique_seeds:
            df_seed = df_raw[df_raw['seed'] == seed]
            if df_seed.empty: continue
            
            fig, axes = plt.subplots(3, 4, figsize=(24, 15))
            fig.suptitle(f"Comparison across Videos (Seed={seed})", fontsize=18)
            
            for row, param in enumerate(params):
                for col, (metric, title, base_color) in enumerate(metrics):
                    ax = axes[row, col]
                    
                    # 遍历每个 Video 画线
                    for i, video in enumerate(unique_videos):
                        df_video_seed = df_seed[df_seed['video'] == video]
                        data = get_slice(df_video_seed, param)
                        
                        if not data.empty and metric in data.columns:
                            ax.plot(data[param], data[metric], marker='o', 
                                    label=f"{video}", linewidth=2, alpha=0.7, color=colors(i))
                    
                    ax.set_title(f"{param.upper()} vs {title}")
                    ax.set_xlabel(param.upper())
                    ax.grid(True, linestyle='--', alpha=0.5)
                    # 只在每行的第一列显示图例，避免太乱
                    if col == 0: ax.legend(fontsize=8)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_name = f"plot_by_seed_{seed}.png"
            plt.savefig(save_name)
            plt.close(fig)
            print(f"   Saved: {save_name}")

        # === 场景 2: 按 Video 分组 (对比 Seed) ===
        print(f"👉 Plotting Group 2: By Video (Comparing Seeds)... Total {len(unique_videos)} plots.")
        for video in unique_videos:
            df_video = df_raw[df_raw['video'] == video]
            if df_video.empty: continue
            
            fig, axes = plt.subplots(3, 4, figsize=(24, 15))
            fig.suptitle(f"Comparison across Seeds (Video={video})", fontsize=18)
            
            for row, param in enumerate(params):
                for col, (metric, title, base_color) in enumerate(metrics):
                    ax = axes[row, col]
                    
                    # 遍历每个 Seed 画线
                    for i, seed in enumerate(unique_seeds):
                        df_video_seed = df_video[df_video['seed'] == seed]
                        data = get_slice(df_video_seed, param)
                        
                        if not data.empty and metric in data.columns:
                            ax.plot(data[param], data[metric], marker='^', 
                                    label=f"Seed {seed}", linewidth=2, alpha=0.7, color=colors(i))
                    
                    ax.set_title(f"{param.upper()} vs {title}")
                    ax.set_xlabel(param.upper())
                    ax.grid(True, linestyle='--', alpha=0.5)
                    if col == 0: ax.legend(fontsize=8)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_name = f"plot_by_video_{video}.png"
            plt.savefig(save_name)
            plt.close(fig)
            print(f"   Saved: {save_name}")

        logger.info("✅ All plots generated successfully.")

    except Exception as e:
        logger.error(f"Plotting Error: {e}")
        import traceback
        traceback.print_exc()

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
        
        video_files = []
        if os.path.exists(INPUT_VIDEO_DIR) and os.path.isdir(INPUT_VIDEO_DIR):
            for f in os.listdir(INPUT_VIDEO_DIR):
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append(os.path.join(INPUT_VIDEO_DIR, f))
        else:
            logger.error(f"❌ Input Directory {INPUT_VIDEO_DIR} not found.")
            return

        logger.info(f"📂 Found {len(video_files)} videos: {[os.path.basename(v) for v in video_files]}")
        
        all_raw_data = []
        fixed = {'fps': DEFAULT_FPS, 'steps': DEFAULT_STEPS, 'cfg': DEFAULT_CFG}
        
        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            logger.info(f"🎬 Processing Video: {video_name}")
            
            # 1. 实验: FPS
            all_raw_data.extend(run_experiment_batch(video_path, video_name, 'fps', FPS_LIST, fixed))
            # 2. 实验: Steps
            all_raw_data.extend(run_experiment_batch(video_path, video_name, 'steps', STEPS_LIST, fixed))
            # 3. 实验: CFG
            all_raw_data.extend(run_experiment_batch(video_path, video_name, 'cfg', CFG_LIST, fixed))

        if not all_raw_data: 
            logger.error("❌ No results generated.")
        else:
            logger.info("💾 Saving All Results...")
            df_raw = pd.DataFrame(all_raw_data)
            df_raw.to_csv("experiment_results_batch_gpu34.csv", index=False)
            plot_all_results(df_raw)
            
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