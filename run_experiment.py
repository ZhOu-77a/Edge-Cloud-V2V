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
import imageio.v3 as iio
from PIL import Image
import base64
import gc
import traceback
import logging


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment_debug.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("🚀 Client Experiment Script initialized on GPU 1.")

try:
    from metric_utils import (
        calc_quality_prompt_alignment, 
        calc_quality_sharpness, 
        calc_quality_structure_ssim,
        calc_smoothness_warp_error
    )
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
SERVER_PORT = 12345
SERVER_LOG_FILE = "server.log"
BASE_URL = f"http://127.0.0.1:{SERVER_PORT}"
CLOUD_URL = f"{BASE_URL}/inference"
HEALTH_URL = f"{BASE_URL}/health"

# === 实验变量 ===
INPUT_VIDEO = "asset/inpaint_video.mp4" 
PROMPT = "A cute cat."
NEGATIVE_PROMPT = "The video is not of a high quality, low resolution, watermark, distortion."
MODEL_NAME = "models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP"
if not os.path.exists(MODEL_NAME):
    ABS_MODEL_PATH = "/home/zhoujh/Edge-Cloud-diffusion/CogVideoX-Fun/" + MODEL_NAME
    if os.path.exists(ABS_MODEL_PATH): MODEL_NAME = ABS_MODEL_PATH
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DTYPE = torch.bfloat16

BANDWIDTH_MBPS = 10.0 
FIXED_DURATION = 4.0 
SAMPLE_SIZE = [384, 672] 
STRENGTH = 0.8 

# FPS_LIST = [4, 6, 8, 10, 12]
# STEPS_LIST = [10, 20, 30, 40, 50]
# CFG_LIST = [0.0, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0]

FPS_LIST = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
STEPS_LIST = [3, 5, 8, 10, 15, 20, 30, 40, 50]
CFG_LIST = [0.0, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0]


DEFAULT_FPS = 8
DEFAULT_STEPS = 50
DEFAULT_CFG = 1.0 

NUM_TRIALS = 3
START_SEED = 43 

OUTPUT_DIR = "output_experiment"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
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

def wait_for_server(max_retries=60): # 重试次数到 60
    session = requests.Session(); session.trust_env = False
    logger.info(f"⏳ Waiting for server at {HEALTH_URL} (max {max_retries*2}s)...")
    for i in range(max_retries):
        try:
            if session.get(HEALTH_URL, timeout=2).status_code == 200:
                logger.info("✅ Server Connected and Ready.")
                return True
        except: 
            if i % 5 == 0: # 每5次打印一次
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
    try:
        raw_frames = iio.imread(video_path, plugin="pyav")
        total_frames = len(raw_frames)
        indices = np.linspace(0, total_frames - 1, target_frames).astype(int)
        
        processed_frames = []
        tg_h, tg_w = target_hw
        
        for idx in indices:
            img = Image.fromarray(raw_frames[idx])
            w, h = img.size
            
            # Center Crop Logic
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
    except Exception as e:
        logger.error(f"❌ Video Load Error: {e}")
        return None

def run_single_trial(fps, steps, cfg, seed, save_video=False):
    # 状态标记
    trial_id = f"FPS={fps} Steps={steps} CFG={cfg} Seed={seed}"
    logger.info(f"▶️ Start Trial: {trial_id}")
    
    gc.collect(); torch.cuda.empty_cache()
    load_vae()
    
    # 1. 计算帧数
    raw_target_frames = int(FIXED_DURATION * fps)
    compression = vae.config.temporal_compression_ratio 
    aligned_frames = int((raw_target_frames - 1) // compression * compression) + 1
    if aligned_frames < raw_target_frames: aligned_frames += compression
    
    # 2. 读取视频
    input_video = load_and_process_video(INPUT_VIDEO, aligned_frames, SAMPLE_SIZE)
    if input_video is None:
        logger.error(f"   ❌ SKIP: Input video load failed for {trial_id}")
        return None

    input_video = input_video.to(DEVICE).to(WEIGHT_DTYPE)
    input_video = 2.0 * input_video - 1.0
    
    t_start_total = time.time()
    
    # 3. Encode
    try:
        with torch.no_grad():
            init_latents = vae.encode(input_video).latent_dist.sample() * vae.config.scaling_factor
            if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
                 init_latents = init_latents - vae.config.shift_factor
    except Exception as e:
        logger.error(f"   ❌ Encode Error: {e}")
        return None
    
    # 4. Request
    try:
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
        resp = session.post(CLOUD_URL, json=payload, timeout=600) # 增加超时时间
        resp.raise_for_status()
        t_req_dur = time.time() - t_req_start
        data = resp.json()
    except Exception as e: 
        logger.error(f"   ❌ Cloud Error: {e}")
        return None

    cloud_proc = data.get("process_time", 0.0)
    result_b64 = data["result_b64"]
    t_down = simulate_latency(len(result_b64), BANDWIDTH_MBPS)
    
    # 5. Decode
    try:
        t_dec_start = time.time()
        latents_out = decode_tensor(result_b64, init_latents.shape)
        with torch.no_grad():
            if hasattr(vae.config, "shift_factor") and vae.config.shift_factor is not None:
                 latents_out = latents_out + vae.config.shift_factor
            video_out = vae.decode(latents_out / vae.config.scaling_factor).sample
        
        video_out = (video_out / 2.0 + 0.5).clamp(0, 1)
        t_decode = time.time() - t_dec_start
    except Exception as e:
        logger.error(f"   ❌ Decode Error: {e}")
        return None
    
    total_latency = time.time() - t_start_total
    
    # 6. Save
    if save_video:
        save_filename = f"fps{fps}_steps{steps}_cfg{cfg}_seed{seed}.mp4"
        save_path = os.path.join(OUTPUT_DIR, save_filename)
        try: 
            save_videos_grid(video_out.to(dtype=torch.float32).cpu(), save_path, fps=fps)
            logger.info(f"   💾 Saved: {save_filename}") # 确认保存成功
        except Exception as e:
            logger.error(f"   ❌ Save Error: {e}")
    
    # 7. Metrics
    try:
        gen_tensor = video_out.squeeze(0).float()
        orig_tensor = (input_video.squeeze(0).float() + 1.0) / 2.0 
        
        q_sem = calc_quality_prompt_alignment(gen_tensor, PROMPT)
        q_shp = calc_quality_sharpness(gen_tensor)
        q_str = calc_quality_structure_ssim(orig_tensor, gen_tensor)
        smooth = calc_smoothness_warp_error(orig_tensor, gen_tensor)
    except Exception as e:
        logger.error(f"   ❌ Metrics Error: {e}")
        return None
    
    return {
        "fps": fps, "steps": steps, "cfg": cfg, "seed": seed,
        "latency": total_latency, "q_sem": q_sem, "q_shp": q_shp, "q_str": q_str, "smooth": smooth
    }

def run_experiment_batch(param_name, param_list, fixed_params):
    raw_results = []        
    logger.info(f"\n=== Experiment: Varying {param_name.upper()} ===")
    
    for val in param_list:
        logger.info(f"👉 Testing {param_name}={val}...")
        for i in range(NUM_TRIALS):
            # 强制所有都保存！
            save_flag = True 
            current_seed = START_SEED + i
            args = fixed_params.copy()
            args[param_name] = val
            res = run_single_trial(args['fps'], args['steps'], args['cfg'], current_seed, save_video=save_flag)
            if res: raw_results.append(res)
    return raw_results

def plot_all_results(df_raw):
    try:
        df_avg = df_raw.groupby(['fps', 'steps', 'cfg']).mean().reset_index()
        def normalize(series):
            if series.max() == series.min(): return series
            return (series - series.min()) / (series.max() - series.min())

        df_avg['norm_sem'] = normalize(df_avg['q_sem'])
        df_avg['norm_shp'] = normalize(df_avg['q_shp'])
        df_avg['norm_str'] = normalize(df_avg['q_str'])
        df_avg['quality_composite'] = (df_avg['norm_sem'] + df_avg['norm_shp'] + df_avg['norm_str']) / 3.0
        
        fig1, axes1 = plt.subplots(3, 3, figsize=(18, 15))
        
        def get_slice(param):
            if param == 'fps': 
                return df_avg[(df_avg['steps']==DEFAULT_STEPS) & (np.isclose(df_avg['cfg'], DEFAULT_CFG))].sort_values('fps')
            if param == 'steps':
                return df_avg[(df_avg['fps']==DEFAULT_FPS) & (np.isclose(df_avg['cfg'], DEFAULT_CFG))].sort_values('steps')
            if param == 'cfg':
                return df_avg[(df_avg['fps']==DEFAULT_FPS) & (df_avg['steps']==DEFAULT_STEPS)].sort_values('cfg')

        params = ['fps', 'steps', 'cfg']
        metrics = [
            ('latency', 'Latency (s) ↓', 'k'),
            ('quality_composite', 'Quality (Composite) ↑', 'purple'),
            ('smooth', 'Smoothness (Warp Error) ↓', 'r')
        ]
        
        for row, param in enumerate(params):
            data = get_slice(param)
            for col, (metric, title, color) in enumerate(metrics):
                ax = axes1[row, col]
                if not data.empty:
                    ax.plot(data[param], data[metric], marker='o', color=color, linewidth=2)
                ax.set_title(f"{param.upper()} vs {title}")
                ax.set_xlabel(param.upper())
                ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig("qoe_main_3x3.png")
        
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
        for idx, param in enumerate(params):
            data = get_slice(param)
            ax = axes2[idx]
            if not data.empty:
                ax.plot(data[param], data['norm_sem'], marker='o', label='Semantic', color='g')
                ax.plot(data[param], data['norm_shp'], marker='s', label='Sharpness', color='b')
                ax.plot(data[param], data['norm_str'], marker='^', label='Structure', color='orange')
            ax.set_title(f"Quality Breakdown vs {param.upper()}")
            ax.set_xlabel(param.upper())
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.savefig("qoe_quality_breakdown.png")
        logger.info("📈 Plots saved.")
    except Exception as e:
        logger.error(f"Plotting Error: {e}")

def main():
    server_process = None
    server_started_by_me = False
    
    if not is_port_in_use(SERVER_PORT):
        logger.info(f"☁️ Server not running. Starting local {SERVER_SCRIPT} on GPU 0...")
        # Server 环境,默认0卡
        server_env = os.environ.copy()
        server_env["CUDA_VISIBLE_DEVICES"] = "0" 
        # 启动进程
        try:
            log_file = open(SERVER_LOG_FILE, "w")
            server_process = subprocess.Popen(
                [sys.executable, SERVER_SCRIPT],
                env=server_env,
                stdout=log_file,
                stderr=log_file
            )
            server_started_by_me = True
            logger.info(f"   Server started with PID {server_process.pid}. Logs: {SERVER_LOG_FILE}")
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return

    try:
        # 等待 Server 就绪
        if not wait_for_server(): 
            logger.error("❌ Server start failed or not reachable. Check server.log for details.")
            if server_started_by_me and server_process:
                server_process.terminate()
            return
        
        # === 开始实验 ===
        all_raw_data = []
        fixed = {'fps': DEFAULT_FPS, 'steps': DEFAULT_STEPS, 'cfg': DEFAULT_CFG}
        
        # 1. 实验: FPS
        all_raw_data.extend(run_experiment_batch('fps', FPS_LIST, fixed))
        # 2. 实验: Steps
        all_raw_data.extend(run_experiment_batch('steps', STEPS_LIST, fixed))
        # 3. 实验: CFG
        all_raw_data.extend(run_experiment_batch('cfg', CFG_LIST, fixed))

        if not all_raw_data: 
            logger.error("❌ No results generated.")
        else:
            logger.info("💾 Saving Results...")
            df_raw = pd.DataFrame(all_raw_data)
            df_raw.to_csv("experiment_results_raw.csv", index=False)
            plot_all_results(df_raw)
            
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user.")
    finally:
        # 退出时清理 Server
        if server_started_by_me and server_process:
            logger.info("🛑 Stopping background server...")
            server_process.terminate()
            server_process.wait()
            try: log_file.close()
            except: pass

if __name__ == "__main__":
    main()