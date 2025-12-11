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

# === 1. 核心配置 ===
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print(f"🚀 Client Experiment Script initialized on GPU 1.", flush=True)

try:
    # 【核心修复】这里使用 metric_utils.py 中定义的新函数名
    from metric_utils import calc_quality_prompt_alignment, calc_smoothness_warp_error, calc_quality_sharpness
    print("📦 Metrics loaded successfully.", flush=True)
except ImportError as e:
    print(f"❌ Metrics Import Error: {e}")
    print("   请检查 metric_utils.py 是否包含了 calc_quality_prompt_alignment, calc_smoothness_warp_error, calc_quality_sharpness")
    sys.exit(1)

# === 项目配置 ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT) if PROJECT_ROOT not in sys.path else None

from videox_fun.models import AutoencoderKLCogVideoX
from videox_fun.utils.utils import save_videos_grid

# 指定 Server 文件名
SERVER_SCRIPT = "cloud_server.py"
SERVER_PORT = 12345
SERVER_LOG_FILE = "server.log"

BASE_URL = f"http://127.0.0.1:{SERVER_PORT}"
CLOUD_URL = f"{BASE_URL}/inference"
HEALTH_URL = f"{BASE_URL}/health"

if not os.path.exists(SERVER_SCRIPT):
    print(f"❌ 错误: 找不到 {SERVER_SCRIPT}")
    sys.exit(1)

# INPUT_VIDEO = "asset/building.mp4"
# PROMPT = "A building in cartoon style."
INPUT_VIDEO = "asset/inpaint_video.mp4" 
PROMPT = "A cute cat."
NEGATIVE_PROMPT = "The video is not of a high quality, low resolution, watermark, distortion."

MODEL_NAME = "models/Diffusion_Transformer/CogVideoX-Fun-V1.1-2b-InP"
if not os.path.exists(MODEL_NAME):
    ABS_MODEL_PATH = "/home/zhoujh/Edge-Cloud-diffusion/CogVideoX-Fun/" + MODEL_NAME
    if os.path.exists(ABS_MODEL_PATH):
        MODEL_NAME = ABS_MODEL_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DTYPE = torch.bfloat16

# === 实验变量 ===
BANDWIDTH_MBPS = 10.0 
FIXED_DURATION = 3.0 
FPS_LIST = [4, 5, 6, 7, 8, 9, 10, 11, 12]
STEPS_LIST = [10, 15, 20, 25, 30, 35, 40, 45, 50]
DEFAULT_FPS = 8
DEFAULT_STEPS = 50

OUTPUT_DIR = "output_experiment"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

vae = None

def load_vae():
    global vae
    if vae is None:
        print(f"📦 Loading Client VAE...", flush=True)
        try:
            vae = AutoencoderKLCogVideoX.from_pretrained(MODEL_NAME, subfolder="vae").to(DEVICE).to(WEIGHT_DTYPE)
        except Exception as e:
            print(f"❌ Failed to load VAE: {e}")
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

def wait_for_server(max_retries=600):
    print("⏳ Waiting for server to be ready...", end="", flush=True)
    session = requests.Session()
    session.trust_env = False
    
    start_time = time.time()
    for i in range(max_retries):
        try:
            resp = session.get(HEALTH_URL, timeout=1)
            if resp.status_code == 200:
                print(f"\n✅ Server is ready! (Took {time.time()-start_time:.1f}s)", flush=True)
                return True
        except:
            pass
        
        if i % 2 == 0:
            print(".", end="", flush=True)
        time.sleep(1)
            
    print("\n❌ Timeout waiting for server!", flush=True)
    print("-" * 20 + " SERVER LOG TAIL " + "-" * 20)
    if os.path.exists(SERVER_LOG_FILE):
        os.system(f"tail -n 20 {SERVER_LOG_FILE}")
    print("-" * 50)
    return False

def simulate_latency(data_bytes, bandwidth_mbps, direction="Upload"):
    bits = data_bytes * 8
    bandwidth_bps = bandwidth_mbps * 1_000_000
    latency = bits / bandwidth_bps
    final_latency = max(0, latency * np.random.uniform(0.95, 1.05))
    time.sleep(final_latency)
    return final_latency

def run_single_trial(fps, steps):
    print(f"\n🧪 Trial: FPS={fps}, Steps={steps}", flush=True)
    load_vae()
    
    target_frames = int(FIXED_DURATION * fps) 
    compression = vae.config.temporal_compression_ratio 
    aligned_frames = int((target_frames - 1) // compression * compression) + 1
    if aligned_frames < target_frames: aligned_frames += compression
    
    if not os.path.exists(INPUT_VIDEO): return None

    try:
        raw_frames = iio.imread(INPUT_VIDEO, plugin="pyav") 
    except Exception as e:
        print(f"❌ Error reading video: {e}")
        return None

    total_orig_frames = len(raw_frames)
    indices = np.linspace(0, total_orig_frames-1, aligned_frames).astype(int)
    sampled_frames = raw_frames[indices]
    
    processed_frames = []
    for f in sampled_frames:
        img = Image.fromarray(f).resize((672, 384))
        processed_frames.append(np.array(img))
    
    video_np = np.stack(processed_frames).astype(np.float32) / 255.0
    video_np = video_np.transpose(0, 3, 1, 2) 
    input_video = torch.from_numpy(video_np).unsqueeze(0).permute(0, 2, 1, 3, 4).to(DEVICE).to(WEIGHT_DTYPE)
    input_video = 2.0 * input_video - 1.0
    
    t_start_total = time.time()
    
    # Encode
    t_enc_start = time.time()
    with torch.no_grad():
        init_latents = vae.encode(input_video).latent_dist.sample()
        init_latents = init_latents * vae.config.scaling_factor
    t_encode = time.time() - t_enc_start
    
    # Request
    latents_b64 = encode_tensor(init_latents)
    payload = {
        "latents_b64": latents_b64,
        "shape": list(init_latents.shape),
        "prompt": PROMPT, "negative_prompt": NEGATIVE_PROMPT,
        "strength": 0.8, "steps": steps, "guidance_scale": 6.0, "seed": 43
    }
    
    # Upload Sim
    t_up = simulate_latency(len(latents_b64), BANDWIDTH_MBPS, "Upload")
    
    try:
        session = requests.Session(); session.trust_env = False 
        t_req_start = time.time()
        resp = session.post(CLOUD_URL, json=payload); resp.raise_for_status()
        t_req_dur = time.time() - t_req_start
        data = resp.json()
    except Exception as e: print(f"❌ Cloud Error: {e}"); return None

    cloud_proc = data.get("process_time", 0.0)
    
    # Download Sim
    result_b64 = data["result_b64"]
    t_down = simulate_latency(len(result_b64), BANDWIDTH_MBPS, "Download")
    actual_rtt = max(0.001, t_req_dur - cloud_proc)
    trans_time = t_up + t_down + actual_rtt

    # Decode
    t_dec_start = time.time()
    latents_out = decode_tensor(result_b64, init_latents.shape)
    with torch.no_grad():
        video_out = vae.decode(latents_out / vae.config.scaling_factor).sample
    video_out = (video_out / 2.0 + 0.5).clamp(0, 1)
    t_decode = time.time() - t_dec_start
    
    # Save
    save_filename = f"fps_{fps}_steps_{steps}.mp4"
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    try:
        video_to_save = video_out.to(dtype=torch.float32).cpu()
        save_videos_grid(video_to_save, save_path, fps=fps)
    except: pass
    
    total_latency = time.time() - t_start_total
    edge_proc_time = t_encode + t_decode
    
    # === Metrics ===
    gen_tensor = video_out.squeeze(0).float()
    orig_tensor = (input_video.squeeze(0).float() + 1.0) / 2.0
    
    # 【核心修复】使用新的函数名调用 metric_utils
    quality_semantic = calc_quality_prompt_alignment(gen_tensor, PROMPT)
    quality_sharpness = calc_quality_sharpness(gen_tensor) 
    smoothness_warp = calc_smoothness_warp_error(orig_tensor, gen_tensor)
    
    print(f"   ⏱️ Total: {total_latency:.2f}s | Cloud: {cloud_proc:.2f}s")
    print(f"   📊 Qual(Sem): {quality_semantic:.4f} | 👁️ Qual(Sharp): {quality_sharpness:.2f} | 📉 Smooth: {smoothness_warp:.2f}", flush=True)
    
    return {
        "fps": fps, "steps": steps,
        "latency": total_latency,
        "quality_semantic": quality_semantic,
        "quality_sharpness": quality_sharpness,
        "smoothness": smoothness_warp
    }

def main():
    server_process = None
    server_started_by_me = False 

    if is_port_in_use(SERVER_PORT):
        print(f"ℹ️ Server running on port {SERVER_PORT}...", flush=True)
    else:
        print(f"☁️ Starting Server...", flush=True)
        server_env = os.environ.copy(); server_env["CUDA_VISIBLE_DEVICES"] = "0"
        log_file = open(SERVER_LOG_FILE, "w")
        server_process = subprocess.Popen([sys.executable, SERVER_SCRIPT], env=server_env, stdout=log_file, stderr=log_file)
        server_started_by_me = True

    try:
        if not wait_for_server(): return
        
        results = []
        for fps in FPS_LIST:
            res = run_single_trial(fps=fps, steps=DEFAULT_STEPS)
            if res: results.append(res)
        for steps in STEPS_LIST:
            if steps == DEFAULT_STEPS and DEFAULT_FPS in FPS_LIST: continue
            res = run_single_trial(fps=DEFAULT_FPS, steps=steps)
            if res: results.append(res)
            
    except KeyboardInterrupt: pass
    finally:
        if server_started_by_me and server_process:
            server_process.terminate(); server_process.wait()
            try: log_file.close()
            except: pass

    if not results: return

    # === Plotting ===
    df = pd.DataFrame(results)
    df.to_csv("experiment_results_final.csv", index=False)
    
    df_fps = df[df['steps'] == DEFAULT_STEPS].sort_values('fps')
    df_steps = df[df['fps'] == DEFAULT_FPS].sort_values('steps')
    
    # 2 行 4 列: Latency, Semantic, Sharpness, Smoothness
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    
    # Helper to plot
    def plot_metric(row_idx, df_data, x_col, y_col, title, color):
        ax = axes[row_idx]
        if y_col == 'latency': idx = 0
        elif y_col == 'quality_semantic': idx = 1
        elif y_col == 'quality_sharpness': idx = 2
        elif y_col == 'smoothness': idx = 3
        
        ax[idx].plot(df_data[x_col], df_data[y_col], marker='o', color=color, linestyle='-')
        ax[idx].set_title(title)
        ax[idx].set_xlabel(x_col.upper())
        ax[idx].grid(True)

    # Row 1: Varying FPS
    plot_metric(0, df_fps, 'fps', 'latency', 'FPS vs Latency (Lower Better)', 'k')
    plot_metric(0, df_fps, 'fps', 'quality_semantic', 'FPS vs Semantic (Higher Better)', 'g')
    plot_metric(0, df_fps, 'fps', 'quality_sharpness', 'FPS vs Sharpness (Higher Better)', 'b')
    plot_metric(0, df_fps, 'fps', 'smoothness', 'FPS vs Smoothness (Lower Better)', 'r')

    # Row 2: Varying Steps
    plot_metric(1, df_steps, 'steps', 'latency', 'Steps vs Latency (Lower Better)', 'k')
    plot_metric(1, df_steps, 'steps', 'quality_semantic', 'Steps vs Semantic (Higher Better)', 'g')
    plot_metric(1, df_steps, 'steps', 'quality_sharpness', 'Steps vs Sharpness (Higher Better)', 'b')
    plot_metric(1, df_steps, 'steps', 'smoothness', 'Steps vs Smoothness (Lower Better)', 'r')
    
    plt.tight_layout()
    plt.savefig("qoe_optimization_curves_final.png")
    print("📈 Evaluation Finished! Check qoe_optimization_curves_final.png")

if __name__ == "__main__":
    main()