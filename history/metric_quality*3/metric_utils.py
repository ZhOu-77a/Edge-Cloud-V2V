import torch
import torch.nn.functional as F
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

# 独立定义设备，防止循环引用
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================================
# 1. Semantic Quality: Prompt Alignment (CLIP)  # calculate similarity between text and frames
# =========================================================================
def calc_quality_prompt_alignment(video_tensor, prompt, model_id="openai/clip-vit-base-patch32"):
    try:
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
        
        if video_tensor.ndim == 5: video_tensor = video_tensor.squeeze(0)
        video_tensor = video_tensor.permute(1, 0, 2, 3) 
        # if len(video_tensor) > 16: video_tensor = video_tensor[::2]

        inputs_text = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_feat = model.get_text_features(**inputs_text)
            text_feat = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True) # get the norm
        
        # Statistical characteristics
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        
        video_resized = F.interpolate(video_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        video_norm = (video_resized - mean) / std
        
        with torch.no_grad():
            img_feats = model.get_image_features(pixel_values=video_norm)
            img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        
        scores = img_feats @ text_feat.t()
        return scores.mean().item()
    except Exception as e:
        print(f"Error in Semantic Metric: {e}")
        return 0.0

# =========================================================================
# 2. Perceptual Quality: Sharpness (Laplacian)
# =========================================================================
def calc_quality_sharpness(video_tensor):
    try:
        if video_tensor.ndim == 5: video_tensor = video_tensor.squeeze(0)
        if video_tensor.shape[0] == 3: video_tensor = video_tensor.permute(1, 0, 2, 3)
        
        gray = 0.299 * video_tensor[:, 0:1, :, :] + \
               0.587 * video_tensor[:, 1:2, :, :] + \
               0.114 * video_tensor[:, 2:3, :, :]
        
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                              dtype=torch.float32, device=device).view(1, 1, 3, 3)
        
        edges = F.conv2d(gray, kernel, padding=1)
        sharpness_score = edges.var().item()
        
        return sharpness_score * 1000.0
    except Exception as e:
        print(f"Error in Sharpness Metric: {e}")
        return 0.0

# =========================================================================
# 3. Structural Quality: SSIM (Structure Similarity vs Input)
# =========================================================================
def calc_quality_structure_ssim(orig_tensor, gen_tensor):
    """
    计算生成视频与原视频的结构相似度 (SSIM)。
    衡量生成内容是否保留了原视频的布局和结构。
    """
    try:
        # 简化版 SSIM 实现 (PyTorch Native)
        def gaussian_window(window_size, sigma):
            gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
            return gauss/gauss.sum()

        def create_window(window_size, channel):
            _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(channel, 1, window_size, window_size).contiguous().to(device)
            return window

        def ssim(img1, img2, window, window_size=11, channel=3):
            mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
            mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1*mu2
            sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
            sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
            sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
            C1 = 0.01**2
            C2 = 0.03**2
            ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()

        # 数据预处理
        if orig_tensor.ndim == 5: orig_tensor = orig_tensor.squeeze(0)
        if gen_tensor.ndim == 5: gen_tensor = gen_tensor.squeeze(0)
        # Ensure [F, C, H, W]
        if orig_tensor.shape[0] == 3: orig_tensor = orig_tensor.permute(1, 0, 2, 3)
        if gen_tensor.shape[0] == 3: gen_tensor = gen_tensor.permute(1, 0, 2, 3)
        
        # 简单对齐：截取相同长度
        min_len = min(len(orig_tensor), len(gen_tensor))
        orig_t = orig_tensor[:min_len]
        gen_t = gen_tensor[:min_len]
        
        window = create_window(11, 3)
        score = ssim(orig_t, gen_t, window, 11, 3)
        return score.item()

    except Exception as e:
        print(f"Error in Structure Metric: {e}")
        return 0.0

# =========================================================================
# 4. Smoothness: Warp Error
# =========================================================================
def warp(x, flow):
    B, C, H, W = x.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(x.device)
    vgrid = grid + flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, align_corners=True, padding_mode='zeros')
    mask = torch.ones_like(x[:, :1, :, :])
    mask = F.grid_sample(mask, vgrid, align_corners=True, padding_mode='zeros')
    mask[mask < 0.999] = 0
    mask[mask > 0] = 1
    return output, mask

def calc_smoothness_warp_error(orig_video, gen_video):
    try:
        weights = Raft_Large_Weights.DEFAULT
        flow_model = raft_large(weights=weights).to(device).eval()
        transforms = weights.transforms()
        
        if orig_video.ndim == 5: orig_video = orig_video.squeeze(0)
        if gen_video.ndim == 5: gen_video = gen_video.squeeze(0)
        if orig_video.shape[0] == 3: orig_video = orig_video.permute(1, 0, 2, 3)
        if gen_video.shape[0] == 3: gen_video = gen_video.permute(1, 0, 2, 3)

        T = min(len(orig_video), len(gen_video))
        errors = []
        with torch.no_grad():
            for i in range(T - 1):
                img1 = orig_video[i:i+1]
                img2 = orig_video[i+1:i+2]
                img1_t, img2_t = transforms(img1, img2)
                flow = flow_model(img1_t, img2_t)[-1]
                
                gen_t = gen_video[i:i+1]
                gen_t1_target = gen_video[i+1:i+2]
                warped_gen_t, mask = warp(gen_t, flow)
                
                diff = (warped_gen_t - gen_t1_target) * 255.0
                squared_diff = diff ** 2 * mask
                valid_pixels = mask.sum() * 3
                if valid_pixels > 0:
                    mse = squared_diff.sum() / valid_pixels
                    errors.append(mse.item())
        return np.mean(errors) if errors else 0.0
    except Exception as e:
        print(f"Error in Warp: {e}")
        return 0.0