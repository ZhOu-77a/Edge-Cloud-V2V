import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import torchvision.transforms as T
from torchvision.io import read_video

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================================
# 1. CLIP Score (Source: StreamV2V clip_score.py)
# =========================================================================
def calc_clip_score(video_path, prompt, model_id="openai/clip-vit-base-patch32"):
    try:
        # Load Model
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        video_embs = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return 0.0, 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # CV2 BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                
                with torch.no_grad():
                    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    # StreamV2V logic: collect image embeddings
                    video_embs.append(outputs.image_embeds)
                    # Text embedding is recalculated every frame in reference, but is constant.
                    text_embeds = outputs.text_embeds
            else:
                break
        cap.release()

        if not video_embs: return 0.0, 0.0

        video_embs = torch.cat(video_embs, dim=0) # [T, 512]

        # Metric A: Text Alignment Score (Prompt Score)
        # Reference: text_score = cos(text_embeds, video_embs)
        text_score = cos(text_embeds, video_embs).mean().cpu().item()

        # Metric B: Temporal Consistency Score
        # Reference: cos(emb1, emb2)
        if len(video_embs) > 1:
            emb1 = video_embs[:-1]
            emb2 = video_embs[1:]
            consistency_score = cos(emb1, emb2).mean().cpu().item()
        else:
            consistency_score = 1.0

        return text_score, consistency_score

    except Exception as e:
        print(f"Error in CLIP Score: {e}")
        return 0.0, 0.0

# =========================================================================
# 2. Warp Error (Source: StreamV2V warp_error.py)
# =========================================================================

def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij') # [H, W]
    stacks = [x, y]
    if homogeneous:
        ones = torch.ones_like(x)
        stacks.append(ones)
    grid = torch.stack(stacks, dim=0).float() # [2, H, W]
    grid = grid[None].repeat(b, 1, 1, 1) # [B, 2, H, W]
    if device is not None:
        grid = grid.to(device)
    return grid

def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    # img: [B, C, H, W]
    if sample_coords.size(1) != 2:
        sample_coords = sample_coords.permute(0, 3, 1, 2)
    b, _, h, w = sample_coords.shape
    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1
    grid = torch.stack([x_grid, y_grid], dim=-1) # [B, H, W, 2]
    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1) # [B, H, W]
        return img, mask
    return img

def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()
    grid = coords_grid(b, h, w, device=flow.device) + flow  # [B, 2, H, W]
    return bilinear_sample(feature, grid, padding_mode=padding_mode, return_mask=mask)

def forward_backward_consistency_check(fwd_flow, bwd_flow, alpha=0.01, beta=0.5):
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1) 
    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow) 
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow) 
    
    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1) 
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)
    
    threshold = alpha * flow_mag + beta
    
    fwd_occ = (diff_fwd > threshold).float() 
    bwd_occ = (diff_bwd > threshold).float()
    
    return fwd_occ, bwd_occ

def preprocess_raft(batch):
    transforms = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
    ])
    return transforms(batch)

def calculate_error(frame1, frame2, mask):

    
    frame1_norm = frame1 
    frame2_norm = frame2
    # mask = mask.numpy().astype(np.uint8) # 这里传入的 mask 已经是 numpy uint8
    
    pixels_to_consider = (mask == 0) # 0 means valid
    
    error = np.abs(frame1_norm - frame2_norm)[pixels_to_consider].mean()

    return error

def calc_warp_error(ref_video_path, edit_video_path):
    try:
        # Load RAFT
        model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
        
        # Read Videos [T, H, W, C] (0-255 uint8)
        # Permute to [T, C, H, W] for torch processing
        ref_frames, _, _ = read_video(str(ref_video_path), output_format="TCHW")
        edit_frames, _, _ = read_video(str(edit_video_path), output_format="TCHW")
        
        # Resize Edit to match Ref height/width (StreamV2V logic)
        ref_height, ref_width = ref_frames.shape[2], ref_frames.shape[3]
        # Reference uses interpolate
        if ref_frames.shape[2:] != edit_frames.shape[2:]:
            edit_frames = F.interpolate(edit_frames.float(), size=(ref_height, ref_width), mode='bilinear', align_corners=False).to(torch.uint8)

        # Align Length
        num_frames = min(ref_frames.shape[0], edit_frames.shape[0])
        ref_frames = ref_frames[:num_frames]
        edit_frames = edit_frames[:num_frames]
        
        error_list = []
        for i in range(num_frames - 1):
            img1 = ref_frames[i]
            img2 = ref_frames[i+1]
            
            # RAFT Input: [Batch, C, H, W] normalized
            # To fix OOM: Process batch sequentially instead of stacking 2 pairs
            img1_batch = img1.unsqueeze(0)
            img2_batch = img2.unsqueeze(0)
            
            fwd_batch = preprocess_raft(img1_batch).to(device) # img1 -> img2
            bwd_batch = preprocess_raft(img2_batch).to(device) # img2 -> img1
            
            with torch.no_grad():
                # Forward Flow
                fwd_flow = model(fwd_batch, bwd_batch)[-1] 
                # Backward Flow
                bwd_flow = model(bwd_batch, fwd_batch)[-1]
            
            h, w = fwd_flow.shape[2:]
            
            # Consistency Check
            fwd_occ, bwd_occ = forward_backward_consistency_check(fwd_flow, bwd_flow)
            
            # Prepare Images for Warping (numpy uint8 [H, W, C])
            # Reference casts to uint8 explicitly
            edit_image_1 = edit_frames[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            edit_image_2 = edit_frames[i+1].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            
            # Flow: [H, W, 2]
            flow_np = fwd_flow[0].permute(1, 2, 0).cpu().numpy()
            
            # Grid
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            
            # Warping (cv2.remap)
            map_x = (grid_x + flow_np[..., 0]).astype(np.float32)
            map_y = (grid_y + flow_np[..., 1]).astype(np.float32)
            
            warped_image = cv2.remap(edit_image_1, map_x, map_y, cv2.INTER_LINEAR)
            
            # Mask
            occlusion_mask = fwd_occ[0].cpu().numpy().astype(np.uint8)
            
            # Error Calculation (Logic restored to match StreamV2V reference)
            err = calculate_error(warped_image, edit_image_2, occlusion_mask)
            error_list.append(err)
            
        return np.mean(error_list) if error_list else 0.0

    except Exception as e:
        print(f"Error in Warp Error: {e}")
        # import traceback
        # traceback.print_exc()
        return 0.0