import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
import os

# ══════════════════════════════════════════════════════════════════
# 1. MODEL ARCHITECTURE (Matches optimalcolloidpython.ipynb exactly)
# ══════════════════════════════════════════════════════════════════
class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class ResBlock(nn.Module):
    def __init__(self, ch, drop=0.10):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.drop  = drop

    def forward(self, x):
        h = self.conv1(F.relu(self.bn1(x)))
        h = F.dropout(h, self.drop, self.training)
        h = self.conv2(F.relu(self.bn2(h)))
        return x + h

class AttentionGate(nn.Module):
    def __init__(self, f_g, f_x, f_int):
        super().__init__()
        self.W_g = nn.Conv2d(f_g, f_int, 1, bias=False)
        self.W_x = nn.Conv2d(f_x, f_int, 1, bias=False)
        self.psi = nn.Conv2d(f_int, 1, 1, bias=True)
        self.bn  = nn.BatchNorm2d(1)

    def forward(self, g, x):
        g_up = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        attn = torch.sigmoid(self.bn(self.psi(F.relu(self.W_g(g_up) + self.W_x(x)))))
        return x * attn

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBnRelu(in_ch, out_ch)
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ) for d in dilations
        ])
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(dilations) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        gap   = F.interpolate(self.gap(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        feats.append(gap)
        return self.project(torch.cat(feats, dim=1))

class OptimalColloidNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(ConvBnRelu(1, 32), ResBlock(32))
        self.enc1 = nn.Sequential(nn.MaxPool2d(2), ConvBnRelu(32,  64),  ResBlock(64))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), ConvBnRelu(64,  128), ResBlock(128), ResBlock(128))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), ConvBnRelu(128, 256), ResBlock(256), ResBlock(256))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), ConvBnRelu(256, 512), ResBlock(512, drop=0.15), ResBlock(512, drop=0.15))
        self.bottleneck = ASPP(512, 512, dilations=[1, 2, 4, 8])
        self.ag4 = AttentionGate(512, 256, 128)
        self.ag3 = AttentionGate(256, 128,  64)
        self.ag2 = AttentionGate(128,  64,  32)
        self.ag1 = AttentionGate( 64,  32,  16)
        self.up4  = UpBlock(512, 256)
        self.dec4 = nn.Sequential(ConvBnRelu(512, 256), ResBlock(256))
        self.up3  = UpBlock(256, 128)
        self.dec3 = nn.Sequential(ConvBnRelu(256, 128), ResBlock(128))
        self.up2  = UpBlock(128, 64)
        self.dec2 = nn.Sequential(ConvBnRelu(128, 64),  ResBlock(64))
        self.up1  = UpBlock(64, 32)
        self.dec1 = nn.Sequential(ConvBnRelu(64, 32),   ResBlock(32))
        self.refine_hmap   = nn.Sequential(ConvBnRelu(32, 32), ResBlock(32), ConvBnRelu(32, 32))
        self.refine_mask   = nn.Sequential(ConvBnRelu(32, 32), ResBlock(32), ConvBnRelu(32, 32))
        self.refine_offset = nn.Sequential(ConvBnRelu(32, 32), ResBlock(32), ConvBnRelu(32, 32))
        self.heatmap_head = nn.Sequential(ConvBnRelu(32, 32), nn.Conv2d(32, 1, 1))
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)
        self.mask_head = nn.Sequential(ConvBnRelu(32, 32), nn.Conv2d(32, 1, 1))
        nn.init.constant_(self.mask_head[-1].bias, -2.19)
        self.offset_head = nn.Sequential(ConvBnRelu(32, 32), nn.Conv2d(32, 2, 1))

    def forward(self, x):
        s  = self.stem(x)
        e1 = self.enc1(s)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bottleneck(e4)
        d4 = self.dec4(torch.cat([self.up4(b),  self.ag4(b,  e3)], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), self.ag3(d4, e2)], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), self.ag2(d3, e1)], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), self.ag1(d2, s)],  1))
        hmap_out   = self.heatmap_head(self.refine_hmap(d1))
        mask_out   = self.mask_head(self.refine_mask(d1))
        offset_out = self.offset_head(self.refine_offset(d1)) 
        return hmap_out, mask_out, offset_out


# ══════════════════════════════════════════════════════════════════
# 2. INFERENCE & DETECTION HELPERS
# ══════════════════════════════════════════════════════════════════
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = device.type == 'cuda'

def infer_tta(model, img_tensor):
    model.eval()
    augmented, aug_params = [], []

    # Apply 8-fold test-time augmentation (rotations and flips)
    for rot_k in (0, 1):
        for hflip in (False, True):
            for vflip in (False, True):
                t = img_tensor.clone()
                if hflip: t = torch.flip(t, dims=[3])
                if vflip: t = torch.flip(t, dims=[2])
                if rot_k > 0: t = torch.rot90(t, rot_k, dims=[2, 3])
                augmented.append(t)
                aug_params.append((rot_k, hflip, vflip))

    batch = torch.cat(augmented, dim=0)
    with torch.no_grad():
        if USE_AMP:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                hp_batch, mp_batch, op_batch = model(batch)
        else:
            hp_batch, mp_batch, op_batch = model(batch)

    hm_batch = torch.sigmoid(hp_batch[:, 0])

    aug_hmaps, aug_offsets = [], []
    for i, (rot_k, hflip, vflip) in enumerate(aug_params):
        hm, off = hm_batch[i], op_batch[i]
        
        # Reverse augmentations to realign predictions
        if rot_k > 0:
            hm = torch.rot90(hm, -rot_k, dims=[0, 1])
            for _ in range(4 - rot_k):
                off = torch.stack([-off[1].clone(), off[0].clone()], dim=0)
            off = torch.rot90(off, -rot_k, dims=[1, 2])
        if vflip:
            hm, off = torch.flip(hm, [0]), torch.flip(off, [1])
            off[0] = -off[0]
        if hflip:
            hm, off = torch.flip(hm, [1]), torch.flip(off, [2])
            off[1] = -off[1]

        aug_hmaps.append(hm)
        aug_offsets.append(off)

    return (torch.stack(aug_hmaps).mean(0).cpu().float().numpy(),
            torch.stack(aug_offsets).mean(0).cpu().float().numpy())

def detect_centers(hmap, offset_map, threshold=0.15, min_dist=3):
    raw_peaks = peak_local_max(hmap, min_distance=min_dist, threshold_abs=threshold, exclude_border=False)
    if len(raw_peaks) == 0:
        return np.empty((0, 2), dtype=np.float32)

    refined = []
    for pk in raw_peaks:
        r, c = pk[0], pk[1]
        dy, dx = offset_map[:, r, c] 
        refined.append([r + dy, c + dx])
    return np.array(refined, dtype=np.float32)

def pad_to_multiple(img, multiple=16):
    h, w = img.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
    return img, h, w

def estimate_optimal_scale(first_frame_gray, target_radius=10.0):
    print("Auto-estimating particle scale from the first frame...")
    blurred = cv2.GaussianBlur(first_frame_gray, (5, 5), 0)
    
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=10, 
        param1=50,  
        param2=25, 
        minRadius=2, 
        maxRadius=150
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        median_radius = np.median(circles[:, 2])
        print(f"  -> Estimated real particle radius: {median_radius}px")
        
        scale_factor = target_radius / median_radius
        scale_factor = np.clip(scale_factor, 0.25, 4.0)
        print(f"  -> Calculated auto-scale factor: {scale_factor:.2f}x")
        return float(scale_factor)
    else:
        print("  -> Could not auto-detect particles. Defaulting to 1.0x scale.")
        return 1.0


# ══════════════════════════════════════════════════════════════════
# 3. MAIN VIDEO PROCESSING LOOP
# ══════════════════════════════════════════════════════════════════
def process_video(video_path, model_weights_path, output_path):
    print(f"Loading model from {model_weights_path}...")
    model = OptimalColloidNet().to(device)
    
    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

    print(f"Processing {total_frames} frames...")
    frame_idx = 0
    scale_factor = 1.0 

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Auto-scaling logic on the first frame
        if frame_idx == 1:
            scale_factor = estimate_optimal_scale(gray, target_radius=10.0)
            
        # Resize input for the Neural Network
        if scale_factor != 1.0:
            new_w = int(gray.shape[1] * scale_factor)
            new_h = int(gray.shape[0] * scale_factor)
            nn_input_gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            nn_input_gray = gray.copy()

        nn_input_gray = nn_input_gray.astype(np.float32)
        f_min, f_max = nn_input_gray.min(), nn_input_gray.max()
        if f_max > f_min:
            nn_input_gray = (nn_input_gray - f_min) / (f_max - f_min)
            
        padded_img, padded_h, padded_w = pad_to_multiple(nn_input_gray, multiple=16)
        
        tensor = torch.from_numpy(padded_img[None, None, ...]).to(device)
        hmap, offset = infer_tta(model, tensor)
        centers = detect_centers(hmap, offset)
        
        annotated_frame = frame.copy()
        for center in centers:
            cy_scaled, cx_scaled = center
            
            # Filter out padding artifacts
            if cy_scaled < padded_h and cx_scaled < padded_w:
                # Scale coordinates back to match original video resolution
                cx_orig = cx_scaled / scale_factor
                cy_orig = cy_scaled / scale_factor
                
                cv2.circle(annotated_frame, (int(round(cx_orig)), int(round(cy_orig))), radius=3, color=(0, 255, 0), thickness=-1)
                cv2.circle(annotated_frame, (int(round(cx_orig)), int(round(cy_orig))), radius=5, color=(0, 100, 0), thickness=1)

        cv2.putText(annotated_frame, f"Particles: {len(centers)} | Scale: {scale_factor:.2f}x", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        out.write(annotated_frame)
        
        if frame_idx % 10 == 0:
            print(f"Processed frame {frame_idx}/{total_frames}")

    cap.release()
    out.release()
    print(f"Done! Output saved to: {output_path}")

# ══════════════════════════════════════════════════════════════════
# EXECUTION
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # Make sure to update these paths before running!
    INPUT_VIDEO  = "sample_microscopy.mp4" 
    MODEL_WEIGHTS = "best_checkpoint.pt"
    OUTPUT_VIDEO = "annotated_output.mp4"
    
    process_video(INPUT_VIDEO, MODEL_WEIGHTS, OUTPUT_VIDEO)