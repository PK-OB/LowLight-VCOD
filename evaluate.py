# evaluate.py
# (기존 기능 100% 유지 + 특징 맵 시각화 추가)

import torch
import torch.nn as nn
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torch.utils.data import DataLoader, Subset, Dataset
from torch.nn.functional import grid_sample
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
import logging
from PIL import Image
from torchvision import transforms
import time
import cv2  # [추가] 시각화용

# 사용자 정의 모듈 임포트
from config import cfg
from models.main_model import DCNetStyleVCOD
from utils.py_sod_metrics import SODMetrics

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- [추가] 특징 맵 캡처를 위한 전역 변수 및 Hook 함수 ---
captured_feat = None
def hook_fn(module, input, output):
    global captured_feat
    # output shape: (B*T, C, H, W)
    captured_feat = output.detach().cpu()
# -----------------------------------------------------

def unnormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """정규화된 이미지 텐서를 원래 이미지로 되돌리는 함수"""
    tensor = tensor.clone()
    tensor_cpu = tensor.cpu()
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    tensor_cpu.mul_(std_tensor).add_(mean_tensor)
    tensor_cpu = torch.clamp(tensor_cpu, 0, 1)
    return tensor_cpu

# ▼▼▼ [유지] 기존의 꼼꼼한 데이터셋 클래스 (변경 없음) ▼▼▼
class EvalFolderImageMaskDataset(Dataset):
    def __init__(self, root_dir, original_data_root, image_folder_name, mask_folder_name, clip_len=8, resolution=(224, 224), is_train=True, use_augmentation=True):
        self.root_dir = root_dir
        self.original_data_root = original_data_root
        self.clip_len = clip_len
        self.resolution = resolution
        self.is_train = is_train 
        self.use_augmentation = use_augmentation
        self.image_folder_name = image_folder_name
        self.mask_folder_name = mask_folder_name
        
        self.clips = []
        logger.info(f"Scanning dataset in '{root_dir}' to create video clips...")

        if not os.path.exists(self.root_dir):
            logger.error(f"Dataset root directory not found: {self.root_dir}")
            return 

        scan_dirs = []
        flat_img_dir = os.path.join(self.root_dir, self.image_folder_name)
        if os.path.isdir(flat_img_dir):
            logger.info("Detected flat dataset structure.")
            scan_dirs.append(self.root_dir)
        else:
            logger.info("Detected species-based dataset structure.")
            try:
                for sub_dir in sorted(os.listdir(self.root_dir)):
                    sub_dir_path = os.path.join(self.root_dir, sub_dir)
                    if os.path.isdir(sub_dir_path) and os.path.isdir(os.path.join(sub_dir_path, self.image_folder_name)):
                        scan_dirs.append(sub_dir_path)
            except Exception as e:
                logger.error(f"Error scanning species structure: {e}")
        
        if not scan_dirs:
            logger.warning(f"No valid data directories found in {self.root_dir}")
        
        for video_root_path in scan_dirs:
            image_dir = os.path.join(video_root_path, self.image_folder_name)
            mask_dir = os.path.join(video_root_path, self.mask_folder_name)

            if os.path.exists(image_dir) and os.path.exists(mask_dir):
                frames = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
                
                if len(frames) >= clip_len:
                    for i in range(len(frames) - clip_len + 1):
                        image_paths = [os.path.join(image_dir, frames[i+j]) for j in range(clip_len)]
                        valid_mask_paths = []
                        all_masks_found = True
                        for j in range(clip_len):
                            base_name = os.path.splitext(frames[i+j])[0]
                            found_mask = None
                            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                                potential_mask = os.path.join(mask_dir, base_name + ext)
                                if os.path.exists(potential_mask):
                                    found_mask = potential_mask
                                    break
                            if found_mask:
                                valid_mask_paths.append(found_mask)
                            else:
                                all_masks_found = False
                                break
                        
                        if all_masks_found and len(valid_mask_paths) == clip_len:
                            self.clips.append((image_paths, valid_mask_paths))

        logger.info(f"Found {len(self.clips)} video clips.")
        
        self.image_transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        self.original_image_transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(), 
        ])
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        image_paths, mask_paths = self.clips[idx]
        image_clip_tensors = []
        mask_clip_tensors = []
        original_image_clip_tensors = [] 
        apply_flip = self.is_train and self.use_augmentation and random.random() > 0.5
        
        try:
            for img_path, msk_path in zip(image_paths, mask_paths):
                image = Image.open(img_path).convert("RGB")
                mask = Image.open(msk_path).convert("L")
                
                relative_path = os.path.relpath(img_path, self.root_dir)
                original_img_path = os.path.join(self.original_data_root, relative_path)
                
                if not os.path.exists(original_img_path):
                    return None
                
                original_image = Image.open(original_img_path).convert("RGB")
                
                if self.is_train and self.use_augmentation:
                    if apply_flip:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)
                        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                        original_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)
                    image = self.color_jitter(image)

                image_clip_tensors.append(self.image_transform(image))
                mask_clip_tensors.append(self.mask_transform(mask))
                original_image_clip_tensors.append(self.original_image_transform(original_image))

        except Exception: return None

        if not image_clip_tensors: return None
        return torch.stack(image_clip_tensors), torch.stack(mask_clip_tensors), torch.stack(original_image_clip_tensors)
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# ▼▼▼ [수정됨] 특징 맵 포함 6행 시각화 함수 ▼▼▼
def visualize_predictions(model, dataset, device, eval_cfg, common_cfg, run_name="Evaluation"):
    logger.info(f"--- Generating visualization for {run_name} ---")

    # Hook 등록 (중요)
    hook_handle = model.decoder.register_forward_hook(hook_fn)

    if isinstance(dataset, Subset):
        dataset_to_sample = dataset.dataset
        indices_to_sample_from = dataset.indices
    else:
        dataset_to_sample = dataset
        indices_to_sample_from = list(range(len(dataset)))

    num_samples_available = len(indices_to_sample_from)
    if num_samples_available < 1: return

    num_samples_to_check = min(5, num_samples_available)
    random.seed(int(time.time()))
    actual_indices = [indices_to_sample_from[i] for i in random.sample(range(num_samples_available), num_samples_to_check)]

    # [변경] 6행(Rows)으로 확장 (입력, 특징맵, 주간, 복원, GT마스크, 예측마스크)
    fig, axes = plt.subplots(6, num_samples_to_check, figsize=(4 * num_samples_to_check, 24), squeeze=False) 
    fig.suptitle(f'{run_name} Prediction Visualization', fontsize=16)

    model.eval()
    
    with torch.no_grad():
        for i, idx in enumerate(tqdm(actual_indices, desc=f"Visualizing ({run_name})")):
            try:
                clip_image_paths, _ = dataset_to_sample.clips[idx]
                logger.info(f"  Sample {i+1}: {clip_image_paths[0]}")
            except: pass

            data_sample = dataset_to_sample[idx]
            if data_sample is None: continue

            video_clip, mask_clip, original_day_clip = data_sample
            video_clip_batch = video_clip.unsqueeze(0).to(device)

            try:
                # 추론 실행 (이때 Hook이 작동하여 captured_feat에 특징맵 저장됨)
                predicted_logits_seq, reconstructed_images_flat = model(video_clip_batch)
                predicted_mask_seq = torch.sigmoid(predicted_logits_seq)
            except Exception as e:
                logger.error(f"Inference failed for sample {idx}: {e}")
                continue

            # 시각화할 프레임 인덱스 (중간 프레임)
            frame_idx = min(common_cfg.get('clip_len', 8) // 2, video_clip.shape[0] - 1)
            
            # --- 특징 맵 처리 (Heatmap 생성) ---
            heatmap_vis = None
            if captured_feat is not None:
                # captured_feat: (B*T, C, H, W) -> (B, T, C, H, W)
                b_sz, t_sz = video_clip_batch.shape[:2]
                feat_reshaped = rearrange(captured_feat, '(b t) c h w -> b t c h w', b=b_sz, t=t_sz)
                
                # 해당 프레임의 특징 맵 추출
                feat = feat_reshaped[0, frame_idx] # (C, H, W)
                
                # 평균 -> 정규화 -> 컬러맵
                heatmap = torch.mean(feat, dim=0).numpy()
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                
                # 원본 크기로 리사이즈 (H, W)
                target_h, target_w = video_clip.shape[-2:]
                heatmap_resized = cv2.resize(heatmap, (target_w, target_h))
                
                # JET Colormap 적용
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                heatmap_vis = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) # RGB로 변환
            # --------------------------------

            first_row_title = "Original Night" if "Night" in run_name else "Original Day"

            plot_titles = [
                first_row_title, 
                "Shared Features",       # [New]
                "Original Day (GT)", 
                "Reconstructed Day", 
                "Ground Truth Mask", 
                "Predicted Mask"
            ]
            
            tensors_to_plot = [
                video_clip[frame_idx],          # 0: Input
                heatmap_vis,                    # 1: Feature Map (numpy)
                original_day_clip[frame_idx],   # 2: GT Day
                None,                           # 3: Recon Day
                mask_clip[frame_idx],           # 4: GT Mask
                predicted_mask_seq.squeeze(0)[frame_idx] # 5: Pred Mask
            ]

            if reconstructed_images_flat is not None and "Night" in run_name:
                b, t, c, h, w = video_clip_batch.shape
                recon_seq = reconstructed_images_flat.view(b, t, c, h, w)
                tensors_to_plot[3] = recon_seq.squeeze(0)[frame_idx]

            for row, (data, title) in enumerate(zip(tensors_to_plot, plot_titles)):
                ax = axes[row, i]
                if data is not None:
                    try:
                        if row == 0: # Input (Normalize 복원)
                            img_np = unnormalize(data).numpy().transpose(1, 2, 0)
                            ax.imshow(np.clip(img_np, 0, 1))
                        elif row == 1: # Feature Map (RGB Numpy)
                            ax.imshow(data) 
                        elif row == 2 or row == 3: # Day / Recon
                            img_np = data.cpu().numpy().transpose(1, 2, 0)
                            ax.imshow(np.clip(img_np, 0, 1))
                        else: # Masks
                            img_np = data.squeeze().cpu().numpy()
                            ax.imshow(img_np, cmap='gray')
                        
                        ax.set_title(title if row != 0 else f"Sample {idx}\n{title}")
                    except Exception: ax.set_title("Error")
                else:
                    ax.set_title("N/A")
                ax.axis('off')

    hook_handle.remove() # Hook 제거

    base_name = os.path.basename(eval_cfg.get('visualization_path', 'viz.png'))
    if base_name.lower().endswith('.png'): base_name = base_name[:-4]
    
    dir_name = os.path.dirname(eval_cfg.get('visualization_path', 'viz.png'))
    os.makedirs(dir_name, exist_ok=True)
    
    save_name = f"{run_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{base_name}_feat.png"
    save_path = os.path.join(dir_name, save_name)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved visualization to: {save_path}")
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# ▼▼▼ [유지] 기존의 꼼꼼한 평가 로직 (OOM 처리 등 포함) ▼▼▼
def run_evaluation_pass(run_name, data_root, original_data_root, model, raft_model, raft_transforms, device, eval_cfg, common_cfg, model_config_source, checkpoint_path, calculate_warping_error):
    logger.info(f"--- Starting Evaluation Pass: {run_name} ---")
    l1_loss = nn.L1Loss().to(device)

    try:
        model_input_resolution = model_config_source.get('resolution', (224, 224))
        test_dataset = EvalFolderImageMaskDataset(
            root_dir=data_root, 
            original_data_root=original_data_root, 
            image_folder_name=eval_cfg.get('eval_image_folder_name', 'Imgs'),
            mask_folder_name=eval_cfg.get('eval_mask_folder_name', 'GT'),
            clip_len=common_cfg.get('clip_len', 8),
            resolution=model_input_resolution, 
            is_train=False, use_augmentation=False
        )
        if len(test_dataset) == 0: return
    except Exception as e: logger.error(f"Dataset init failed: {e}"); return

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None and isinstance(x,tuple) and len(x)==3, batch))
        return torch.utils.data.dataloader.default_collate(batch) if batch else (None,)*3

    test_loader = DataLoader(test_dataset, batch_size=eval_cfg.get('batch_size', 16), shuffle=False, num_workers=common_cfg.get('num_workers', 4), collate_fn=collate_fn)
    
    # [FIX] Metrics 초기화
    metrics = SODMetrics()
    metrics.reset()  # 명시적 초기화
    
    total_we = 0.0; cnt_we = 0; total_enh = 0.0; cnt_enh = 0

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc=f"Evaluating ({run_name})"):
            if batch_data[0] is None: continue
            video_clip, gt_masks, orig_day = [x.to(device, non_blocking=True) for x in batch_data]
            b, t, c, h, w = video_clip.shape

            try:
                logits, recon = model(video_clip)
                preds = torch.sigmoid(logits)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.error("OOM error. Skipping batch."); torch.cuda.empty_cache(); continue
                continue
            except: continue

            # 1. Metrics - [FIX] 데이터 형식 수정
            preds_flat = rearrange(preds, 'b t c h w -> (b t) c h w')
            gt_flat = gt_masks.view(b*t, 1, h, w)
            
            # [수정] squeeze(1)로 채널 차원 제거, GT 이진화 제거
            preds_np = (preds_flat.squeeze(1).cpu().numpy() * 255).astype(np.uint8)  # (B*T, H, W)
            gts_np = (gt_flat.squeeze(1).cpu().numpy() * 255).astype(np.uint8)  # (B*T, H, W)
            
            # [디버그] 첫 배치만 로깅
            if not hasattr(run_evaluation_pass, '_debug_logged'):
                logger.info(f"[DEBUG] Pred shape: {preds_np.shape}, dtype: {preds_np.dtype}, range: [{preds_np.min()}, {preds_np.max()}]")
                logger.info(f"[DEBUG] GT shape: {gts_np.shape}, dtype: {gts_np.dtype}, unique values: {np.unique(gts_np)}")
                logger.info(f"[DEBUG] Sample 0 - Pred mean: {preds_np[0].mean():.4f}, GT mean: {gts_np[0].mean():.4f}")
                run_evaluation_pass._debug_logged = True
            
            # [수정] 빈 GT 체크 제거 - metric 내부에서 처리
            for i in range(b*t):
                try:
                    metrics.step(pred=preds_np[i], gt=gts_np[i])
                except Exception as e:
                    logger.warning(f"Metric calculation failed for sample {i}: {e}")

            # 2. Enhancement Loss (Night only)
            if recon is not None and "Night" in run_name:
                loss = l1_loss(recon, orig_day.view(b*t, c, h, w)).item()
                if not np.isnan(loss): total_enh += loss * (b*t); cnt_enh += (b*t)

            # 3. Warping Error
            if t > 1 and calculate_warping_error and raft_model:
                try:
                    img1 = video_clip[:, :-1].reshape(-1, c, h, w)
                    img2 = video_clip[:, 1:].reshape(-1, c, h, w)
                    if not torch.equal(img1, img2):
                        tf_img1, tf_img2 = raft_transforms(img1, img2)
                        flows = raft_model(tf_img1, tf_img2)[-1]
                        flows = F.interpolate(flows, (h, w), mode='bilinear', align_corners=False)
                        
                        # 복잡한 Grid Sample 로직 (기존과 동일)
                        for i in range(t-1):
                            grid = torch.stack(torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')[::-1], 2).float()
                            disp = flows.view(b, t-1, 2, h, w)[:, i].permute(0,2,3,1)
                            grid_w = (grid + disp).mul(torch.tensor([2/(w-1), 2/(h-1)], device=device)).sub(1)
                            warped = grid_sample(preds[:, i], grid_w, align_corners=False)
                            loss_we = l1_loss(warped, preds[:, i+1]).item()
                            if not np.isnan(loss_we): total_we += loss_we * b; cnt_we += b
                except: pass

    res = metrics.get_results()
    avg_we = total_we/cnt_we if cnt_we else float('nan')
    avg_enh = total_enh/cnt_enh if cnt_enh else float('nan')

    logger.info(f"\n--- Results ({run_name}) ---")
    logger.info(f"Sm: {res.get('Sm', 0):.4f}, MAE: {res.get('MAE', 0):.4f}, WE: {avg_we:.4f}, EnhL1: {avg_enh:.4f}")
    
    if eval_cfg.get('generate_visualization', True):
        visualize_predictions(model, test_dataset, device, eval_cfg, common_cfg, run_name)

def main():
    common_cfg = cfg.common; eval_cfg = cfg.evaluate
    device = torch.device(f"cuda:{common_cfg.get('gpu_ids', '0').split(',')[0]}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    ckpt_path = eval_cfg.get('checkpoint_path')
    if not os.path.isfile(ckpt_path): return
    
    logger.info(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Auto-detect Enhancement Head
    conf_src = ckpt.get('config', {}).train if ckpt.get('config') else cfg.train
    state_dict = ckpt.get('model_state_dict', ckpt)
    has_head = any('enhance_head' in k for k in state_dict.keys())
    logger.info(f"Auto-detected enhancement head: {has_head}")
    
    model = DCNetStyleVCOD(
        backbone_name=conf_src.get('backbone_name', 'swin_small_patch4_window7_224'),
        input_size=conf_src.get('resolution', (224, 224)),
        num_frames=common_cfg.get('clip_len', 8),
        pretrained=False,
        gru_hidden_dim=conf_src.get('gru_hidden_dim', 128),
        decoder_channel=conf_src.get('decoder_channel', 64),
        use_enhancement=has_head 
    ).to(device)
    
    # 키 불일치 방지
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    raft_model = None; raft_tf = None
    if eval_cfg.get('calculate_warping_error', True):
        try:
            w = Raft_Large_Weights.DEFAULT
            raft_model = raft_large(weights=w, progress=False).to(device).eval()
            raft_tf = w.transforms()
        except: pass

    run_evaluation_pass("Night", eval_cfg.get('eval_folder_data_root'), eval_cfg.get('eval_original_data_root'), 
                        model, raft_model, raft_tf, device, eval_cfg, common_cfg, conf_src, ckpt_path, True)
    
    run_evaluation_pass("Day (Original)", eval_cfg.get('eval_original_data_root'), eval_cfg.get('eval_original_data_root'),
                        model, raft_model, raft_tf, device, eval_cfg, common_cfg, conf_src, ckpt_path, True)

if __name__ == '__main__':
    main()