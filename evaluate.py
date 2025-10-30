# evaluate.py
# (Bug Fix 7: 'Day (Original)' 평가 시 Enhancement/Reconstruction 로직 건너뛰기)

import torch
import torch.nn as nn # L1Loss 사용
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torch.utils.data import DataLoader, Subset, Dataset # <-- Dataset 임포트 추가
from torch.nn.functional import grid_sample
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange # 모델 출력 처리용
import logging # 로깅 추가
from PIL import Image # <-- PIL 임포트 추가
from torchvision import transforms # <-- transforms 임포트 추가
import time # <-- [개선] 랜덤 시드용 time 임포트

# 사용자 정의 모듈 임포트
from config import cfg
# ▼▼▼ [최신 모델] DCNetStyleVCOD 임포트 ▼▼▼
from models.main_model import DCNetStyleVCOD
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
# from datasets.folder_mask_dataset import FolderImageMaskDataset # <-- 원본 임포트 제거
from utils.py_sod_metrics import SODMetrics
# utils.logger는 별도 설정 없이 기본 logging 사용

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def unnormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """정규화된 이미지 텐서를 원래 이미지로 되돌리는 함수 (run_experiment.py와 동일)"""
    tensor = tensor.clone()
    tensor_cpu = tensor.cpu()
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    tensor_cpu.mul_(std_tensor).add_(mean_tensor)
    tensor_cpu = torch.clamp(tensor_cpu, 0, 1)
    return tensor_cpu

# ▼▼▼ [유지] evaluate.py 전용 데이터셋 클래스 (구조 자동 감지) ▼▼▼
class EvalFolderImageMaskDataset(Dataset):
    """
    evaluate.py 전용 데이터셋 로더.
    1. Root/Species/Imgs (종별 구조)
    2. Root/Imgs (플랫 구조)
    두 가지 데이터셋 구조를 자동으로 감지하여 클립을 생성합니다.
    """
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

        # --- [개선된 로직] 루트 경로 유효성 검사 ---
        if not os.path.exists(self.root_dir):
            logger.error(f"Dataset root directory not found: {self.root_dir}")
            logger.error("Please check the 'eval_folder_data_root' / 'eval_original_data_root' path in your config.py.")
            # Found 0 video clips.
            return # __init__을 여기서 종료
        # --- [개선 완료] ---

        # --- [수정된 로직] 데이터셋 구조 감지 ---
        scan_dirs = []
        # 1. 플랫 구조 (Root/Imgs) 확인
        flat_img_dir = os.path.join(self.root_dir, self.image_folder_name)
        if os.path.isdir(flat_img_dir):
            logger.info("Detected flat dataset structure (e.g., Root/Imgs).")
            scan_dirs.append(self.root_dir)
        # 2. 종별 구조 (Root/Species/Imgs) 확인
        else:
            logger.info("Detected species-based dataset structure (e.g., Root/Species/Imgs).")
            try:
                # 이제 self.root_dir이 존재함이 보장됨
                for sub_dir in sorted(os.listdir(self.root_dir)):
                    sub_dir_path = os.path.join(self.root_dir, sub_dir)
                    # 종별 폴더 안에 Imgs 폴더가 있는지 한 번 더 확인
                    if os.path.isdir(sub_dir_path) and os.path.isdir(os.path.join(sub_dir_path, self.image_folder_name)):
                        scan_dirs.append(sub_dir_path)
            except Exception as e:
                logger.error(f"Error scanning species structure in {self.root_dir}: {e}")
        
        if not scan_dirs:
            logger.warning(f"No valid data directories found in {self.root_dir} (structure may be incorrect).")
        
        # 스캔할 디렉토리 리스트(scan_dirs)를 순회
        for video_root_path in scan_dirs:
            image_dir = os.path.join(video_root_path, self.image_folder_name)
            mask_dir = os.path.join(video_root_path, self.mask_folder_name)

            if os.path.exists(image_dir) and os.path.exists(mask_dir):
                # 프레임 정렬
                frames = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
                
                if len(frames) >= clip_len:
                    # 클립 생성
                    for i in range(len(frames) - clip_len + 1):
                        image_paths = [os.path.join(image_dir, frames[i+j]) for j in range(clip_len)]
                        
                        # 마스크 경로 탐색 (유연성)
                        valid_mask_paths = []
                        all_masks_found = True
                        for j in range(clip_len):
                            base_name = os.path.splitext(frames[i+j])[0]
                            found_mask = None
                            for ext in ['.png', '.jpg', '.jpeg', '.bmp']: # 일반적인 마스크 확장자
                                potential_mask = os.path.join(mask_dir, base_name + ext)
                                if os.path.exists(potential_mask):
                                    found_mask = potential_mask
                                    break
                            if found_mask:
                                valid_mask_paths.append(found_mask)
                            else:
                                all_masks_found = False
                                break # 이 클립은 유효하지 않음
                        
                        if all_masks_found and len(valid_mask_paths) == clip_len:
                            self.clips.append((image_paths, valid_mask_paths))
        # --- [로직 수정 끝] ---

        logger.info(f"Found {len(self.clips)} video clips.")
        
        # Transforms (원본과 동일)
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
            transforms.ToTensor(), # [0, 1] 범위로 변환
        ])
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        # __getitem__ 로직은 두 구조 모두에서 동일하게 작동 (relpath 덕분)
        image_paths, mask_paths = self.clips[idx]
        
        image_clip_tensors = []
        mask_clip_tensors = []
        original_image_clip_tensors = [] 

        apply_flip = self.is_train and self.use_augmentation and random.random() > 0.5
        
        try:
            for img_path, msk_path in zip(image_paths, mask_paths):
                image = Image.open(img_path).convert("RGB")
                mask = Image.open(msk_path).convert("L")
                
                # relpath는 self.root_dir 기준으로 경로를 생성하므로
                # (1) /Root/Species/Imgs -> Species/Imgs
                # (2) /Root/Imgs -> Imgs
                # 두 경우 모두 올바른 상대 경로를 생성합니다.
                relative_path = os.path.relpath(img_path, self.root_dir)
                original_img_path = os.path.join(self.original_data_root, relative_path)
                
                if not os.path.exists(original_img_path):
                    # `eval_original_data_root` 경로도 확인해야 함
                    logger.warning(f"Original day image not found at {original_img_path}. Skipping clip.")
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

        except FileNotFoundError as e:
            logger.warning(f"Warning: File not found, skipping this clip. Details: {e}")
            return None
        except Exception as e:
            logger.warning(f"Warning: Error loading clip {idx}, {img_path}. Details: {e}")
            return None

        if not image_clip_tensors or not mask_clip_tensors or not original_image_clip_tensors:
             return None

        image_clip = torch.stack(image_clip_tensors, dim=0)
        mask_clip = torch.stack(mask_clip_tensors, dim=0)
        original_image_clip = torch.stack(original_image_clip_tensors, dim=0)
        
        return image_clip, mask_clip, original_image_clip
# ▲▲▲ [클래스 추가 끝] ▲▲▲

# ▼▼▼ [개선] run_name 추가, 랜덤 시드 수정 ▼▼▼
def visualize_predictions(model, dataset, device, eval_cfg, common_cfg, run_name="Evaluation"):
    """모델 예측 시각화 (4x5 플롯: 야간/복원/GT마스크/예측마스크)"""
    logger.info(f"--- Generating visualization for {run_name} ---")

    if isinstance(dataset, Subset):
        dataset_to_sample = dataset.dataset
        indices_to_sample_from = dataset.indices
    else:
        dataset_to_sample = dataset
        indices_to_sample_from = list(range(len(dataset)))

    num_samples_available = len(indices_to_sample_from)
    if num_samples_available < 1:
        logger.warning("Dataset empty, skipping visualization.")
        return
    # 시각화 샘플 수를 5개로 고정
    num_samples_to_check = min(5, num_samples_available)

    # --- [개선] 진정한 랜덤 샘플링을 위해 현재 시간으로 시드 설정 ---
    random.seed(int(time.time()))
    # --- [개선 완료] ---
    
    random_subset_indices = random.sample(range(num_samples_available), num_samples_to_check)
    actual_indices = [indices_to_sample_from[i] for i in random_subset_indices]

    fig, axes = plt.subplots(4, num_samples_to_check, figsize=(4 * num_samples_to_check, 16), squeeze=False) # squeeze=False 추가
    fig.suptitle(f'{run_name} Prediction Visualization: Random Samples', fontsize=16)

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(tqdm(actual_indices, desc=f"Visualizing ({run_name})")):
            try:
                data_sample = dataset_to_sample[idx]
            except Exception as e:
                logger.warning(f"Error loading sample {idx} for viz: {e}. Skipping.")
                continue

            if data_sample is None or len(data_sample) != 3:
                logger.warning(f"Invalid data for sample {idx} for viz. Skipping.")
                continue

            video_clip, mask_clip, original_day_clip = data_sample

            if video_clip.nelement() == 0 or mask_clip.nelement() == 0:
                 logger.warning(f"Empty tensor in sample {idx} for viz. Skipping.")
                 continue

            video_clip_batch = video_clip.unsqueeze(0).to(device)

            try:
                # DCNetStyleVCOD도 동일한 (logits, recon) 튜플을 반환
                predicted_logits_seq, reconstructed_images_flat = model(video_clip_batch)
                predicted_mask_seq = torch.sigmoid(predicted_logits_seq)
            except RuntimeError as e: # [개선] OOM 오류를 더 구체적으로 잡기
                if "CUDA out of memory" in str(e):
                    logger.error(f"OOM during visualization for sample {idx}. Skipping sample.")
                else:
                    logger.error(f"Model forward pass failed during visualization for sample {idx}: {e}")
                for row in range(4):
                     axes[row, i].set_title(f"Sample {idx}\n(Error)")
                     axes[row, i].axis('off')
                continue
            except Exception as e:
                logger.error(f"Model forward pass failed during visualization for sample {idx}: {e}")
                for row in range(4):
                     axes[row, i].set_title(f"Sample {idx}\n(Error)")
                     axes[row, i].axis('off')
                continue

            clip_len_actual = video_clip.shape[0]
            frame_idx = min(common_cfg.get('clip_len', 8) // 2, clip_len_actual - 1)
            
            # [개선] run_name에 따라 첫 번째 행 제목 변경
            first_row_title = "Original Night" if "Night" in run_name else "Original Day"

            plot_titles = [first_row_title, "Reconstructed Day", "Ground Truth Mask", "Predicted Mask"]
            tensors_to_plot = [
                video_clip[frame_idx] if frame_idx < video_clip.shape[0] else None,
                None,
                mask_clip[frame_idx] if frame_idx < mask_clip.shape[0] else None,
                predicted_mask_seq.squeeze(0)[frame_idx] if frame_idx < predicted_mask_seq.shape[1] else None
            ]

            use_enhancement_viz = reconstructed_images_flat is not None
            
            # ▼▼▼ [BUG FIX 7] "Night" 평가 시에만 복원 이미지 시각화 ▼▼▼
            if use_enhancement_viz and "Night" in run_name: 
                 try:
                    b_viz, t_viz, c_viz, h_viz, w_viz = video_clip_batch.shape
                    recon_images_seq = reconstructed_images_flat.view(b_viz, t_viz, c_viz, h_viz, w_viz)
                    if frame_idx < recon_images_seq.shape[1]:
                        tensors_to_plot[1] = recon_images_seq.squeeze(0)[frame_idx]
                 except Exception as e_recon_viz:
                    logger.warning(f"Viz sample {idx}: Error processing reconstructed image: {e_recon_viz}")
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            for row, (tensor, title) in enumerate(zip(tensors_to_plot, plot_titles)):
                ax = axes[row, i]
                if tensor is not None and tensor.nelement() > 0:
                    try:
                        if row == 0: # 원본 야간 또는 주간
                            img_np = unnormalize(tensor).numpy().transpose(1, 2, 0)
                            ax.imshow(np.clip(img_np, 0, 1))
                        elif row == 1: # 복원 주간
                            img_np = tensor.cpu().numpy().transpose(1, 2, 0) # 이미 0~1
                            ax.imshow(np.clip(img_np, 0, 1))
                        else: # 마스크 (GT, Pred)
                            img_np = tensor.squeeze().cpu().numpy()
                            ax.imshow(img_np, cmap='gray')

                        full_title = f"Sample {idx}\n{title}" if row == 0 else title
                        ax.set_title(full_title)
                    except Exception as e_plot:
                         ax.set_title(f"Plot Error: {e_plot}")
                else:
                    # ▼▼▼ [BUG FIX 7] "Day" 평가 시 "Not Available"이 올바르게 표시됨 ▼▼▼
                    ax.set_title(f"{title}\n(Not Available)")
                ax.axis('off')

    # [개선] 저장 경로에 run_name 포함
    base_name = os.path.basename(eval_cfg.get('visualization_path', 'evaluation_visualization.png'))
    dir_name = os.path.dirname(eval_cfg.get('visualization_path', 'evaluation_visualization.png'))
    
    # 파일 이름에서 .png 확장자 제거 (있을 경우)
    if base_name.lower().endswith('.png'):
        base_name = base_name[:-4]
        
    new_base_name = f"{run_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{base_name}.png"
    save_path = os.path.join(dir_name, new_base_name)
    
    try:
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            logger.info(f"Created directory for visualization: {dir_name}")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path)
        plt.close(fig)
        logger.info(f"Visualization image saved to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save visualization image to {save_path}: {e}")
# ▲▲▲ [개선 완료] ▲▲▲

# ▼▼▼ [신규] 평가 로직을 별도 함수로 분리 ▼▼▼
# --- [BUG FIX 6] calculate_warping_error 인자 추가 ---
def run_evaluation_pass(run_name, data_root, original_data_root, model, raft_model, raft_transforms, device, eval_cfg, common_cfg, model_config_source, checkpoint_path, calculate_warping_error):
    """
    주어진 데이터 경로에 대해 전체 평가 프로세스를 실행합니다.
    """
    logger.info(f"--- Starting Evaluation Pass: {run_name} ---")
    
    # L1 Loss (Warping, Enhancement 용)
    l1_loss = nn.L1Loss().to(device)

    # --- 데이터셋 및 데이터로더 ---
    try:
        eval_dataset_type = eval_cfg.get('eval_dataset_type', 'folder')
        if eval_dataset_type == 'folder':
            if not data_root or not original_data_root: 
                raise ValueError(f"'{run_name}' pass: 'data_root' ({data_root}) or 'original_data_root' ({original_data_root}) is missing or invalid.")

            logger.info(f"Using EvalFolderImageMaskDataset from: {data_root} & {original_data_root}")
            
            model_input_resolution = model_config_source.get('resolution', (224, 224))
            
            test_dataset = EvalFolderImageMaskDataset(
                root_dir=data_root, # <-- [개선] 함수 인자 사용
                original_data_root=original_data_root, # <-- [개선] 함수 인자 사용
                image_folder_name=eval_cfg.get('eval_image_folder_name', 'Imgs'),
                mask_folder_name=eval_cfg.get('eval_mask_folder_name', 'GT'),
                clip_len=common_cfg.get('clip_len', 8),
                resolution=model_input_resolution, 
                is_train=False, use_augmentation=False
            )
        else: 
            raise ValueError(f"Unsupported eval_dataset_type: {eval_dataset_type}")
        
        if len(test_dataset) == 0: 
            logger.error(f"Eval dataset for '{run_name}' is empty. Check paths and dataset structure.")
            return # 이 평가 패스 중단

    except Exception as e: 
        logger.error(f"Failed to init eval dataset for '{run_name}': {e}"); return

    # collate_fn
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None and isinstance(x,tuple) and len(x)==3, batch))
        if not batch: return None, None, None
        try: return torch.utils.data.dataloader.default_collate(batch)
        except Exception as e: logger.warning(f"Eval collate error ({run_name}), skip batch: {e}"); return None, None, None

    test_loader = DataLoader(
        test_dataset, batch_size=eval_cfg.get('batch_size', 16), shuffle=False,
        num_workers=common_cfg.get('num_workers', 4), pin_memory=True, collate_fn=collate_fn
    )

    # --- 평가 지표 계산 ---
    metrics = SODMetrics()
    total_warping_error = 0.0; temporal_comparison_count = 0
    total_enhancement_loss = 0.0; enhancement_comparison_count = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc=f"Evaluating ({run_name})")
        for batch_data in progress_bar:
            if batch_data[0] is None: continue
            # video_clip은 이제 run_name에 따라 야간 또는 주간 클립이 됨
            video_clip, ground_truth_masks, original_day_images = batch_data
            if video_clip.nelement() == 0 or ground_truth_masks.nelement() == 0: continue

            video_clip = video_clip.to(device, non_blocking=True)
            ground_truth_masks = ground_truth_masks.to(device, non_blocking=True)
            original_day_images = original_day_images.to(device, non_blocking=True)
            b, t, c, h, w = video_clip.shape

            try:
                predicted_logits_seq, reconstructed_images_flat = model(video_clip)
                predicted_masks_seq = torch.sigmoid(predicted_logits_seq)
            except RuntimeError as e: # [개선] OOM 오류를 더 구체적으로 잡기
                if "CUDA out of memory" in str(e):
                    logger.error(f"Eval ({run_name}): Model forward failed (CUDA OOM). Skip batch.")
                    torch.cuda.empty_cache() # 캐시 비우기
                    continue # 다음 배치로
                else:
                    logger.error(f"Eval ({run_name}): Model forward failed: {e}. Skip batch."); continue
            except Exception as e: 
                logger.error(f"Eval ({run_name}): Model forward failed: {e}. Skip batch."); continue

            try:
                # 1. SOD Metrics
                predicted_masks_flat_eval = rearrange(predicted_masks_seq, 'b t c h w -> (b t) c h w')
                masks_flat_target_eval = ground_truth_masks.view(b*t, 1, h, w)
                
                preds_np_batch = (predicted_masks_flat_eval.cpu().numpy() * 255).astype(np.uint8)
                gts_np_batch = (masks_flat_target_eval.cpu().numpy() > 0.5).astype(np.uint8) * 255

                for i in range(b*t):
                    pred_mask_uint8 = preds_np_batch[i].squeeze()
                    gt_mask_uint8 = gts_np_batch[i].squeeze()
                    if gt_mask_uint8.max() > 0: 
                        metrics.step(pred=pred_mask_uint8, gt=gt_mask_uint8)

                # 2. Enhancement L1 Loss
                use_enhancement_eval = reconstructed_images_flat is not None and eval_cfg.get('calculate_enhancement_loss', True)
                
                # ▼▼▼ [BUG FIX 7] "Night" 평가 시에만 Enhancement Loss 계산 ▼▼▼
                if use_enhancement_eval and "Night" in run_name:
                    original_images_flat_target_eval = original_day_images.view(b*t, c, h, w)
                    loss_enh_batch = l1_loss(reconstructed_images_flat, original_images_flat_target_eval).item()
                    if not np.isnan(loss_enh_batch) and not np.isinf(loss_enh_batch):
                         total_enhancement_loss += loss_enh_batch * (b*t)
                         enhancement_comparison_count += (b*t)
                # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

                # 3. Warping Error (입력 비디오(video_clip) 기준)
                # --- [BUG FIX 6] calculate_warping_error 변수 사용 ---
                if t > 1 and calculate_warping_error and raft_model is not None:
                    try:
                        img1_batch_we = video_clip[:, :-1].reshape(-1, c, h, w)
                        img2_batch_we = video_clip[:, 1:].reshape(-1, c, h, w)
                        if not torch.equal(img1_batch_we, img2_batch_we):
                            img1_tf_we, img2_tf_we = raft_transforms(img1_batch_we, img2_batch_we)
                            with torch.no_grad():
                                flows_we = raft_model(img1_tf_we.contiguous(), img2_tf_we.contiguous())[-1]
                            
                            flows_rs_we = F.interpolate(flows_we, size=(h, w), mode='bilinear', align_corners=False)
                            flows_un_we = flows_rs_we.view(b, t - 1, 2, h, w)
                            
                            batch_temporal_error_sum = 0.0; valid_comps = 0
                            for frame_idx in range(t - 1):
                                flow_i_we = flows_un_we[:, frame_idx]
                                grid_y_we, grid_x_we = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
                                grid_we = torch.stack((grid_x_we, grid_y_we), 2).float()
                                disp_we = flow_i_we.permute(0, 2, 3, 1)
                                w_grid_we = grid_we + disp_we
                                w_grid_we[..., 0]=2.0*w_grid_we[..., 0]/(w-1)-1.0
                                w_grid_we[..., 1]=2.0*w_grid_we[..., 1]/(h-1)-1.0
                                
                                mask_t_s = predicted_masks_seq[:, frame_idx]
                                mask_tp1_s = predicted_masks_seq[:, frame_idx+1]
                                mask_t_w = grid_sample(mask_t_s, w_grid_we, mode='bilinear', padding_mode='border', align_corners=False)
                                
                                step_err = l1_loss(mask_t_w, mask_tp1_s).item()
                                
                                if not np.isnan(step_err) and not np.isinf(step_err):
                                     batch_temporal_error_sum += step_err * b 
                                     valid_comps += b
                                     
                            if valid_comps > 0: 
                                total_warping_error += batch_temporal_error_sum
                                temporal_comparison_count += valid_comps
                                
                    except RuntimeError as e_we: # [개선] OOM 오류 잡기
                         if "CUDA out of memory" in str(e_we):
                             logger.warning(f"Eval ({run_name}): Warping Error calc failed (CUDA OOM). Skip for batch.")
                             torch.cuda.empty_cache() # 캐시 비우기
                         else:
                             logger.warning(f"Eval ({run_name}): Warping Error calc failed: {e_we}. Skip for batch.")
                    except Exception as e_we: 
                        logger.warning(f"Eval ({run_name}): Warping Error calc failed: {e_we}. Skip for batch.")

            except Exception as e_metric: 
                logger.error(f"Eval ({run_name}): Metric calculation error: {e_metric}. Skip batch metrics.")

    # --- 최종 결과 ---
    results = metrics.get_results()
    avg_warping_error = total_warping_error / temporal_comparison_count if temporal_comparison_count > 0 else float('nan')
    avg_enhancement_loss = total_enhancement_loss / enhancement_comparison_count if enhancement_comparison_count > 0 else float('nan')

    logger.info(f"\n--- Evaluation Results ({run_name}) ---")
    # --- [BUG FIX 5] checkpoint_path 사용 ---
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Dataset: {data_root}")
    logger.info(f"S-measure (Sm) ↑:           {results.get('Sm', float('nan')):.4f}")
    logger.info(f"E-measure (Em) ↑:           {results.get('Em', float('nan')):.4f}")
    logger.info(f"Weighted F-measure (wFm) ↑: {results.get('wFm', float('nan')):.4f}")
    logger.info(f"Mean Absolute Error (MAE) ↓:{results.get('MAE', float('nan')):.4f}")
    logger.info(f"Adaptive F-measure (adpFm)↑:{results.get('adpFm', float('nan')):.4f}")
    logger.info(f"Max F-measure (F-beta) ↑:   {results.get('F-beta', float('nan')):.4f}")
    logger.info(f"Warping Error (Temporal) ↓: {avg_warping_error:.4f}")
    # ▼▼▼ [BUG FIX 7] "Day" 평가 시 nan이 올바르게 표시됨 ▼▼▼
    logger.info(f"Enhancement L1 Loss ↓:      {avg_enhancement_loss:.4f}")
    logger.info("--------------------------")

    # 시각화
    if eval_cfg.get('generate_visualization', True) and len(test_dataset) > 0:
        visualize_predictions(model, test_dataset, device, eval_cfg, common_cfg, run_name=run_name)
# ▲▲▲ [함수 분리 끝] ▲▲▲


def main():
    common_cfg = cfg.common
    eval_cfg = cfg.evaluate
    try:
        train_cfg_temp = cfg.train
    except AttributeError:
        logger.warning("cfg.train not found in config.py. Using default values for model init fallback.")
        train_cfg_temp = {} # 빈 딕셔너리로 대체

    # GPU 설정
    gpu_id = common_cfg.get('gpu_ids', '0').split(',')[0]
    if torch.cuda.is_available():
         device = torch.device(f"cuda:{gpu_id}")
    else:
         device = torch.device("cpu")
         logger.warning("CUDA not available, running on CPU.")

    logger.info("--- Starting Evaluation ---")
    logger.info(f"Using device: {device}")

    # ▼▼▼ checkpoint_path를 여기서 정의 ▼▼▼
    checkpoint_path = eval_cfg.get('checkpoint_path')
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
         logger.error(f"Checkpoint path '{checkpoint_path}' not found or invalid.")
         return

    model_config_source = None
    checkpoint = None
    try:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        # weights_only=False가 중요 (config 객체를 로드하기 위해)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # 모델 아키텍처 결정을 위해 config 로드
        if 'config' in checkpoint and checkpoint['config']:
            logger.info("Config found in checkpoint. Initializing model from checkpoint's config.")
            # config.py의 Config 클래스 객체에서 .train 속성을 가져옵니다.
            # 이 속성은 딕셔너리입니다.
            model_config_source = checkpoint['config'].train 
        else:
            logger.warning("No 'config' key found in checkpoint. Falling back to current config.py settings.")
            model_config_source = train_cfg_temp # 현재 config.py의 train 설정을 사용
    
    except ImportError as e:
         logger.error(f"ImportError loading checkpoint: {e}. Check if required classes (like config.Config) are defined.")
         logger.warning("Falling back to current config.py settings for model init.")
         model_config_source = train_cfg_temp # 현재 config.py의 train 설정을 사용
         # 체크포인트 객체는 있으나 config 로드 실패 시, state_dict만이라도 로드 시도
         if checkpoint is None:
             checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True) # weights_only=True로 재시도
    except Exception as e:
        logger.error(f"Failed to load checkpoint '{checkpoint_path}': {e}. Aborting.")
        return
    
    if checkpoint is None:
        logger.error("Checkpoint could not be loaded. Aborting.")
        return
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    # ▼▼▼ [수정된 로직] 결정된 config로 모델 초기화 ▼▼▼
    try:
        # run_experiment.py와 동일한 기본값 사용
        # [버그 수정] getattr() -> .get() 으로 변경
        model = DCNetStyleVCOD(
            backbone_name=model_config_source.get('backbone_name', 'swin_small_patch4_window7_224'),
            input_size=model_config_source.get('resolution', (224, 224)),
            num_frames=common_cfg.get('clip_len', 8), # num_frames는 common_cfg에서
            pretrained=False, # 평가 시에는 체크포인트 로드하므로 False
            gru_hidden_dim=model_config_source.get('gru_hidden_dim', 128), 
            decoder_channel=model_config_source.get('decoder_channel', 64), # <-- .get()으로 수정
            # ▼▼▼ [BUG FIX 7] 모델 초기화 시에는 Enhancement Head를 항상 켜도록 수정 ▼▼▼
            # (체크포인트가 Head 가중치를 가지고 있으므로, 로드 시 필요)
            use_enhancement=True # eval_cfg.get('calculate_enhancement_loss', True) 
        ).to(device)
        logger.info(f"Model '{type(model).__name__}' created for evaluation.")
        
        # --- 모델 상태 로드 ---
        model_state_dict = checkpoint.get('model_state_dict')
        if not model_state_dict:
             logger.warning("Checkpoint key 'model_state_dict' not found. Trying to load entire object.")
             model_state_dict = checkpoint # 전체 객체가 state_dict일 수 있음
        
        # strict=True로 설정하여 아키텍처가 정말 일치하는지 확인
        load_result = model.load_state_dict(model_state_dict, strict=True) 
        
        logger.info(f"Checkpoint load result: {load_result}")
        if load_result.missing_keys: logger.warning(f"Missing keys during load: {load_result.missing_keys}")
        if load_result.unexpected_keys: logger.warning(f"Unexpected keys during load: {load_result.unexpected_keys}")

        model.eval()
        logger.info("Checkpoint loaded successfully.")

    except Exception as e:
        logger.error(f"Failed to create model or load state_dict: {e}")
        logger.error("This often means the checkpoint architecture (even if config was loaded) does not match the current model definition.")
        return
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    # RAFT 모델 (Warping Error용)
    raft_model = None; raft_transforms = None
    # --- [BUG FIX 6] calculate_warping_error를 여기서 정의 ---
    calculate_warping_error = eval_cfg.get('calculate_warping_error', True)
    if calculate_warping_error:
        try:
            raft_weights = Raft_Large_Weights.DEFAULT; raft_transforms = raft_weights.transforms()
            raft_model = raft_large(weights=raft_weights, progress=False).to(device); raft_model.eval()
            logger.info("RAFT model loaded.")
        except Exception as e: 
            logger.error(f"Failed to load RAFT model: {e}. Warping Error disabled.")
            calculate_warping_error = False # <-- 실패 시 False로 설정
    else: 
        logger.info("Warping Error calculation disabled.")

    # ▼▼▼ [개선] main 함수의 평가 로직을 2회 호출로 변경 ▼▼▼
    
    # --- Run 1: Night Evaluation ---
    run_evaluation_pass(
        run_name="Night",
        data_root=eval_cfg.get('eval_folder_data_root'),
        original_data_root=eval_cfg.get('eval_original_data_root'),
        model=model,
        raft_model=raft_model,
        raft_transforms=raft_transforms,
        device=device,
        eval_cfg=eval_cfg,
        common_cfg=common_cfg,
        model_config_source=model_config_source,
        checkpoint_path=checkpoint_path, # <-- 전달
        calculate_warping_error=calculate_warping_error # <-- [BUG FIX 6] 전달
    )

    logger.info("="*50)

    # --- Run 2: Day (Original) Evaluation ---
    run_evaluation_pass(
        run_name="Day (Original)",
        data_root=eval_cfg.get('eval_original_data_root'), # Day images as input
        original_data_root=eval_cfg.get('eval_original_data_root'), # Day images as reference
        model=model,
        raft_model=raft_model,
        raft_transforms=raft_transforms,
        device=device,
        eval_cfg=eval_cfg,
        common_cfg=common_cfg,
        model_config_source=model_config_source,
        checkpoint_path=checkpoint_path, # <-- 전달
        calculate_warping_error=calculate_warping_error # <-- [BUG FIX 6] 전달
    )
    # ▲▲▲ [개선 완료] ▲▲▲

if __name__ == '__main__':
    main()
