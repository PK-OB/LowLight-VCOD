# evaluate.py
# (최신 DCNetStyleVCOD 모델 아키텍처에 맞게 수정된 버전)

import torch
import torch.nn as nn # L1Loss 사용
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torch.utils.data import DataLoader, Subset # Subset 타입 확인용
from torch.nn.functional import grid_sample
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange # 모델 출력 처리용
import logging # 로깅 추가

# 사용자 정의 모듈 임포트
from config import cfg
# ▼▼▼ [최신 모델] DCNetStyleVCOD 임포트 ▼▼▼
from models.main_model import DCNetStyleVCOD
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
from datasets.folder_mask_dataset import FolderImageMaskDataset # Multi-Task 데이터셋
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

def visualize_predictions(model, dataset, device, eval_cfg, common_cfg):
    """모델 예측 시각화 (4x5 플롯: 야간/복원/GT마스크/예측마스크)"""
    logger.info("--- Generating visualization ---")

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
    num_samples_to_check = min(20, num_samples_available)

    random_subset_indices = random.sample(range(num_samples_available), num_samples_to_check)
    actual_indices = [indices_to_sample_from[i] for i in random_subset_indices]

    fig, axes = plt.subplots(4, num_samples_to_check, figsize=(4 * num_samples_to_check, 16), squeeze=False) # squeeze=False 추가
    fig.suptitle('Prediction Visualization: Random Samples', fontsize=16)

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(tqdm(actual_indices, desc="Visualizing")):
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
            except Exception as e:
                logger.error(f"Model forward pass failed during visualization for sample {idx}: {e}")
                for row in range(4):
                     axes[row, i].set_title(f"Sample {idx}\n(Error)")
                     axes[row, i].axis('off')
                continue

            clip_len_actual = video_clip.shape[0]
            frame_idx = min(common_cfg.get('clip_len', 8) // 2, clip_len_actual - 1)

            plot_titles = ["Original Night", "Reconstructed Day", "Ground Truth Mask", "Predicted Mask"]
            tensors_to_plot = [
                video_clip[frame_idx] if frame_idx < video_clip.shape[0] else None,
                None,
                mask_clip[frame_idx] if frame_idx < mask_clip.shape[0] else None,
                predicted_mask_seq.squeeze(0)[frame_idx] if frame_idx < predicted_mask_seq.shape[1] else None
            ]

            use_enhancement_viz = reconstructed_images_flat is not None
            if use_enhancement_viz:
                 try:
                    b_viz, t_viz, c_viz, h_viz, w_viz = video_clip_batch.shape
                    recon_images_seq = reconstructed_images_flat.view(b_viz, t_viz, c_viz, h_viz, w_viz)
                    if frame_idx < recon_images_seq.shape[1]:
                        tensors_to_plot[1] = recon_images_seq.squeeze(0)[frame_idx]
                 except Exception as e_recon_viz:
                    logger.warning(f"Viz sample {idx}: Error processing reconstructed image: {e_recon_viz}")

            for row, (tensor, title) in enumerate(zip(tensors_to_plot, plot_titles)):
                ax = axes[row, i]
                if tensor is not None and tensor.nelement() > 0:
                    try:
                        if row == 0: # 원본 야간
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
                    ax.set_title(f"{title}\n(Not Available)")
                ax.axis('off')

    save_path = eval_cfg.get('visualization_path', 'evaluation_visualization.png')
    save_dir = os.path.dirname(save_path)
    try:
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Created directory for visualization: {save_dir}")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path)
        plt.close(fig)
        logger.info(f"Visualization image saved to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save visualization image to {save_path}: {e}")


def main():
    common_cfg = cfg.common
    eval_cfg = cfg.evaluate
    # 학습 시 사용된 설정도 일부 필요 (모델 초기화 등)
    try:
        # train_cfg에서 모델 하이퍼파라미터를 읽어와야 함
        train_cfg_temp = cfg.train
    except AttributeError:
        logger.warning("cfg.train not found in config.py. Using default values for model init.")
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

    # ▼▼▼ [최신 모델] DCNetStyleVCOD 초기화 로직 ▼▼▼
    try:
        model = DCNetStyleVCOD(
            backbone_name=train_cfg_temp.get('backbone_name', 'swin_small_patch4_window7_224'),
            input_size=train_cfg_temp.get('resolution', (224, 224)),
            num_frames=common_cfg.get('clip_len', 8),
            pretrained=False, # 평가 시에는 체크포인트 로드
            # DCNetStyleVCOD에 필요한 파라미터 (cfg.train에서 읽어옴)
            gru_hidden_dim=train_cfg_temp.get('gru_hidden_dim', 128),
            decoder_channel=train_cfg_temp.get('decoder_channel', 64),
            use_enhancement=eval_cfg.get('calculate_enhancement_loss', True)
        ).to(device)
        logger.info(f"Model '{type(model).__name__}' created for evaluation.")
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        return
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    # 체크포인트 로드
    checkpoint_path = eval_cfg.get('checkpoint_path')
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
         logger.error(f"Checkpoint path '{checkpoint_path}' not found or invalid.")
         return

    try:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        # weights_only=False로 설정하여 객체(config)도 로드 시도 (권장)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model_state_dict = checkpoint.get('model_state_dict')
        if not model_state_dict:
             logger.warning("Checkpoint key 'model_state_dict' not found. Trying to load entire object.")
             model_state_dict = checkpoint
        
        # ▼▼▼ [중요] 체크포인트의 config와 현재 config 비교 ▼▼▼
        # 체크포인트에 config가 저장되어 있는지 확인
        if 'config' in checkpoint and checkpoint['config']:
            logger.info("Config found in checkpoint. Verifying parameters...")
            ckpt_train_cfg = checkpoint['config'].train
            
            # 모델 구조에 영향을 주는 중요 파라미터 비교
            mismatch = False
            params_to_check = ['backbone_name', 'gru_hidden_dim', 'decoder_channel']
            for param in params_to_check:
                ckpt_val = ckpt_train_cfg.get(param)
                curr_val = train_cfg_temp.get(param)
                # 기본값 처리 (config에 명시되지 않았을 경우)
                if param == 'gru_hidden_dim' and curr_val is None: curr_val = 128
                if param == 'decoder_channel' and curr_val is None: curr_val = 64
                
                if ckpt_val != curr_val:
                    logger.warning(f"Config mismatch! Param: '{param}' | Checkpoint: {ckpt_val} | Current Cfg: {curr_val}")
                    mismatch = True
                    
            if mismatch:
                 logger.error("Model structure mismatch detected based on config. ABORTING.")
                 logger.error("Please ensure your config.py matches the one used for training, OR")
                 logger.error("use a checkpoint that matches the current config.py settings.")
                 return # *** 중요: 불일치 시 중단 ***
            else:
                 logger.info("Config parameters match. Proceeding with loading.")
        else:
            logger.warning("No 'config' key found in checkpoint. Cannot verify parameters. Trying to load weights anyway...")
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # 모델 상태 로드 (strict=True로 변경하여 정확한 일치 확인)
        # 만약 위 config 검증 로직을 사용한다면, strict=True가 더 안전합니다.
        # 만약 config 검증 없이 유연하게 로드하려면 strict=False를 사용하세요.
        load_result = model.load_state_dict(model_state_dict, strict=True) 
        
        logger.info(f"Checkpoint load result: {load_result}")
        if load_result.missing_keys: logger.warning(f"Missing keys during load: {load_result.missing_keys}")
        if load_result.unexpected_keys: logger.warning(f"Unexpected keys during load: {load_result.unexpected_keys}")

        model.eval()
        logger.info("Checkpoint loaded successfully.")
    except ImportError as e:
         logger.error(f"ImportError loading checkpoint: {e}. Check if required classes (like config.Config) are defined and importable.")
         return
    except Exception as e:
        # strict=True로 인해 발생하는 size mismatch 오류가 여기서 잡힙니다.
        logger.error(f"Failed to load checkpoint '{checkpoint_path}': {e}")
        logger.error("This often means the checkpoint architecture does not match the current model definition OR config.py settings.")
        return

    # RAFT 모델 (Warping Error용)
    raft_model = None; raft_transforms = None
    calculate_warping_error = eval_cfg.get('calculate_warping_error', True)
    if calculate_warping_error:
        try:
            raft_weights = Raft_Large_Weights.DEFAULT; raft_transforms = raft_weights.transforms()
            raft_model = raft_large(weights=raft_weights, progress=False).to(device); raft_model.eval()
            logger.info("RAFT model loaded.")
        except Exception as e: logger.error(f"Failed to load RAFT model: {e}. Warping Error disabled."); calculate_warping_error = False
    else: logger.info("Warping Error calculation disabled.")
    l1_loss = nn.L1Loss().to(device) # .to(device) 추가

    # --- 데이터셋 및 데이터로더 ---
    try:
        eval_dataset_type = eval_cfg.get('eval_dataset_type', 'folder')
        if eval_dataset_type == 'folder':
            eval_folder_root = eval_cfg.get('eval_folder_data_root')
            eval_original_root = eval_cfg.get('eval_original_data_root')
            if not eval_folder_root or not eval_original_root: raise ValueError("'eval_folder_data_root' or 'eval_original_data_root' missing.")

            logger.info(f"Using FolderDataset from: {eval_folder_root} & {eval_original_root}")
            test_dataset = FolderImageMaskDataset(
                root_dir=eval_folder_root, original_data_root=eval_original_root,
                image_folder_name=eval_cfg.get('eval_image_folder_name', 'Imgs'),
                mask_folder_name=eval_cfg.get('eval_mask_folder_name', 'GT'),
                clip_len=common_cfg.get('clip_len', 8),
                # 해상도도 cfg.train에서 읽어오도록 통일
                resolution=train_cfg_temp.get('resolution', (224, 224)),
                is_train=False, use_augmentation=False
            )
        else: raise ValueError(f"Unsupported eval_dataset_type: {eval_dataset_type}")
        if len(test_dataset) == 0: raise ValueError("Eval dataset is empty.")
    except Exception as e: logger.error(f"Failed to init eval dataset: {e}"); return

    # collate_fn
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None and isinstance(x,tuple) and len(x)==3, batch))
        if not batch: return None, None, None
        try: return torch.utils.data.dataloader.default_collate(batch)
        except Exception as e: logger.warning(f"Eval collate error, skip batch: {e}"); return None, None, None

    test_loader = DataLoader(
        test_dataset, batch_size=eval_cfg.get('batch_size', 16), shuffle=False,
        num_workers=common_cfg.get('num_workers', 4), pin_memory=True, collate_fn=collate_fn
    )

    # --- 평가 지표 계산 ---
    metrics = SODMetrics()
    total_warping_error = 0.0; temporal_comparison_count = 0
    total_enhancement_loss = 0.0; enhancement_comparison_count = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating")
        for batch_data in progress_bar:
            if batch_data[0] is None: continue
            video_clip, ground_truth_masks, original_day_images = batch_data
            if video_clip.nelement() == 0 or ground_truth_masks.nelement() == 0: continue

            video_clip = video_clip.to(device, non_blocking=True)
            ground_truth_masks = ground_truth_masks.to(device, non_blocking=True)
            original_day_images = original_day_images.to(device, non_blocking=True)
            b, t, c, h, w = video_clip.shape

            try:
                predicted_logits_seq, reconstructed_images_flat = model(video_clip)
                predicted_masks_seq = torch.sigmoid(predicted_logits_seq)
            except Exception as e: logger.error(f"Eval: Model forward failed: {e}. Skip batch."); continue

            try:
                # 1. SOD Metrics
                predicted_masks_flat_eval = rearrange(predicted_masks_seq, 'b t c h w -> (b t) c h w')
                masks_flat_target_eval = ground_truth_masks.view(b*t, 1, h, w)
                
                # .cpu() 호출을 루프 밖으로 이동하여 오버헤드 감소
                preds_np_batch = (predicted_masks_flat_eval.cpu().numpy() * 255).astype(np.uint8)
                gts_np_batch = (masks_flat_target_eval.cpu().numpy() > 0.5).astype(np.uint8) * 255

                for i in range(b*t):
                    pred_mask_uint8 = preds_np_batch[i].squeeze()
                    gt_mask_uint8 = gts_np_batch[i].squeeze()
                    # GT가 비어있지 않은 경우에만 메트릭 계산
                    if gt_mask_uint8.max() > 0: 
                        metrics.step(pred=pred_mask_uint8, gt=gt_mask_uint8)

                # 2. Enhancement L1 Loss
                use_enhancement_eval = reconstructed_images_flat is not None and eval_cfg.get('calculate_enhancement_loss', True)
                if use_enhancement_eval:
                    original_images_flat_target_eval = original_day_images.view(b*t, c, h, w)
                    loss_enh_batch = l1_loss(reconstructed_images_flat, original_images_flat_target_eval).item()
                    if not np.isnan(loss_enh_batch) and not np.isinf(loss_enh_batch):
                         total_enhancement_loss += loss_enh_batch * (b*t)
                         enhancement_comparison_count += (b*t)

                # 3. Warping Error
                if t > 1 and calculate_warping_error and raft_model is not None:
                    try:
                        img1_batch_we = video_clip[:, :-1].reshape(-1, c, h, w)
                        img2_batch_we = video_clip[:, 1:].reshape(-1, c, h, w)
                        if not torch.equal(img1_batch_we, img2_batch_we):
                            img1_tf_we, img2_tf_we = raft_transforms(img1_batch_we, img2_batch_we)
                            # RAFT forward는 no_grad() 컨텍스트 내에서도 명시적으로 no_grad() 추가 (메모리 절약)
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
                                     batch_temporal_error_sum += step_err * b # 배치 크기(b) 곱하기
                                     valid_comps += b
                                     
                            if valid_comps > 0: 
                                total_warping_error += batch_temporal_error_sum
                                temporal_comparison_count += valid_comps
                                
                    except Exception as e_we: logger.warning(f"Eval: Warping Error calc failed: {e_we}. Skip for batch.")

            except Exception as e_metric: logger.error(f"Eval: Metric calculation error: {e_metric}. Skip batch metrics.")

    # --- 최종 결과 ---
    results = metrics.get_results()
    avg_warping_error = total_warping_error / temporal_comparison_count if temporal_comparison_count > 0 else float('nan')
    avg_enhancement_loss = total_enhancement_loss / enhancement_comparison_count if enhancement_comparison_count > 0 else float('nan')

    logger.info("\n--- Evaluation Results ---")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Dataset: {eval_cfg.get('eval_folder_data_root', 'N/A')}")
    logger.info(f"S-measure (Sm) ↑:           {results.get('Sm', float('nan')):.4f}")
    logger.info(f"E-measure (Em) ↑:           {results.get('Em', float('nan')):.4f}")
    logger.info(f"Weighted F-measure (wFm) ↑: {results.get('wFm', float('nan')):.4f}")
    logger.info(f"Mean Absolute Error (MAE) ↓:{results.get('MAE', float('nan')):.4f}")
    logger.info(f"Adaptive F-measure (adpFm)↑:{results.get('adpFm', float('nan')):.4f}")
    logger.info(f"Max F-measure (F-beta) ↑:   {results.get('F-beta', float('nan')):.4f}")
    logger.info(f"Warping Error (Temporal) ↓: {avg_warping_error:.4f}")
    logger.info(f"Enhancement L1 Loss ↓:      {avg_enhancement_loss:.4f}")
    logger.info("--------------------------")

    # 시각화
    if eval_cfg.get('generate_visualization', True) and len(test_dataset) > 0:
        visualize_predictions(model, test_dataset, device, eval_cfg, common_cfg)

if __name__ == '__main__':
    main()