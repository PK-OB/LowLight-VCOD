# pk-ob/jed-vcod/JED-VCOD-cc543b29cefb3a45b940bfd01f42c33af7a6bb25/run_experiment.py
# (최종 버전: 체크포인트 로딩/저장, Multi-Task Loss, Video Swin 모델 로딩)

import torch
import torch.nn as nn
import os
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights # Temporal loss 용도
from torch.utils.data import DataLoader, random_split, Subset
from torch.nn.functional import grid_sample
import torch.nn.functional as F # <-- NameError 방지
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, CosineAnnealingLR # <-- CosineAnnealingLR 추가
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
import argparse # <-- Argument Parser 임포트
from einops import rearrange # <-- Video Swin 모델에서 사용

from config import cfg
# ▼▼▼ 수정된 main_model 임포트 ▼▼▼
from models.main_model import VideoSwinVCOD
from datasets.folder_mask_dataset import FolderImageMaskDataset # <-- Multi-Task 데이터셋 임포트
from utils.losses import DiceLoss, FocalLoss
from utils.logger import setup_logger
from utils.cutmix import cutmix_data # <-- Multi-Task CutMix 임포트

def unnormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """정규화된 이미지 텐서를 원래 이미지로 되돌리는 함수 (변경 없음)"""
    tensor = tensor.clone()
    # CPU 텐서로 이동 후 연산 (GPU 메모리 절약 및 호환성)
    tensor_cpu = tensor.cpu()
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)

    # 정규화 역연산
    tensor_cpu.mul_(std_tensor).add_(mean_tensor)
    # 클램핑 추가 (부동소수점 오류로 인한 범위 초과 방지)
    tensor_cpu = torch.clamp(tensor_cpu, 0, 1)
    return tensor_cpu


def verify_and_save_samples(dataset, common_cfg, train_cfg):
    """
    데이터셋 샘플을 이미지로 저장하여 로딩 확인 (Multi-Task 데이터셋 반영 버전)
    """
    logging.info("--- Verifying dataset samples ---")
    
    # Subset 객체 처리
    if isinstance(dataset, Subset):
        original_dataset = dataset.dataset
        indices_to_sample_from = dataset.indices
    else:
        original_dataset = dataset
        indices_to_sample_from = list(range(len(dataset)))

    num_samples_available = len(indices_to_sample_from)
    if num_samples_available < 1:
        logging.warning("Dataset is empty, cannot verify samples.")
        return
    num_samples_to_check = min(5, num_samples_available)

    fig, axes = plt.subplots(3, num_samples_to_check, figsize=(4 * num_samples_to_check, 12))
    fig.suptitle('Dataset Verification: Random Samples (Night / Mask / Day)', fontsize=16)

    # 실제 사용할 인덱스 샘플링
    random_subset_indices = random.sample(range(num_samples_available), num_samples_to_check)
    actual_indices = [indices_to_sample_from[i] for i in random_subset_indices]

    for i, idx in enumerate(actual_indices):
        try:
            data_sample = original_dataset[idx] # 원본 데이터셋에서 가져옴
        except IndexError:
            logging.warning(f"Index {idx} out of bounds for dataset. Skipping sample verification.")
            continue
        except Exception as e:
            logging.warning(f"Error loading sample {idx}: {e}. Skipping sample verification.")
            continue

        if data_sample is None:
            logging.warning(f"Loaded None for sample {idx}. Skipping sample verification.")
            continue
        if len(data_sample) != 3:
            logging.warning(f"Sample {idx} did not return 3 items. Skipping sample verification.")
            continue

        video_clip, mask_clip, original_day_clip = data_sample
        clip_len_actual = video_clip.shape[0] # 실제 클립 길이 확인
        frame_index_to_show = min(common_cfg['clip_len'] // 2, clip_len_actual - 1) # 인덱스 범위 확인

        ax_row1 = axes[0, i] if num_samples_to_check > 1 else axes[0]
        ax_row2 = axes[1, i] if num_samples_to_check > 1 else axes[1]
        ax_row3 = axes[2, i] if num_samples_to_check > 1 else axes[2]

        # 1. 야간 이미지
        try:
            if frame_index_to_show < video_clip.shape[0]:
                image_tensor = video_clip[frame_index_to_show]
                image_to_show = unnormalize(image_tensor).numpy().transpose(1, 2, 0)
                ax_row1.imshow(np.clip(image_to_show, 0, 1))
                ax_row1.set_title(f"Sample index {idx}\nNight Image")
            else: ax_row1.set_title(f"Sample index {idx}\n(Night Image OOB)")
        except Exception as e: ax_row1.set_title(f"Sample index {idx}\n(Night Img Err: {e})")
        ax_row1.axis('off')

        # 2. 마스크
        try:
            if frame_index_to_show < mask_clip.shape[0]:
                mask_tensor = mask_clip[frame_index_to_show]
                mask_to_show = mask_tensor.squeeze().cpu().numpy() # CPU로 이동
                ax_row2.imshow(mask_to_show, cmap='gray')
                ax_row2.set_title(f"Mask")
            else: ax_row2.set_title(f"(Mask OOB)")
        except Exception as e: ax_row2.set_title(f"(Mask Err: {e})")
        ax_row2.axis('off')

        # 3. 주간 이미지
        try:
            if frame_index_to_show < original_day_clip.shape[0]:
                day_image_tensor = original_day_clip[frame_index_to_show]
                day_image_to_show = day_image_tensor.cpu().numpy().transpose(1, 2, 0) # CPU로 이동
                ax_row3.imshow(np.clip(day_image_to_show, 0, 1))
                ax_row3.set_title(f"Day Image (GT)")
            else: ax_row3.set_title(f"(Day Image OOB)")
        except Exception as e: ax_row3.set_title(f"(Day Img Err: {e})")
        ax_row3.axis('off')


    debug_dir = os.path.join(train_cfg['log_dir'], 'debug_images')
    os.makedirs(debug_dir, exist_ok=True)
    save_path = os.path.join(debug_dir, f"{train_cfg.get('experiment_name', 'exp')}_dataset_verification.png")
    try:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path)
        plt.close(fig)
        logging.info(f"Verification image saved to: {save_path}")
    except Exception as e:
        logging.error(f"Failed to save verification image: {e}")


def train_one_epoch(model, raft_model, raft_transforms, train_loader, optimizer, device, focal_loss, dice_loss, l1_loss, writer, epoch, train_cfg, common_cfg):
    """
    한 에포크 학습 로직 (Video Swin 모델 출력 처리, 오류 처리 강화)
    """
    model.train()
    epoch_loss = 0.0
    processed_batches = 0 # 유효 배치 카운트
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    
    # 그래디언트 누적 설정
    accumulation_steps = train_cfg.get('accumulation_steps', 1)
    optimizer.zero_grad() # 에포크 시작 시 초기화

    for i, batch in enumerate(progress_bar):
        # --- 배치 유효성 검사 ---
        if batch is None: 
            logging.warning(f"Epoch {epoch+1}, Batch {i}: Received None from DataLoader, skipping.")
            continue
        if not isinstance(batch, (list, tuple)) or len(batch) != 3:
             logging.warning(f"Epoch {epoch+1}, Batch {i}: Expected 3 items, got {type(batch)} len {len(batch) if isinstance(batch, (list, tuple)) else 'N/A'}, skipping.")
             continue
        video_clip, ground_truth_masks, original_day_images = batch
        if not all(isinstance(t, torch.Tensor) for t in batch) or \
           video_clip.nelement() == 0 or ground_truth_masks.nelement() == 0 or original_day_images.nelement() == 0:
            logging.warning(f"Epoch {epoch+1}, Batch {i}: Received invalid or empty tensor(s), skipping.")
            continue
            
        # --- 데이터 GPU 이동 ---
        try:
            video_clip = video_clip.to(device, non_blocking=True)
            ground_truth_masks = ground_truth_masks.to(device, non_blocking=True)
            original_day_images = original_day_images.to(device, non_blocking=True)
        except Exception as e:
            logging.error(f"Epoch {epoch+1}, Batch {i}: Error moving data to device {device}: {e}. Skipping batch.")
            continue

        b, t, c, h, w = video_clip.shape

        # --- CutMix --- (모델 입력 전 수행)
        apply_cutmix = train_cfg.get('use_cutmix', False) and train_cfg.get('cutmix_beta', 0) > 0 and random.random() < train_cfg.get('cutmix_prob', 0)
        
        images_flat_orig = video_clip.view(b*t, c, h, w)
        masks_flat_orig = ground_truth_masks.view(b*t, 1, h, w)
        original_images_flat_orig = original_day_images.view(b*t, c, h, w)

        if apply_cutmix:
            try:
                images_flat_cutmix, masks_flat_cutmix, original_images_flat_cutmix = cutmix_data(
                    images_flat_orig.clone(), masks_flat_orig.clone(), original_images_flat_orig.clone(), 
                    train_cfg['cutmix_beta'], use_cuda=(device.type == 'cuda')
                )
                video_clip_input = images_flat_cutmix.view(b, t, c, h, w)
                masks_flat_target = masks_flat_cutmix
                original_images_flat_target = original_images_flat_cutmix
            except Exception as e:
                logging.warning(f"Epoch {epoch+1}, Batch {i}: CutMix failed: {e}. Using original data.")
                video_clip_input = video_clip
                masks_flat_target = masks_flat_orig
                original_images_flat_target = original_images_flat_orig
        else:
            video_clip_input = video_clip
            masks_flat_target = masks_flat_orig
            original_images_flat_target = original_images_flat_orig

        # --- 모델 Forward ---
        try:
            # 모델은 (B, T, C, H, W) 입력을 가정
            predicted_masks_seq, reconstructed_images_flat = model(video_clip_input)
        except Exception as e:
            logging.error(f"Epoch {epoch+1}, Batch {i}: Model forward pass failed: {e}. Skipping batch.")
            # 오류 발생 시 그래디언트 초기화하고 다음 배치로
            optimizer.zero_grad() 
            continue

        # --- Loss 계산 ---
        try:
            # 1. 분할 손실
            predicted_masks_flat = rearrange(predicted_masks_seq, 'b t c h w -> (b t) c h w')
            loss_focal = focal_loss(predicted_masks_flat, masks_flat_target)
            loss_dice = dice_loss(predicted_masks_flat, masks_flat_target)
            if torch.isnan(loss_focal) or torch.isinf(loss_focal): loss_focal = torch.tensor(0.0).to(device)
            if torch.isnan(loss_dice) or torch.isinf(loss_dice): loss_dice = torch.tensor(0.0).to(device)
            loss_seg = loss_focal + train_cfg.get('dice_weight', 1.0) * loss_dice

            # 2. 강화 손실
            loss_enhancement = torch.tensor(0.0).to(device)
            use_enhancement = reconstructed_images_flat is not None and train_cfg.get('lambda_enhancement', 0) > 0
            if use_enhancement:
                loss_enhancement = l1_loss(reconstructed_images_flat, original_images_flat_target)
                if torch.isnan(loss_enhancement) or torch.isinf(loss_enhancement): loss_enhancement = torch.tensor(0.0).to(device)

            # 3. 시간적 손실
            loss_temporal = torch.tensor(0.0).to(device)
            if t > 1 and train_cfg.get('lambda_temporal', 0) > 0 and raft_model is not None:
                # Temporal loss 계산 (내부 오류 처리 포함)
                try:
                    img1_batch_t = video_clip[:, :-1].reshape(-1, c, h, w)
                    img2_batch_t = video_clip[:, 1:].reshape(-1, c, h, w)
                    
                    if not torch.equal(img1_batch_t, img2_batch_t): 
                        img1_transformed_t, img2_transformed_t = raft_transforms(img1_batch_t, img2_batch_t)
                        with torch.no_grad():
                            list_of_flows_t = raft_model(img1_transformed_t.contiguous(), img2_transformed_t.contiguous())
                            flows_t = list_of_flows_t[-1]
                        flows_resized_t = F.interpolate(flows_t, size=(h, w), mode='bilinear', align_corners=False) 
                        flows_unbatched_t = flows_resized_t.view(b, t - 1, 2, h, w)

                        current_temporal_loss = 0.0
                        valid_temporal_comps = 0
                        for frame_idx in range(t - 1):
                            flow_i_t = flows_unbatched_t[:, frame_idx]
                            grid_y_t, grid_x_t = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
                            grid_t = torch.stack((grid_x_t, grid_y_t), 2).float() 
                            displacement_t = flow_i_t.permute(0, 2, 3, 1) 
                            warped_grid_t = grid_t + displacement_t
                            warped_grid_t[..., 0] = 2.0 * warped_grid_t[..., 0] / (w - 1) - 1.0
                            warped_grid_t[..., 1] = 2.0 * warped_grid_t[..., 1] / (h - 1) - 1.0
                            
                            mask_t_seq = predicted_masks_seq[:, frame_idx]
                            mask_t_plus_1_seq = predicted_masks_seq[:, frame_idx+1]
                            mask_t_warped_seq = grid_sample(mask_t_seq, warped_grid_t, mode='bilinear', padding_mode='border', align_corners=False)
                            
                            step_loss = l1_loss(mask_t_warped_seq, mask_t_plus_1_seq)
                            if not torch.isnan(step_loss) and not torch.isinf(step_loss):
                                current_temporal_loss += step_loss
                                valid_temporal_comps += 1
                                
                        loss_temporal = current_temporal_loss / valid_temporal_comps if valid_temporal_comps > 0 else torch.tensor(0.0).to(device)
                except Exception as e:
                    logging.warning(f"E{epoch+1} B{i}: Temporal loss calc error: {e}. Skip.")
                    loss_temporal = torch.tensor(0.0).to(device)


            # 총 손실
            total_loss = loss_seg + \
                         train_cfg.get('lambda_temporal', 0.5) * loss_temporal + \
                         train_cfg.get('lambda_enhancement', 0.5) * loss_enhancement

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logging.error(f"!!! E{epoch+1} B{i}: Total Loss is NaN/Inf after combining. Seg:{loss_seg.item():.4f}, Enh:{loss_enhancement.item():.4f}, Temp:{loss_temporal.item():.4f}. Skip backward.")
                optimizer.zero_grad() # 다음 배치를 위해 초기화
                continue 

            # 총 손실 정규화 (Accumulation 시)
            # loss_norm = total_loss / accumulation_steps
            loss_norm = total_loss # Accumulation 안 할 경우

        except Exception as e:
            logging.error(f"Epoch {epoch+1}, Batch {i}: Error calculating loss: {e}. Skipping batch.")
            optimizer.zero_grad()
            continue

        # --- Backward & Optimize ---
        try:
            loss_norm.backward()
        except Exception as e:
             logging.error(f"Epoch {epoch+1}, Batch {i}: Backward pass failed: {e}. Skipping optimizer step.")
             optimizer.zero_grad() # 오류 시 그래디언트 초기화
             continue

        # 그래디언트 누적 스텝 확인 및 옵티마이저 실행
        if (i + 1) % accumulation_steps == 0:
            try:
                # 그래디언트 클리핑 (옵티마이저 스텝 전에 수행)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.get('clip_grad_norm', 1.0))
                
                optimizer.step() # 옵티마이저 스텝
                optimizer.zero_grad() # 그래디언트 초기화

                # 로깅 (Accumulation 스텝 완료 시)
                current_loss_display = total_loss.item() # 누적 전 loss 값
                epoch_loss += current_loss_display * accumulation_steps # 누적분 반영
                processed_batches += accumulation_steps # 처리된 배치 수 증가

                global_step = epoch * len(train_loader) + i
                writer.add_scalar('Loss/Step/Total', current_loss_display, global_step)
                writer.add_scalar('Loss/Step/Segmentation', loss_seg.item(), global_step)
                if use_enhancement: writer.add_scalar('Loss/Step/Enhancement', loss_enhancement.item(), global_step)
                writer.add_scalar('Loss/Step/Temporal', loss_temporal.item(), global_step)
                writer.add_scalar('Gradient_Norm', grad_norm.item(), global_step) # 그래디언트 놈 로깅
                progress_bar.set_postfix(loss=f"{current_loss_display:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

            except Exception as e:
                logging.error(f"Epoch {epoch+1}, Batch {i}: Optimizer step or logging failed: {e}.")
                optimizer.zero_grad() # 오류 시 초기화


    # --- 에포크 종료 후 처리 ---
    # 남은 그래디언트 처리 (배치 수가 accumulation_steps의 배수가 아닐 경우)
    # if len(train_loader) % accumulation_steps != 0:
    #     try:
    #         grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.get('clip_grad_norm', 1.0))
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         logging.info(f"Epoch {epoch+1}: Final optimizer step performed.")
    #     except Exception as e:
    #         logging.error(f"Epoch {epoch+1}: Final optimizer step failed: {e}.")
    #         optimizer.zero_grad()


    # 디버그 이미지 저장 (에포크 종료 후)
    if (epoch + 1) % train_cfg.get('debug_image_interval', 5) == 0:
        # ... (이전 디버그 이미지 저장 코드와 동일) ...
        try:
            # 변수 존재 확인 및 유효성 검사 추가
            if 'predicted_masks_seq' in locals() and predicted_masks_seq.nelement() > 0 and \
               'ground_truth_masks' in locals() and ground_truth_masks.nelement() > 0 and \
               'original_day_images' in locals() and original_day_images.nelement() > 0 and \
               'video_clip' in locals() and video_clip.nelement() > 0:
                
                b_last, t_last, c_last, h_last, w_last = video_clip.shape # 마지막 배치 크기
                idx_to_save = 0 
                frame_to_save = 0
                
                pred_to_save = torch.sigmoid(predicted_masks_seq[idx_to_save, frame_to_save]).cpu()
                gt_to_save = ground_truth_masks[idx_to_save, frame_to_save].cpu()
                orig_night_to_save = unnormalize(video_clip[idx_to_save, frame_to_save]).cpu() 
                orig_day_to_save = original_day_images[idx_to_save, frame_to_save].cpu()
                
                images_to_log = {
                    "01_Prediction": pred_to_save, "02_GroundTruth_Mask": gt_to_save,
                    "05_Processed_Night_Input": orig_night_to_save, "04_Original_Day_GT": orig_day_to_save,
                }

                use_enhancement_debug = reconstructed_images_flat is not None and train_cfg.get('lambda_enhancement', 0) > 0
                if use_enhancement_debug and reconstructed_images_flat.nelement() > 0:
                     try:
                        # reconstructed_images_flat는 (B*T) 형태이므로 view 필요
                        recon_images_seq = reconstructed_images_flat.view(b_last, t_last, c_last, h_last, w_last)
                        recon_to_save = recon_images_seq[idx_to_save, frame_to_save].cpu()
                        images_to_log["03_Reconstructed_Day"] = recon_to_save
                     except Exception as e_recon:
                         logging.warning(f"Epoch {epoch+1}: Could not get reconstructed image for debug: {e_recon}")


                debug_dir = os.path.join(train_cfg['log_dir'], 'debug_images')
                os.makedirs(debug_dir, exist_ok=True)
                
                for name, img_tensor in images_to_log.items():
                    if img_tensor.nelement() > 0: # 빈 텐서 저장 방지
                        save_image(img_tensor, os.path.join(debug_dir, f'epoch_{epoch+1}_{name}.png'))
                        writer.add_image(f'Images/{name}', img_tensor, epoch)

            else:
                 logging.warning(f"Epoch {epoch+1}: Cannot save debug images, relevant tensors are missing or empty.")
        except NameError:
             logging.warning(f"Epoch {epoch+1}: Cannot save debug images, variables undefined (likely due to skipped batches).")
        except Exception as e:
            logging.warning(f"Epoch {epoch+1}: Error saving debug images: {e}")

    # 평균 에포크 손실 계산
    avg_epoch_loss = epoch_loss / processed_batches if processed_batches > 0 else 0.0
    return avg_epoch_loss


def validate(model, val_loader, device, focal_loss, dice_loss, l1_loss, train_cfg):
    """
    검증 로직 (Video Swin 모델 출력 처리, 오류 처리 강화)
    """
    model.eval()
    val_loss = 0.0
    val_seg_loss = 0.0
    val_enhance_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating")
        for batch in progress_bar:
            # --- 배치 유효성 검사 ---
            if batch is None: 
                logging.warning(f"Validation: Received None, skipping.")
                continue
            if not isinstance(batch, (list, tuple)) or len(batch) != 3:
                 logging.warning(f"Validation: Expected 3 items, got {type(batch)} len {len(batch) if isinstance(batch, (list, tuple)) else 'N/A'}, skipping.")
                 continue
            video_clip, ground_truth_masks, original_day_images = batch
            if not all(isinstance(t, torch.Tensor) for t in batch) or \
               video_clip.nelement() == 0 or ground_truth_masks.nelement() == 0 or original_day_images.nelement() == 0:
                logging.warning(f"Validation: Received invalid or empty tensor(s), skipping.")
                continue

            # --- 데이터 GPU 이동 ---
            try:
                video_clip = video_clip.to(device, non_blocking=True)
                ground_truth_masks = ground_truth_masks.to(device, non_blocking=True)
                original_day_images = original_day_images.to(device, non_blocking=True)
            except Exception as e:
                logging.error(f"Validation: Error moving data to device {device}: {e}. Skipping batch.")
                continue

            b, t, c, h, w = video_clip.shape
            
            # --- 모델 Forward ---
            try:
                predicted_masks_seq, reconstructed_images_flat = model(video_clip)
            except Exception as e:
                logging.error(f"Validation: Model forward pass failed: {e}. Skipping batch.")
                continue
            
            # --- Loss 계산 ---
            try:
                predicted_masks_flat = rearrange(predicted_masks_seq, 'b t c h w -> (b t) c h w')
                masks_flat_target = ground_truth_masks.view(b*t, 1, h, w)
                original_images_flat_target = original_day_images.view(b*t, c, h, w)

                # 1. 분할 손실
                loss_focal = focal_loss(predicted_masks_flat, masks_flat_target)
                loss_dice = dice_loss(predicted_masks_flat, masks_flat_target)
                if torch.isnan(loss_focal) or torch.isinf(loss_focal): loss_focal = torch.tensor(0.0).to(device)
                if torch.isnan(loss_dice) or torch.isinf(loss_dice): loss_dice = torch.tensor(0.0).to(device)
                loss_seg = loss_focal + train_cfg.get('dice_weight', 1.0) * loss_dice
                
                # 2. 강화 손실
                loss_enhancement = torch.tensor(0.0).to(device)
                use_enhancement = reconstructed_images_flat is not None and train_cfg.get('lambda_enhancement', 0) > 0
                if use_enhancement:
                    loss_enhancement = l1_loss(reconstructed_images_flat, original_images_flat_target)
                    if torch.isnan(loss_enhancement) or torch.isinf(loss_enhancement): loss_enhancement = torch.tensor(0.0).to(device)
                
                # 3. 총 손실 (Validation에서는 Temporal Loss 제외)
                loss = loss_seg + train_cfg.get('lambda_enhancement', 0.5) * loss_enhancement
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    val_seg_loss += loss_seg.item()
                    if use_enhancement:
                        val_enhance_loss += loss_enhancement.item()
                    num_batches += 1
                    progress_bar.set_postfix(val_loss=f"{loss.item():.4f}")
                else:
                     logging.warning(f"Validation: NaN or Inf detected in validation loss, skipping batch result.")
            except Exception as e:
                logging.error(f"Validation: Error calculating loss: {e}. Skipping batch result.")

    if num_batches == 0:
        logging.error("Validation: No valid batches processed. Returning zero loss.")
        return 0.0, 0.0, 0.0
        
    avg_total_loss = val_loss / num_batches
    avg_seg_loss = val_seg_loss / num_batches
    avg_enhance_loss = val_enhance_loss / num_batches if use_enhancement and num_batches > 0 else 0.0
    
    return avg_total_loss, avg_seg_loss, avg_enhance_loss


# --- 메인 실행 함수 ---
def main(args): 
    common_cfg = cfg.common
    train_cfg = cfg.train
    
    logger = setup_logger(train_cfg.get('log_dir', 'logs'), train_cfg.get('experiment_name', 'exp'))
    writer = SummaryWriter(os.path.join(train_cfg.get('log_dir', 'logs'), 'tensorboard', train_cfg.get('experiment_name', 'exp')))

    # GPU 설정
    if common_cfg.get('gpu_ids'):
        os.environ["CUDA_VISIBLE_DEVICES"] = common_cfg['gpu_ids']
        gpu_ids = list(range(len(common_cfg['gpu_ids'].split(','))))
        if not torch.cuda.is_available():
             logger.error("CUDA is not available but GPUs specified. Check CUDA setup.")
             device = torch.device("cpu"); gpu_ids = None
        else: device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        gpu_ids = None; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 모델 초기화
    try:
        model = VideoSwinVCOD(
            backbone_name=train_cfg.get('backbone_name', 'swin_small_patch4_window7_224'),
            input_size=train_cfg.get('resolution', (224, 224)),
            num_frames=common_cfg.get('clip_len', 8),
            pretrained=train_cfg.get('backbone_pretrained', True),
            decoder_channel=train_cfg.get('decoder_channel', 256),
            use_enhancement=train_cfg.get('lambda_enhancement', 0) > 0
        )
        logger.info(f"Model '{type(model).__name__}' created.")
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        return
        
    # 멀티 GPU
    if gpu_ids and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.to(device)

    # 옵티마이저
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=train_cfg.get('lr', 1e-5), 
            weight_decay=train_cfg.get('weight_decay', 0.05) 
        )
        logger.info(f"Optimizer: AdamW")
    except Exception as e:
         logger.error(f"Failed to create optimizer: {e}")
         return

    # 스케줄러
    try:
        scheduler_name = train_cfg.get('scheduler_name', 'CosineAnnealingLR') 
        if scheduler_name == 'CosineAnnealingLR':
             scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg.get('epochs', 100), eta_min=train_cfg.get('eta_min', 1e-7))
             logger.info(f"Scheduler: CosineAnnealingLR")
        elif scheduler_name == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=train_cfg.get('T_0', 50), T_mult=train_cfg.get('T_mult', 1), eta_min=train_cfg.get('eta_min', 1e-7))
            logger.info(f"Scheduler: CosineAnnealingWarmRestarts")
        else: 
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=train_cfg.get('scheduler_factor', 0.1), patience=train_cfg.get('scheduler_patience', 10))
            logger.info(f"Scheduler: ReduceLROnPlateau")
    except Exception as e:
         logger.error(f"Failed to create scheduler: {e}")
         return

    # RAFT 모델
    raft_model = None
    raft_transforms = None
    if train_cfg.get('lambda_temporal', 0) > 0:
        try:
            raft_weights = Raft_Large_Weights.DEFAULT
            raft_transforms = raft_weights.transforms()
            raft_model = raft_large(weights=raft_weights, progress=False).to(device)
            raft_model.eval()
            logger.info("RAFT model loaded for temporal loss.")
        except Exception as e:
            logger.error(f"Failed to load RAFT model: {e}. Temporal loss disabled.")
            train_cfg['lambda_temporal'] = 0 
            
    # 손실 함수
    focal_loss = FocalLoss(); dice_loss = DiceLoss(); l1_loss = nn.L1Loss()
    
    # 체크포인트 로딩
    start_epoch = 0; best_val_loss = np.inf
    if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
        logger.info(f"Resuming from: {args.resume_checkpoint}")
        try:
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            # 모델 로드 (유연하게)
            model_state_dict = checkpoint['model_state_dict']
            current_model_dict = model.state_dict()
            new_state_dict={k:v for k,v in model_state_dict.items() if k in current_model_dict and v.size()==current_model_dict[k].size()}
            # DataParallel 접두사 처리
            if isinstance(model, nn.DataParallel) and not all(k.startswith('module.') for k in new_state_dict.keys()):
                new_state_dict = {'module.' + k: v for k, v in new_state_dict.items()}
            elif not isinstance(model, nn.DataParallel) and all(k.startswith('module.') for k in new_state_dict.keys()):
                new_state_dict = {k.replace('module.', ''): v for k, v in new_state_dict.items()}

            ignored = [k for k in model_state_dict if k not in new_state_dict]
            missing = [k for k in current_model_dict if k not in new_state_dict]
            if ignored: logger.warning(f"Ignored keys: {ignored}")
            if missing: logger.warning(f"Missing keys: {missing}")
            current_model_dict.update(new_state_dict)
            load_result = model.load_state_dict(current_model_dict, strict=False)
            logger.info(f"Model load result: {load_result}") # 로드 결과 로깅

            if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else: logger.warning("Optimizer state not found.")
            if 'scheduler_state_dict' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else: logger.warning("Scheduler state not found.")

            start_epoch = checkpoint.get('epoch', -1) + 1
            best_val_loss = checkpoint.get('best_val_loss', np.inf)
            logger.info(f"Loaded Checkpoint. Epoch: {start_epoch}, Best Val Loss: {best_val_loss:.4f}")
        except Exception as e:
            logger.error(f"Failed load checkpoint '{args.resume_checkpoint}': {e}. Start from scratch."); start_epoch = 0; best_val_loss = np.inf
    elif args.resume_checkpoint: logger.warning(f"Checkpoint not found: '{args.resume_checkpoint}'. Start from scratch.")
    else: logger.info("No checkpoint. Start from scratch.")

    # 데이터셋 및 데이터로더
    # run_experiment.py (DCNet 이전 버전, main 함수 내부 데이터 분할 로직 수정)

    # ... (로거, 디바이스, 모델, 옵티마이저, 스케줄러, 손실 함수 등 초기화 코드) ...

    # --- 데이터셋 로딩 및 분할 (수정된 로직) ---
    try:
        if train_cfg.get('dataset_type') != 'folder':
            raise ValueError("Only 'folder' dataset_type is supported.")
        folder_root = train_cfg.get('folder_data_root')
        original_root = train_cfg.get('original_data_root')
        if not folder_root or not original_root:
            raise ValueError("'folder_data_root' or 'original_data_root' missing in config.")

        common_args = {
            "root_dir": folder_root,
            "original_data_root": original_root,
            "image_folder_name": train_cfg.get('image_folder_name', 'Imgs'),
            "mask_folder_name": train_cfg.get('mask_folder_name', 'GT'),
            "clip_len": common_cfg.get('clip_len', 8),
            "resolution": train_cfg.get('resolution', (224, 224)),
        }

        # Augmentation 적용된 전체 학습 데이터셋 로드 (Subset 생성용)
        train_dataset_full_aug = FolderImageMaskDataset(**common_args, is_train=True, use_augmentation=True)
        # Augmentation 미적용 전체 데이터셋 로드 (분할 기준 및 검증용)
        val_dataset_full_noaug = FolderImageMaskDataset(**common_args, is_train=False, use_augmentation=False)

        if len(val_dataset_full_noaug) == 0:
            raise ValueError("Dataset is empty. Check paths and data.")

        logger.info(f"Loaded dataset from: {folder_root} & {original_root}")
        logger.info(f"Total clips found: {len(val_dataset_full_noaug)}")

        # --- 정확한 80:20 분할 로직 ---
        split_ratio = train_cfg.get('train_val_split_ratio', 0.8) # Config에서 비율 읽기 (없으면 0.8)
        total_size = len(val_dataset_full_noaug)
        train_size = int(split_ratio * total_size)
        val_size = total_size - train_size

        # 재현성을 위한 시드 고정 (옵션)
        split_seed = train_cfg.get('split_seed', 42)
        generator = torch.Generator().manual_seed(split_seed)

        logger.info(f"Splitting dataset ({total_size} clips) into {train_size} train ({split_ratio*100:.1f}%) and {val_size} validation using seed {split_seed}.")

        # Augmentation 없는 데이터셋을 기준으로 random_split 실행
        # train_indices_subset: 학습용 인덱스를 가진 Subset (Aug 미적용 데이터 기반)
        # val_dataset: 검증용 Subset (Aug 미적용 데이터 기반)
        train_indices_subset, val_dataset = random_split(val_dataset_full_noaug, [train_size, val_size], generator=generator)

        # 실제 학습에 사용할 train_dataset 생성
        # train_indices_subset에서 인덱스만 추출하여, Augmentation 적용된 데이터셋에서 Subset 생성
        train_dataset = Subset(train_dataset_full_aug, train_indices_subset.indices)

        # 검증 데이터셋은 random_split 결과(val_dataset) 그대로 사용 (Aug 미적용)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("Dataset split resulted in empty train or validation set.")
        # --- 분할 로직 끝 ---

    except Exception as e:
        logger.error(f"Dataset setup or split failed: {e}")
        return # 데이터 준비 실패 시 종료

    # collate_fn (3개 항목 처리, 이전 멀티태스크 버전과 동일)
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None and isinstance(x,tuple) and len(x)==3, batch))
        if not batch: return None
        try: return torch.utils.data.dataloader.default_collate(batch)
        except Exception as e: logger.warning(f"Collate error, skip batch: {e}"); return None

    # 데이터로더 생성 (train_dataset, val_dataset 사용)
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.get('batch_size',4), shuffle=True, num_workers=common_cfg.get('num_workers',4), pin_memory=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.get('val_batch_size',8), shuffle=False, num_workers=common_cfg.get('num_workers',4), pin_memory=True, collate_fn=collate_fn)
    logger.info(f"Dataset split confirmed: {len(train_dataset)} train, {len(val_dataset)} val.") # <-- 이제 로그와 실제 크기 일치!

# ... (샘플 검증, 학습 루프 등 나머지 코드는 이전 멀티태스크 버전과 동일) ...

    # 샘플 이미지 검증
    if start_epoch == 0 and len(train_dataset) > 0 :
        verify_and_save_samples(train_dataset, common_cfg, train_cfg)

    early_stop_counter = 0

    # --- 메인 학습 루프 ---
    logger.info("--- Starting Training ---")
    epochs = train_cfg.get('epochs', 100)
    for epoch in range(start_epoch, epochs): 
        # Train
        train_loss = train_one_epoch(model, raft_model, raft_transforms, train_loader, optimizer, device, focal_loss, dice_loss, l1_loss, writer, epoch, train_cfg, common_cfg)
        if train_loss < 0: logger.error(f"Epoch {epoch+1}: Train stopped due to critical error (NaN/Inf)."); break 
            
        # Validate
        val_loss, val_seg_loss, val_enhance_loss = validate(model, val_loader, device, focal_loss, dice_loss, l1_loss, train_cfg)
        
        # 로깅
        lr_log = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Total Loss: {val_loss:.4f} (Seg: {val_seg_loss:.4f}, Enh: {val_enhance_loss:.4f}) | LR: {lr_log:.6f}")
        writer.add_scalar('Loss/Epoch/Train_Total', train_loss, epoch); writer.add_scalar('Loss/Epoch/Val_Total', val_loss, epoch)
        writer.add_scalar('Loss/Epoch/Val_Segmentation', val_seg_loss, epoch); writer.add_scalar('Loss/Epoch/Val_Enhancement', val_enhance_loss, epoch)
        writer.add_scalar('Learning_Rate', lr_log, epoch)

        # Scheduler Step
        if isinstance(scheduler, ReduceLROnPlateau): scheduler.step(val_loss)
        else: scheduler.step() # Cosine 스케줄러는 에포크마다 step

        # 체크포인트 저장 (Best)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss; early_stop_counter = 0
            checkpoint_dir = train_cfg.get('checkpoint_dir', 'checkpoints')
            checkpoint_name = train_cfg.get('checkpoint_name', 'best_model.pth')
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_path = os.path.join(checkpoint_dir, checkpoint_name)
            state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            try:
                torch.save({'epoch': epoch, 'model_state_dict': state_to_save, 'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(), 'best_val_loss': best_val_loss, 'config': cfg}, save_path)
                logger.info(f"Val loss improve {best_val_loss:.4f}. Saved best model: {save_path}")
            except Exception as e: logger.error(f"Failed save checkpoint epoch {epoch+1}: {e}")
        else:
            early_stop_counter += 1
            logger.info(f"Val loss not improve. Counter: {early_stop_counter}/{train_cfg.get('patience', 10)}")

        # Early Stopping
        patience = train_cfg.get('patience', 0) 
        if patience > 0 and early_stop_counter >= patience:
            logger.info(f"Early stop triggered after {patience} epochs no improve."); break

    writer.close()
    logger.info("--- Training Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="JED-VCOD Training Script with Video Swin")
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint file to resume training.')
    args = parser.parse_args()
    main(args)