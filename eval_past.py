# pk-ob/jed-vcod/JED-VCOD-cc543b29cefb3a45b940bfd01f42c33af7a6bb25/evaluate.py

import torch
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torch.utils.data import DataLoader
from torch.nn.functional import grid_sample
import torch.nn.functional as F # (NameError 해결을 위해 import)
from tqdm import tqdm
from utils.py_sod_metrics import SODMetrics

from config import cfg

# import 문 수정 (main_model 추가)
from models.eval_models import JED_VCOD_Fauna_Simplified_Eval, YourSOTAVCODModel, YourSOTAEnhancerModel
from models.main_model import JED_VCOD_Fauna_Simplified 

# 'FolderImageMaskDataset' 임포트
from datasets.moca_video_dataset import MoCAVideoDataset
from datasets.folder_mask_dataset import FolderImageMaskDataset 

def unnormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """정규화된 이미지 텐서를 원래 이미지로 되돌리는 함수"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def visualize_predictions(model, dataset, device, eval_cfg, common_cfg):
    """모델의 예측 결과를 시각화하여 이미지 파일로 저장하는 함수"""
    print("\n--- Generating visualization ---")
    
    # Subset과 일반 Dataset 모두 처리
    if isinstance(dataset, torch.utils.data.Subset):
        dataset_to_sample = dataset.dataset
        indices_to_sample_from = dataset.indices
    else:
        dataset_to_sample = dataset
        indices_to_sample_from = list(range(len(dataset)))
        
    if len(indices_to_sample_from) < 5:
        print(f"Dataset has only {len(indices_to_sample_from)} samples, skipping visualization.")
        return
        
    random_indices_from_list = random.sample(indices_to_sample_from, 5)

    # ▼▼▼ 수정된 부분: 4x5 플롯 (야간/주간(복원)/마스크(GT)/마스크(Pred)) ▼▼▼
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle('Prediction Visualization: 5 Random Samples', fontsize=16)
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(tqdm(random_indices_from_list, desc="Visualizing")):
            data_sample = dataset_to_sample[idx] # 원본 데이터셋에서 idx로 가져옴
            if data_sample is None: continue
            
            # ▼▼▼ 수정된 부분: 3개 항목 수신 ▼▼▼
            video_clip, mask_clip, original_day_clip = data_sample
            
            # mask_clip이 비어있는 경우(예: 일부 샘플에 GT가 없는 경우) 처리
            if mask_clip is None or mask_clip.nelement() == 0:
                print(f"Skipping sample {idx} due to empty mask_clip.")
                continue
                
            video_clip_batch = video_clip.unsqueeze(0).to(device)
            
            # ▼▼▼ 수정된 부분: 모델이 2개 항목 반환 ▼▼▼
            predicted_logits, reconstructed_images_flat = model(video_clip_batch)
            predicted_mask = torch.sigmoid(predicted_logits)
            
            # (B*T, C, H, W) -> (B, T, C, H, W)
            b, t, c, h, w = video_clip_batch.shape
            reconstructed_images = reconstructed_images_flat.view(b, t, c, h, w)
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            frame_idx = common_cfg['clip_len'] // 2
            
            # 1. 원본 야간 이미지 (입력)
            image_tensor = video_clip[frame_idx]
            image_to_show = unnormalize(image_tensor).cpu().numpy().transpose(1, 2, 0)
            axes[0, i].imshow(np.clip(image_to_show, 0, 1))
            axes[0, i].set_title(f"Sample {idx}\nOriginal Night Image")
            axes[0, i].axis('off')

            # ▼▼▼ 수정된 부분: 2. 복원된 주간 이미지 (출력) ▼▼▼
            recon_image_tensor = reconstructed_images.squeeze(0)[frame_idx]
            recon_image_to_show = recon_image_tensor.cpu().numpy().transpose(1, 2, 0)
            axes[1, i].imshow(np.clip(recon_image_to_show, 0, 1))
            axes[1, i].set_title("Reconstructed Day Image")
            axes[1, i].axis('off')
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            # 3. 정답 마스크
            gt_mask_tensor = mask_clip[frame_idx]
            gt_mask_to_show = gt_mask_tensor.squeeze().cpu().numpy()
            axes[2, i].imshow(gt_mask_to_show, cmap='gray')
            axes[2, i].set_title("Ground Truth Mask")
            axes[2, i].axis('off')
            
            # 4. 예측 마스크
            pred_mask_tensor = predicted_mask.squeeze(0)[frame_idx]
            pred_mask_to_show = pred_mask_tensor.squeeze().cpu().numpy()
            axes[3, i].imshow(pred_mask_to_show, cmap='gray')
            axes[3, i].set_title("Predicted Mask")
            axes[3, i].axis('off')

    # 이미지를 저장하기 전에 해당 경로의 폴더가 존재하는지 확인하고, 없으면 생성합니다.
    save_path = eval_cfg['visualization_path']
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    print(f"Visualization image saved to: {save_path}")

def main():
    common_cfg = cfg.common
    eval_cfg = cfg.evaluate

    gpu_id = common_cfg['gpu_ids'].split(',')[0]
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Evaluation ---")
    print(f"Experiment Type: {eval_cfg['experiment']}")
    print(f"Using device: {device}")

    enhancer_model = None
    if eval_cfg['experiment'] == 'baseline_b':
        if not eval_cfg.get('enhancer_checkpoint_path'): # .get()으로 안전하게 접근
            raise ValueError("Baseline B requires 'enhancer_checkpoint_path' in config.py")
        enhancer_model = YourSOTAEnhancerModel().to(device)
        enhancer_model.load_state_dict(torch.load(eval_cfg['enhancer_checkpoint_path'], map_location=device))
        enhancer_model.eval()
        print(f"Loaded Enhancer model from: {eval_cfg['enhancer_checkpoint_path']}")

    # 모델 초기화 로직 (main_model 사용)
    if eval_cfg['experiment'] == 'proposed':
        model = JED_VCOD_Fauna_Simplified() # <--- 학습에 사용한 모델
    elif eval_cfg['experiment'] == 'ablation_1':
        model = JED_VCOD_Fauna_Simplified_Eval(use_dae=True) # (기존 eval 모델)
    elif eval_cfg['experiment'] == 'ablation_2':
        model = JED_VCOD_Fauna_Simplified_Eval(use_dae=False) # (기존 eval 모델)
    elif eval_cfg['experiment'] in ['baseline_a', 'baseline_b']:
        model = YourSOTAVCODModel()
    else:
        raise ValueError(f"Unknown experiment: {eval_cfg['experiment']}")

    state_dict = torch.load(eval_cfg['checkpoint_path'], map_location=device)
    if all(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 가중치 로드 (엄격하게)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print("="*50)
        print("!!! 모델 가중치 로드 중 에러 발생 !!!")
        print("학습(main_model.py)과 평가(evaluate.py)에 사용된 모델 아키텍처가 여전히 다른지 확인하세요.")
        print(f"에러 메시지: {e}")
        print("="*50)
        return
        
    model.to(device)
    model.eval()
    print(f"Loaded Detection model from: {eval_cfg['checkpoint_path']}")

    raft_weights = Raft_Large_Weights.DEFAULT
    raft_transforms = raft_weights.transforms()
    raft_model = raft_large(weights=raft_weights, progress=False).to(device)
    raft_model.eval()
    l1_loss = torch.nn.L1Loss()


    # config의 'eval_dataset_type'에 따라 분기 처리
    # -----------------------------------------------------------------
    eval_dataset_type = eval_cfg.get('eval_dataset_type', 'moca_csv') # 기본값을 'moca_csv'로 설정

    if eval_dataset_type == 'folder':
        print(f"Using 'FolderImageMaskDataset' for evaluation from: {eval_cfg['eval_folder_data_root']}")
        if not eval_cfg.get('eval_folder_data_root'):
            raise ValueError("eval_dataset_type is 'folder', but 'eval_folder_data_root' is not set in config.py")
        
        # ▼▼▼ 수정된 부분: 'eval_original_data_root'를 config에서 읽어와 전달 ▼▼▼
        if not eval_cfg.get('eval_original_data_root'):
             raise ValueError("config.py의 evaluate 섹션에 'eval_original_data_root' (원본 주간 테스트셋 경로) 설정이 필요합니다.")
        
        print(f"Loading original day images for evaluation from: {eval_cfg['eval_original_data_root']}")
        
        test_dataset = FolderImageMaskDataset(
            root_dir=eval_cfg['eval_folder_data_root'],
            original_data_root=eval_cfg['eval_original_data_root'], # <-- 수정됨
            image_folder_name=eval_cfg.get('eval_image_folder_name', 'Imgs'),
            mask_folder_name=eval_cfg.get('eval_mask_folder_name', 'GT'),
            clip_len=common_cfg['clip_len'],
            is_train=False,  # 평가 모드
            use_augmentation=False # 평가 시 증강 사용 안 함
        )
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    
    elif eval_dataset_type == 'moca_csv':
        print(f"Using 'MoCAVideoDataset' (CSV-based) for evaluation from: {eval_cfg['data_root']}")
        test_dataset = MoCAVideoDataset(
            synthetic_data_root=eval_cfg['data_root'],
            annotation_file=eval_cfg['annotation_file'], 
            clip_len=common_cfg['clip_len']
        )
    else:
        raise ValueError(f"Unknown eval_dataset_type: {eval_dataset_type}")
    # -----------------------------------------------------------------
    

    # collate_fn 추가 (데이터셋에서 None 반환 시 처리)
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch: return None, None, None # <-- 3개 반환하도록 수정
        return torch.utils.data.dataloader.default_collate(batch)
    
    test_loader = DataLoader(test_dataset, batch_size=eval_cfg['batch_size'], shuffle=False, num_workers=common_cfg['num_workers'], collate_fn=collate_fn)

    metrics = SODMetrics()
    total_warping_error = 0.0
    temporal_comparison_count = 0
    
    # ▼▼▼ 수정된 부분: Enhancement Loss(L1)를 평가 시에도 계산 ▼▼▼
    total_enhancement_loss = 0.0
    enhancement_comparison_count = 0
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating"):
            # collate_fn에서 (None, None, None)을 반환한 경우 건너뜁니다.
            if batch_data[0] is None:
                continue
                
            # ▼▼▼ 수정된 부분: 3개 항목 수신 ▼▼▼
            video_clip, ground_truth_masks, original_day_images = batch_data
            
            # collate fn을 통과했음에도 ground_truth_masks가 비어있는 엣지 케이스 처리
            if ground_truth_masks is None:
                continue
                
            video_clip = video_clip.to(device)
            ground_truth_masks = ground_truth_masks.to(device)
            original_day_images = original_day_images.to(device) # <-- 추가
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            video_for_detection = video_clip
            if enhancer_model:
                b, t, c, h, w = video_clip.shape
                enhancer_input = video_clip.view(b * t, c, h, w)
                enhanced_frames = enhancer_model(enhancer_input)
                video_for_detection = enhanced_frames.view(b, t, c, h, w)
            
            # ▼▼▼ 수정된 부분: 모델이 2개 항목 반환 ▼▼▼
            predicted_logits, reconstructed_images_flat = model(video_for_detection)
            predicted_masks = torch.sigmoid(predicted_logits)
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            # predicted_masks의 shape (B, T, 1, H, W)를 가져옵니다.
            b, t, _, h, w = predicted_masks.shape
            
            # ▼▼▼ 수정된 부분: Enhancement Loss(L1) 계산 ▼▼▼
            original_images_flat = original_day_images.view(b*t, 3, h, w)
            total_enhancement_loss += l1_loss(reconstructed_images_flat, original_images_flat).item() * (b*t)
            enhancement_comparison_count += (b*t)
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            for i in range(b):
                for j in range(t):
                    pred_mask_np = predicted_masks[i, j].squeeze().cpu().numpy()
                    
                    # ground_truth_masks가 배치 크기보다 적게 반환되는 경우 방지
                    if i >= ground_truth_masks.shape[0] or j >= ground_truth_masks.shape[1]:
                        continue
                        
                    gt_mask_np = ground_truth_masks[i, j].squeeze().cpu().numpy()
                    
                    pred_mask_uint8 = (pred_mask_np * 255).astype(np.uint8)
                    gt_mask_uint8 = (gt_mask_np > 0.5).astype(np.uint8) * 255

                    # 정답 마스크에 객체가 있는 경우에만 지표를 계산합니다.
                    if gt_mask_uint8.max() > 0:
                        metrics.step(pred=pred_mask_uint8, gt=gt_mask_uint8)

            if t > 1 and total_warping_error >= 0: # Warping Error 계산을 원치 않으면 -1로 설정
                try:
                    # video_clip의 H, W를 사용해야 합니다 (마스크와 크기가 다를 수 있음)
                    _, _, _, h_img, w_img = video_clip.shape
                    img1_batch = video_clip[:, :-1].reshape(-1, 3, h_img, w_img)
                    img2_batch = video_clip[:, 1:].reshape(-1, 3, h_img, w_img)
                    
                    # RAFT는 특정 크기의 입력이 필요할 수 있으므로 transforms 적용
                    img1_transformed, img2_transformed = raft_transforms(img1_batch, img2_batch)
                    flows = raft_model(img1_transformed, img2_transformed)[-1]
                    
                    # flow의 H, W는 raft_transforms에 의해 결정됩니다.
                    _, _, h_flow, w_flow = flows.shape
                    flows_unbatched = flows.view(b, t - 1, 2, h_flow, w_flow)
                    
                    # 그리드 생성 시 flow의 크기(h_flow, w_flow) 기준
                    grid_y, grid_x = torch.meshgrid(torch.arange(h_flow, device=device), torch.arange(w_flow, device=device), indexing='ij')
                    grid = torch.stack((grid_x, grid_y), 2).float()

                    for i in range(t - 1):
                        flow_i = flows_unbatched[:, i] # (B, 2, h_flow, w_flow)
                        
                        # grid_sample을 위해 마스크를 flow 크기로 리사이즈
                        mask_t = F.interpolate(predicted_masks[:, i], size=(h_flow, w_flow), mode='bilinear', align_corners=False)
                        mask_t_plus_1 = F.interpolate(predicted_masks[:, i+1], size=(h_flow, w_flow), mode='bilinear', align_corners=False)
                        
                        displacement = flow_i.permute(0, 2, 3, 1) # (B, h_flow, w_flow, 2)
                        warped_grid = grid + displacement
                        
                        # 정규화
                        warped_grid[..., 0] = 2.0 * warped_grid[..., 0] / (w_flow - 1) - 1.0
                        warped_grid[..., 1] = 2.0 * warped_grid[..., 1] / (h_flow - 1) - 1.0
                        
                        mask_t_warped = grid_sample(mask_t, warped_grid, mode='bilinear', padding_mode='border', align_corners=False)
                        
                        total_warping_error += l1_loss(mask_t_warped, mask_t_plus_1).item()
                        temporal_comparison_count += b
                except Exception as e:
                    # SyntaxError가 발생했던 부분 (한 줄로 수정됨)
                    print(f"\nWarping Error calculation failed: {e}") 
                    print("Skipping Warping Error calculation for subsequent batches.")
                    total_warping_error = -1 # 오류 발생 시 더 이상 계산 안 함

    results = metrics.get_results()
    avg_warping_error = total_warping_error / temporal_comparison_count if temporal_comparison_count > 0 else 0
    
    # ▼▼▼ 수정된 부분: Enhancement L1 Loss 평균 계산 ▼▼▼
    avg_enhancement_loss = total_enhancement_loss / enhancement_comparison_count if enhancement_comparison_count > 0 else 0
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    print("\n--- Evaluation Results ---")
    print(f"S-measure (Sm):           {results.get('Sm', float('nan')):.4f}")
    print(f"E-measure (Em):           {results.get('Em', float('nan')):.4f}")
    print(f"Weighted F-measure (wFm): {results.get('wFm', float('nan')):.4f}")
    print(f"Mean Absolute Error (MAE):{results.get('MAE', float('nan')):.4f}")
    print(f"Warping Error:            {avg_warping_error:.4f}")
    # ▼▼▼ 수정된 부분: Enhancement L1 Loss 출력 ▼▼▼
    print(f"Enhancement L1 Loss (↓): {avg_enhancement_loss:.4f}")
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    print("--------------------------")

    visualize_predictions(model, test_dataset, device, eval_cfg, common_cfg)

if __name__ == '__main__':
    main()