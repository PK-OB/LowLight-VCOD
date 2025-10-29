import torch
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torch.utils.data import DataLoader
from torch.nn.functional import grid_sample
from tqdm import tqdm
from utils.py_sod_metrics import SODMetrics

from config import cfg
from models.eval_models import JED_VCOD_Fauna_Simplified_Eval, YourSOTAVCODModel, YourSOTAEnhancerModel
from datasets.moca_video_dataset import MoCAVideoDataset

def unnormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """정규화된 이미지 텐서를 원래 이미지로 되돌리는 함수"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def visualize_predictions(model, dataset, device, eval_cfg, common_cfg):
    """모델의 예측 결과를 시각화하여 이미지 파일로 저장하는 함수"""
    print("\n--- Generating visualization ---")
    if len(dataset) < 5:
        print("Dataset has fewer than 5 samples, skipping visualization.")
        return

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle('Prediction Visualization: 5 Random Samples', fontsize=16)
    random_indices = random.sample(range(len(dataset)), 5)

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(tqdm(random_indices, desc="Visualizing")):
            video_clip, mask_clip = dataset[idx]
            video_clip_batch = video_clip.unsqueeze(0).to(device)
            
            predicted_logits = model(video_clip_batch)
            predicted_mask = torch.sigmoid(predicted_logits)

            frame_idx = common_cfg['clip_len'] // 2
            
            # 1. 원본 이미지
            image_tensor = video_clip[frame_idx]
            image_to_show = unnormalize(image_tensor).cpu().numpy().transpose(1, 2, 0)
            axes[0, i].imshow(np.clip(image_to_show, 0, 1))
            axes[0, i].set_title(f"Sample {idx}\nOriginal Image")
            axes[0, i].axis('off')

            # 2. 정답 마스크
            gt_mask_tensor = mask_clip[frame_idx]
            gt_mask_to_show = gt_mask_tensor.squeeze().cpu().numpy()
            axes[1, i].imshow(gt_mask_to_show, cmap='gray')
            axes[1, i].set_title("Ground Truth Mask")
            axes[1, i].axis('off')
            
            # 3. 예측 마스크
            pred_mask_tensor = predicted_mask.squeeze(0)[frame_idx]
            pred_mask_to_show = pred_mask_tensor.squeeze().cpu().numpy()
            axes[2, i].imshow(pred_mask_to_show, cmap='gray')
            axes[2, i].set_title("Predicted Mask")
            axes[2, i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(eval_cfg['visualization_path'])
    print(f"Visualization image saved to: {eval_cfg['visualization_path']}")

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
        if not eval_cfg['enhancer_checkpoint_path']:
            raise ValueError("Baseline B requires 'enhancer_checkpoint_path' in config.py")
        enhancer_model = YourSOTAEnhancerModel().to(device)
        enhancer_model.load_state_dict(torch.load(eval_cfg['enhancer_checkpoint_path'], map_location=device))
        enhancer_model.eval()
        print(f"Loaded Enhancer model from: {eval_cfg['enhancer_checkpoint_path']}")

    if eval_cfg['experiment'] in ['proposed', 'ablation_1']:
        model = JED_VCOD_Fauna_Simplified_Eval(use_dae=True)
    elif eval_cfg['experiment'] == 'ablation_2':
        model = JED_VCOD_Fauna_Simplified_Eval(use_dae=False)
    elif eval_cfg['experiment'] in ['baseline_a', 'baseline_b']:
        model = YourSOTAVCODModel()
    else:
        raise ValueError(f"Unknown experiment: {eval_cfg['experiment']}")

    state_dict = torch.load(eval_cfg['checkpoint_path'], map_location=device)
    if all(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded Detection model from: {eval_cfg['checkpoint_path']}")

    raft_weights = Raft_Large_Weights.DEFAULT
    raft_transforms = raft_weights.transforms()
    raft_model = raft_large(weights=raft_weights, progress=False).to(device)
    raft_model.eval()
    l1_loss = torch.nn.L1Loss()

    test_dataset = MoCAVideoDataset(
        synthetic_data_root=eval_cfg['data_root'],
        annotation_file=eval_cfg['annotation_file'], 
        clip_len=common_cfg['clip_len']
    )
    test_loader = DataLoader(test_dataset, batch_size=eval_cfg['batch_size'], shuffle=False, num_workers=common_cfg['num_workers'])

    metrics = SODMetrics()
    total_warping_error = 0.0
    temporal_comparison_count = 0

    with torch.no_grad():
        for video_clip, ground_truth_masks in tqdm(test_loader, desc="Evaluating"):
            video_clip = video_clip.to(device)
            ground_truth_masks = ground_truth_masks.to(device)

            video_for_detection = video_clip
            if enhancer_model:
                b, t, c, h, w = video_clip.shape
                enhancer_input = video_clip.view(b * t, c, h, w)
                enhanced_frames = enhancer_model(enhancer_input)
                video_for_detection = enhanced_frames.view(b, t, c, h, w)
            
            predicted_logits = model(video_for_detection)
            predicted_masks = torch.sigmoid(predicted_logits)

            b, t, c, h, w = predicted_masks.shape

            for i in range(b):
                for j in range(t):
                    pred_mask_np = predicted_masks[i, j].squeeze().cpu().numpy()
                    gt_mask_np = ground_truth_masks[i, j].squeeze().cpu().numpy()
                    
                    # float32 (0~1) 예측값을 uint8 (0~255) 이미지 형식으로 변환합니다.
                    pred_mask_uint8 = (pred_mask_np * 255).astype(np.uint8)
                    
                    # ground truth도 0 또는 255 값을 갖는 uint8 형식으로 변환합니다.
                    gt_mask_uint8 = (gt_mask_np > 0.5).astype(np.uint8) * 255

                    if gt_mask_uint8.max() > 0:
                        metrics.step(pred=pred_mask_uint8, gt=gt_mask_uint8)

            if t > 1:
                img1_batch = video_clip[:, :-1].reshape(-1, 3, h, w)
                img2_batch = video_clip[:, 1:].reshape(-1, 3, h, w)
                img1_transformed, img2_transformed = raft_transforms(img1_batch, img2_batch)
                flows = raft_model(img1_transformed, img2_transformed)[-1]
                flows_unbatched = flows.view(b, t - 1, 2, h, w)

                for i in range(t - 1):
                    flow_i = flows_unbatched[:, i]
                    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
                    grid = torch.stack((grid_x, grid_y), 2).float()
                    displacement = flow_i.permute(0, 2, 3, 1)
                    warped_grid = grid + displacement
                    warped_grid[..., 0] = 2.0 * warped_grid[..., 0] / (w - 1) - 1.0
                    warped_grid[..., 1] = 2.0 * warped_grid[..., 1] / (h - 1) - 1.0
                    
                    mask_t = predicted_masks[:, i]
                    mask_t_plus_1 = predicted_masks[:, i + 1]
                    mask_t_warped = grid_sample(mask_t, warped_grid, mode='bilinear', padding_mode='border', align_corners=False)
                    
                    total_warping_error += l1_loss(mask_t_warped, mask_t_plus_1).item()
                    temporal_comparison_count += b

    results = metrics.get_results()
    avg_warping_error = total_warping_error / temporal_comparison_count if temporal_comparison_count > 0 else 0

    print("\n--- Evaluation Results ---")
    print(f"S-measure (Sm):           {results['Sm']:.4f}")
    print(f"E-measure (Em):           {results['Em']:.4f}")
    print(f"Weighted F-measure (wFm): {results['wFm']:.4f}")
    print(f"Mean Absolute Error (MAE):{results['MAE']:.4f}")
    print(f"Warping Error:            {avg_warping_error:.4f}")
    print("--------------------------")

    visualize_predictions(model, test_dataset, device, eval_cfg, common_cfg)

if __name__ == '__main__':
    main()