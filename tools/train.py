import torch
import os
import argparse
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import grid_sample
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

from models.main_model import JED_VCOD_Fauna_Simplified
from datasets.moca_video_dataset import MoCAVideoDataset
from utils.losses import DiceLoss

def train_one_epoch(model, raft_model, raft_transforms, train_loader, optimizer, device, bce_loss, dice_loss, l1_loss, lambda_temporal):
    """한 에포크 동안 모델을 학습시키는 함수"""
    model.train()
    epoch_loss = 0
    for video_clip, ground_truth_masks in tqdm(train_loader, desc="Training"):
        video_clip = video_clip.to(device)
        ground_truth_masks = ground_truth_masks.to(device)

        optimizer.zero_grad()
        predicted_masks = model(video_clip)
        loss_seg = bce_loss(predicted_masks, ground_truth_masks) + dice_loss(predicted_masks, ground_truth_masks)

        loss_temporal = 0
        b, t, c, h, w = video_clip.shape

        if t > 1:
            img1_batch = video_clip[:, :-1].reshape(-1, c, h, w)
            img2_batch = video_clip[:, 1:].reshape(-1, c, h, w)
            img1_transformed, img2_transformed = raft_transforms(img1_batch, img2_batch)
            with torch.no_grad():
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
                mask_t_warped = grid_sample(predicted_masks[:, i], warped_grid, mode='bilinear', padding_mode='border', align_corners=False)
                loss_temporal += l1_loss(mask_t_warped, predicted_masks[:, i+1])
        
        total_loss = loss_seg + lambda_temporal * (loss_temporal / (t - 1) if t > 1 else 0)
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
        
    return epoch_loss / len(train_loader)

def validate(model, val_loader, device, bce_loss, dice_loss):
    """모델의 성능을 검증 데이터셋으로 평가하는 함수"""
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for video_clip, ground_truth_masks in tqdm(val_loader, desc="Validating"):
            video_clip = video_clip.to(device)
            ground_truth_masks = ground_truth_masks.to(device)
            predicted_masks = model(video_clip)
            loss = bce_loss(predicted_masks, ground_truth_masks) + dice_loss(predicted_masks, ground_truth_masks)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def main(args):
    # 1. 초기화 및 멀티 GPU 설정
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        gpu_ids = list(range(len(args.gpu_ids.split(','))))
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        gpu_ids = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    model = JED_VCOD_Fauna_Simplified()
    if gpu_ids and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # ▼▼▼ 수정된 부분 ▼▼▼
    # verbose=True 인자를 삭제했습니다.
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_factor, patience=args.scheduler_patience)

    # RAFT 모델 로드 (시간적 손실 계산용)
    raft_weights = Raft_Large_Weights.DEFAULT
    raft_transforms = raft_weights.transforms()
    raft_model = raft_large(weights=raft_weights, progress=False).to(device)
    raft_model.eval()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    l1_loss = torch.nn.L1Loss()

    # 2. 데이터 로더 설정
    full_dataset = MoCAVideoDataset(
        synthetic_data_root=args.data_root,
        annotation_file=args.annotation_file,
        clip_len=args.clip_len
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 3. Early Stopping 설정
    patience = args.patience
    best_val_loss = np.inf
    early_stop_counter = 0

    # 4. 학습 루프
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, raft_model, raft_transforms, train_loader, optimizer, device, bce_loss, dice_loss, l1_loss, args.lambda_temporal)
        val_loss = validate(model, val_loader, device, bce_loss, dice_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, os.path.join(args.checkpoint_dir, "best_model2.pth"))
            print(f"Validation loss improved. Saved best model to {args.checkpoint_dir}/best_model2.pth")
        else:
            early_stop_counter += 1
            print(f"Validation loss did not improve. Counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="JED-VCOD-Fauna Training Script")
    
    # 데이터 및 경로 관련
    parser.add_argument('--data_root', type=str, default='data/Night-Camo-Fauna/', help='합성 데이터셋의 루트 경로')
    parser.add_argument('--annotation_file', type=str, default='data/MoCA/Annotations/annotations_modified.csv', help='어노테이션 파일 경로')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='모델 가중치를 저장할 디렉토리')
    
    # 학습 하이퍼파라미터
    parser.add_argument('--epochs', type=int, default=50, help='총 학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=2, help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-4, help='학습률 (learning rate)')
    parser.add_argument('--clip_len', type=int, default=8, help='비디오 클립의 길이 (프레임 수)')
    parser.add_argument('--lambda_temporal', type=float, default=0.1, help='시간적 손실의 가중치')
    
    # 시스템 관련
    parser.add_argument('--num_workers', type=int, default=4, help='데이터 로딩에 사용할 워커 수')
    parser.add_argument('--gpu_ids', type=str, default=None, help='사용할 GPU ID (예: "0,1,2")')
    
    # Early Stopping
    parser.add_argument('--patience', type=int, default=10, help='Early stopping을 위한 대기 에포크 수')

    # Scheduler 관련
    parser.add_argument('--scheduler_patience', type=int, default=3, help='학습률 스케줄러 대기 에포크 수')
    parser.add_argument('--scheduler_factor', type=float, default=0.1, help='학습률 감소 비율')

    args = parser.parse_args()
    main(args)