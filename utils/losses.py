# pk-ob/lowlight-vcod/LowLight-VCOD-3129585e63b4dbf2e754bab7cdcb562f81d620a6/models/losses.py
# (Boundary Loss 추가 및 CombinedLoss 수정 버전)

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class DiceLoss(nn.Module):
    """표준 Dice Loss (변경 없음)"""
    def __init__(self): super().__init__()
    def forward(self, i, t, s=1):
        try: i=torch.sigmoid(i); i=i.reshape(-1); t=t.reshape(-1); inter=(i*t).sum(); d=(2.*inter+s)/(i.sum()+t.sum()+s); return 1-d
        except Exception as e: logger.error(f"DiceLoss error: {e}"); return torch.tensor(1.0).to(i.device)

class FocalLoss(nn.Module):
    """표준 Focal Loss (변경 없음)"""
    def __init__(self, a=0.25, g=2.0, r='mean'): super().__init__(); self.a=a; self.g=g; self.r=r
    def forward(self, i, t):
        try: bce=F.binary_cross_entropy_with_logits(i,t.float(),reduction='none'); pt=torch.exp(-bce); F_l=self.a*(1-pt)**self.g*bce; return torch.mean(F_l) if self.r=='mean' else (torch.sum(F_l) if self.r=='sum' else F_l)
        except Exception as e: logger.error(f"FocalLoss error: {e}"); return torch.tensor(1.0).to(i.device)

class StructureLoss(nn.Module):
    """Dice + L1 기반 Structure Loss (변경 없음)"""
    def __init__(self): super().__init__()
    def forward(self, pred_logits, gt_mask):
        try:
            pred_sig=torch.sigmoid(pred_logits); gt_float=gt_mask.float()
            inter=(pred_sig*gt_float).sum(dim=(-1,-2)); union=pred_sig.sum(dim=(-1,-2))+gt_float.sum(dim=(-1,-2))
            obj_loss=1.0-(2.*inter+1e-8)/(union+1e-8) # Dice 유사
            reg_loss=torch.abs(pred_sig-gt_float).mean(dim=(-1,-2)) # L1
            str_loss=(obj_loss+reg_loss).mean()
            if torch.isnan(str_loss) or torch.isinf(str_loss): return torch.tensor(1.0).to(pred_logits.device)
            return str_loss
        except Exception as e: logger.error(f"StructureLoss error: {e}"); return torch.tensor(1.0).to(pred_logits.device)

# ▼▼▼ [검증된 방식 2] Boundary Loss 추가 ▼▼▼
class BoundaryLoss(nn.Module):
    """
    GT 마스크의 경계 영역에서 L1 Loss를 계산하는 간단한 경계 손실
    """
    def __init__(self, kernel_size=3, reduction='mean'):
        super(BoundaryLoss, self).__init__()
        self.reduction = reduction
        # 1채널 입력, 1채널 출력, kernel_size, groups=1
        laplacian_kernel = torch.tensor(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
            dtype=torch.float32
        ).reshape(1, 1, kernel_size, kernel_size)
        self.register_buffer('laplacian_kernel', laplacian_kernel)

    def forward(self, pred_logits, gt_mask):
        try:
            pred_sig = torch.sigmoid(pred_logits)
            gt_float = gt_mask.float()

            # GT의 경계 맵 생성 (Padding='same'과 동일하게)
            # 1채널 conv이므로 gt_float의 채널(C=1)과 커널(in_channels=1)이 맞음
            gt_boundary = F.conv2d(gt_float, self.laplacian_kernel, padding=(self.laplacian_kernel.shape[-1] // 2))
            gt_boundary = (gt_boundary > 0.1).float() # 경계 영역 (True/False -> 1.0/0.0)

            # 경계 영역에서의 L1 Loss 계산
            boundary_loss_map = torch.abs(pred_sig - gt_float) * gt_boundary
            
            # 정규화
            if self.reduction == 'mean':
                total_boundary_pixels = gt_boundary.sum()
                if total_boundary_pixels > 0:
                    return boundary_loss_map.sum() / total_boundary_pixels # 경계 영역 픽셀 수로 정규화
                else:
                    return boundary_loss_map.mean() # 경계가 없는 경우 (매우 드묾)
            else: # 'sum'
                return boundary_loss_map.sum()
        except Exception as e:
            logger.error(f"BoundaryLoss error: {e}"); return torch.tensor(0.0).to(pred_logits.device) # 오류 시 0 반환
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# ▼▼▼ CombinedLoss 수정 (Boundary Loss 추가) ▼▼▼
class CombinedLoss(nn.Module):
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0,
                 dice_weight=1.0, structure_weight=1.0,
                 enhancement_weight=0.0,
                 boundary_weight=0.0): # <-- boundary_weight 추가
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        self.structure_loss = StructureLoss()
        self.boundary_loss = BoundaryLoss() # <-- Boundary Loss 인스턴스화
        self.l1_loss = nn.L1Loss() # <-- Enhancement 용 L1 Loss
        
        self.dice_weight = dice_weight
        self.structure_weight = structure_weight
        self.enhancement_weight = enhancement_weight
        self.boundary_weight = boundary_weight # <-- 가중치 저장

    def forward(self, pred_logits, gt_mask, reconstructed_img=None, original_img=None):
        loss_focal = self.focal_loss(pred_logits, gt_mask)
        loss_dice = self.dice_loss(pred_logits, gt_mask)
        loss_structure = self.structure_loss(pred_logits, gt_mask)
        
        # Segmentation Loss (기본)
        loss_seg = loss_focal + self.dice_weight * loss_dice + self.structure_weight * loss_structure

        # Boundary Loss (경계)
        loss_boundary = torch.tensor(0.0, device=pred_logits.device)
        if self.boundary_weight > 0:
            loss_boundary = self.boundary_loss(pred_logits, gt_mask)
            if torch.isnan(loss_boundary) or torch.isinf(loss_boundary): 
                logger.warning("Boundary Loss is NaN/Inf, setting to 0.")
                loss_boundary = torch.tensor(0.0).to(pred_logits.device)
        
        # Enhancement Loss (복원)
        loss_enhance = torch.tensor(0.0, device=pred_logits.device)
        if self.enhancement_weight > 0 and reconstructed_img is not None and original_img is not None:
            if reconstructed_img.shape[-2:] != original_img.shape[-2:]:
                 reconstructed_img = F.interpolate(reconstructed_img, size=original_img.shape[-2:], mode='bilinear', align_corners=False)
            if reconstructed_img.shape == original_img.shape:
                loss_enhance = self.l1_loss(reconstructed_img, original_img)
                if torch.isnan(loss_enhance) or torch.isinf(loss_enhance): 
                    logger.warning("Enhancement Loss is NaN/Inf, setting to 0.")
                    loss_enhance = torch.tensor(0.0).to(pred_logits.device)
            else: 
                logger.warning(f"Enh loss skipped: Shape mismatch Recon{reconstructed_img.shape} vs Orig{original_img.shape}")

        # Total Loss (모두 합산)
        total_loss = loss_seg + self.boundary_weight * loss_boundary + self.enhancement_weight * loss_enhance

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error(f"NaN/Inf total loss! Seg:{loss_seg.item():.4f}, Bnd:{loss_boundary.item():.4f}, Enh:{loss_enhance.item():.4f}")
            # NaN/Inf 발생 시, 0이 아닌 손실(예: loss_seg)이나 큰 값 반환 (학습 중단 방지)
            if not (torch.isnan(loss_seg) or torch.isinf(loss_seg)):
                 total_loss = loss_seg # Seg Loss만이라도 반환
            else:
                 # requires_grad=True를 설정해야 backward() 호출 가능
                 total_loss = torch.tensor(10.0, device=pred_logits.device, requires_grad=True) 
            
        loss_dict = {
            'total': total_loss.detach().clone(),
            'focal': loss_focal.detach().clone(), 'dice': loss_dice.detach().clone(),
            'structure': loss_structure.detach().clone(), 'boundary': loss_boundary.detach().clone(), # <-- 경계 Loss 로깅
            'enhancement': loss_enhance.detach().clone()
        }
        return total_loss, loss_dict
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲