import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Dice loss.
        :param inputs: A tensor of shape (B, T, C, H, W) or (B, C, H, W), logits from the model.
        :param targets: A tensor of the same shape as inputs, ground truth.
        """
        inputs = torch.sigmoid(inputs)

        # ▼▼▼ 수정된 부분 ▼▼▼
        # .view(-1)을 .reshape(-1)으로 변경하여 non-contiguous tensor 오류를 해결합니다.
        inputs_flat = inputs.reshape(-1)
        targets_flat = targets.reshape(-1)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        intersection = (inputs_flat * targets_flat).sum()
        dice = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)

        return 1 - dice

class FocalLoss(nn.Module):
    """클래스 불균형 문제를 해결하기 위한 Focal Loss"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) # pt = p if y=1, 1-p if y=0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss