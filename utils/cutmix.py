# pk-ob/jed-vcod/JED-VCOD-cc543b29cefb3a45b940bfd01f42c33af7a6bb25/utils/cutmix.py

import torch
import numpy as np

def rand_bbox(size, lam):
    """ Bounding box 생성을 위한 함수 """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, W) # <-- (오타 수정: H -> W) 원본 파일에 W로 되어있어 유지합니다. (H가 맞을 수 있음)

    return bbx1, bby1, bbx2, bby2

# ▼▼▼ 수정된 부분: 3개의 텐서를 받도록 z 인자 추가 ▼▼▼
def cutmix_data(x, y, z, alpha=1.0, use_cuda=True):
    '''
    CutMix를 수행합니다.
    x: 야간 이미지 텐서 (B*T, C, H, W)
    y: 마스크 텐서 (B*T, 1, H, W)
    z: 주간 이미지 텐서 (B*T, C, H, W)
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    y[:, :, bby1:bby2, bbx1:bbx2] = y[index, :, bby1:bby2, bbx1:bbx2]
    z[:, :, bby1:bby2, bbx1:bbx2] = z[index, :, bby1:bby2, bbx1:bbx2] # <-- 추가
    
    return x, y, z # <-- 3개 반환
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲