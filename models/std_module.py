# pk-ob/jed-vcod/JED-VCOD-cc543b29cefb3a45b940bfd01f42c33af7a6bb25/models/std_module.py
# (Stacked ConvLSTM, Residual Refinement Head 적용 버전)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple # <-- 타입 힌트 추가

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Attention Block (변경 없음)"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvLSTMCell(nn.Module):
    """ConvLSTMCell (변경 없음)"""
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))

# ▼▼▼ 2. SOTA 개선: Residual Refinement Block ▼▼▼
class ResidualRefinementBlock(nn.Module):
    """
    간단한 Residual Block을 사용한 특징 정제 모듈
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualRefinementBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 입력 채널과 출력 채널이 다를 경우 1x1 Conv로 맞춰줌 (옵션)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity # Residual Connection
        out = self.relu(out)
        return out
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

class STDModule(nn.Module):
    """
    SOTA 개선:
    1. Stacked ConvLSTM (2 layers)
    2. Residual Refinement Head
    (Concat + 1x1 Conv 융합은 이전 버전에서 이미 적용됨)
    """
    def __init__(self, in_channels_list, hidden_dim=128, lstm_layers=2): # <-- lstm_layers 추가
        super(STDModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers # <-- LSTM 레이어 수
        self.feature_fusion_convs = nn.ModuleList()
        total_fused_channels = 0

        for in_channels in in_channels_list:
            self.feature_fusion_convs.append(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
            )
            total_fused_channels += hidden_dim
            
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_fused_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.attention = SEBlock(hidden_dim)
        
        # ▼▼▼ 1. SOTA 개선: Stacked ConvLSTM ▼▼▼
        self.conv_lstm_cells = nn.ModuleList()
        for i in range(self.lstm_layers):
            input_dim = hidden_dim if i == 0 else hidden_dim # 첫 레이어 입력=fused, 나머지는 이전 LSTM 출력
            self.conv_lstm_cells.append(
                ConvLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=(3, 3))
            )
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # ▼▼▼ 2. SOTA 개선: Residual Refinement Head ▼▼▼
        # ConvLSTM 최종 출력(hidden_dim)을 받아 정제
        self.refinement_block = ResidualRefinementBlock(hidden_dim, hidden_dim // 2)
        # 최종 1x1 Conv로 마스크 생성
        self.final_conv = nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)
        # (기존 seg_head 제거)
        # self.seg_head = nn.Sequential(...)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    def _init_hidden(self, batch_size, image_size, device):
        # Stacked LSTM을 위한 초기 hidden state 리스트 생성
        init_states = []
        for _ in range(self.lstm_layers):
            init_states.append(self.conv_lstm_cells[0].init_hidden(batch_size, image_size, device))
        return init_states

    def forward(self, multi_scale_features_seq_list, target_size):
        if len(multi_scale_features_seq_list) > 1:
            lstm_size = multi_scale_features_seq_list[1].shape[3:] # (H/8, W/8) 기준
        else:
            lstm_size = multi_scale_features_seq_list[0].shape[3:]

        seq_len = multi_scale_features_seq_list[0].shape[0]
        batch_size = multi_scale_features_seq_list[0].shape[1]
        device = multi_scale_features_seq_list[0].device
        
        # Stacked LSTM 초기화
        hidden_states: List[Tuple[torch.Tensor, torch.Tensor]] = self._init_hidden(batch_size, lstm_size, device)

        outputs = []
        for t in range(seq_len):
            # 특징 융합 (Concat + Conv)
            features_resized_list = []
            for i, features_seq in enumerate(multi_scale_features_seq_list):
                feature_t = self.feature_fusion_convs[i](features_seq[t])
                feature_t_resized = F.interpolate(feature_t, size=lstm_size, mode='bilinear', align_corners=False)
                features_resized_list.append(feature_t_resized)
            
            fused_feature_cat = torch.cat(features_resized_list, dim=1)
            fused_feature = self.fusion_conv(fused_feature_cat)
            fused_feature = self.attention(fused_feature)

            # ▼▼▼ 1. SOTA 개선: Stacked ConvLSTM Forward ▼▼▼
            current_input = fused_feature
            next_hidden_states = []
            for i in range(self.lstm_layers):
                h, c = hidden_states[i] # 현재 레이어의 hidden state
                next_h, next_c = self.conv_lstm_cells[i](input_tensor=current_input, cur_state=(h, c))
                next_hidden_states.append((next_h, next_c))
                current_input = next_h # 다음 레이어의 입력은 현재 레이어의 hidden output
            
            hidden_states = next_hidden_states # 다음 time step을 위해 hidden state 업데이트
            lstm_output = current_input # 마지막 레이어의 hidden output 사용
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            # ▼▼▼ 2. SOTA 개선: Residual Refinement Head ▼▼▼
            refined_feature = self.refinement_block(lstm_output)
            pred_mask = self.final_conv(refined_feature)
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            # 최종 출력 업샘플링
            pred_mask_upsampled = F.interpolate(pred_mask, size=target_size, mode='bilinear', align_corners=False)
            outputs.append(pred_mask_upsampled)
        
        return torch.stack(outputs, dim=0).permute(1, 0, 2, 3, 4)