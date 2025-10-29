# pk-ob/jed-vcod/JED-VCOD-cc543b29cefb3a45b940bfd01f42c33af7a6bb25/models/dae_module.py
# (ResNet Encoder, Attention Gate, Residual Decoder Blocks 적용 버전)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ▼▼▼ 3. SOTA 개선: Basic Residual Block ▼▼▼
class BasicResidualBlock(nn.Module):
    """
    ResNet의 기본 Residual Block (2개의 3x3 Conv)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# (ConvBlock은 이제 사용되지 않으므로 주석 처리 또는 삭제 가능)
# class ConvBlock(nn.Module): ... 

class AttentionGate(nn.Module):
    """Attention U-Net 게이트 (변경 없음)"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g_prime = self.W_g(g)
        x_prime = self.W_x(x)
        if g_prime.shape[2:] != x_prime.shape[2:]:
            g_prime = F.interpolate(g_prime, size=x_prime.shape[2:], mode='bilinear', align_corners=False)
        psi_input = self.relu(g_prime + x_prime)
        alpha = self.psi(psi_input)
        return x * alpha

class DAEModule(nn.Module):
    """
    SOTA 개선:
    1. 인코더 (ResNet-34)
    2. AttentionGate
    3. 디코더 블록을 BasicResidualBlock으로 교체
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]): # features는 하위 호환용
        super().__init__()
        
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        if in_channels != 3:
            self.init_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.init_conv = resnet.conv1 # 1/2

        self.encoder0_0 = nn.Sequential(resnet.bn1, resnet.relu) # 1/2
        self.encoder0_1 = resnet.maxpool # 1/4
        self.encoder1 = resnet.layer1 # 64 ch, 1/4
        self.encoder2 = resnet.layer2 # 128 ch, 1/8
        self.encoder3 = resnet.layer3 # 256 ch, 1/16
        self.encoder4 = resnet.layer4 # 512 ch, 1/32
        
        # ▼▼▼ 3. SOTA 개선: 병목 구간도 Residual Block 사용 (옵션, 성능 향상 기대) ▼▼▼
        self.bottleneck = BasicResidualBlock(512, 1024, stride=1) # stride=1 유지
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        self.decoder_channels = [512, 256, 128, 64] 
        self.skip_channels = [256, 128, 64, 64] 

        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        
        in_ch = 1024 # Bottleneck 출력
        
        for i in range(len(self.decoder_channels)):
            dec_ch = self.decoder_channels[i] 
            skip_ch = self.skip_channels[i]   
            
            self.upconvs.append(
                nn.ConvTranspose2d(in_ch, dec_ch, kernel_size=2, stride=2)
            )
            self.attention_gates.append(
                AttentionGate(F_g=dec_ch, F_l=skip_ch, F_int=dec_ch // 2)
            )
            
            # ▼▼▼ 3. SOTA 개선: 디코더 블록을 BasicResidualBlock으로 교체 ▼▼▼
            self.decoder_blocks.append(
                BasicResidualBlock(dec_ch + skip_ch, dec_ch, stride=1)
            )
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
            
            in_ch = dec_ch 

        # 이미지 복원 헤드
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(self.decoder_channels[-1], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x_in):
        # --- Encoder ---
        x0 = self.init_conv(x_in) 
        x0_relu = self.encoder0_0(x0) 
        e1_in = self.encoder0_1(x0_relu) 
        e1 = self.encoder1(e1_in) 
        e2 = self.encoder2(e1) 
        e3 = self.encoder3(e2) 
        e4 = self.encoder4(e3) 
        skip_connections = [e3, e2, e1, x0_relu] 
        
        # --- Bottleneck ---
        b = self.bottleneck(e4)

        multi_scale_features = []
        x = b 

        # --- Decoder ---
        for i in range(len(self.decoder_channels)):
            x = self.upconvs[i](x) 
            skip_connection = skip_connections[i] 
            gated_skip_connection = self.attention_gates[i](g=x, x=skip_connection)
            concat_skip = torch.cat((gated_skip_connection, x), dim=1)
            x = self.decoder_blocks[i](concat_skip) # <-- Residual Block 통과
            multi_scale_features.append(x)

        # --- Reconstruction Head ---
        reconstructed_image_half = self.reconstruction_head(x)
        reconstructed_image = F.interpolate(
            reconstructed_image_half, 
            size=x_in.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        return multi_scale_features, reconstructed_image