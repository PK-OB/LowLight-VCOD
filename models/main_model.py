# pk-ob/jed-vcod/JED-VCOD-cc543b29cefb3a45b940bfd01f42c33af7a6bb25/models/main_model.py
# (Video Swin 백본 출력 Permute 로직 추가 버전)

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange
import logging # 로깅 추가

logger = logging.getLogger(__name__) # 로거 설정

# --- UPerNet 스타일 Decoder ---
class PPM(nn.Module):
    """Pyramid Pooling Module used in UPerNet (이전과 동일)"""
    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.pool_layers = nn.ModuleList()
        for scale in pool_scales:
            self.pool_layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.conv_last = nn.Sequential(
                nn.Conv2d(in_channels + len(pool_scales) * out_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        h, w = x.shape[-2:]
        features = [x]
        for pool_layer in self.pool_layers:
            pooled_features = pool_layer(x)
            upsampled_features = F.interpolate(pooled_features, size=(h, w), mode='bilinear', align_corners=False)
            features.append(upsampled_features)
        out = torch.cat(features, dim=1)
        out = self.conv_last(out)
        return out

class UPerNetDecoder(nn.Module):
    """Simplified UPerNet Decoder Head (이전과 동일)"""
    def __init__(self, backbone_channels, decoder_channel=256):
        super().__init__()
        self.ppm = PPM(backbone_channels[-1], decoder_channel)
        self.fpn_stages = nn.ModuleList()
        for i in range(len(backbone_channels) - 1, -1, -1):
            lateral_conv = nn.Conv2d(backbone_channels[i], decoder_channel, 1)
            output_conv = nn.Conv2d(decoder_channel, decoder_channel, 3, padding=1)
            self.fpn_stages.append(nn.ModuleDict({
                'lateral': lateral_conv,
                'output': output_conv
            }))
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(len(backbone_channels) * decoder_channel, decoder_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        fpn_features = [self.ppm(features[-1])] 
        for i in range(len(features) - 2, -1, -1):
            prev_feature = fpn_features[0] 
            current_feature = features[i]  
            lateral = self.fpn_stages[len(features) - 1 - i]['lateral'](current_feature)
            top_down = F.interpolate(prev_feature, size=lateral.shape[-2:], mode='bilinear', align_corners=False) + lateral
            fpn_output = self.fpn_stages[len(features) - 1 - i]['output'](top_down)
            fpn_features.insert(0, fpn_output)
        h, w = fpn_features[0].shape[-2:]
        for i in range(1, len(fpn_features)):
            fpn_features[i] = F.interpolate(fpn_features[i], size=(h, w), mode='bilinear', align_corners=False)
        fused_features = torch.cat(fpn_features, dim=1)
        fused_features = self.fusion_conv(fused_features)
        return fused_features

# --- Enhancement Head ---
class SimpleEnhancementHead(nn.Module):
    """간단한 Enhancement Head (이전과 동일)"""
    def __init__(self, in_channels, mid_channels=64, out_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.conv(x)

# --- Main Model ---
class VideoSwinVCOD(nn.Module):
    def __init__(self, 
                 backbone_name='swin_small_patch4_window7_224', 
                 input_size=(224, 224), 
                 num_frames=8, 
                 pretrained=True, 
                 decoder_channel=256,
                 use_enhancement=True):
        super().__init__()
        self.num_frames = num_frames
        self.use_enhancement = use_enhancement

        # 1. Video Swin Transformer 백본 로드
        try:
             self.backbone = timm.create_model(
                 backbone_name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3) 
             )
             # backbone_channels 순서 확인 (timm 최신 버전은 channels() 사용)
             if hasattr(self.backbone.feature_info, 'channels'):
                 backbone_channels = self.backbone.feature_info.channels() 
             else: # 구버전 호환
                 backbone_channels = [info['num_chs'] for info in self.backbone.feature_info]
                 
             logger.info(f"Loaded backbone '{backbone_name}' with feature channels: {backbone_channels}")
             # Swin Transformer 출력 형태 확인 (대부분 B, H, W, C) - Permute 필요성 확인
             self.backbone_output_format_bhwc = True # True로 가정
             dummy_input = torch.randn(2, 3, input_size[0], input_size[1]) # B, C, H, W
             dummy_features = self.backbone(dummy_input)
             if isinstance(dummy_features, list) and len(dummy_features) > 0 and isinstance(dummy_features[0], torch.Tensor):
                  # 첫 번째 특징 맵의 채널 차원 확인
                  if dummy_features[0].shape[1] == backbone_channels[0]:
                       self.backbone_output_format_bhwc = False # B, C, H, W 형태면 Permute 불필요
                       logger.info("Backbone output format detected as B, C, H, W.")
                  else:
                       logger.info("Backbone output format detected as B, H, W, C (Permute needed).")
             else:
                  logger.warning("Could not determine backbone output format, assuming B, H, W, C.")

        except Exception as e:
            logger.error(f"Failed to load backbone '{backbone_name}': {e}.")
            raise e
            
        # 2. UPerNet Decoder Head
        self.decoder = UPerNetDecoder(backbone_channels, decoder_channel)

        # 3. Segmentation 최종 출력 레이어
        self.seg_final_conv = nn.Conv2d(decoder_channel, 1, kernel_size=1)

        # 4. (옵션) Enhancement Head
        if self.use_enhancement:
            self.enhance_head = SimpleEnhancementHead(decoder_channel, mid_channels=64, out_channels=3)
            logger.info("Enhancement head enabled.")

    def forward(self, x):
        # 입력: x shape = (B, T, C, H, W)
        b, t, c, h, w = x.shape

        # 백본 입력 형태로 변환: (B, T, C, H, W) -> (B*T, C, H, W)
        x_flat = rearrange(x, 'b t c h w -> (b t) c h w').contiguous() # contiguous 추가

        # Video Swin Transformer 백본 Forward
        features_flat = self.backbone(x_flat) # 각 스테이지 출력 리스트

        # ▼▼▼ [오류 수정] 백본 출력 Permute ▼▼▼
        features_processed = []
        if isinstance(features_flat, list):
            for i, feat in enumerate(features_flat):
                if isinstance(feat, torch.Tensor):
                    # B, H, W, C 형태일 경우 B, C, H, W로 변경
                    if self.backbone_output_format_bhwc and len(feat.shape) == 4 and feat.shape[1] != self.backbone.feature_info.channels()[i]:
                         feat_permuted = feat.permute(0, 3, 1, 2).contiguous()
                         features_processed.append(feat_permuted)
                         # print(f"DEBUG: Permuted Stage {i} feature from {feat.shape} to {feat_permuted.shape}") # 디버그용 출력
                    else:
                         features_processed.append(feat.contiguous()) # B, C, H, W 형태면 그대로 사용
                         # print(f"DEBUG: Stage {i} feature shape OK: {feat.shape}") # 디버그용 출력
                else:
                    logger.warning(f"Backbone stage {i} output is not a tensor: {type(feat)}. Decoder might fail.")
                    # 오류를 피하기 위해 None 또는 더미 텐서를 추가할 수 있으나, 일단 그대로 전달 시도
                    features_processed.append(feat) 
        else:
             logger.error(f"Backbone output is not a list ({type(features_flat)}). Decoder expects a list of features.")
             # 치명적 오류, 빈 리스트 반환 또는 예외 발생
             features_processed = []

        if not features_processed or len(features_processed) != len(self.backbone.feature_info.channels()):
             logger.error("Processed features list is invalid. Cannot proceed.")
             # 오류 상황 처리: 임시 더미 출력 반환 (학습 스크립트 오류 방지용)
             dummy_mask = torch.zeros((b, t, 1, h, w), device=x.device)
             dummy_recon = torch.zeros((b * t, c, h, w), device=x.device) if self.use_enhancement else None
             return dummy_mask, dummy_recon
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # UPerNet Decoder Forward (이제 채널 차원이 올바른 특징 리스트 사용)
        decoder_output_flat = self.decoder(features_processed) # (B*T, decoder_channel, H/4, W/4)

        # Segmentation 예측
        seg_logits_small_flat = self.seg_final_conv(decoder_output_flat) # (B*T, 1, H/4, W/4)
        
        # 원본 크기로 업샘플링
        seg_logits_flat = F.interpolate(
            seg_logits_small_flat, size=(h, w), mode='bilinear', align_corners=False
        ) # (B*T, 1, H, W)

        # (옵션) Enhancement 예측
        reconstructed_images_flat = None
        if self.use_enhancement:
            enhance_output_flat = self.enhance_head(decoder_output_flat) # (B*T, 3, H/4, W/4)
            reconstructed_images_flat = F.interpolate(
                enhance_output_flat, size=(h, w), mode='bilinear', align_corners=False
            ) # (B*T, 3, H, W)

        # 최종 출력 형태 복원: (B*T, ...) -> (B, T, ...)
        predicted_masks_seq = rearrange(seg_logits_flat, '(b t) c h w -> b t c h w', b=b)

        # Multi-task 출력 반환
        if self.use_enhancement:
            return predicted_masks_seq, reconstructed_images_flat 
        else:
            return predicted_masks_seq, None