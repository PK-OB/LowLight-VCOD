# pk-ob/lowlight-vcod/LowLight-VCOD-3129585e63b4dbf2e754bab7cdcb562f81d620a6/models/main_model.py
# (Decoder Refinement Head 강화 버전)

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange
import logging

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- ConvGRU Cell ---
class ConvGRUCell(nn.Module):
    """(변경 없음)"""
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvGRUCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv_gates = nn.Conv2d(input_dim + hidden_dim, 2 * hidden_dim, kernel_size, padding=padding, bias=bias)
        self.conv_candidate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding, bias=bias)

    def forward(self, input_tensor, h_cur):
        combined = torch.cat([input_tensor, h_cur], dim=1)
        gates = self.conv_gates(combined)
        reset_gate, update_gate = gates.chunk(2, 1)

        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)

        combined_reset = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        candidate = torch.tanh(self.conv_candidate(combined_reset))

        h_next = (1 - update_gate) * h_cur + update_gate * candidate
        return h_next

    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size
        return torch.zeros(batch_size, self.hidden_dim, height, width, device=device)

# --- Cascaded Refinement Decoder ---
class RefinementBlock(nn.Module):
    """(변경 없음)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
             self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
             self.shortcut = nn.Identity()
             
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(identity + out) # 간단한 잔차 연결
        return out

# (PPM 클래스 추가 - UPerNetDecoder에서 가져옴)
class PPM(nn.Module):
    """Pyramid Pooling Module (변경 없음)"""
    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
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

class CascadedDecoder(nn.Module):
    """Cascaded Decoder (final_conv 제거)"""
    def __init__(self, backbone_channels, gru_hidden_dim, decoder_channel=64):
        super().__init__()
        num_stages = len(backbone_channels) # 4
        
        # PPM (가장 깊은 특징 사용)
        self.ppm = PPM(backbone_channels[-1], decoder_channel * 4) # 예: 256 ch
        
        self.refinement_stages = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        # Stage 3: ConvGRU 출력 + PPM 출력 융합
        in_ch_stage3 = gru_hidden_dim + (decoder_channel * 4) # GRU + PPM
        self.refinement_stages.append(RefinementBlock(in_ch_stage3, decoder_channel)) # (H/32)

        # Stage 2, 1
        for i in range(num_stages - 2, 0, -1): # i = 2 (Stage 2), 1 (Stage 1)
            in_ch = decoder_channel + backbone_channels[i] # 이전 stage 출력 + 현재 stage 백본 특징
            self.refinement_stages.append(RefinementBlock(in_ch, decoder_channel))
            self.upsample_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

        # Stage 0
        in_ch_stage0 = decoder_channel + backbone_channels[0]
        self.refinement_stages.append(RefinementBlock(in_ch_stage0, decoder_channel))
        self.upsample_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        
        # ▼▼▼ [검증된 방식 1] final_conv 제거 ▼▼▼
        # self.final_conv = nn.Conv2d(decoder_channel, 1, kernel_size=1)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    def forward(self, backbone_features, gru_output):
        # backbone_features: 리스트 [feat_s0(H/4), feat_s1(H/8), feat_s2(H/16), feat_s3(H/32)] (얕->깊)
        # gru_output: ConvGRU의 최종 hidden state (BT, hidden, H/32, W/32)

        # PPM 처리 (Swin S3 특징 사용)
        ppm_out = self.ppm(backbone_features[3]) # (BT, dec_ch*4, H/32, W/32)

        # Stage 3 (가장 깊은 레벨)
        x = torch.cat([gru_output, ppm_out], dim=1) # GRU + PPM
        x = self.refinement_stages[0](x) # (BT, dec_ch, H/32, W/32)
        
        # Stage 2, 1 (Top-down)
        # refinement_stages[0] -> stage 3 결과
        # refinement_stages[1] -> stage 2 처리 블록
        # refinement_stages[2] -> stage 1 처리 블록
        for i in range(1, len(self.refinement_stages) - 1): # i = 1 (Stage 2), 2 (Stage 1)
             # Upsample (H/32->H/16, H/16->H/8)
             x = self.upsample_layers[i-1](x)
             # Concat (bb_f 인덱스는 3-i -> 2, 1)
             x = torch.cat([x, backbone_features[3 - i]], dim=1) 
             x = self.refinement_stages[i](x) # Refine

        # Stage 0 (마지막 refinement)
        x = self.upsample_layers[-1](x) # H/4 크기로 업샘플링
        x = torch.cat([x, backbone_features[0]], dim=1) # Concat S0
        x = self.refinement_stages[-1](x) # (BT, dec_ch, H/4, W/4)

        # ▼▼▼ [검증된 방식 1] 최종 H/4 특징 반환 ▼▼▼
        return x 
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# (SimpleEnhancementHead 클래스는 이전과 동일)
class SimpleEnhancementHead(nn.Module):
    def __init__(self, in_channels, mid_channels=64, out_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1), nn.Sigmoid()
        )
    def forward(self, x): return self.conv(x)


# ▼▼▼ [검증된 방식 1] 최종 마스크 정제를 위한 헤드 추가 ▼▼▼
class SegmentationRefinementHead(nn.Module):
    """ H/4 특징을 H/1 마스크 로짓으로 정제하는 헤드 """
    def __init__(self, in_channels, mid_channels=32, out_channels=1):
        super(SegmentationRefinementHead, self).__init__()
        # H/4 -> H/2
        self.up_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True)
        ) # (BT, mid_channels, H/2, W/2)
        
        # H/2 -> H/1
        self.up_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True)
        ) # (BT, mid_channels, H/1, W/1)
        
        self.final_conv = nn.Conv2d(mid_channels, out_channels, 1) # 최종 마스크 (Logits)

    def forward(self, x): # x: (BT, in_channels, H/4, W/4)
        x = self.up_conv1(x)
        x = self.up_conv2(x)
        x = self.final_conv(x) # (BT, 1, H/1, W/1)
        return x
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲


# --- Main Model ---
class DCNetStyleVCOD(nn.Module):
    def __init__(self,
                 backbone_name='swin_small_patch4_window7_224', input_size=(224, 224),
                 num_frames=8, pretrained=True, gru_hidden_dim=128,
                 decoder_channel=64, use_enhancement=True):
        super().__init__()
        self.num_frames = num_frames; self.gru_hidden_dim = gru_hidden_dim; self.use_enhancement = use_enhancement

        # 1. Swin Backbone
        try:
             self.backbone=timm.create_model(backbone_name,pretrained=pretrained,features_only=True,out_indices=(0,1,2,3))
             if hasattr(self.backbone.feature_info,'channels'): self.bb_ch=self.backbone.feature_info.channels()
             else: self.bb_ch=[info['num_chs'] for info in self.backbone.feature_info]
             logger.info(f"Loaded backbone '{backbone_name}' CH: {self.bb_ch}")
             self.needs_permute=False; dummy=torch.randn(2,3,input_size[0],input_size[1])
             with torch.no_grad(): dummy_f=self.backbone(dummy)
             if isinstance(dummy_f,list) and dummy_f and isinstance(dummy_f[0],torch.Tensor):
                 if len(dummy_f[0].shape)==4 and dummy_f[0].shape[-1]==self.bb_ch[0]: self.needs_permute=True; logger.info("Backbone output: BHWC (Permute needed).")
                 else: logger.info("Backbone output: BCHW (No permute needed).")
             else: logger.warning("Assuming backbone output: BCHW.")
        except Exception as e: logger.error(f"Backbone load fail: {e}."); raise e

        # 2. ConvGRU
        self.conv_gru = ConvGRUCell(self.bb_ch[-1], gru_hidden_dim, kernel_size=3); logger.info(f"ConvGRU init: in={self.bb_ch[-1]}, hid={gru_hidden_dim}")

        # 3. Cascaded Decoder
        self.decoder = CascadedDecoder(self.bb_ch, gru_hidden_dim, decoder_channel); logger.info(f"Cascaded Decoder init: out_ch={decoder_channel}")

        # ▼▼▼ [검증된 방식 1] Seg Head 변경 ▼▼▼
        self.seg_head = SegmentationRefinementHead(decoder_channel, mid_channels=decoder_channel // 2)
        logger.info(f"Segmentation Refinement Head enabled (in={decoder_channel}).")
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        if self.use_enhancement:
            self.enhance_head = SimpleEnhancementHead(decoder_channel, mid_channels=decoder_channel // 2)
            logger.info("Enhancement head enabled.")
        else: self.enhance_head = None

    def forward(self, x):
        b, t, c, h, w = x.shape
        x_flat = rearrange(x, 'b t c h w -> (b t) c h w').contiguous()
        features_flat_list = self.backbone(x_flat)

        # 백본 출력 처리 (Permute & 유효성 검사)
        features_processed = []
        if isinstance(features_flat_list, list) and len(features_flat_list) == len(self.bb_ch):
            for i, feat in enumerate(features_flat_list):
                if isinstance(feat, torch.Tensor):
                    exp_ch = self.bb_ch[i]
                    if self.needs_permute and len(feat.shape)==4 and feat.shape[-1]==exp_ch: feat=feat.permute(0,3,1,2).contiguous()
                    if len(feat.shape)==4 and feat.shape[1]==exp_ch: features_processed.append(feat)
                    else: logger.error(f"S{i} shape err: {feat.shape}, exp CH: {exp_ch}. Stop."); return torch.zeros((b,t,1,h,w),device=x.device), None
                else: logger.error(f"S{i} not tensor: {type(feat)}. Stop."); return torch.zeros((b,t,1,h,w),device=x.device), None
        else: logger.error(f"Backbone out invalid. Got {type(features_flat_list)}. Stop."); return torch.zeros((b,t,1,h,w),device=x.device), None


        # 시간 모델링 (ConvGRU)
        gru_in_feat = features_processed[-1]; gru_h,gru_w = gru_in_feat.shape[-2:]
        gru_in_seq = rearrange(gru_in_feat, '(b t) c h w -> b t c h w', b=b)
        gru_hidden = self.conv_gru.init_hidden(b, (gru_h, gru_w), device=x.device); gru_outputs = []
        for frame_idx in range(t): gru_hidden = self.conv_gru(gru_in_seq[:, frame_idx], gru_hidden); gru_outputs.append(gru_hidden)
        gru_outputs_tensor = torch.stack(gru_outputs, dim=1); gru_outputs_flat = rearrange(gru_outputs_tensor, 'b t c h w -> (b t) c h w')

        # Cascaded Decoder (H/4 특징 반환)
        final_decoder_feat_flat = self.decoder(features_processed, gru_outputs_flat) # (BT, dec_ch, H/4, W/4)

        # ▼▼▼ [검증된 방식 1] Seg Head 수정 ▼▼▼
        # H/4 특징을 입력받아 H/1 크기 로짓으로 정제
        mask_logits_flat = self.seg_head(final_decoder_feat_flat) # (BT, 1, H, W)
        predicted_masks_seq = rearrange(mask_logits_flat, '(b t) c h w -> b t c h w', b=b)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # Enhancement 예측 (H/4 특징 사용, 원본 크기로 업샘플링)
        reconstructed_images_flat = None
        if self.use_enhancement and self.enhance_head is not None:
            enhance_output_small_flat = self.enhance_head(final_decoder_feat_flat) # (BT, 3, H/4, W/4)
            reconstructed_images_flat = F.interpolate(enhance_output_small_flat, size=(h, w), mode='bilinear', align_corners=False)

        return predicted_masks_seq, reconstructed_images_flat