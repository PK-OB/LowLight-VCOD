# models/main_model_exp5.py
# (Exp 5용: ConvGRU 제거 버전)

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange
import logging

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# (ConvGRUCell 클래스는 사용 안 하지만 에러 방지용으로 남겨둠)
class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvGRUCell, self).__init__()
        # ... (내용 생략) ...

class RefinementBlock(nn.Module):
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
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        out = self.relu(identity + out)
        return out

class PPM(nn.Module):
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
    def __init__(self, backbone_channels, gru_hidden_dim, decoder_channel=64):
        super().__init__()
        num_stages = len(backbone_channels)
        self.ppm = PPM(backbone_channels[-1], decoder_channel * 4)
        self.refinement_stages = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        in_ch_stage3 = gru_hidden_dim + (decoder_channel * 4)
        self.refinement_stages.append(RefinementBlock(in_ch_stage3, decoder_channel))

        for i in range(num_stages - 2, 0, -1):
            in_ch = decoder_channel + backbone_channels[i]
            self.refinement_stages.append(RefinementBlock(in_ch, decoder_channel))
            self.upsample_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

        in_ch_stage0 = decoder_channel + backbone_channels[0]
        self.refinement_stages.append(RefinementBlock(in_ch_stage0, decoder_channel))
        self.upsample_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

    def forward(self, backbone_features, gru_output):
        ppm_out = self.ppm(backbone_features[3])
        x = torch.cat([gru_output, ppm_out], dim=1)
        x = self.refinement_stages[0](x)
        
        for i in range(1, len(self.refinement_stages) - 1):
             x = self.upsample_layers[i-1](x)
             x = torch.cat([x, backbone_features[3 - i]], dim=1) 
             x = self.refinement_stages[i](x)

        x = self.upsample_layers[-1](x)
        x = torch.cat([x, backbone_features[0]], dim=1)
        x = self.refinement_stages[-1](x)
        return x 

class SimpleEnhancementHead(nn.Module):
    def __init__(self, in_channels, mid_channels=64, out_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1), nn.Sigmoid()
        )
    def forward(self, x): return self.conv(x)

class SegmentationRefinementHead(nn.Module):
    def __init__(self, in_channels, mid_channels=32, out_channels=1):
        super(SegmentationRefinementHead, self).__init__()
        self.up_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True)
        )
        self.up_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.up_conv1(x)
        x = self.up_conv2(x)
        x = self.final_conv(x)
        return x

class DCNetStyleVCOD(nn.Module):
    def __init__(self,
                 backbone_name='swin_small_patch4_window7_224', input_size=(224, 224),
                 num_frames=8, pretrained=True, gru_hidden_dim=128,
                 decoder_channel=64, use_enhancement=True):
        super().__init__()
        self.num_frames = num_frames; self.gru_hidden_dim = gru_hidden_dim; self.use_enhancement = use_enhancement

        try:
             self.backbone=timm.create_model(backbone_name,pretrained=pretrained,features_only=True,out_indices=(0,1,2,3))
             if hasattr(self.backbone.feature_info,'channels'): self.bb_ch=self.backbone.feature_info.channels()
             else: self.bb_ch=[info['num_chs'] for info in self.backbone.feature_info]
             logger.info(f"Loaded backbone '{backbone_name}' CH: {self.bb_ch}")
             self.needs_permute=False; dummy=torch.randn(2,3,input_size[0],input_size[1])
             with torch.no_grad(): dummy_f=self.backbone(dummy)
             if isinstance(dummy_f,list) and dummy_f and isinstance(dummy_f[0],torch.Tensor):
                 if len(dummy_f[0].shape)==4 and dummy_f[0].shape[-1]==self.bb_ch[0]: self.needs_permute=True
        except Exception as e: logger.error(f"Backbone load fail: {e}."); raise e

        # ▼▼▼ [수정 1] ConvGRU 제거 -> Simple Conv Adapter ▼▼▼
        # self.conv_gru = ConvGRUCell(self.bb_ch[-1], gru_hidden_dim, kernel_size=3)
        self.conv_adapter = nn.Conv2d(self.bb_ch[-1], gru_hidden_dim, kernel_size=3, padding=1)
        logger.info(f"Exp5 Mode: ConvGRU Replaced with Conv2d (Temporal Context Disabled).")
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        self.decoder = CascadedDecoder(self.bb_ch, gru_hidden_dim, decoder_channel)
        self.seg_head = SegmentationRefinementHead(decoder_channel, mid_channels=decoder_channel // 2)

        if self.use_enhancement:
            self.enhance_head = SimpleEnhancementHead(decoder_channel, mid_channels=decoder_channel // 2)
        else: self.enhance_head = None

    def forward(self, x):
        b, t, c, h, w = x.shape
        x_flat = rearrange(x, 'b t c h w -> (b t) c h w').contiguous()
        features_flat_list = self.backbone(x_flat)

        features_processed = []
        if isinstance(features_flat_list, list):
            for i, feat in enumerate(features_flat_list):
                if self.needs_permute and len(feat.shape)==4 and feat.shape[-1]==self.bb_ch[i]: 
                    feat=feat.permute(0,3,1,2).contiguous()
                features_processed.append(feat)

        # ▼▼▼ [수정 2] 시간적 반복문(Loop) 제거 ▼▼▼
        # gru_hidden = self.conv_gru.init_hidden(...) 
        # for frame_idx in range(t): ...
        
        gru_in_feat = features_processed[-1] # (B*T, C, H, W)
        gru_outputs_flat = self.conv_adapter(gru_in_feat) # (B*T, Hidden, H, W) -> No temporal mixing
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        final_decoder_feat_flat = self.decoder(features_processed, gru_outputs_flat)
        mask_logits_flat = self.seg_head(final_decoder_feat_flat)
        predicted_masks_seq = rearrange(mask_logits_flat, '(b t) c h w -> b t c h w', b=b)

        reconstructed_images_flat = None
        if self.use_enhancement and self.enhance_head is not None:
            enhance_output_small_flat = self.enhance_head(final_decoder_feat_flat)
            reconstructed_images_flat = F.interpolate(enhance_output_small_flat, size=(h, w), mode='bilinear', align_corners=False)

        return predicted_masks_seq, reconstructed_images_flat