# pk-ob/jed-vcod/JED-VCOD-cc543b29cefb3a45b940bfd01f42c33af7a6bb25/models/main_model.py
# (DCNet 스타일 + Enhancement Head 복원 버전)

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
    def __init__(self, i_d, h_d, k_s, b=True): super().__init__(); self.h_d=h_d; p=k_s//2; self.conv_g=nn.Conv2d(i_d+h_d,2*h_d,k_s,p=p,b=b); self.conv_c=nn.Conv2d(i_d+h_d,h_d,k_s,p=p,b=b)
    def forward(self, i_t, h_c): cmb=torch.cat([i_t,h_c],dim=1); g=self.conv_g(cmb); r_g,u_g=g.chunk(2,1); r_g=torch.sigmoid(r_g); u_g=torch.sigmoid(u_g); cmb_r=torch.cat([i_t,r_g*h_c],dim=1); cdd=torch.tanh(self.conv_c(cmb_r)); h_n=(1-u_g)*h_c+u_g*cdd; return h_n
    def init_hidden(self, b_s, i_s, dv): h,w=i_s; return torch.zeros(b_s,self.h_d,h,w,device=dv)

# --- Cascaded Refinement Decoder ---
class RefinementBlock(nn.Module):
    def __init__(self, i_c, o_c): super().__init__(); self.c1=nn.Conv2d(i_c,o_c,3,padding=1,bias=False); self.b1=nn.BatchNorm2d(o_c); self.relu=nn.ReLU(inplace=True); self.c2=nn.Conv2d(o_c,o_c,3,padding=1,bias=False); self.b2=nn.BatchNorm2d(o_c); self.sc=nn.Conv2d(i_c,o_c,1,bias=False) if i_c!=o_c else nn.Identity()
    def forward(self, x): i=self.sc(x); o=self.c1(x); o=self.b1(o); o=self.relu(o); o=self.c2(o); o=self.b2(o); o=self.relu(i+o); return o

class CascadedDecoder(nn.Module):
    def __init__(self, bb_ch, gru_h_d, dec_ch=64):
        super().__init__(); n_st=len(bb_ch); self.ref_st=nn.ModuleList(); self.up_l=nn.ModuleList()
        in_ch_s3=gru_h_d+bb_ch[3]; self.ref_st.append(RefinementBlock(in_ch_s3,dec_ch))
        for i in range(n_st-2,0,-1): in_ch=dec_ch+bb_ch[i]; self.ref_st.append(RefinementBlock(in_ch,dec_ch)); self.up_l.append(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False))
        in_ch_s0=dec_ch+bb_ch[0]; self.ref_st.append(RefinementBlock(in_ch_s0,dec_ch)); self.up_l.append(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False))
        self.final_conv=nn.Conv2d(dec_ch,1,1)

    def forward(self, bb_f, gru_o):
        x=torch.cat([gru_o,bb_f[3]],dim=1); x=self.ref_st[0](x) # Stage 3 + GRU -> Refine
        for i in range(1,len(self.ref_st)-1): # Stage 2, 1
            x=self.up_l[i-1](x); x=torch.cat([x,bb_f[3-i]],dim=1); x=self.ref_st[i](x)
        x=self.up_l[-1](x); x=torch.cat([x,bb_f[0]],dim=1); x=self.ref_st[-1](x) # Stage 0 -> Refine
        m_l=self.final_conv(x) # Mask logits prediction
        return m_l, x # Return logits and final decoder feature map

# --- Enhancement Head ---
class SimpleEnhancementHead(nn.Module):
    def __init__(self, in_channels, mid_channels=64, out_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1), nn.Sigmoid()
        )
    def forward(self, x): return self.conv(x)

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
                 if len(dummy_f[0].shape)==4 and dummy_f[0].shape[-1]==self.bb_ch[0]: self.needs_permute=True; logger.info("Backbone output: BHWC.")
                 else: logger.info("Backbone output: BCHW.")
             else: logger.warning("Assuming backbone output: BCHW.")
        except Exception as e: logger.error(f"Backbone load fail: {e}."); raise e

        # 2. ConvGRU
        self.conv_gru = ConvGRUCell(self.bb_ch[-1], gru_hidden_dim, kernel_size=3)
        logger.info(f"ConvGRU init: in={self.bb_ch[-1]}, hid={gru_hidden_dim}")

        # 3. Cascaded Decoder
        self.decoder = CascadedDecoder(self.bb_ch, gru_hidden_dim, decoder_channel)
        logger.info(f"Cascaded Decoder init: out_ch={decoder_channel}")

        # 4. Enhancement Head
        if self.use_enhancement:
            self.enhance_head = SimpleEnhancementHead(decoder_channel, mid_channels=decoder_channel // 2)
            logger.info("Enhancement head enabled.")
        else: self.enhance_head = None

    def forward(self, x):
        b, t, c, h, w = x.shape
        x_flat = rearrange(x, 'b t c h w -> (b t) c h w').contiguous()
        features_flat_list = self.backbone(x_flat)

        # 백본 출력 처리
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
        gru_in_feat = features_processed[-1]
        gru_h, gru_w = gru_in_feat.shape[-2:]
        gru_in_seq = rearrange(gru_in_feat, '(b t) c h w -> b t c h w', b=b)
        gru_hidden = self.conv_gru.init_hidden(b, (gru_h, gru_w), device=x.device)
        gru_outputs = []
        for frame_idx in range(t):
            gru_hidden = self.conv_gru(gru_in_seq[:, frame_idx], gru_hidden)
            gru_outputs.append(gru_hidden)
        gru_outputs_tensor = torch.stack(gru_outputs, dim=1)
        gru_outputs_flat = rearrange(gru_outputs_tensor, 'b t c h w -> (b t) c h w')

        # Cascaded Decoder
        mask_logits_small_flat, final_decoder_feat_flat = self.decoder(features_processed, gru_outputs_flat)

        # Segmentation 업샘플링
        mask_logits_flat = F.interpolate(mask_logits_small_flat, size=(h, w), mode='bilinear', align_corners=False)
        predicted_masks_seq = rearrange(mask_logits_flat, '(b t) c h w -> b t c h w', b=b)

        # Enhancement 예측
        reconstructed_images_flat = None
        if self.use_enhancement and self.enhance_head is not None:
            enhance_output_small_flat = self.enhance_head(final_decoder_feat_flat)
            reconstructed_images_flat = F.interpolate(enhance_output_small_flat, size=(h, w), mode='bilinear', align_corners=False)

        return predicted_masks_seq, reconstructed_images_flat # 멀티태스크 출력