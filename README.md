# Low-Light Video Camouflaged Object Detection

저조도 비디오 환경에서 위장 객체 탐지를 수행하는 멀티태스크 딥러닝 모델.

## Model Architecture

**DCNetStyleVCOD**: Swin Transformer 기반 비디오 위장 객체 탐지 모델
- **Backbone**: Swin Transformer Base (384×384, ImageNet-22K pretrained)
- **Temporal Module**: ConvGRU (프레임 간 정보 융합)
- **Decoder**: Cascaded Refinement with PPM
- **Multi-task Heads**: Segmentation + Low-light Enhancement

## Key Features

- 384×384 고해상도 입력 처리
- 멀티태스크 학습 (분할 + 저조도 복원)
- RAFT 기반 시간적 일관성 손실
- Boundary Loss로 경계 정확도 향상
- Quantization 지원 (모델 크기 75% 감소)

---

## Requirements

```bash
conda create -n vcod_env python=3.8
conda activate vcod_env
pip install -r requirements.txt
```

주요 의존성:
- PyTorch >= 1.7.0
- timm >= 1.0.20
- albumentations >= 2.0.8
- tensorboard >= 2.20.0

---

## Dataset Structure

```
data/MoCA-Mask/
├── Seq_Train/          # 원본 주간 이미지 (학습)
│   └── {species}/
│       ├── Imgs/       # 비디오 프레임
│       └── GT/         # 마스크 어노테이션
├── Seq_Train_Night/    # 합성 야간 이미지 (학습)
├── Seq_Test/           # 원본 주간 이미지 (평가)
└── Seq_Test_Night/     # 합성 야간 이미지 (평가)
```

데이터 전처리:
```bash
python3 data_re.py --train_ratio 0.8
```

---

## Training

### 기본 학습
```bash
python3 run_experiment.py
```

### 주요 설정 (`config.py`)

```python
train = {
    # 모델
    'backbone_name': 'swin_base_patch4_window12_384.ms_in22k_ft_in1k',
    'decoder_channel': 128,
    'resolution': (384, 384),
    
    # 학습
    'lr': 5e-5,
    'batch_size': 4,
    'epochs': 5000,
    
    # 손실 가중치
    'lambda_boundary': 0.5,
    'lambda_temporal': 0.3,
    'lambda_enhancement': 0.5,
    
    # Quantization (경량화)
    'use_quantization': False,  # True로 변경하여 QAT 활성화
}
```

### Quantization-Aware Training (QAT)
```python
# config.py
'use_quantization': True,
'quantization_type': 'qat',
```

모델 크기 75% 감소, 추론 속도 2-4배 향상.

---

## Evaluation

```bash
python3 evaluate.py
```

평가 지표:
- S-measure (Structure)
- E-measure (Enhanced alignment)
- MAE (Mean Absolute Error)
- Warping Error (시간적 일관성)

결과 저장: `evaluation_results/`

---

## Model Quantization (Post-Training)

학습 완료 후 Dynamic Quantization 적용:

```python
from utils.quantization import prepare_model_for_quantization, save_quantized_model
from models.main_model import DCNetStyleVCOD
import torch

# 모델 로드
model = DCNetStyleVCOD(...)
model.load_state_dict(torch.load('checkpoints/best_model.pth'))

# Quantization 적용
quantized_model = prepare_model_for_quantization(
    model, 
    quantization_type='dynamic',
    backend='fbgemm'
)

# 저장
save_quantized_model(quantized_model, 'checkpoints/quantized_model.pth')
```

---

## Project Structure

```
JED-VCOD/
├── config.py                   # 중앙 설정 파일
├── run_experiment.py           # 학습 스크립트
├── evaluate.py                 # 평가 스크립트
├── models/
│   ├── main_model.py          # DCNetStyleVCOD 모델
│   ├── dae_module.py          # Denoising Autoencoder
│   └── std_module.py          # Spatio-temporal 모듈
├── datasets/
│   ├── folder_mask_dataset.py # 메인 데이터 로더
│   └── moca_video_dataset.py  # MoCA CSV 데이터 로더
├── utils/
│   ├── losses.py              # 손실 함수 (Focal, Dice, Boundary)
│   ├── quantization.py        # Quantization 유틸리티
│   ├── cutmix.py              # CutMix 데이터 증강
│   └── py_sod_metrics.py      # 평가 지표
└── tools/
    ├── synthesize_night_images.py  # 야간 이미지 합성
    └── train.py                     # 레거시 학습 스크립트
```

---

## Loss Functions

1. **Segmentation Loss**
   - Focal Loss (클래스 불균형 처리)
   - Dice Loss (영역 겹침 최적화)
   - Boundary Loss (경계 정확도, λ=0.5)

2. **Temporal Loss** (λ=0.3)
   - RAFT optical flow 기반
   - 프레임 간 일관성 보장

3. **Enhancement Loss** (λ=0.5)
   - L1 Loss
   - 저조도 → 주간 이미지 복원

---

## Implementation Details

| 항목 | 값 |
|-----|-----|
| Backbone | Swin Base (88M params) |
| Input Resolution | 384×384 |
| Decoder Channels | 128 |
| Batch Size | 4 |
| Learning Rate | 5e-5 |
| Optimizer | AdamW (weight_decay=0.05) |
| Scheduler | CosineAnnealingLR |
| Gradient Clipping | 1.0 |

---

## Quantization Options

| Type | 적용 시점 | 모델 크기 | 추론 속도 | 정확도 손실 |
|------|---------|---------|---------|-----------|
| Dynamic | 추론 시 | -75% | 2-4배 | 1-2% |
| Static | 추론 시 (Calibration) | -75% | 3-5배 | 1-2% |
| QAT | 학습 중 | -75% | 3-5배 | 0.5-1% |

Backend:
- `fbgemm`: x86 CPU (Intel, AMD)
- `qnnpack`: ARM CPU (모바일, 엣지 디바이스)

---

## Key Changes (Latest Version)

1. **Data Leakage 수정**: Test → Train 데이터셋 경로 수정
2. **고해상도 처리**: 224×224 → 384×384 (정보량 3배 증가)
3. **백본 업그레이드**: Swin Small → Swin Base 384
4. **과적합 방지**: Decoder 채널 256 → 128
5. **학습 최적화**: Learning rate 1e-5 → 5e-5
6. **경량화 지원**: Quantization 기능 추가

---

## Citation

```bibtex
@article{jed-vcod-2024,
  title={Low-Light Video Camouflaged Object Detection},
  author={...},
  journal={...},
  year={2024}
}
```

---

## License

MIT License

---

## Contact

For questions or issues, please open an issue on GitHub.