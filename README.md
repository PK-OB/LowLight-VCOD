네, 프로젝트의 `README.md` 문서를 작성해 드릴게요. 각 파일과 함수가 어떤 역할을 하는지 명확하게 정리하면 앞으로 프로젝트를 관리하고 다른 사람과 협업할 때 정말 큰 도움이 되죠. 말씀하신 대로 함수별 역할을 개조식으로 깔끔하게 정리해 보겠습니다.

-----

# 저조도 환경에서의 위장 야생동물 탐지 (JED-VCOD)

## 1\. 프로젝트 개요

본 프로젝트는 저조도 및 야간 환경에서 위장한 야생동물을 탐지하는 딥러닝 모델, `JED-VCOD-Fauna`를 구현합니다. 이 모델은 '탐지 지향적 강화(Detection-Oriented Enhancement)' 패러다임을 기반으로, 저조도 이미지 강화를 통한 특징 추출과 시공간적 움직임 분석을 결합하여 정확한 동물 탐지를 수행합니다.

## 2\. 프로젝트 구조

```
JED-VCOD/
├── data/
│   └── MoCA/
│       ├── Annotations/
│       │   └── annotations.csv
│       └── Videos/
│           └── ... (동물 영상 폴더)
├── datasets/
│   └── moca_video_dataset.py
├── models/
│   ├── dae_module.py
│   ├── main_model.py
│   └── std_module.py
├── utils/
│   └── losses.py
├── train.py
└── README.md
```

## 3\. 코드별 분석

### `train.py`

  - **역할**: 모델 학습을 위한 메인 스크립트입니다. 데이터셋 로딩, 모델 초기화, 학습 루프 실행 등 전체 프로세스를 총괄합니다.
  - **main()**:
      - 모델, 손실 함수(`BCE`, `DiceLoss`), 옵티마이저(`AdamW`)를 초기화합니다.
      - `MoCAVideoDataset`을 이용해 학습 및 검증 데이터로더를 생성합니다.
      - 지정된 에포크(epoch)만큼 학습 루프를 반복 실행합니다.
      - 각 배치(batch)마다 모델의 예측을 수행하고, 손실을 계산한 뒤 역전파를 통해 모델의 가중치를 업데이트합니다.

-----

### `main_model.py`

  - **역할**: DAE와 STD 모듈을 결합하여 최종 `JED-VCOD` 모델을 정의합니다.
  - **JED\_VCOD\_Fauna\_Simplified.\_\_init\_\_()**:
      - `DAEModule`과 `STDModule`의 인스턴스를 생성하여 모델의 구성요소를 초기화합니다.
  - **JED\_VCOD\_Fauna\_Simplified.forward()**:
      - 모델의 순전파 로직을 정의합니다.
      - 입력된 비디오 클립(`(B, T, C, H, W)`)을 프레임 단위로 분리하여 `DAEModule`에 입력합니다.
      - `DAEModule`에서 강화된 다중 스케일 특징 맵을 다시 시퀀스 형태로 변환합니다.
      - 변환된 특징 맵 시퀀스를 `STDModule`에 전달하여 최종 마스크를 예측하고 반환합니다.

-----

### `dae_module.py` (Detection-Aware Enhancement)

  - **역할**: U-Net 기반의 인코더-디코더 구조를 사용하여 입력된 프레임의 특징을 강화하고, 후속 모듈이 객체를 더 잘 탐지할 수 있도록 돕습니다.
  - **ConvBlock()**:
      - U-Net의 기본 구성 블록으로, 2개의 `Conv2d`, `BatchNorm2d`, `ReLU` 활성화 함수로 구성됩니다.
  - **DAEModule.\_\_init\_\_()**:
      - U-Net의 인코더, 디코더, 병목(bottleneck) 구간을 `ConvBlock`을 이용해 구성합니다.
  - **DAEModule.forward()**:
      - 인코더 경로에서 점진적으로 특징 맵을 다운샘플링하며 `skip connection` 정보를 저장합니다.
      - 디코더 경로에서 다시 업샘플링하며 각 단계의 `skip connection` 정보를 결합(concatenate)합니다.
      - 디코더의 각 레벨에서 생성된 다중 스케일 특징 맵들을 리스트에 담아 반환합니다.

-----

### `std_module.py` (Spatio-Temporal Detection)

  - **역할**: DAE 모듈에서 받은 강화된 특징 맵들을 이용해, 비디오 시퀀스 내에서 시간적 맥락을 고려하여 최종 객체 마스크를 예측합니다.
  - **ConvLSTMCell()**:
      - 컨볼루션 연산을 사용하는 LSTM 셀을 정의합니다. 이미지의 공간적 구조를 유지하면서 시간적 정보를 처리하는 데 사용됩니다.
  - **STDModule.\_\_init\_\_()**:
      - 다중 스케일 특징 맵들을 융합하기 위한 `Conv2d` 레이어들을 초기화합니다.
      - 시퀀스 데이터를 처리할 `ConvLSTMCell`을 초기화합니다.
      - 최종 마스크를 생성할 분할 헤드(`seg_head`)를 초기화합니다.
  - **STDModule.forward()**:
      - DAE로부터 받은 다중 스케일 특징 맵들을 프레임(시간) 단위로 반복 처리합니다.
      - 각 프레임에서 모든 스케일의 특징 맵들을 하나의 `fused_feature`로 융합합니다.
      - 융합된 특징 맵을 `ConvLSTMCell`에 입력하여 시간적 정보를 업데이트합니다.
      - `ConvLSTM`의 출력을 `seg_head`에 통과시켜 작은 크기의 마스크를 예측합니다.
      - 예측된 마스크를 `F.interpolate`를 통해 원래 이미지 크기로 업샘플링하여 최종 결과를 반환합니다.

-----

### `moca_video_dataset.py`

  - **역할**: `annotations.csv` 파일과 이미지 파일들을 바탕으로, 모델 학습에 필요한 비디오 클립과 정답 마스크 텐서를 생성하는 PyTorch `Dataset` 클래스입니다.
  - **MoCAVideoDataset.\_\_init\_\_()**:
      - `annotations.csv` 파일을 `pandas`로 읽어옵니다.
      - 이미지와 마스크에 적용할 변환(리사이즈, 텐서 변환, 정규화 등)을 정의합니다.
      - 전체 비디오를 지정된 길이(`clip_len`)의 클립 단위로 자르는 작업을 준비합니다.
  - **MoCAVideoDataset.\_\_len\_\_()**:
      - 데이터셋의 총 클립 개수를 반환합니다.
  - **MoCAVideoDataset.\_\_getitem\_\_()**:
      - 특정 인덱스(`idx`)에 해당하는 비디오 클립을 로드합니다.
      - 클립에 포함된 각 프레임 이미지 파일을 읽고, `spatial_coordinates` 정보를 이용해 정답 마스크를 생성합니다.
      - 이미지와 마스크에 미리 정의된 변환을 적용하여 텐서로 만들고 반환합니다.

-----

### `utils/losses.py`

  - **역할**: 학습에 사용될 커스텀 손실 함수를 정의합니다.
  - **DiceLoss.forward()**:
      - 이미지 분할(segmentation) 태스크에서 널리 사용되는 Dice Loss를 계산합니다.
      - 예측 마스크와 정답 마스크 사이의 겹치는 영역을 최대화하는 방향으로 모델을 학습시키는 역할을 합니다.