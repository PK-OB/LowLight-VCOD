알겠습니다. 요청에 맞춰 `입니다`를 제외하고 보고서 및 기술 문서에서 자주 사용되는 '-(으)ㅁ' 종결 어미를 사용하여 `README.md` 내용을 다시 작성했습니다.

---

# Low-Light Video Camouflaged Object Detection

본 리포지토리는 저조도 비디오 환경에서 위장 객체 탐지(VCOD)를 수행하는 `DCNetStyleVCOD` 모델의 공식 구현 코드임.

본 모델은 위장 객체 분할(Primary Task)과 저조도 강화(Auxiliary Task)를 동시에 수행하는 멀티태스크 아키텍처를 기반으로 하며, 시간적 일관성을 위한 ConvGRU, 경계 손실, 시간적 손실 등을 포함함.

## 📝 프로젝트 구조 및 파일 설명

### 1. 핵심 스크립트

* **`run_experiment.py`**
    * **메인 학습 스크립트**. `config.py` 설정을 기반으로 `DCNetStyleVCOD` 모델의 학습, 검증, 체크포인트 저장을 수행함.
    * 모든 손실 함수(Focal, Dice, L1, Boundary, Temporal)의 조합 및 `train_one_epoch` 로직이 구현되어 있음.

* **`evaluate.py`**
    * **메인 평가 스크립트**. 학습된 `DCNetStyleVCOD` 체크포인트를 로드하여 성능을 평가함.
    * `utils/py_sod_metrics.py`를 사용하여 표준 VCOD/SOD 지표(Sm, Em, MAE 등)를 계산함.
    * 저조도 강화(Enhancement L1 Loss) 및 시간적 일관성(Warping Error)도 함께 평가함.
    * `evaluation_results/`에 4-row (야간원본, 복원주간, GT마스크, 예측마스크) 시각화 이미지를 저장함.

* **`config.py`**
    * 프로젝트의 모든 하이퍼파라미터와 경로를 관리하는 **중앙 설정 파일**.
    * `common`: GPU ID, 클립 길이 등 공통 설정
    * `train`: 학습 데이터 경로(`folder_data_root`, `original_data_root`), 모델 파라미터(백본, LR, 손실 가중치) 등 학습 관련 모든 설정
    * `evaluate`: 평가 데이터 경로, 체크포인트 경로, 시각화 저장 경로 등 평가 관련 설정

### 2. 모델 아키텍처 (`models/`)

* **`models/main_model.py`**
    * 본 프로젝트의 **핵심 모델(`DCNetStyleVCOD`)**이 정의된 파일.
    * `Swin Transformer` 백본, `ConvGRUCell` (시간 모델링), `PPM` 및 `RefinementBlock` (계층적 디코더), `SegmentationRefinementHead`, `SimpleEnhancementHead` (멀티태스크 출력 헤드)의 조합으로 구성됨.

* **`models/dae_module.py`**
    * ResNet-34 기반의 Denoising Autoencoder (DAE) 모듈.
    * `models/eval_models.py`에서 이전 버전 모델의 강화부로 사용됨.

* **`models/std_module.py`**
    * Stacked ConvLSTM 및 Residual Refinement Head를 포함하는 시공간 분할 모듈.
    * `models/eval_models.py`에서 이전 버전 모델의 분할부로 사용됨.

* **`models/eval_models.py`**
    * `JED_VCOD_Fauna_Simplified_Eval` 등, `main_model.py` 이전의 실험/베이스라인 모델 아키텍처를 정의함.

### 3. 데이터 로더 (`datasets/`)

* **`datasets/folder_mask_dataset.py`**
    * **핵심 데이터 로더**. `config.py`에 명시된 폴더 구조를 기반으로 데이터를 로드함.
    * `__getitem__`은 멀티태스크 학습을 위해 `(image_clip, mask_clip, original_image_clip)` 3-튜플을 반환함.

* **`datasets/moca_video_dataset.py`**
    * 원본 MoCA CSV 어노테이션 파일을 파싱하여 비디오 클립을 생성하는 레거시 데이터 로더.

* **`datasets/moca_box_dataset.py`**
    * `evaluate_box.py`에서 사용되며, 마스크 대신 바운딩 박스(BBox) 어노테이션을 로드함.

### 4. 유틸리티 및 도구 (`utils/`, `tools/`)

* **`utils/losses.py`**
    * 학습에 사용되는 손실 함수(`DiceLoss`, `FocalLoss`, `BoundaryLoss` 등)를 정의함.

* **`utils/cutmix.py`**
    * CutMix 데이터 증강을 구현함. 멀티태스크 학습을 위해 3개의 텐서(야간이미지, 마스크, 주간이미지)에 동일한 CutMix를 적용하도록 수정됨.

* **`utils/py_sod_metrics.py`**
    * `evaluate.py`에서 사용되는 표준 SOD/VCOD 평가 지표(S-measure, E-measure, MAE 등) 계산 라이브러리.

* **`utils/logger.py`**
    * 학습 로그를 파일과 콘솔에 출력하기 위한 로거 설정 스크립트.

* **`tools/synthesize_night_images.py`**
    * Diffusers `StableDiffusionInstructPix2PixPipeline`을 사용하여 원본 주간 이미지를 "make it look like night" 프롬프트로 변환, **야간 데이터셋을 합성**하는 스크립트.
    * `tools/synthesize_night_images_mask.py`는 특정 폴더 구조(`Imgs`)에 맞게 수정된 버전.

* **`data_re.py`**
    * 데이터 전처리 스크립트. 기존의 Train/Test 데이터셋을 병합한 후, **비디오 프레임 순서를 유지하며** (`shuffle` 대신 순서 기반 분할) 설정된 비율(예: 80%)에 따라 새로운 Train/Test 폴더로 재분배함.
    * `config.py`에서 사용되는 `Seq_Train`, `Seq_Test` 폴더를 생성함.

* **`evaluate_dataset_quality.py`**
    * `tools/synthesize_night_images.py`로 생성된 합성 야간 데이터셋과 원본 주간 데이터셋 간의 SSIM, LPIPS를 계산하여 품질을 정량적으로 평가함.

* **`tools/preprocess_csv.py`** / **`tools/check_csv.py`**
    * 원본 MoCA CSV 어노테이션 파일을 파싱하거나 검사하는 유틸리티 스크립트.

### 5. 환경 설정

* **`requirements.txt`**
    * `pip` 기반의 파이썬 패키지 의존성 목록.

* **`vcod_env.yml`**
    * `Conda` 환경을 재생성하기 위한 `yml` 파일.

* **`.gitignore`**
    * Git 형상 관리에서 제외할 파일 및 폴더(데이터, 로그, 체크포인트 등) 목록.

### 6. 레거시 및 기타 평가 스크립트

* **`evaluate_past.py`**
    * `models/eval_models.py`에 정의된 이전 버전 모델(`JED_VCOD_Fauna_Simplified_Eval`)을 평가하기 위한 스크립트. `moca_video_dataset`을 사용함.

* **`evaluate_box.py`**
    * 모델의 분할 마스크 예측을 바운딩 박스(BBox)로 변환한 후, GT BBox와 IoU를 비교하여 Precision, Recall, F1-Score를 계산함.

* **`tools/train.py`**
    * `run_experiment.py` 이전의 레거시 학습 스크립트로 보이며, `JED_VCOD_Fauna_Simplified` 모델을 `moca_video_dataset`으로 학습시킴.