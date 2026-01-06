# pk-ob/jed-vcod/JED-VCOD-cc543b29cefb3a45b940bfd01f42c33af7a6bb25/config.py

class Config:
    # --- 공통 설정 ---
    common = {
        'gpu_ids': '2,3',
        'clip_len': 8,
        'num_workers': 12,
    }

    # --- 학습 설정 ---
    train = {
        'experiment_name': 'JED-VCOD_VideoSwin_UPerNet', # 실험 이름 변경
        'dataset_type': 'folder', 
        'folder_data_root': '/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Seq_Train_Night',
        'original_data_root': '/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Seq_Train', 
        'image_folder_name': 'Imgs',
        'mask_folder_name': 'GT',
        
        # --- 모델 설정 ---
        'backbone_name': 'swin_base_patch4_window12_384.ms_in22k_ft_in1k', # Swin Base 384 (pretrained)
        'backbone_pretrained': True,
        'decoder_channel': 128, # UPerNet 디코더 채널 수 (256->128: 과적합 방지)
        'resolution': (384, 384), # 모델 입력 해상도 (Swin Base 384 네이티브 지원)
        
        # --- 학습 하이퍼파라미터 (Transformer용 조정 예시) ---
        'epochs': 5000, # Transformer는 더 많은 에포크 필요할 수 있음
        'batch_size': 4, # 해상도 증가로 배치 크기 감소 (8->4)
        'val_batch_size': 8, # 검증 시에는 더 큰 배치 사용 가능
        'lr': 5e-5,          # 학습률 조정 (1e-5->5e-5: 수렴 속도 개선)
        'weight_decay': 0.05, # Weight Decay 높이기
        'clip_grad_norm': 1.0, # 그래디언트 클리핑 최대 값
        # 'accumulation_steps': 2, # GPU 메모리 부족 시 그래디언트 누적 사용

        # --- 스케줄러 (Transformer용 조정 예시) ---
        'scheduler_name': 'CosineAnnealingLR', # ReduceLROnPlateau보다 Cosine 추천
        'eta_min': 1e-7,                 # Cosine 스케줄러 최소 LR
        # 'T_0': 50,  # CosineAnnealingWarmRestarts 사용 시
        # 'T_mult': 1, 
        # 'scheduler_patience': 10, # ReduceLROnPlateau 사용 시
        # 'scheduler_factor': 0.1,w

        # --- 손실 가중치 ---
        'lambda_enhancement': 0.5, # Enhancement Loss 가중치 (0이면 비활성화)
        'lambda_temporal': 0.3, # Temporal Loss 가중치 (0이면 비활성화)
        'lambda_boundary': 0.5,   # 경계 손실 가중치 (1.0->0.5: 균형 조정)
        'dice_weight': 1.0,     # Dice Loss 가중치 (Focal 대비)
        'structure_weight': 1.0,  # Structure Loss 가중치
        
        # --- 기타 ---
        'use_cutmix': False, # CutMix 활성화 (데이터 증강)
        'cutmix_beta': 1.0,
        'cutmix_prob': 0.5,
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'checkpoint_name': '0106_quantization_0.5_0.3_0.5.pth', # 체크포인트 이름 변경
        'debug_image_interval': 10, # 이미지 저장 빈도 조정
        'patience': 300, # Early stopping patience 늘리기 (Transformer 학습 안정화 시간 고려)
        
        # --- Quantization (양자화) 설정 ---
        'use_quantization': True, # Quantization 사용 여부
        'quantization_type': 'dynamic', # 'dynamic', 'static', 'qat' (Quantization-Aware Training)
        'quantization_backend': 'fbgemm', # 'fbgemm' (x86), 'qnnpack' (ARM)
        'quantization_dtype': 'qint8', # 'qint8', 'float16'
    }
    
    # --- 평가 설정 ---
    evaluate = {
        'experiment': 'proposed',
        'batch_size': 8,
        'visualization_path': 'evaluation_results/1223/1030_1746.png', # (저장 경로 예시)
        'checkpoint_path': 'checkpoints/1030_1746.pth', #           (평가할 모델 경로)
        
        'visualization_hardcoded_paths': [
            '''
            '/home/sjy/paper/JED-VCOD/0_test_img/00004.jpg',
            '/home/sjy/paper/JED-VCOD/0_test_img/00010.jpg',
            '/home/sjy/paper/JED-VCOD/0_test_img/00011.jpg',
            '/home/sjy/paper/JED-VCOD/0_test_img/00019.jpg',
            '/home/sjy/paper/JED-VCOD/0_test_img/00026.jpg',
            '/home/sjy/paper/JED-VCOD/0_test_img/00066.jpg'
            '''
        ],

        # ▼▼▼ 1. 여기에 새로운 설정을 추가합니다 ▼▼▼
        # -----------------------------------------------------------------
        'eval_dataset_type': 'folder',  # 'folder' 또는 'moca_csv' (기본값)
        
        # 'folder' 타입을 사용할 경우
        'eval_folder_data_root': '/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Seq_Test_Night', # <-- 여기에 새 테스트셋 경로 입력!
        
        # ▼▼▼ 수정된 부분 ▼▼▼
        # (평가 시에도 원본 주간 이미지가 필요합니다. Test_Night의 원본 경로로 수정해주세요)
        'eval_original_data_root': '/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Seq_Test',
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        'eval_image_folder_name': 'Imgs',
        'eval_mask_folder_name': 'GT',
        # -----------------------------------------------------------------
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # 'moca_csv' 타입을 사용할 경우 (기존 설정)
        'data_root': 'data/Night-Camo-Fauna/',
        'annotation_file': 'data/MoCA/Annotations/annotations.csv',
    }

    # --- 박스 평가 설정 ---
    evaluate_box = {
        'annotation_file': 'data/MoCA/Annotations/annotations.csv',
        'checkpoint_path': 'checkpoints/best_model.pth',
        'iou_threshold': 0.5,
        'batch_size': 1,
    }

cfg = Config()

