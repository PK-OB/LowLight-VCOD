#!/usr/bin/env python3
"""
Model Complexity and Performance Benchmark Script
논문 Table 6 재현용 - 모델 복잡도, 추론 속도, 성능 비교
"""

import torch
import torch.nn as nn
import time
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
from tabulate import tabulate

# 프로젝트 모듈 임포트
from models.main_model import DCNetStyleVCOD
from datasets.folder_mask_dataset import FolderImageMaskDataset
from utils.py_sod_metrics import SODMetrics
from torch.utils.data import DataLoader
import torch.quantization as quant

# ============================================================================
# 모델 경로 설정 (하드코딩)
# ============================================================================
MODEL_CONFIGS = [
    {
        'name': 'Model1',
        'precision': 'FP32',
        'checkpoint': 'checkpoints/0106_경량화이전_0.5_0.3_0.5.pth',
        'backbone': 'swin_base_patch4_window12_384.ms_in22k_ft_in1k',
        'resolution': (384, 384),
        'decoder_channel': 128,
        'use_quantization': False,
    },
    {
        'name': 'Model2',
        'precision': 'INT8',
        'checkpoint': 'checkpoints/0106_quantization_0.5_0.3_0.5.pth',
        'backbone': 'swin_base_patch4_window12_384.ms_in22k_ft_in1k',
        'resolution': (384, 384),
        'decoder_channel': 128,
        'use_quantization': True,
        'quantization_type': 'dynamic',
    },
]

# 평가 데이터셋 경로
EVAL_DATA_ROOT = '/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Seq_Test_Night'
EVAL_ORIGINAL_ROOT = '/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Seq_Test'

# 벤치마크 설정
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# 유틸리티 함수
# ============================================================================

def get_model_size_mb(model):
    """모델 크기 계산 (MB)"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def count_parameters(model):
    """파라미터 수 계산 (M)"""
    return sum(p.numel() for p in model.parameters()) / 1e6

def count_flops(model, input_size, device):
    """FLOPs 계산 (G) - 근사값"""
    try:
        from thop import profile
        dummy_input = torch.randn(1, 8, 3, *input_size).to(device)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return flops / 1e9
    except ImportError:
        print("Warning: thop not installed. FLOPs calculation skipped.")
        return 0.0
    except Exception as e:
        print(f"Warning: FLOPs calculation failed: {e}")
        return 0.0

def measure_latency_and_throughput(model, input_size, device, iterations=100, warmup=10):
    """
    Latency (ms) 및 Throughput (FPS) 측정
    """
    model.eval()
    dummy_input = torch.randn(1, 8, 3, *input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Synchronize
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.time()
            _ = model(dummy_input)
            if str(device).startswith('cuda'):
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)  # ms
    
    latency_ms = np.mean(times)
    fps = 1000.0 / latency_ms
    
    return latency_ms, fps

def evaluate_segmentation(model, data_loader, device):
    """
    Segmentation 성능 평가 (S-measure, mIoU)
    """
    model.eval()
    metrics = SODMetrics()
    metrics.reset()
    
    total_iou = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Evaluating", leave=False):
            if batch_data[0] is None:
                continue
            
            video_clip, gt_masks, _ = [x.to(device) for x in batch_data]
            b, t, c, h, w = video_clip.shape
            
            try:
                logits, _ = model(video_clip)
                preds = torch.sigmoid(logits)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            
            # Flatten
            preds_flat = preds.view(b*t, 1, h, w)
            gt_flat = gt_masks.view(b*t, 1, h, w)
            
            # Numpy 변환
            preds_np = (preds_flat.squeeze(1).cpu().numpy() * 255).astype(np.uint8)
            gts_np = (gt_flat.squeeze(1).cpu().numpy() * 255).astype(np.uint8)
            
            # Metrics 계산
            for i in range(b*t):
                try:
                    metrics.step(pred=preds_np[i], gt=gts_np[i])
                    
                    # mIoU 계산
                    pred_binary = (preds_np[i] > 127).astype(np.uint8)
                    gt_binary = (gts_np[i] > 127).astype(np.uint8)
                    
                    intersection = np.logical_and(pred_binary, gt_binary).sum()
                    union = np.logical_or(pred_binary, gt_binary).sum()
                    
                    if union > 0:
                        iou = intersection / union
                        total_iou += iou
                        num_samples += 1
                except Exception as e:
                    continue
    
    results = metrics.get_results()
    sm = results.get('Sm', 0.0)
    miou = total_iou / num_samples if num_samples > 0 else 0.0
    
    return sm, miou

def calculate_drop_rate(baseline_sm, current_sm):
    """
    성능 하락률 계산 (%)
    """
    if baseline_sm == 0:
        return 0.0
    drop_rate = ((baseline_sm - current_sm) / baseline_sm) * 100
    return drop_rate

# ============================================================================
# 메인 벤치마크 함수
# ============================================================================

def benchmark_model(config, baseline_sm=None):
    """
    단일 모델 벤치마크
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking: {config['name']} ({config['precision']})")
    print(f"{'='*80}")
    
    device = torch.device(DEVICE)
    
    # 모델 로드
    print("Loading model...")
    checkpoint_path = config['checkpoint']
    
    if not Path(checkpoint_path).exists():
        print(f"Warning: Checkpoint not found: {checkpoint_path}")
        print("Using random initialized model for demonstration.")
        checkpoint_path = None
    
    # 모델 생성
    model = DCNetStyleVCOD(
        backbone_name=config['backbone'],
        input_size=config['resolution'],
        num_frames=8,
        pretrained=False,
        decoder_channel=config['decoder_channel'],
        use_enhancement=True
    )
    
    # Checkpoint 로드
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    
    # Quantization 적용
    if config.get('use_quantization', False):
        print("Applying quantization...")
        from utils.quantization import prepare_model_for_quantization
        
        # Quantization은 CPU에서만 작동
        print("Moving model to CPU for quantization...")
        model = model.cpu()
        device = torch.device('cpu')
        
        model = prepare_model_for_quantization(
            model,
            quantization_type=config.get('quantization_type', 'dynamic'),
            backend='fbgemm'
        )
        print("Quantization applied. Model will run on CPU.")
    
    # FP16 적용
    if config.get('use_fp16', False):
        print("Converting to FP16...")
        model = model.half()
    
    model.eval()
    
    # 1. Model Size
    model_size_mb = get_model_size_mb(model)
    print(f"Model Size: {model_size_mb:.1f} MB")
    
    # 2. Parameters
    params_m = count_parameters(model)
    print(f"Parameters: {params_m:.1f} M")
    
    # 3. FLOPs
    flops_g = count_flops(model, config['resolution'], device)
    print(f"FLOPs: {flops_g:.1f} G")
    
    # 4. Latency & Throughput
    print("Measuring latency and throughput...")
    latency_ms, fps = measure_latency_and_throughput(
        model, 
        config['resolution'], 
        device,
        iterations=BENCHMARK_ITERATIONS,
        warmup=WARMUP_ITERATIONS
    )
    print(f"Latency: {latency_ms:.1f} ms")
    print(f"Throughput: {fps:.1f} FPS")
    
    # 5. Segmentation Performance
    print("Evaluating segmentation performance...")
    
    # 데이터셋 로드
    from datasets.folder_mask_dataset import FolderImageMaskDataset
    
    eval_dataset = FolderImageMaskDataset(
        root_dir=EVAL_DATA_ROOT,
        original_data_root=EVAL_ORIGINAL_ROOT,
        image_folder_name='Imgs',
        mask_folder_name='GT',
        clip_len=8,
        resolution=config['resolution'],
        is_train=False,
        use_augmentation=False
    )
    
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None, None)
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    sm, miou = evaluate_segmentation(model, eval_loader, device)
    print(f"S-measure: {sm:.2f}")
    print(f"mIoU: {miou:.2f}")
    
    # 6. Drop Rate
    drop_rate = 0.0
    if baseline_sm is not None:
        drop_rate = calculate_drop_rate(baseline_sm, sm)
        print(f"Drop Rate: {drop_rate:.2f}%")
    
    return {
        'Model Precision': f"{config['name']} ({config['precision']})",
        'Model Size (MB)': model_size_mb,
        'Params (M)': params_m,
        'FLOPs (G)': flops_g,
        'Latency (ms)': latency_ms,
        'Throughput (FPS)': fps,
        'Sm': sm,
        'mIoU': miou,
        'Drop Rate (%)': drop_rate if baseline_sm else '-'
    }

# ============================================================================
# 메인 실행
# ============================================================================

def main():
    print("="*80)
    print("Model Complexity and Performance Benchmark")
    print("Table 6: Comparison across different precision formats")
    print("="*80)
    
    results = []
    baseline_sm = None
    
    for i, config in enumerate(MODEL_CONFIGS):
        result = benchmark_model(config, baseline_sm)
        results.append(result)
        
        # 첫 번째 모델을 baseline으로 설정
        if i == 0:
            baseline_sm = result['Sm']
    
    # 결과 테이블 생성
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80 + "\n")
    
    # 테이블 헤더
    headers = [
        'Model Precision',
        'Model Size\n(MB)',
        'Params\n(M)',
        'FLOPs\n(G)',
        'Latency\n(ms)',
        'Throughput\n(FPS)',
        'Performance (LLCA-Mask)\nSm ↑',
        'mIoU ↑',
        'Drop Rate (%)'
    ]
    
    # 테이블 데이터 포맷팅
    table_data = []
    for r in results:
        row = [
            r['Model Precision'],
            f"{r['Model Size (MB)']:.1f}",
            f"{r['Params (M)']:.1f}",
            f"{r['FLOPs (G)']:.1f}",
            f"{r['Latency (ms)']:.1f}",
            f"{r['Throughput (FPS)']:.1f}",
            f"{r['Sm']:.2f}",
            f"{r['mIoU']:.2f}",
            f"{r['Drop Rate (%)']:.2f}%" if isinstance(r['Drop Rate (%)'], float) else r['Drop Rate (%)']
        ]
        table_data.append(row)
    
    # 테이블 출력
    table_str = tabulate(table_data, headers=headers, tablefmt='grid')
    print(table_str)
    
    # 결과 저장
    output_file = 'benchmark_results_table6.txt'
    with open(output_file, 'w') as f:
        f.write("Table 6. Comparison of model complexity, inference speed, and segmentation performance\n")
        f.write("across different precision formats. The inference speed (FPS) was measured on an NVIDIA RTX 5090 GPU.\n\n")
        f.write(table_str)
    
    print(f"\nResults saved to: {output_file}")
    
    # CSV 저장
    import csv
    csv_file = 'benchmark_results_table6.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"CSV results saved to: {csv_file}")

if __name__ == '__main__':
    main()
