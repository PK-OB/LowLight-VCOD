# utils/quantization.py
"""
Quantization 유틸리티 모듈
모델 양자화를 위한 헬퍼 함수들
"""

import torch
import torch.quantization as quant
import logging
import os

logger = logging.getLogger(__name__)


def prepare_model_for_quantization(model, quantization_type='dynamic', backend='fbgemm'):
    """
    모델을 양자화를 위해 준비
    
    Args:
        model: PyTorch 모델
        quantization_type: 'dynamic', 'static', 'qat'
        backend: 'fbgemm' (x86) or 'qnnpack' (ARM)
    
    Returns:
        준비된 모델
    """
    logger.info(f"Preparing model for {quantization_type} quantization with {backend} backend")
    
    # Backend 설정
    torch.backends.quantized.engine = backend
    
    if quantization_type == 'dynamic':
        # Dynamic Quantization (가장 간단, 추론 시에만 적용)
        # 주로 Linear, LSTM 레이어에 적용
        quantized_model = quant.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        logger.info("Applied dynamic quantization")
        return quantized_model
        
    elif quantization_type == 'static':
        # Static Quantization (Calibration 필요)
        model.eval()
        model.qconfig = quant.get_default_qconfig(backend)
        model_prepared = quant.prepare(model)
        logger.info("Model prepared for static quantization (calibration needed)")
        return model_prepared
        
    elif quantization_type == 'qat':
        # Quantization-Aware Training
        model.train()
        model.qconfig = quant.get_default_qat_qconfig(backend)
        model_prepared = quant.prepare_qat(model)
        logger.info("Model prepared for quantization-aware training")
        return model_prepared
        
    else:
        logger.error(f"Unknown quantization type: {quantization_type}")
        return model


def calibrate_model(model, data_loader, device, num_batches=100):
    """
    Static Quantization을 위한 Calibration
    
    Args:
        model: 준비된 모델 (prepare 후)
        data_loader: Calibration 데이터 로더
        device: 디바이스
        num_batches: Calibration에 사용할 배치 수
    """
    logger.info(f"Starting calibration with {num_batches} batches")
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break
                
            if batch is None or not isinstance(batch, (list, tuple)) or len(batch) != 3:
                continue
                
            video_clip, _, _ = batch
            try:
                video_clip = video_clip.to(device)
                _ = model(video_clip)
            except Exception as e:
                logger.warning(f"Calibration batch {i} failed: {e}")
                continue
                
    logger.info("Calibration completed")


def convert_to_quantized(model):
    """
    준비된 모델을 최종 양자화 모델로 변환
    
    Args:
        model: Calibration 완료된 모델
    
    Returns:
        양자화된 모델
    """
    logger.info("Converting to quantized model")
    quantized_model = quant.convert(model)
    logger.info("Conversion completed")
    return quantized_model


def save_quantized_model(model, save_path):
    """
    양자화된 모델 저장
    
    Args:
        model: 양자화된 모델
        save_path: 저장 경로
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Quantized model saved to {save_path}")


def load_quantized_model(model, load_path, device='cpu'):
    """
    양자화된 모델 로드
    
    Args:
        model: 모델 인스턴스
        load_path: 로드 경로
        device: 디바이스 (양자화 모델은 주로 CPU)
    
    Returns:
        로드된 모델
    """
    model.load_state_dict(torch.load(load_path, map_location=device))
    logger.info(f"Quantized model loaded from {load_path}")
    return model


def compare_model_size(original_model, quantized_model):
    """
    원본 모델과 양자화 모델의 크기 비교
    
    Args:
        original_model: 원본 모델
        quantized_model: 양자화 모델
    """
    def get_model_size(model):
        torch.save(model.state_dict(), "temp.p")
        size = os.path.getsize("temp.p") / 1e6  # MB
        os.remove("temp.p")
        return size
    
    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)
    
    logger.info(f"Original model size: {original_size:.2f} MB")
    logger.info(f"Quantized model size: {quantized_size:.2f} MB")
    logger.info(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
    
    return original_size, quantized_size


def print_quantization_info(model):
    """
    모델의 양자화 정보 출력
    
    Args:
        model: 양자화된 모델
    """
    logger.info("Quantization configuration:")
    for name, module in model.named_modules():
        if hasattr(module, 'qconfig') and module.qconfig is not None:
            logger.info(f"  {name}: {module.qconfig}")
