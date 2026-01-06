# utils/py_sod_metrics.py
# (Bug Fix 3: OpenCV dtype 오류 수정)

import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage.filters import gaussian_filter
import cv2  # <-- [BUG FIX 2] cv2 임포트 추가
from PIL import Image # <-- [BUG FIX 2] Image 임포트 추가 (shape 다를 때 대비)

class SODMetrics:
    def __init__(self):
        self.metric_calculators = [
            MAE(),
            Emeasure(),
            Smeasure(),
            WeightedFmeasure(),
            Fmeasure(),
            # ... 다른 지표 추가 가능
        ]
        self.results = None

    def step(self, pred: np.ndarray, gt: np.ndarray):
        """
        하나의 예측/정답 쌍에 대해 모든 지표 계산을 수행합니다.
        :param pred: [0, 255] 범위의 uint8 numpy 배열 (예측 마스크)
        :param gt: [0, 255] 범위의 uint8 numpy 배열 (정답 마스크)
        """
        # [FIX] Shape 검증 강화
        if pred.ndim != 2:
            raise ValueError(f"Pred must be 2D, got shape {pred.shape}")
        if gt.ndim != 2:
            raise ValueError(f"GT must be 2D, got shape {gt.shape}")
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")
        
        # 데이터 타입 변환
        if pred.dtype != np.uint8:
            pred = np.clip(pred * 255, 0, 255).astype(np.uint8)
        if gt.dtype != np.uint8:
            gt = np.clip(gt, 0, 255).astype(np.uint8)

        for calculator in self.metric_calculators:
            calculator.step(pred, gt)

    def get_results(self) -> dict:
        """
        모든 지표의 최종 평균 결과를 반환합니다.
        :return: 지표 이름을 키로, 평균값을 값으로 하는 딕셔너리
        """
        # [FIX] 캐싱 제거 - 매번 새로 계산
        results = {}
        for calculator in self.metric_calculators:
            results.update(calculator.get_results())
        return results
    
    def reset(self):
        """
        평가 재시작 시 호출 - 모든 누적값 초기화
        """
        for calculator in self.metric_calculators:
            calculator.num_samples = 0
            calculator.total_value = 0
            calculator.total_values = []
            if hasattr(calculator, 'adaptive_f_results'):
                calculator.adaptive_f_results = []
            if hasattr(calculator, 'precisions'):
                calculator.precisions = []
            if hasattr(calculator, 'recalls'):
                calculator.recalls = []
            if hasattr(calculator, 'weighted_f_results'):
                calculator.weighted_f_results = []

# ==========================================================
# 내부 지표 계산기 클래스 (Base, MAE, F-measure, E-measure 등)
# ==========================================================

class _BaseMetric:
    """
    모든 지표 계산기의 기본 클래스.
    """
    def __init__(self):
        self.num_samples = 0
        self.total_value = 0
        self.total_values = [] # E-measure, F-measure 등 곡선용

    def step(self, pred: np.ndarray, gt: np.ndarray):
        raise NotImplementedError

    def get_results(self) -> dict:
        raise NotImplementedError

    def _prepare_data(self, pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        데이터를 전처리하는 공통 함수.
        pred: [0, 255] uint8
        gt: [0, 255] uint8
        """
        gt = gt > 128 # GT는 0.5 (128) 기준으로 이진화
        pred = pred / 255.0 # Pred는 [0, 1] float로
        
        # ▼▼▼ [치명적 오류 수정] ▼▼▼
        # 이 재정규화 로직은 Saliency Map의 절대적인 자신감을 무시하고
        # [0, 0.1] 범위의 희미한 예측도 [0, 1]로 강제 스트레칭하여
        # 지표를 심각하게 왜곡시킵니다.
        
        # if pred.max() != pred.min(): # <-- 주석 처리 (BUG FIX 1)
        #     pred = (pred - pred.min()) / (pred.max() - pred.min()) # <-- 주석 처리 (BUG FIX 1)
        # ▲▲▲ [수정 완료] ▲▲▲
            
        return pred, gt


class MAE(_BaseMetric):
    """
    Mean Absolute Error (MAE)
    """
    def __init__(self):
        super(MAE, self).__init__()

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = self._prepare_data(pred, gt)
        # [FIX] 빈 GT 처리 통일
        if gt.sum() == 0:
            # 빈 GT: 예측이 0에 가까울수록 좋음
            mae = np.mean(pred)
        else:
            mae = np.mean(np.abs(pred - gt))
        self.total_value += mae
        self.num_samples += 1

    def get_results(self) -> dict:
        avg_mae = self.total_value / self.num_samples if self.num_samples > 0 else 0
        return {"MAE": avg_mae}


class Fmeasure(_BaseMetric):
    """
    F-measure (Precision, Recall, F-beta)
    """
    def __init__(self, beta2=0.3): # 0.3은 S-measure와 동일한 설정
        super(Fmeasure, self).__init__()
        self.beta2 = beta2
        self.adaptive_f_results = []
        self.precisions = []
        self.recalls = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = self._prepare_data(pred, gt)
        gt_size = gt.sum()
        if gt_size == 0:
            # GT가 비어있으면 이 샘플은 무시 (또는 1.0 처리)
            # 여기서는 일관성을 위해 무시
            return

        # 1. Adaptive F-measure (adpFm)
        adaptive_threshold = 2 * pred.mean()
        if adaptive_threshold > 1.0:
            adaptive_threshold = 1.0
        
        binary_pred_adaptive = (pred >= adaptive_threshold)
        tp_adaptive = (binary_pred_adaptive & gt).sum()
        
        precision_adaptive = tp_adaptive / (binary_pred_adaptive.sum() + 1e-12)
        recall_adaptive = tp_adaptive / (gt_size + 1e-12)
        
        f_beta_adaptive = (1 + self.beta2) * (precision_adaptive * recall_adaptive) / (self.beta2 * precision_adaptive + recall_adaptive + 1e-12)
        self.adaptive_f_results.append(f_beta_adaptive)

        # 2. Precision-Recall Curve (Max F-measure용)
        thresholds = np.linspace(0, 1, 256)
        sample_precisions = []
        sample_recalls = []

        for th in thresholds:
            binary_pred = (pred >= th)
            tp = (binary_pred & gt).sum()
            
            precision = tp / (binary_pred.sum() + 1e-12)
            recall = tp / (gt_size + 1e-12)
            
            sample_precisions.append(precision)
            sample_recalls.append(recall)

        self.precisions.append(np.array(sample_precisions))
        self.recalls.append(np.array(sample_recalls))
        self.num_samples += 1

    def get_results(self) -> dict:
        if self.num_samples == 0:
            return {"adpFm": 0, "maxFm": 0, "avgP": 0, "avgR": 0}

        # 1. Adaptive F-measure
        avg_adpfm = np.mean(np.array(self.adaptive_f_results))

        # 2. Max F-measure
        avg_precision_curve = np.mean(np.stack(self.precisions, axis=0), axis=0)
        avg_recall_curve = np.mean(np.stack(self.recalls, axis=0), axis=0)

        f_betas = (1 + self.beta2) * (avg_precision_curve * avg_recall_curve) / (self.beta2 * avg_precision_curve + avg_recall_curve + 1e-12)
        
        max_fm = np.max(f_betas)
        
        return {
            "adpFm": avg_adpfm,
            "maxFm": max_fm,
            "avgP": np.mean(avg_precision_curve), # 참고용
            "avgR": np.mean(avg_recall_curve)   # 참고용
        }


class WeightedFmeasure(Fmeasure):
    """
    Weighted F-measure
    """
    def __init__(self, beta2=1.0): # F-measure 논문 기본값은 1.0
        super(WeightedFmeasure, self).__init__(beta2=beta2)
        self.weighted_f_results = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = self._prepare_data(pred, gt)
        
        if gt.sum() == 0:
            # GT가 비어있으면, 1-pred의 평균 (배경을 잘 예측했는지)
            wfm = np.mean(1 - pred)
            self.weighted_f_results.append(wfm)
        else:
            # "Enhanced-alignment measure for binary foreground map evaluation" (2014)
            # 논문 저자 코드를 기반으로 한 구현
            
            # 1. 픽셀 가중치 계산 (GT 기준)
            # [BUG FIX 2] cv2 함수를 사용
            dst = cv2.distanceTransform((gt * 255).astype(np.uint8), cv2.DIST_L2, 0)
            bw = (dst == 0) # GT 영역
            
            # 가중치 맵 w
            w = np.ones_like(gt) * 5.0 # 기본 가중치
            
            # ▼▼▼ [BUG FIX 3] OpenCV dtype 오류 수정 ▼▼▼
            # 1. dst를 float64로 변환 (Laplacian 입력/출력 타입 일치)
            dst_64f = dst.astype(np.float64) 
            # 2. float64 입력으로 Laplacian 계산
            lap = cv2.Laplacian(dst_64f, cv2.CV_64F)
            # ▲▲▲ [수정 완료] ▲▲▲
            
            # 라플라시안 계산 시 0으로 나누는 오류 방지
            lap[bw] = 1.0 # 0이 되는 것을 방지 (어차피 아래에서 덮어쓰므로)
            
            w_bw = 1 + (dst[bw] / lap[bw])
            # w_bw 값이 무한대가 되거나 너무 큰 경우를 대비 (안정성)
            w_bw[lap[bw] == 0] = 1.0 
            
            w[bw] = w_bw

            # 2. Weighted Precision/Recall
            tp_matrix = w * (pred * gt) # 가중치가 적용된 TP
            
            precision_num = tp_matrix.sum()
            precision_den = (w * pred).sum() + 1e-12
            
            recall_num = tp_matrix.sum()
            recall_den = (w * gt).sum() + 1e-12
            
            precision = precision_num / precision_den
            recall = recall_num / recall_den

            # 3. Weighted F-measure
            wfm = (1 + self.beta2) * (precision * recall) / (self.beta2 * precision + recall + 1e-12)
            self.weighted_f_results.append(wfm)
            
        self.num_samples += 1

    def get_results(self) -> dict:
        avg_wfm = np.mean(np.array(self.weighted_f_results)) if self.num_samples > 0 else 0
        return {"wFm": avg_wfm}


class Emeasure(_BaseMetric):
    """
    Enhanced-alignment measure (E-measure)
    """
    def __init__(self):
        super(Emeasure, self).__init__()

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = self._prepare_data(pred, gt)
        
        # 256개 임계값에 대한 E-measure 곡선 계산
        thresholds = np.linspace(0, 1, 256)
        sample_ems = []
        
        for th in thresholds:
            if th == 0 or th == 1: # 양 끝값 제외
                continue
                
            # Alignment Term 계산
            binary_pred = (pred >= th).astype(np.float64)
            mu_pred = np.mean(binary_pred)
            mu_gt = np.mean(gt)
            
            align_matrix = 2 * (mu_pred * mu_gt) / (mu_pred**2 + mu_gt**2 + 1e-12)
            
            # Enhanced Alignment Term
            phi = (binary_pred - mu_pred) * (gt - mu_gt)
            phi_numerator = (phi + 1)**2 / 4 # (0~1)
            
            # [BUG FIX 2] 분모 안정성 강화
            mean_pred_sq = np.mean((binary_pred - mu_pred)**2)
            mean_gt_sq = np.mean((gt - mu_gt)**2)
            phi_denominator = mean_pred_sq + mean_gt_sq + 1e-12
            
            enhanced_align_matrix = phi_numerator / phi_denominator
            
            em = np.mean(align_matrix * enhanced_align_matrix)
            sample_ems.append(em)

        self.total_values.append(np.array(sample_ems))
        self.num_samples += 1

    def get_results(self) -> dict:
        if self.num_samples == 0:
            return {"Em": 0}
        
        # 전체 데이터셋에 대한 평균 곡선
        avg_em_curve = np.mean(np.stack(self.total_values, axis=0), axis=0)
        max_em = np.max(avg_em_curve) # 최대값 (Em)
        
        return {"Em": max_em}


class Smeasure(_BaseMetric):
    """
    Structural measure (S-measure)
    """
    def __init__(self, alpha=0.5):
        super(Smeasure, self).__init__()
        self.alpha = alpha

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = self._prepare_data(pred, gt)
        
        gt_fg = (gt > 0)
        gt_bg = (gt == 0)

        # 1. Object-aware S-measure (So)
        s_object = 0
        if gt_fg.sum() > 0:
            pred_fg = pred[gt_fg]
            mu_pred_fg = np.mean(pred_fg)
            sigma_pred_fg = np.std(pred_fg)
            
            # 구조적 유사도 (x)
            s_x = 2 * mu_pred_fg / (mu_pred_fg**2 + 1 + 1e-12) # GT의 mu는 1.0
            # 분포 유사도 (y)
            # [BUG FIX 2] sigma_gt가 0이므로 s_y 계산 수정
            s_y = 2 * (sigma_pred_fg * 0) / (sigma_pred_fg**2 + 0**2 + 1e-12)
            # S-Measure 논문 저자 코드는 다음을 사용함:
            if sigma_pred_fg == 0:
                s_y = 1.0 if mu_pred_fg == 0 else 0.0 # GT(0)와 sigma가 같음
            else:
                s_y = 2 * sigma_pred_fg * 0 / (sigma_pred_fg**2 + 1e-12) # 0
            
            # S-Measure 원본 논문(2017)의 Eq. (4)
            sigma_cross = np.mean((pred_fg - mu_pred_fg) * (1.0 - 1.0)) # 0
            s_y = (2 * sigma_cross + 1e-12) / (sigma_pred_fg**2 + 0**2 + 1e-12) # 0
            # 저자 MATLAB 코드는 이진 GT(mu=1, std=0)에 대해 다르게 처리함
            # mu_x*mu_y*2 / (mu_x^2 + mu_y^2)
            # std_x*std_y*2 / (std_x^2 + std_y^2)
            
            # 여기서는 mu=1, std=0 인 GT와의 유사도를 직접 계산
            mu_gt_fg = 1.0
            sigma_gt_fg = 0.0
            
            s_x = (2 * mu_pred_fg * mu_gt_fg + 1e-12) / (mu_pred_fg**2 + mu_gt_fg**2 + 1e-12)
            s_y = (2 * sigma_pred_fg * sigma_gt_fg + 1e-12) / (sigma_pred_fg**2 + sigma_gt_fg**2 + 1e-12)
            
            s_object = self.alpha * s_x + (1 - self.alpha) * s_y
            
            if np.isnan(s_object):
                # 단일 값 픽셀 등 예외 처리
                s_object = 1.0 if mu_pred_fg > 0.5 else 0.0

        # 2. Region-aware S-measure (Sr)
        # 4개의 영역 (GT=1, P>T), (GT=0, P<=T), (GT=1, P<=T), (GT=0, P>T)
        pred_mean = pred.mean()
        pred_fg_reg = (pred > pred_mean)
        pred_bg_reg = (pred <= pred_mean)
        
        gt_fg_reg = gt_fg
        gt_bg_reg = gt_bg

        # 픽셀 수
        w1 = (gt_fg_reg & pred_fg_reg).sum()
        w2 = (gt_bg_reg & pred_bg_reg).sum()
        w3 = (gt_fg_reg & pred_bg_reg).sum()
        w4 = (gt_bg_reg & pred_fg_reg).sum()
        
        total_w = w1 + w2 + w3 + w4
        
        # 영역 가중치
        w1 /= (total_w + 1e-12)
        w2 /= (total_w + 1e-12)
        w3 /= (total_w + 1e-12)
        w4 /= (total_w + 1e-12)
        
        # 영역 유사도
        s1 = 1.0 # (GT=1, P=1)
        s2 = 1.0 # (GT=0, P=0)
        
        # [BUG FIX 2] 영역이 0일 경우 NaN 방지
        pred_w3 = pred[gt_fg_reg & pred_bg_reg]
        s3 = np.mean(pred_w3) if pred_w3.size > 0 else 0 # (GT=1, P=0)
        
        pred_w4 = pred[gt_bg_reg & pred_fg_reg]
        s4 = 1.0 - np.mean(pred_w4) if pred_w4.size > 0 else 1 # (GT=0, P=1)
        
        # S-region
        s_region = w1*s1 + w2*s2 + w3*s3 + w4*s4

        # 3. S-measure
        sm = 0.5 * s_object + 0.5 * s_region # 논문에서는 0.5:0.5 사용
        
        # [FIX] NaN 처리 개선
        if np.isnan(sm) or np.isinf(sm):
            if gt.sum() == 0:
                # GT가 완전히 비어있으면 배경 예측 점수
                sm = 1.0 - pred.mean()
            else:
                sm = 0.0

        self.total_value += sm
        self.num_samples += 1

    def get_results(self) -> dict:
        avg_sm = self.total_value / self.num_samples if self.num_samples > 0 else 0
        return {"Sm": avg_sm}