# Benchmark Table 6 실행 가이드

## 사전 준비

### 1. 필요한 패키지 설치
```bash
pip install thop tabulate
```

### 2. 모델 체크포인트 준비
스크립트 상단의 `MODEL_CONFIGS`에서 체크포인트 경로 수정:
```python
MODEL_CONFIGS = [
    {
        'name': 'Model1',
        'precision': 'FP32',
        'checkpoint': 'checkpoints/your_fp32_model.pth',  # ← 수정
        ...
    },
    ...
]
```

## 실행 방법

```bash
python3 benchmark_table6.py
```

## 출력 형식

### 콘솔 출력
```
+-------------------+-------------+----------+-----------+------------+---------------+------------------------+--------+---------------+
| Model Precision   | Model Size  | Params   | FLOPs     | Latency    | Throughput    | Performance (LLCA-Mask)| mIoU ↑ | Drop Rate (%) |
|                   | (MB)        | (M)      | (G)       | (ms)       | (FPS)         | Sm ↑                   |        |               |
+===================+=============+==========+===========+============+===============+========================+========+===============+
| Model1 (FP32)     | 350.5       | 88.2     | 245.3     | 125.3      | 8.0           | 0.85                   | 0.82   | -             |
+-------------------+-------------+----------+-----------+------------+---------------+------------------------+--------+---------------+
| Model2 (INT8 ↓2%) | 87.6        | 88.2     | 245.3     | 45.2       | 22.1          | 0.83                   | 0.80   | 2.35%         |
+-------------------+-------------+----------+-----------+------------+---------------+------------------------+--------+---------------+
```

### 파일 저장
- `benchmark_results_table6.txt`: 테이블 형식
- `benchmark_results_table6.csv`: CSV 형식

## 주요 지표 설명

| 지표 | 설명 | 단위 |
|------|------|------|
| Model Size | 모델 파일 크기 | MB (소수점 1자리) |
| Params | 파라미터 수 | M (소수점 1자리) |
| FLOPs | 연산량 | G (소수점 1자리) |
| Latency | 추론 지연시간 | ms (소수점 1자리) |
| Throughput | 초당 처리 프레임 | FPS (소수점 1자리) |
| Sm | S-measure (구조 유사도) | 0-1 (소수점 2자리) |
| mIoU | Mean Intersection over Union | 0-1 (소수점 2자리) |
| Drop Rate | 성능 하락률 (baseline 대비) | % (소수점 2자리) |

## 커스터마이징

### 벤치마크 반복 횟수 조정
```python
WARMUP_ITERATIONS = 10      # Warmup 반복 횟수
BENCHMARK_ITERATIONS = 100  # 측정 반복 횟수
```

### GPU 선택
```python
DEVICE = 'cuda:0'  # GPU 0번 사용
DEVICE = 'cuda:1'  # GPU 1번 사용
```

### 평가 데이터셋 변경
```python
EVAL_DATA_ROOT = '/path/to/your/test/data'
EVAL_ORIGINAL_ROOT = '/path/to/your/original/data'
```

## 문제 해결

### thop 설치 실패 시
FLOPs 계산을 건너뛰고 0.0으로 표시됩니다. 수동 설치:
```bash
pip install thop
```

### OOM 오류 발생 시
평가 배치 크기를 1로 유지하고, 모델을 하나씩 벤치마크:
```python
MODEL_CONFIGS = [
    MODEL_CONFIGS[0]  # 첫 번째 모델만
]
```

### 체크포인트 없을 시
랜덤 초기화 모델로 테스트 (경고 메시지 출력):
```
Warning: Checkpoint not found: ...
Using random initialized model for demonstration.
```

## 예상 실행 시간

- 모델 1개당: 약 5-10분
- 전체 3개 모델: 약 15-30분

(데이터셋 크기와 GPU 성능에 따라 다름)
