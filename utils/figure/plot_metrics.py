import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import io

# ---------------------------------------------------------
# 1. 데이터 준비 (CSV 파일이 있다면 pd.read_csv('data.csv') 사용)
# ---------------------------------------------------------
csv_data = """Model,Params,FLOPs,Time
SINet-V2,26.5,8.4,18
SLT-Net,32.5,65.8,35
Ours(FP32),50.0,75.2,42
Ours(INT8),12.5,18.5,10"""

# 문자열을 데이터프레임으로 변환 (실제 사용 시엔 아래 주석을 풀고 파일 경로를 넣으세요)
df = pd.read_csv(io.StringIO(csv_data))
# df = pd.read_csv('data.csv') 

# ---------------------------------------------------------
# 2. 그래프 스타일 설정 (논문용 깔끔한 스타일)
# ---------------------------------------------------------
sns.set_style("whitegrid") # 격자 무늬 배경
plt.rcParams.update({
    'font.family': 'serif',          # 논문에서 선호하는 세리프 폰트
    'font.size': 12,                 # 기본 폰트 크기
    'axes.titlesize': 14,            # 제목 크기
    'axes.labelsize': 12,            # 축 라벨 크기
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300                # 고해상도 (인쇄용)
})

# ---------------------------------------------------------
# 3. 그래프 그리기 (1행 3열 구조)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # 가로 18인치, 세로 6인치

# 그릴 대상 리스트 정의 (데이터 컬럼명, Y축 라벨, 제목, 색상 테마)
metrics = [
    ('Params', 'Parameters (M)', 'Model Size', 'Blues'),
    ('FLOPs', 'GFLOPs', 'Computational Cost', 'Oranges'),
    ('Time', 'Inference Time (ms)', 'Latency', 'Greens')
]

for i, (col, ylabel, title, color_map) in enumerate(metrics):
    ax = axes[i]
    
    # 'Ours' 모델 강조를 위한 색상 지정 로직
    # 기본은 회색, Ours가 포함된 모델은 강조색 사용
    bar_colors = []
    base_color = sns.color_palette(color_map)[3] # 테마색 추출
    
    for model_name in df['Model']:
        if 'Ours' in model_name:
            bar_colors.append(base_color)  # 우리 모델은 진한 색
        else:
            bar_colors.append('lightgray') # 비교 모델은 연한 회색

    # 막대 그래프 생성
    bars = ax.bar(df['Model'], df[col], color=bar_colors, edgecolor='black', linewidth=0.8)
    
    # 설정
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.7) # Y축 그리드만 표시
    ax.set_axisbelow(True) # 그리드가 막대 뒤로 가도록 설정

    # 막대 위에 수치 표시 (Bar Label)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (height * 0.02),
                f'{height}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # X축 라벨 회전 (이름이 길 경우)
    # ax.tick_params(axis='x', rotation=15) 

# ---------------------------------------------------------
# 4. 마무리 및 저장
# ---------------------------------------------------------
plt.tight_layout() # 간격 자동 조정
plt.savefig('comparison_result.png', dpi=300, bbox_inches='tight') # 이미지 저장
plt.show() # 화면 출력

print("그래프가 'comparison_result.png'로 저장되었습니다.")