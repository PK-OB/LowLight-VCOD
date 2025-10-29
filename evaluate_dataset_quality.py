import os
import torch
import lpips
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# --- 설정 ---
# 1. 원본 주간 데이터셋의 최상위 폴더 경로
ORIGINAL_ROOT_DIR = '/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Test' # 예: 'data/MoCA-Mask/Train'

# 2. 합성 야간 데이터셋의 최상위 폴더 경로
SYNTHETIC_ROOT_DIR = '/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Test_Night' # 예: 'data/Night-Camo-Fauna'
# --- 설정 끝 ---

# ▼▼▼ 수정된 부분 ▼▼▼
def find_image_pairs(original_root, synthetic_root):
    """
    두 개의 최상위 디렉토리 구조를 탐색하여
    상대 경로가 동일한 이미지 쌍의 전체 경로 리스트를 반환합니다.
    """
    image_pairs = []
    print(f"Scanning for original images in: {original_root}")
    # 원본 디렉토리를 기준으로 모든 파일을 찾습니다.
    for dirpath, _, filenames in os.walk(original_root):
        # 'Imgs' 폴더에 있는 이미지만 대상으로 합니다.
        if os.path.basename(dirpath) == 'Imgs':
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    original_path = os.path.join(dirpath, filename)
                    
                    # 원본 경로에서 상대 경로를 추출합니다.
                    # 예: data/MoCA/Train/species/Imgs/001.jpg -> species/Imgs/001.jpg
                    relative_path = os.path.relpath(original_path, original_root)
                    
                    # 상대 경로를 사용하여 합성 이미지의 전체 경로를 만듭니다.
                    synthetic_path = os.path.join(synthetic_root, relative_path)
                    
                    # 두 파일이 모두 존재하는 경우에만 쌍으로 추가합니다.
                    if os.path.exists(synthetic_path):
                        image_pairs.append((original_path, synthetic_path))
    return image_pairs
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

def main():
    # GPU 사용 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # LPIPS 모델 초기화 (AlexNet 기반)
    loss_fn_lpips = lpips.LPIPS(net='alex').to(device)

    # 이미지 전처리 파이프라인
    preprocess_lpips = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    preprocess_ssim = transforms.Compose([
        transforms.Resize((256, 256)),
    ])

    # ▼▼▼ 수정된 부분 ▼▼▼
    # 수정된 함수를 호출하여 이미지 쌍 목록을 가져옵니다.
    image_pairs = find_image_pairs(ORIGINAL_ROOT_DIR, SYNTHETIC_ROOT_DIR)

    if not image_pairs:
        print("Error: No common image files found between the two directory structures.")
        return
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    print(f"Found {len(image_pairs)} common image pairs to evaluate.")

    ssim_scores = []
    lpips_scores = []

    # ▼▼▼ 수정된 부분 ▼▼▼
    # 파일 이름 대신 (원본 경로, 합성 경로) 쌍으로 루프를 돕니다.
    for original_path, synthetic_path in tqdm(image_pairs, desc="Calculating Metrics"):
        try:
            # 이미지 불러오기
            img_original = Image.open(original_path).convert("RGB")
            img_synthetic = Image.open(synthetic_path).convert("RGB")

            # --- SSIM 계산 ---
            img_orig_ssim = np.array(preprocess_ssim(img_original).convert("L"))
            img_synth_ssim = np.array(preprocess_ssim(img_synthetic).convert("L"))
            score_ssim = ssim(img_orig_ssim, img_synth_ssim, win_size=7, data_range=255)
            ssim_scores.append(score_ssim)

            # --- LPIPS 계산 ---
            img_orig_lpips = preprocess_lpips(img_original).unsqueeze(0).to(device)
            img_synth_lpips = preprocess_lpips(img_synthetic).unsqueeze(0).to(device)
            
            with torch.no_grad():
                score_lpips = loss_fn_lpips(img_orig_lpips, img_synth_lpips).item()
            lpips_scores.append(score_lpips)

        except Exception as e:
            print(f"\nCould not process {os.path.basename(original_path)}. Error: {e}")
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    # 평균 점수 계산
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0
    avg_lpips = np.mean(lpips_scores) if lpips_scores else 0

    # 결과 테이블 출력
    print("\n--- 합성 데이터셋 품질 평가 결과 ---")
    print("=" * 40)
    print(f"{'평가 항목':<20} | {'평균 SSIM ↑':<10} | {'평균 LPIPS ↓':<10}")
    print("-" * 40)
    print(f"{'원본(주간) vs 합성(야간)':<20} | {avg_ssim:<10.4f} | {avg_lpips:<10.4f}")
    print("-" * 40)
    print("비고: SSIM이 높고 LPIPS가 적절히 높으면, 구조는 유지하며 스타일은 성공적으로 변환되었음을 의미함.")
    print("=" * 40)

if __name__ == '__main__':
    main()