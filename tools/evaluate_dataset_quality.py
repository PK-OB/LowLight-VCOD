import os
import torch
import lpips
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# ==============================================================================
# 사용자 설정: 이 부분의 경로를 자신의 환경에 맞게 직접 수정하세요.
# ==============================================================================
# 1. 원본 주간 데이터셋의 최상위 폴더 경로
#    (예: .../MoCA-Mask/Seq_Test)
ORIGINAL_ROOT_DIR = '/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Seq_Train'

# 2. 합성 야간 데이터셋의 최상위 폴더 경로
#    (예: .../MoCA-Mask/Seq_Test_Night)
SYNTHETIC_ROOT_DIR = '/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Seq_Train_Night'
# ==============================================================================


def find_image_pairs(original_root, synthetic_root, image_folder_name="Imgs"):
    """
    두 개의 최상위 디렉토리 구조(예: Root/species/Imgs/*.jpg)를 탐색하여
    상대 경로가 동일한 이미지 쌍의 전체 경로 리스트를 반환합니다.
    """
    image_pairs = []
    print(f"Scanning for original images in: {original_root}")
    
    # 원본 디렉토리를 기준으로 모든 파일을 찾습니다.
    for dirpath, dirnames, filenames in os.walk(original_root):
        
        # 'Imgs' 폴더에 있는 이미지만 대상으로 합니다.
        if os.path.basename(dirpath) == image_folder_name:
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    original_path = os.path.join(dirpath, filename)
                    
                    # 원본 경로에서 상대 경로를 추출합니다.
                    # 예: .../Seq_Test/arctic_fox/Imgs/001.jpg
                    #    -> arctic_fox/Imgs/001.jpg
                    relative_path = os.path.relpath(original_path, original_root)
                    
                    # 상대 경로를 사용하여 합성 이미지의 전체 경로를 만듭니다.
                    synthetic_path = os.path.join(synthetic_root, relative_path)
                    
                    # 두 파일이 모두 존재하는 경우에만 쌍으로 추가합니다.
                    if os.path.exists(synthetic_path):
                        image_pairs.append((original_path, synthetic_path))
                        
    return image_pairs


def main():
    # GPU 사용 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # LPIPS 모델 초기화 (AlexNet 기반)
    loss_fn_lpips = lpips.LPIPS(net='alex').to(device)

    # LPIPS 계산을 위한 전처리 ([-1, 1] 범위로 정규화)
    preprocess_lpips = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [-1, 1] 범위
    ])
    
    # SSIM 계산을 위한 전처리 (간단한 리사이즈)
    preprocess_ssim = transforms.Compose([
        transforms.Resize((256, 256)),
    ])

    # 수정된 함수를 호출하여 이미지 쌍 목록을 가져옵니다.
    image_pairs = find_image_pairs(ORIGINAL_ROOT_DIR, SYNTHETIC_ROOT_DIR)

    if not image_pairs:
        print("오류: 두 디렉토리 구조에서 공통 이미지 파일을 찾을 수 없습니다.")
        print(f"원본 경로: {ORIGINAL_ROOT_DIR}")
        print(f"합성 경로: {SYNTHETIC_ROOT_DIR}")
        return

    print(f"비교할 공통 이미지 쌍 {len(image_pairs)}개를 찾았습니다.")

    ssim_scores = []
    lpips_scores = []

    # (원본 경로, 합성 경로) 쌍으로 루프를 돕니다.
    for original_path, synthetic_path in tqdm(image_pairs, desc="Calculating Metrics"):
        try:
            # 이미지 불러오기
            img_original = Image.open(original_path).convert("RGB")
            img_synthetic = Image.open(synthetic_path).convert("RGB")

            # --- 1. SSIM 계산 (구조적 유사성) ---
            # SSIM은 그레이스케일로 변환하여 계산합니다.
            img_orig_ssim_gray = np.array(preprocess_ssim(img_original).convert("L"))
            img_synth_ssim_gray = np.array(preprocess_ssim(img_synthetic).convert("L"))
            
            # data_range는 픽셀의 최대값 (0-255)
            score_ssim = ssim(img_orig_ssim_gray, img_synth_ssim_gray, data_range=255)
            ssim_scores.append(score_ssim)

            # --- 2. LPIPS 계산 (인지적 유사성) ---
            img_orig_lpips = preprocess_lpips(img_original).unsqueeze(0).to(device)
            img_synth_lpips = preprocess_lpips(img_synthetic).unsqueeze(0).to(device)
            
            with torch.no_grad():
                score_lpips = loss_fn_lpips(img_orig_lpips, img_synth_lpips).item()
            lpips_scores.append(score_lpips)

        except Exception as e:
            print(f"\n파일 처리 중 오류 발생 (Skip): {os.path.basename(original_path)}. 오류: {e}")
            
    # --- 3. 최종 결과 집계 ---
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0
    avg_lpips = np.mean(lpips_scores) if lpips_scores else 0

    print("\n" + "=" * 50)
    print("--- 합성 데이터셋 품질 평가 결과 ---")
    print(f" 원본 경로: {ORIGINAL_ROOT_DIR}")
    print(f" 합성 경로: {SYNTHETIC_ROOT_DIR}")
    print(f" 비교 샘플: {len(ssim_scores)} 개")
    print("-" * 50)
    print(f" (↑) 평균 SSIM (구조 유지): {avg_ssim:.4f}")
    print(f" (↓) 평균 LPIPS (인지적 차이): {avg_lpips:.4f}")
    print("-" * 50)
    print("해석:")
    print("- SSIM이 1.0에 가까울수록: 원본의 구조와 특징이 잘 보존되었습니다. (뭉개짐이 적음)")
    print("- LPIPS가 낮을수록: 두 이미지가 인지적으로 유사합니다.")
    print("  (스타일 변환이므로 0.0이 아니라 적절히 낮은 값이 나오는 것이 이상적입니다.)")
    print("=" * 50)

if __name__ == '__main__':
    main()