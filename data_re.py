import os
import shutil
import random # Shuffle은 사용하지 않지만, random 모듈은 유지 (혹시 모를 확장을 위해)
from collections import defaultdict
from tqdm import tqdm
import argparse
import logging
from pathlib import Path

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

###########################################################################
### 사용자 설정 ###
###########################################################################
# 1. 기존 데이터셋 경로 (합칠 Train/Test 경로들을 리스트로 지정)
ORIGINAL_DATA_ROOTS_TO_MERGE = [
    "/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Train",
    "/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Test"
]
NIGHT_DATA_ROOTS_TO_MERGE = [
    "/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Train_Night",
    "/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Test_Night"
]

# 2. 새롭게 생성될 데이터셋 경로 (올바르게 분할된 결과 저장 경로)
NEW_ORIGINAL_TRAIN_ROOT = "/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Seq_Train"      # 예: 순서 분할 주간 Train
NEW_ORIGINAL_TEST_ROOT = "/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Seq_Test"       # 예: 순서 분할 주간 Test
NEW_NIGHT_TRAIN_ROOT = "/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Seq_Train_Night"  # 예: 순서 분할 야간 Train
NEW_NIGHT_TEST_ROOT = "/home/sjy/paper/JED-VCOD/data/MoCA-Mask/Seq_Test_Night"   # 예: 순서 분할 야간 Test

# 3. 폴더 이름
IMAGE_FOLDER_NAME = "Imgs"
MASK_FOLDER_NAME = "GT"

# 4. 분할 비율 (학습 데이터 비율) 및 시드 (시드는 여기서는 사용 안 함)
DEFAULT_TRAIN_RATIO = 0.8
# DEFAULT_SEED = 42 # 순서 기반 분할에는 시드 불필요
###########################################################################

def find_all_species_files_sequential(root_dirs_to_merge, img_folder=IMAGE_FOLDER_NAME, gt_folder=MASK_FOLDER_NAME):
    """
    여러 루트 디렉토리에서 모든 종별 이미지 및 GT 파일 목록을 병합하고,
    **파일 이름 순서대로 정렬된 리스트**를 반환.
    """
    all_species_files = defaultdict(lambda: {'img': [], 'gt': []})
    seen_files_identifier = set() # 중복 파일 (종/파일명 기준) 방지

    logger.info(f"데이터 스캔 및 병합 시작 (순서 정렬): {root_dirs_to_merge}")

    for root_dir in root_dirs_to_merge:
        if not root_dir or not os.path.exists(root_dir):
            logger.warning(f"경로 없음 (Skip): {root_dir}"); continue
        logger.info(f"'{root_dir}' 스캔 중...")
        try:
            for species_name in os.listdir(root_dir):
                species_path = os.path.join(root_dir, species_name)
                if not os.path.isdir(species_path): continue

                img_path = os.path.join(species_path, img_folder)
                gt_path = os.path.join(species_path, gt_folder)

                if os.path.exists(img_path):
                    # 파일 목록 가져오기 (정렬되지 않음)
                    img_filenames = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                    # ▼▼▼ 파일 이름 순서대로 정렬 ▼▼▼
                    img_filenames.sort()
                    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

                    for filename in img_filenames:
                        base_name = os.path.splitext(filename)[0]
                        file_identifier = f"{species_name}/{filename}"

                        if file_identifier in seen_files_identifier: continue # 중복 방지

                        img_file = os.path.join(img_path, filename)

                        # GT 파일 찾기
                        gt_file = None
                        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                            potential_gt = os.path.join(gt_path, base_name + ext)
                            if os.path.exists(potential_gt): gt_file = potential_gt; break

                        if gt_file: # GT 존재 시 추가
                            all_species_files[species_name]['img'].append(img_file)
                            all_species_files[species_name]['gt'].append(gt_file)
                            seen_files_identifier.add(file_identifier)

        except Exception as e: logger.error(f"'{root_dir}' 스캔 오류: {e}")

    # 최종 정렬 확인 (이미 sort() 했지만, 병합 후 최종 확인)
    for species_name in all_species_files:
        # 경로 문자열 자체를 다시 정렬 (혹시 모를 순서 섞임 방지)
        img_list = all_species_files[species_name]['img']
        gt_list = all_species_files[species_name]['gt']
        if img_list and gt_list: # 파일이 있을 때만
            combined = sorted(zip(img_list, gt_list), key=lambda pair: pair[0]) # 이미지 경로 기준으로 정렬
            all_species_files[species_name]['img'] = [p[0] for p in combined]
            all_species_files[species_name]['gt'] = [p[1] for p in combined]

    total_files = sum(len(f['img']) for f in all_species_files.values())
    logger.info(f"총 {len(all_species_files)}종, {total_files}개 유효 쌍 병합 및 정렬 완료.")
    if total_files == 0: logger.error("병합 데이터 없음. 경로/폴더 구조 확인.")
    return all_species_files


def split_and_copy_files_sequential(all_species_files, train_ratio, new_train_root, new_test_root, img_folder=IMAGE_FOLDER_NAME, gt_folder=MASK_FOLDER_NAME):
    """
    **종별 & 시간 순서**로 파일 목록을 분할하고 새 디렉토리로 복사
    """
    if not new_train_root or not new_test_root: logger.error("새 경로 미지정."); return 0, 0
    if new_train_root == new_test_root: logger.error("Train/Test 경로 동일 불가."); return 0, 0
    if os.path.exists(new_train_root) and os.listdir(new_train_root): logger.warning(f"대상 Train '{new_train_root}' 비어있지 않음.")
    if os.path.exists(new_test_root) and os.listdir(new_test_root): logger.warning(f"대상 Test '{new_test_root}' 비어있지 않음.")
    os.makedirs(new_train_root, exist_ok=True); os.makedirs(new_test_root, exist_ok=True)
    logger.info(f"\n'{os.path.basename(new_train_root)}' 및 '{os.path.basename(new_test_root)}'로 **종별 순서({train_ratio:.1f})** 분할/복사 시작...")

    total_train = 0; total_test = 0

    for species_name, files in tqdm(all_species_files.items(), desc="Species Progress"):
        img_files = files.get('img', []); gt_files = files.get('gt', [])
        if not img_files or len(img_files) != len(gt_files):
            logger.warning(f"'{species_name}' 종: 파일 수 오류. Skip."); continue

        # --- 시간 순서(파일명 정렬 기준)로 비율 기반 분할 ---
        num_files = len(img_files)
        num_train = int(num_files * train_ratio)

        # 최소 1개 보장 (파일 2개 이상 시)
        if num_files >= 2:
            if num_train == 0: num_train = 1
            if num_train == num_files: num_train = num_files - 1 # Test에도 최소 1개

        # **섞지 않고 순서대로 자르기**
        train_img_files = img_files[:num_train]
        train_gt_files = gt_files[:num_train]
        test_img_files = img_files[num_train:]
        test_gt_files = gt_files[num_train:]
        # ------------------------------------------------

        logger.debug(f"'{species_name}': {num_files}개 -> Train {len(train_img_files)}개 / Test {len(test_img_files)}개 (순서 유지)")
        total_train += len(train_img_files)
        total_test += len(test_img_files)

        # 복사 함수
        def copy_files_to_dest(img_list, gt_list, dest_root, species, img_f, gt_f):
            count = 0
            img_dest = os.path.join(dest_root, species, img_f); gt_dest = os.path.join(dest_root, species, gt_f)
            os.makedirs(img_dest, exist_ok=True); os.makedirs(gt_dest, exist_ok=True)
            for img_src, gt_src in zip(img_list, gt_list):
                try:
                    if os.path.exists(img_src) and os.path.exists(gt_src):
                        shutil.copy2(img_src, os.path.join(img_dest, os.path.basename(img_src)))
                        shutil.copy2(gt_src, os.path.join(gt_dest, os.path.basename(gt_src)))
                        count += 1
                    else: logger.warning(f"소스 누락: {img_src} or {gt_src}")
                except Exception as e: logger.error(f"파일 복사 오류 ({os.path.basename(dest_root)}): {e}")
            return count

        # Train / Test 복사 실행
        copy_files_to_dest(train_img_files, train_gt_files, new_train_root, species_name, img_folder, gt_folder)
        copy_files_to_dest(test_img_files, test_gt_files, new_test_root, species_name, img_folder, gt_folder)

    logger.info("\n종별 순서 분할 및 복사 완료!")
    logger.info(f"총 학습 파일 쌍 수 (모든 종 합계): {total_train}")
    logger.info(f"총 테스트 파일 쌍 수 (모든 종 합계): {total_test}")
    return total_train, total_test

def main(args):
    # 경로 유효성 검사
    all_paths_valid = True
    for path_list in [ORIGINAL_DATA_ROOTS_TO_MERGE, NIGHT_DATA_ROOTS_TO_MERGE]:
        for path in path_list:
            if not path or not os.path.exists(path):
                logger.error(f"오류: 입력 경로 없음 - {path}"); all_paths_valid = False
    if not all_paths_valid: logger.error("스크립트 상단 경로 설정을 확인하세요."); return

    # 랜덤 시드는 여기서는 사용 안 함 (파일 순서 고정)
    logger.info(f"Train 비율: {args.train_ratio:.1f}")

    # 1. 원본 주간 데이터 처리 (병합 -> 종별 순서 분할 -> 복사)
    logger.info("="*30 + " 원본 주간 데이터 처리 " + "="*30)
    all_original_files = find_all_species_files_sequential(ORIGINAL_DATA_ROOTS_TO_MERGE)
    if not all_original_files:
        logger.error("처리할 원본 주간 데이터 없음. 중단."); return
    else:
        split_and_copy_files_sequential(all_original_files, args.train_ratio,
                                        NEW_ORIGINAL_TRAIN_ROOT, NEW_ORIGINAL_TEST_ROOT)

    # 2. 야간 데이터 처리 (병합 -> 종별 순서 분할 -> 복사)
    logger.info("\n" + "="*30 + " 야간 데이터 처리 " + "="*30)
    all_night_files = find_all_species_files_sequential(NIGHT_DATA_ROOTS_TO_MERGE)
    if not all_night_files:
         logger.error("처리할 야간 데이터 없음. 중단.")
    else:
        split_and_copy_files_sequential(all_night_files, args.train_ratio,
                                        NEW_NIGHT_TRAIN_ROOT, NEW_NIGHT_TEST_ROOT)

    # --- 최종 안내 ---
    logger.info("\n" + "="*70)
    logger.info("모든 작업이 완료되었습니다!")
    logger.info("다음 단계를 **반드시** 수행하세요:")
    logger.info(f"1. 생성된 디렉토리 확인:")
    logger.info(f"   - 주간 Train: {os.path.abspath(NEW_ORIGINAL_TRAIN_ROOT)}")
    logger.info(f"   - 주간 Test:  {os.path.abspath(NEW_ORIGINAL_TEST_ROOT)}")
    logger.info(f"   - 야간 Train: {os.path.abspath(NEW_NIGHT_TRAIN_ROOT)}")
    logger.info(f"   - 야간 Test:  {os.path.abspath(NEW_NIGHT_TEST_ROOT)}")
    logger.info("   각 디렉토리 안에 **종별 폴더**가 있고, 파일들이 **시간 순서**대로 비율에 맞게 분배되었는지 확인하세요.")
    logger.info("\n2. `config.py` 파일의 경로들을 **새로운 경로**로 업데이트하세요:")
    logger.info(f"   - train['original_data_root'] = '{os.path.abspath(NEW_ORIGINAL_TRAIN_ROOT)}'")
    logger.info(f"   - train['folder_data_root'] = '{os.path.abspath(NEW_NIGHT_TRAIN_ROOT)}'")
    logger.info(f"   - evaluate['eval_original_data_root'] = '{os.path.abspath(NEW_ORIGINAL_TEST_ROOT)}'") # 검증/테스트용
    logger.info(f"   - evaluate['eval_folder_data_root'] = '{os.path.abspath(NEW_NIGHT_TEST_ROOT)}'")   # 검증/테스트용
    logger.info("\n3. `run_experiment.py` 확인/수정:")
    logger.info("   - **데이터 분할 로직**: 스크립트 내 `random_split` 또는 카테고리 분할 로직은 **제거**해야 합니다.")
    logger.info("     이제 파일 시스템에서 Train/Valid가 분리되었으므로, 간단히 로드합니다:")
    logger.info("     `train_ds = FolderImageMaskDataset(root_dir=cfg.train['folder_data_root'], ...)`")
    logger.info("     `val_ds = FolderImageMaskDataset(root_dir=cfg.evaluate['eval_folder_data_root'], ...)`") # 검증 시 Test 경로 사용
    logger.info("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="데이터셋 Train/Test 구조를 **종별 & 시간 순서** 비율에 따라 병합 후 재분배합니다.")
    parser.add_argument('--train_ratio', type=float, default=DEFAULT_TRAIN_RATIO,
                        help=f'각 종 내에서 학습 데이터로 사용할 비율 (프레임 순서 기준, 기본값: {DEFAULT_TRAIN_RATIO})')
    # 시드는 사용하지 않으므로 제거
    # parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='랜덤 시드')

    args = parser.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        logger.error("오류: --train_ratio는 0과 1 사이 값이어야 함."); exit()

    main(args)