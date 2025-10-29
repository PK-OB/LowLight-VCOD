import os
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
from tqdm import tqdm
import glob

# ==============================================================================
# 사용자 설정: 이 부분의 경로를 자신의 환경에 맞게 직접 수정하세요.
# ==============================================================================
# 1. 원본 MoCA 데이터셋의 최상위 폴더 경로
MOCA_ROOT_PATH = "/home/sjy/paper/VCOD_Night/dataset/MoCA"

# 2. 변환된 이미지가 저장될 최상위 폴더 경로
OUTPUT_ROOT_PATH = "/home/sjy/paper/VCOD_Night/dataset/MoCA/MoCA_Night"
# ==============================================================================


def main():
    """
    MoCA 데이터셋의 주간 이미지를 InstructPix2Pix를 사용해 야간 이미지로 변환합니다.
    """
    # 1. GPU 사용 가능 여부 확인
    if not torch.cuda.is_available():
        print("오류: 이 스크립트는 CUDA 지원 GPU가 필요합니다.")
        return
    device = "cuda"

    # 2. InstructPix2Pix 모델 로드
    # 이 과정은 처음 실행 시 모델 파일을 다운로드하므로 시간이 걸릴 수 있습니다.
    print("InstructPix2Pix 모델을 로드하는 중입니다...")
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    pipe.to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    print("모델 로드가 완료되었습니다.")

    # 3. 변환할 MoCA 이미지 목록 탐색
    # MoCA/JPEGImages/동물_이름/*.jpg 형태의 모든 이미지 파일을 찾습니다.
    source_image_pattern = os.path.join(MOCA_ROOT_PATH, "JPEGImages", "**", "*.jpg")
    source_image_paths = glob.glob(source_image_pattern, recursive=True)
    
    if not source_image_paths:
        print(f"오류: '{MOCA_ROOT_PATH}' 경로에서 이미지를 찾을 수 없습니다. 경로를 확인해주세요.")
        return
        
    print(f"총 {len(source_image_paths)}개의 이미지를 변환합니다.")

    # 4. 메인 변환 루프
    # tqdm을 사용하여 진행 상황을 시각적으로 표시합니다.
    for image_path in tqdm(source_image_paths, desc="이미지 변환 진행률"):
        try:
            # 원본 이미지 로드
            image = Image.open(image_path).convert("RGB")
            
            # 텍스트 프롬프트 정의
            prompt = "make it look like night"
            
            # InstructPix2Pix 파이프라인 실행
            # image_guidance_scale 값이 높을수록 원본 이미지 구조를 더 유지합니다.
            edited_image = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5).images[0]

            # 5. 결과 이미지 저장
            # 원본과 동일한 폴더 구조를 유지하여 결과물 저장
            relative_path = os.path.relpath(image_path, MOCA_ROOT_PATH)
            output_path = os.path.join(OUTPUT_ROOT_PATH, relative_path)
            
            # 저장할 폴더가 없다면 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            edited_image.save(output_path)

        except Exception as e:
            # 특정 이미지 처리 중 오류 발생 시, 오류를 출력하고 계속 진행
            print(f"\n'{image_path}' 처리 중 오류 발생: {e}")
            continue
            
    print("\n🎉 모든 이미지 변환 작업이 완료되었습니다!")
    print(f"결과물은 '{os.path.abspath(OUTPUT_ROOT_PATH)}' 폴더에 저장되었습니다.")


if __name__ == "__main__":
    # 스크립트 상단에 정의된 경로를 사용하여 메인 함수를 바로 실행합니다.
    main()