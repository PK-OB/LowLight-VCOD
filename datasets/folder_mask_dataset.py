# pk-ob/jed-vcod/JED-VCOD-cc543b29cefb3a45b940bfd01f42c33af7a6bb25/datasets/folder_mask_dataset.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

class FolderImageMaskDataset(Dataset):
    # ▼▼▼ 수정된 부분: 'original_data_root' 인자 추가 ▼▼▼
    def __init__(self, root_dir, original_data_root, image_folder_name, mask_folder_name, clip_len=8, resolution=(224, 224), is_train=True, use_augmentation=True):
        self.root_dir = root_dir
        self.original_data_root = original_data_root # <-- 추가
        self.clip_len = clip_len
        self.resolution = resolution
        self.is_train = is_train
        self.use_augmentation = use_augmentation
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # ▼▼▼ 1. 로직 변경: 모든 이미지 경로를 찾는 대신, 클립의 '시작 프레임' 경로만 찾습니다. ▼▼▼
        self.clips = []
        print(f"Scanning dataset in '{root_dir}' to create video clips...")

        # root_dir 아래의 모든 하위 폴더를 탐색합니다. ('arctic_fox', 'black_cat_1' 등)
        for sub_dir in sorted(os.listdir(root_dir)):
            sub_dir_path = os.path.join(root_dir, sub_dir)
            if not os.path.isdir(sub_dir_path):
                continue

            image_dir = os.path.join(sub_dir_path, image_folder_name)
            mask_dir = os.path.join(sub_dir_path, mask_folder_name)

            if os.path.exists(image_dir) and os.path.exists(mask_dir):
                # 각 비디오(폴더)에 포함된 프레임 목록을 정렬하여 가져옵니다.
                frames = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
                
                # 비디오의 전체 프레임 수가 clip_len보다 길거나 같아야 클립을 만들 수 있습니다.
                if len(frames) >= clip_len:
                    # clip_len 길이의 클립을 생성할 수 있는 모든 시작 지점을 self.clips에 추가합니다.
                    # 예: 10프레임, clip_len=8 -> 시작 지점은 0, 1, 2 (총 3개 클립)
                    for i in range(len(frames) - clip_len + 1):
                        # 각 클립은 (이미지 경로 리스트, 마스크 경로 리스트) 튜플로 저장합니다.
                        image_paths = [os.path.join(image_dir, frames[i+j]) for j in range(clip_len)]
                        mask_paths = [
                            os.path.join(mask_dir, os.path.splitext(frames[i+j])[0] + '.png') for j in range(clip_len)
                        ]
                        
                        # 모든 마스크 파일이 실제로 존재하는지 확인 후 추가합니다.
                        if all(os.path.exists(p) for p in mask_paths):
                            self.clips.append((image_paths, mask_paths))
        
        print(f"Found {len(self.clips)} video clips.")
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        self.image_transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        
        # ▼▼▼ 수정된 부분: 원본 주간 이미지용 Transform (정규화 X, 0~1 스케일) ▼▼▼
        self.original_image_transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(), # [0, 1] 범위로 변환
        ])
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

    def __len__(self):
        # ▼▼▼ 2. 로직 변경: 전체 이미지 수가 아닌, 생성된 '클립'의 총 개수를 반환합니다. ▼▼▼
        return len(self.clips)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    def __getitem__(self, idx):
        # ▼▼▼ 3. 로직 변경: 단일 이미지를 복제하는 대신, 연속된 프레임들을 불러와 실제 클립을 만듭니다. ▼▼▼
        image_paths, mask_paths = self.clips[idx]
        
        image_clip_tensors = []
        mask_clip_tensors = []
        
        # ▼▼▼ 수정된 부분 ▼▼▼
        original_image_clip_tensors = [] # 원본 주간 이미지 텐서 리스트
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # 데이터 증강(Augmentation)을 클립 단위로 일관되게 적용하기 위한 파라미터를 먼저 결정합니다.
        apply_flip = self.is_train and self.use_augmentation and random.random() > 0.5
        
        try:
            # 클립에 포함된 모든 프레임 경로에 대해 반복합니다.
            for img_path, msk_path in zip(image_paths, mask_paths):
                image = Image.open(img_path).convert("RGB")
                mask = Image.open(msk_path).convert("L")
                
                # ▼▼▼ 수정된 부분: 원본 주간 이미지 로드 ▼▼▼
                relative_path = os.path.relpath(img_path, self.root_dir)
                original_img_path = os.path.join(self.original_data_root, relative_path)
                original_image = Image.open(original_img_path).convert("RGB")
                # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

                # is_train 모드이고, use_augmentation이 True일 때만 데이터 증강을 적용합니다.
                if self.is_train and self.use_augmentation:
                    # 클립의 모든 프레임에 동일한 좌우 반전을 적용하여 일관성을 유지합니다.
                    if apply_flip:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)
                        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                        original_image = original_image.transpose(Image.FLIP_LEFT_RIGHT) # <-- 추가
                    
                    # 색상 변형은 각 프레임에 독립적으로 적용될 수 있습니다. (야간 이미지만)
                    image = self.color_jitter(image)

                image_clip_tensors.append(self.image_transform(image))
                mask_clip_tensors.append(self.mask_transform(mask))
                
                # ▼▼▼ 수정된 부분 ▼▼▼
                original_image_clip_tensors.append(self.original_image_transform(original_image))
                # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        except FileNotFoundError as e:
            print(f"Warning: File not found, skipping this clip. Details: {e}")
            return None # DataLoader의 collate_fn에서 None을 처리해야 합니다.

        # 리스트에 담긴 각 프레임 텐서들을 stack하여 최종 클립 텐서를 만듭니다.
        image_clip = torch.stack(image_clip_tensors, dim=0)
        mask_clip = torch.stack(mask_clip_tensors, dim=0)
        
        # ▼▼▼ 수정된 부분 ▼▼▼
        original_image_clip = torch.stack(original_image_clip_tensors, dim=0)
        
        return image_clip, mask_clip, original_image_clip # 3개 항목 반환
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲