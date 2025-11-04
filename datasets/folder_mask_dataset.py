# pk-ob/lowlight-vcod/LowLight-VCOD-1cbc811d7e25b5acb0aa6db157812983363cba26/datasets/folder_mask_dataset.py
# (▼▼▼ [업그레이드] Albumentations + 오류 수정 (Numpy/Tensor 충돌 및 Shape 불일치) ▼▼▼)

import os
import cv2 # Albumentations의 보간법 플래그 및 [FIX]용
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms # <-- 최종 텐서 변환을 위해 일부 유지
import random

# ▼▼▼ [추가] Albumentations 임포트 ▼▼▼
import albumentations as A
from albumentations.pytorch import ToTensorV2
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

class FolderImageMaskDataset(Dataset):
    # ▼▼▼ [수정] Albumentations 파이프라인으로 __init__ 업데이트 ▼▼▼
    def __init__(self, root_dir, original_data_root, image_folder_name, mask_folder_name, clip_len=8, resolution=(224, 224), is_train=True, use_augmentation=True):
        self.root_dir = root_dir
        self.original_data_root = original_data_root 
        self.clip_len = clip_len
        self.resolution = resolution
        self.is_train = is_train
        self.use_augmentation = use_augmentation
        
        # --- 1. 데이터 스캔 (기존과 동일) ---
        self.clips = []
        print(f"Scanning dataset in '{root_dir}' to create video clips...")
        for sub_dir in sorted(os.listdir(root_dir)):
            sub_dir_path = os.path.join(root_dir, sub_dir)
            if not os.path.isdir(sub_dir_path):
                continue

            image_dir = os.path.join(sub_dir_path, image_folder_name)
            mask_dir = os.path.join(sub_dir_path, mask_folder_name)

            if os.path.exists(image_dir) and os.path.exists(mask_dir):
                frames = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
                
                if len(frames) >= clip_len:
                    for i in range(len(frames) - clip_len + 1):
                        image_paths = [os.path.join(image_dir, frames[i+j]) for j in range(clip_len)]
                        mask_paths = [
                            os.path.join(mask_dir, os.path.splitext(frames[i+j])[0] + '.png') for j in range(clip_len)
                        ]
                        
                        if all(os.path.exists(p) for p in mask_paths):
                            self.clips.append((image_paths, mask_paths))
        
        print(f"Found {len(self.clips)} video clips.")

        # --- 2. Albumentations 증강 파이프라인 정의 ---
        
        # 2-1. 기하학적 변형 + 가려짐 (클립 전체에 일관되게 적용)
        self.geometric_occlusion_aug = A.ReplayCompose([
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0, 0.1),
                rotate=(-15, 15),
                shear=(-10, 10),
                p=0.5,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST
            ),
            A.CoarseDropout(
                max_holes=5, max_height=40, max_width=40,
                min_holes=1, min_height=10, min_width=10,
                fill_value=0,
                mask_fill_value=0,
                p=0.3
            )
        ], additional_targets={'mask_day': 'mask'})
        
        # 2-2. 저조도/노이즈 변형 (프레임마다 다르게, 야간 이미지에만 적용)
        self.photometric_aug = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MotionBlur(blur_limit=7, p=0.3),
            A.ImageCompression(quality_lower=40, quality_upper=80, p=0.2)
        ])

        # --- 3. 최종 변환 (Resize, ToTensor, Normalize) ---
        
        # 3-1. 야간 이미지용 (Resize -> Normalize -> ToTensor)
        self.final_transform_night = A.Compose([
            A.Resize(height=resolution[0], width=resolution[1], interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2() # <-- 야간 이미지는 여기서 텐서로 변환
        ])
        
        # 3-2. 마스크용 (Resize (Nearest))
        # ▼▼▼ [FIX] ToTensorV2() 제거. NumPy 배열을 반환하도록 수정 ▼▼▼
        self.final_transform_mask = A.Compose([
            A.Resize(height=resolution[0], width=resolution[1], interpolation=cv2.INTER_NEAREST),
        ])
        
        # 3-3. 주간 원본 이미지용 (Resize)
        # ▼▼▼ [FIX] ToTensorV2() 제거. NumPy 배열을 반환하도록 수정 ▼▼▼
        self.final_transform_day = A.Compose([
            A.Resize(height=resolution[0], width=resolution[1], interpolation=cv2.INTER_LINEAR),
        ])
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    def __len__(self):
        return len(self.clips)

    # ▼▼▼ [수정] Albumentations 파이프라인 + 오류 수정 로직 적용 ▼▼▼
    # [datasets/folder_mask_dataset.py] 파일의
# __getitem__ 함수 (def __getitem__(self, idx): 부터 return ... 까지)
# 전체를 아래 내용으로 교체하세요.

    def __getitem__(self, idx):
        image_paths, mask_paths = self.clips[idx]
        
        images_night_pil, images_mask_pil, images_day_pil = [], [], []
        
        try:
            # --- 1. 클립의 모든 PIL 이미지 로드 ---
            for img_path, msk_path in zip(image_paths, mask_paths):
                images_night_pil.append(Image.open(img_path).convert("RGB"))
                images_mask_pil.append(Image.open(msk_path).convert("L"))
                
                relative_path = os.path.relpath(img_path, self.root_dir)
                original_img_path = os.path.join(self.original_data_root, relative_path)
                images_day_pil.append(Image.open(original_img_path).convert("RGB"))
            
            # --- 2. PIL -> Numpy 변환 ---
            images_night_np = [np.array(img) for img in images_night_pil]
            images_mask_np = [np.array(img) for img in images_mask_pil]
            images_day_np = [np.array(img) for img in images_day_pil]

            # --- 3. 증강 적용 ---
            replay_data = None
            aug_night_list, aug_mask_list, aug_day_list = [], [], []

            for t in range(self.clip_len):
                night_t, mask_t, day_t = images_night_np[t], images_mask_np[t], images_day_np[t]

                # [FIX] Shape 불일치 해결: 증강 전 크기 강제 일치
                night_h, night_w = night_t.shape[:2]
                if mask_t.shape[:2] != (night_h, night_w):
                    mask_t = cv2.resize(mask_t, (night_w, night_h), interpolation=cv2.INTER_NEAREST)
                if day_t.shape[:2] != (night_h, night_w):
                    day_t = cv2.resize(day_t, (night_w, night_h), interpolation=cv2.INTER_LINEAR)

                if self.is_train and self.use_augmentation:
                    # 3-1. 기하학적 + 가려짐 (클립 전체에 일관되게 적용)
                    if t == 0:
                        transformed = self.geometric_occlusion_aug(image=night_t, mask=mask_t, mask_day=day_t)
                        replay_data = transformed['replay']
                    else:
                        transformed = A.ReplayCompose.replay(replay_data, image=night_t, mask=mask_t, mask_day=day_t)
                    
                    night_t = transformed['image']
                    mask_t = transformed['mask']
                    day_t = transformed['mask_day']

                    # 3-2. 저조도/노이즈 (프레임마다 다르게, 야간 이미지에만 적용)
                    night_t = self.photometric_aug(image=night_t)['image']
                
                aug_night_list.append(night_t)
                aug_mask_list.append(mask_t)
                aug_day_list.append(day_t)
            
            # --- 4. 최종 변환 (Resize, Normalize, ToTensor) ---
            tensor_night_list, tensor_mask_list, tensor_day_list = [], [], []
            
            for t in range(self.clip_len):
                # 4-1. 야간 (Resize + Normalize + ToTensor)
                night_tensor = self.final_transform_night(image=aug_night_list[t])['image']
                
                # 4-2. 마스크 (Resize(Nearest) + ToTensor(manual))
                mask_resized_np = self.final_transform_mask(image=aug_mask_list[t])['image']
                mask_tensor = torch.from_numpy(mask_resized_np).float().unsqueeze(0) / 255.0
                
                # 4-3. 주간 (Resize + ToTensor(manual))
                day_resized_np = self.final_transform_day(image=aug_day_list[t])['image']
                day_tensor = torch.from_numpy(day_resized_np.transpose(2, 0, 1)).float() / 255.0
                
                # ▼▼▼ [버그 수정 완료] ▼▼▼
                tensor_night_list.append(night_tensor)
                tensor_mask_list.append(mask_tensor) # <-- tensor_mask_list 자신을 추가하던 버그 수정
                tensor_day_list.append(day_tensor)
                # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        except FileNotFoundError as e:
            print(f"Warning: File not found, skipping this clip. Details: {e}")
            return None 
        except Exception as e_getitem:
            path_info = image_paths[0] if image_paths else 'N/A'
            print(f"Warning: Error processing clip {idx} (path: {path_info}), skipping. Details: {e_getitem}")
            return None 

        # --- 5. 텐서 스택 ---
        if not tensor_night_list or not tensor_mask_list or not tensor_day_list:
             return None
             
        image_clip = torch.stack(tensor_night_list, dim=0)
        mask_clip = torch.stack(tensor_mask_list, dim=0)
        original_image_clip = torch.stack(tensor_day_list, dim=0)
        
        return image_clip, mask_clip, original_image_clip
    
    