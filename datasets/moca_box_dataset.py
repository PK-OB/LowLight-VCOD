import os
import cv2
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MoCABoxDataset(Dataset):
    def __init__(self, synthetic_data_root, annotation_file, clip_len=8, resolution=(224, 224)):
        self.synthetic_data_root = synthetic_data_root
        self.clip_len = clip_len
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # ▼▼▼ 수정된 부분 ▼▼▼
        # 실제 파일의 열 개수인 9개에 맞춰, 사용하지 않는 열의 이름을 추가해줍니다.
        column_names = [
            '#filename', 
            'file_size', 
            'file_attributes', 
            'region_count', 
            'region_id',
            'spatial_coordinates', 
            'metadata',
            'unnamed_7', # 8번째 열
            'unnamed_8'  # 9번째 열
        ]

        # skiprows=10으로 헤더를 포함한 모든 주석을 건너뛰고, header=None으로 헤더가 없다고 알립니다.
        df = pd.read_csv(annotation_file, skiprows=10, header=None)

        # 우리가 직접 정의한 열 이름을 데이터프레임에 부여합니다.
        df.columns = column_names[:len(df.columns)]
        
        self.filename_col = '#filename' 
        
        df['video_name'] = df[self.filename_col].apply(lambda x: os.path.dirname(x).strip('/'))
        self.annotations_df = df
        
        # 비디오 클립 생성
        self.clips = []
        video_groups = self.annotations_df.groupby('video_name')
        for _, group in video_groups:
            sorted_group = group.sort_values(by=self.filename_col)
            if len(sorted_group) >= clip_len:
                for i in range(0, len(sorted_group) - clip_len + 1, clip_len):
                    clip_info = sorted_group.iloc[i : i + clip_len]
                    self.clips.append(clip_info)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        image_clip = []
        gt_boxes_clip = []

        for _, row in clip_info.iterrows():
            base_filename = os.path.basename(row[self.filename_col])
            img_path = os.path.join(self.synthetic_data_root, row['video_name'], base_filename)

            try:
                image = Image.open(img_path).convert("RGB")
                original_w, original_h = image.size
            except FileNotFoundError:
                return None

            boxes = []
            try:
                if 'spatial_coordinates' in row and isinstance(row['spatial_coordinates'], str) and row['spatial_coordinates'] != '[]':
                    coords_str = row['spatial_coordinates'].strip('[]')
                    coords = [float(c) for c in coords_str.split(',')]
                    shape_id = int(coords[0])
                    if shape_id == 2 and len(coords) >= 5:
                        x, y, w, h = coords[1], coords[2], coords[3], coords[4]
                        x_new = x * self.resolution[0] / original_w
                        y_new = y * self.resolution[1] / original_h
                        w_new = w * self.resolution[0] / original_w
                        h_new = h * self.resolution[1] / original_h
                        boxes.append([x_new, y_new, x_new + w_new, y_new + h_new])
            except (ValueError, IndexError):
                pass
            
            image_clip.append(self.transform(image))
            gt_boxes_clip.append(torch.tensor(boxes, dtype=torch.float32))

        if not image_clip or len(image_clip) < self.clip_len:
            return None

        return torch.stack(image_clip, dim=0), gt_boxes_clip