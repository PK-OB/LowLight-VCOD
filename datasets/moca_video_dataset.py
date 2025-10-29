import os
import cv2
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MoCAVideoDataset(Dataset):
    def __init__(self, synthetic_data_root, annotation_file, clip_len=8, resolution=(224, 224)):
        self.synthetic_data_root = synthetic_data_root
        self.clip_len = clip_len
        self.resolution = resolution
        
        # ▼▼▼ 수정된 부분 ▼▼▼
        # 누락되었던 transform 정의 코드를 다시 추가합니다.
        self.transform = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.ToTensor(),
        ])
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
        column_names = [
            'metadata_id', 'file_list', 'flags', 'temporal_coordinates', 'spatial_coordinates',
            'region_shape_attributes', 'metadata', 'unnamed_7', 'unnamed_8'
        ]
        df = pd.read_csv(annotation_file, skiprows=10, header=None)
        df.columns = column_names[:len(df.columns)]
        
        self.filename_col = 'file_list' 
        
        df['video_name'] = df[self.filename_col].apply(lambda x: os.path.dirname(x).strip('/'))
        self.annotations_df = df

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
        # ... (__getitem__ 함수는 이전과 동일) ...
        clip_info = self.clips[idx]
        image_clip = []
        mask_clip = []

        for _, row in clip_info.iterrows():
            base_filename = os.path.basename(row[self.filename_col])
            img_path = os.path.join(self.synthetic_data_root, row['video_name'], base_filename)

            try:
                image = Image.open(img_path).convert("RGB")
                width, height = image.size
            except FileNotFoundError:
                return torch.zeros(self.clip_len, 3, *self.resolution), torch.zeros(self.clip_len, 1, *self.resolution)

            mask = np.zeros((height, width), dtype=np.uint8)
            try:
                if 'spatial_coordinates' in row and isinstance(row['spatial_coordinates'], str) and row['spatial_coordinates'] != '[]':
                    coords_str = row['spatial_coordinates'].strip('[]')
                    coords = [int(float(c)) for c in coords_str.split(',')]
                    shape_id = coords[0]
                    if shape_id == 2 and len(coords) >= 5:
                        x, y, w, h = coords[1], coords[2], coords[3], coords[4]
                        top_left = (x, y)
                        bottom_right = (x + w, y + h)
                        cv2.rectangle(mask, top_left, bottom_right, 255, -1)
            except (ValueError, IndexError):
                pass

            image_clip.append(self.transform(image))
            mask_clip.append(self.mask_transform(Image.fromarray(mask)))
        
        if not image_clip or len(image_clip) < self.clip_len:
             return torch.zeros(self.clip_len, 3, *self.resolution), torch.zeros(self.clip_len, 1, *self.resolution)

        return torch.stack(image_clip, dim=0), torch.stack(mask_clip, dim=0)