import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

from config import cfg
from models.main_model import JED_VCOD_Fauna_Simplified
from datasets.moca_box_dataset import MoCABoxDataset

def mask_to_bbox(mask, threshold=0.5):
    mask_np = (mask.squeeze().cpu().numpy() > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    return [x, y, x + w, y + h]

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def main():
    common_cfg = cfg.common
    eval_cfg = cfg.evaluate_box

    device = torch.device(f"cuda:{common_cfg['gpu_ids'].split(',')[0]}")
    model = JED_VCOD_Fauna_Simplified()
    
    state_dict = torch.load(eval_cfg['checkpoint_path'], map_location=device)
    if all(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    dataset = MoCABoxDataset(
        synthetic_data_root=cfg.train['data_root'],
        annotation_file=eval_cfg['annotation_file'],
        clip_len=common_cfg['clip_len']
    )
    
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return None, None
        return torch.utils.data.dataloader.default_collate(batch)

    loader = DataLoader(dataset, batch_size=eval_cfg['batch_size'], shuffle=False, num_workers=common_cfg['num_workers'], collate_fn=collate_fn)
    
    tp, fp, fn = 0, 0, 0

    with torch.no_grad():
        for video_clip, gt_boxes_clip in tqdm(loader, desc="Evaluating BBoxes"):
            if video_clip is None: continue

            video_clip = video_clip.to(device)
            predicted_logits = model(video_clip)
            predicted_masks = torch.sigmoid(predicted_logits)

            for i in range(predicted_masks.shape[0]):
                for j in range(predicted_masks.shape[1]):
                    pred_box = mask_to_bbox(predicted_masks[i, j])
                    gt_boxes = gt_boxes_clip[j][i]

                    has_gt = gt_boxes.nelement() > 0
                    has_pred = pred_box is not None

                    if not has_gt:
                        if has_pred: fp += 1
                        continue
                    if not has_pred:
                        fn += len(gt_boxes)
                        continue

                    best_iou = 0
                    for gt_box in gt_boxes:
                        iou = calculate_iou(pred_box, gt_box.tolist())
                        if iou > best_iou:
                            best_iou = iou

                    if best_iou >= eval_cfg['iou_threshold']:
                        tp += 1
                        fn += len(gt_boxes) - 1
                    else:
                        fp += 1
                        fn += len(gt_boxes)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- Bounding Box Evaluation Results ---")
    print(f"IoU Threshold: {eval_cfg['iou_threshold']}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")
    print("---------------------------------------")

if __name__ == '__main__':
    main()