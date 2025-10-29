# utils/py_sod_metrics.py

import os
from collections import defaultdict

import numpy as np
import cv2
from scipy.ndimage import convolve, distance_transform_edt as bwdist
from skimage.morphology import disk, ball

"""
A Python implementation of the metrics for Salient Object Detection.
"""


def _prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    gt = gt > 0
    pred = pred / 255
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt


class MAE:
    def __init__(self):
        self.prediction = []
        self.ground_truth = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred, gt)

        self.prediction.append(pred)
        self.ground_truth.append(gt)

    def get_results(self) -> dict:
        prediction = np.array(self.prediction)
        ground_truth = np.array(self.ground_truth)
        abs_error = np.abs(prediction - ground_truth)
        overall_mae = np.mean(abs_error)
        return {"MAE": overall_mae}


class Fmeasure:
    def __init__(self, beta: float = 0.3):
        self.beta = beta
        self.precisions = []
        self.recalls = []
        self.adaptive_fms = []
        self.mean_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        adaptive_fm = self.cal_adaptive_fm(pred, gt)
        self.adaptive_fms.append(adaptive_fm)

        precisions, recalls = self.cal_pr(pred, gt)
        self.precisions.append(precisions)
        self.recalls.append(recalls)

    def cal_adaptive_fm(self, pred, gt):
        # According to the original paper, the mean of the saliency map is used as the threshold.
        # However, the performance of this method is not stable, so we use the F-measure of the mean of the GT.
        # The implementation of the F-measure of the mean of the GT is referenced from
        # https://github.com/DengPingFan/CODToolbox/blob/master/Evaluation/eval-dataset(Image-level)/MAIN_Metric.m#L199
        threshold = 2 * pred.mean()
        if threshold > 1:
            threshold = 1
        binary_pred = pred >= threshold
        area_intersection = np.sum(binary_pred[gt])
        area_pred = np.sum(binary_pred)
        area_gt = np.sum(gt)
        precision = area_intersection / (area_pred + 1e-6)
        recall = area_intersection / (area_gt + 1e-6)
        adaptive_fm = (1 + self.beta) * precision * recall / (self.beta * precision + recall + 1e-6)
        return adaptive_fm

    def cal_pr(self, pred, gt):
        # calculate precision and recall at 255 different thresholds
        pred = (pred * 255).astype(np.uint8)
        thresholds = np.linspace(0, 255, 256)
        precisions = np.zeros(256)
        recalls = np.zeros(256)
        for i, threshold in enumerate(thresholds):
            binary_pred = pred >= threshold
            area_intersection = np.sum(binary_pred[gt])
            area_pred = np.sum(binary_pred)
            area_gt = np.sum(gt)

            precision = area_intersection / (area_pred + 1e-6)
            recall = area_intersection / (area_gt + 1e-6)

            precisions[i] = precision
            recalls[i] = recall
        return precisions, recalls

    def get_results(self) -> dict:
        adaptive_fm = np.mean(np.array(self.adaptive_fms, dtype=np.float64))
        precisions = np.mean(np.array(self.precisions, dtype=np.float64), axis=0)
        recalls = np.mean(np.array(self.recalls, dtype=np.float64), axis=0)
        mean_fm = (1 + self.beta) * precisions * recalls / (self.beta * precisions + recalls + 1e-6)
        return {
            "adaptive_F-beta": adaptive_fm,
            "mean_F-beta": mean_fm.max(),
            "precisions": precisions,
            "recalls": recalls,
        }


class Smeasure:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.scores = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        y = np.mean(gt)
        if y == 0:
            score = 1 - np.mean(pred)
        elif y == 1:
            score = np.mean(pred)
        else:
            score = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
        self.scores.append(score)

    def object(self, pred, gt):
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)

        u = np.mean(gt)
        return u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, 1 - gt)

    def s_object(self, pred, gt):
        x = np.mean(pred[gt])
        sigma_x = np.std(pred[gt])
        return 2 * x / (x ** 2 + 1 + sigma_x + 1e-6)

    def region(self, pred, gt):
        x, y = self.centroid(gt)
        w, h = gt.shape
        if x == -1 and y == -1:
            return 0
        
        # pred_block: a block of the prediction, the center of the block is the center of the gt
        # gt_block: a block of the gt, the center of the block is the center of the gt
        # the size of the block is (2 * x, 2 * y)
        x, y = int(x), int(y)
        x_min, x_max = max(0, 2 * x - w), min(2 * x, w)
        y_min, y_max = max(0, 2 * y - h), min(2 * y, h)

        pred_block = pred[y_min:y_max, x_min:x_max]
        gt_block = gt[y_min:y_max, x_min:x_max]
        
        # in case the gt is a line
        if gt_block.sum() == 0:
            return 0

        # number of foreground pixels in the gt_block
        block_fg = gt_block.sum()
        # number of background pixels in the gt_block
        block_bg = gt_block.size - block_fg
        
        # calculate the mean of the pred_block for the foreground and background
        mu_fg = np.mean(pred_block[gt_block])
        mu_bg = np.mean(pred_block[~gt_block])

        # calculate the score
        score = 0
        if block_fg > 0:
            score += block_fg / gt_block.size * (1 if mu_fg > mu_bg else -1) * (mu_fg - mu_bg) ** 2
        if block_bg > 0:
            score += block_bg / gt_block.size * (1 if mu_fg < mu_bg else -1) * (mu_fg - mu_bg) ** 2
        
        return score

    def centroid(self, gt):
        if np.sum(gt) == 0:
            return -1, -1
        
        # calculate the center of the gt
        x = np.mean(np.where(gt)[1])
        y = np.mean(np.where(gt)[0])
        return x, y
        

    def get_results(self) -> dict:
        score = np.mean(np.array(self.scores, dtype=np.float64))
        return {"S-measure": score}


class Emeasure:
    def __init__(self,):
        self.scores = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        # pred: 2d numpy array of shape (h, w), type float64
        # gt: 2d numpy array of shape (h, w), type bool

        threshold = 2 * pred.mean()
        if threshold > 1:
            threshold = 1

        pred_binary = pred >= threshold
        
        # enhanced-alignment measure
        # Reference: https://github.com/DengPingFan/E-measure/blob/master/E-measure.m
        
        # get the enhanced alignment matrix
        w, h = gt.shape
        gt_fg = gt.sum()
        gt_bg = w * h - gt_fg

        align_matrix = np.zeros((w, h))
        if gt_fg == 0:
            align_matrix[pred_binary] = 1 - pred_binary[pred_binary]
        elif gt_bg == 0:
            align_matrix[pred_binary] = pred_binary[pred_binary]
        else:
            pred_fg = pred_binary.sum()
            pred_bg = w * h - pred_fg

            align_fg = (pred_binary - pred.mean())**2
            align_bg = (1 - pred_binary - (1-pred).mean())**2

            align_matrix[gt] = align_fg[gt]
            align_matrix[~gt] = align_bg[~gt]
        
        score = np.sum(align_matrix) / (w * h -1 + 1e-6)
        self.scores.append(score)


    def get_results(self) -> dict:
        score = np.mean(np.array(self.scores, dtype=np.float64))
        return {"E-measure": score}


class WeightedFmeasure:
    def __init__(self, beta: float = 0.3):
        self.beta = beta
        self.weighted_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        if np.sum(gt) == 0:
            self.weighted_fms.append(1 - np.mean(pred))
            return
        
        # Reference: https://github.com/DengPingFan/CODToolbox/blob/master/Evaluation/eval-dataset(Image-level)/MAIN_Metric.m#L274
        # a weighted F-measure can be calculated as follows:
        # Fw = (1 + β2) · (Prec · Recall) / (β2 · Prec + Recall)
        # where the precision and recall are weighted by the pixel’s importance.

        # center-priori
        # build a disk-like shape gt to calculate the center-priori
        wc = self.mat_weight(gt)

        # calculate the precision and recall
        precisions, recalls = self.cal_pr(pred, gt, wc)

        weighted_fm = (1 + self.beta) * precisions * recalls / (self.beta * precisions + recalls + 1e-6)
        self.weighted_fms.append(weighted_fm)


    def cal_pr(self, pred, gt, wc):
        pred = (pred * 255).astype(np.uint8)
        thresholds = np.linspace(0, 255, 256)
        precisions = np.zeros(256)
        recalls = np.zeros(256)
        for i, threshold in enumerate(thresholds):
            binary_pred = pred >= threshold
            
            # number of true positive pixels
            # according to https://github.com/DengPingFan/CODToolbox/blob/master/Evaluation/eval-dataset(Image-level)/script/Onekey-Evaluate-function/Evaluate_COD_Metrics.m#L159
            # the true positive pixels are the intersection of the binary prediction and the ground truth
            tp = binary_pred * gt
            
            # according to https://github.com/DengPingFan/CODToolbox/blob/master/Evaluation/eval-dataset(Image-level)/script/Onekey-Evaluate-function/Evaluate_COD_Metrics.m#L164
            # the precision is the sum of the weighted true positive pixels divided by the sum of the binary prediction
            precision = np.sum(wc[tp]) / (np.sum(binary_pred) + 1e-6)
            recall = np.sum(wc[tp]) / (np.sum(wc[gt]) + 1e-6)
            precisions[i] = precision
            recalls[i] = recall
        return precisions, recalls


    def mat_weight(self, gt, ksize=5):
        # build a disk-like shape gt to calculate the center-priori
        # the shape of the disk is determined by the ksize
        # the center of the disk is the center of the gt
        # the value of the disk is determined by the distance from the center
        # the closer to the center, the larger the value
        
        # in case the gt is a line
        if np.sum(gt) == 0:
            return np.zeros_like(gt)
        
        # distance transform
        # the distance transform is the distance from the pixel to the nearest background pixel
        # the distance is calculated by the Euclidean distance
        # the background is the pixels with value 0
        # the foreground is the pixels with value 1
        d = bwdist(gt==0)
        d[d > ksize] = ksize
        d = d / ksize
        return d
        
    def get_results(self) -> dict:
        weighted_fms = np.mean(np.array(self.weighted_fms, dtype=np.float64), axis=0)
        return {"weighted_F-beta": weighted_fms.max()}


class SODMetrics:
    def __init__(self, ):
        self.metrics = [MAE(), Fmeasure(), Smeasure(), Emeasure(), WeightedFmeasure()]

    def step(self, pred: np.ndarray, gt: np.ndarray):
        assert pred.shape == gt.shape, f"pred.shape: {pred.shape}, gt.shape: {gt.shape}"
        assert pred.dtype == np.uint8, f"pred.dtype: {pred.dtype}"
        assert gt.dtype == np.uint8, f"gt.dtype: {gt.dtype}"
        
        for metric in self.metrics:
            metric.step(pred, gt)
    
    def get_results(self) -> dict:
        results = {}
        for metric in self.metrics:
            results.update(metric.get_results())
        
        # rename keys
        results["Sm"] = results.pop("S-measure")
        results["wFm"] = results.pop("weighted_F-beta")
        results["adpFm"] = results.pop("adaptive_F-beta")
        results["Em"] = results.pop("E-measure")
        results["F-beta"] = results.pop("mean_F-beta")

        return results