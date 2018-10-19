#!/usr/bin/env python3

import numpy as np

import sklearn.ensemble

import pytoolkit as tk

MIN_THRESHOLD = 0.3
MAX_THRESHOLD = 0.7


def create_data(y, pred):
    """Create adaptive threshold data."""
    threshold_X = create_input_data(pred)

    mask_neg = y.max(axis=(1, 2, 3)) == 0
    mask_pos = np.logical_not(mask_neg)
    y_pos = y > 0.5

    threshold_y = np.empty((len(y),))

    for i in tk.tqdm(np.where(mask_pos)[0], desc='ath'):
        threshold_list = np.linspace(MIN_THRESHOLD, MAX_THRESHOLD, 1000)
        iou_list = []
        for th in threshold_list:
            pred_pos = pred[i] > th
            inter = np.logical_and(pred_pos, y_pos[i])
            union = np.logical_or(pred_pos, y_pos[i])
            iou = np.sum(inter) / max(np.sum(union), 1)
            iou_list.append(iou)
        best_index = np.argmax(iou_list)
        threshold_y[i] = threshold_list[best_index]

    th_neg = (pred[mask_neg].max(axis=(1, 2, 3)) + 1) / 2
    threshold_y[mask_neg] = np.minimum(np.maximum(th_neg, MIN_THRESHOLD), MAX_THRESHOLD)

    threshold_y = np.array(threshold_y)

    return threshold_X, threshold_y


def create_input_data(pred):
    """閾値を予測する用の入力。"""
    threshold_X = np.swapaxes([
        np.max(pred, axis=(1, 2, 3)),
        np.mean(pred, axis=(1, 2, 3)),
        np.percentile(pred, 10, axis=(1, 2, 3)),
        np.percentile(pred, 20, axis=(1, 2, 3)),
        np.percentile(pred, 30, axis=(1, 2, 3)),
        np.percentile(pred, 40, axis=(1, 2, 3)),
        np.percentile(pred, 50, axis=(1, 2, 3)),
        np.percentile(pred, 60, axis=(1, 2, 3)),
        np.percentile(pred, 70, axis=(1, 2, 3)),
        np.percentile(pred, 80, axis=(1, 2, 3)),
        np.percentile(pred, 90, axis=(1, 2, 3)),
    ], 0, 1)
    return threshold_X


def create_estimator(threshold_X, threshold_y):
    """学習。"""
    logger = tk.log.get(__name__)
    estimator = sklearn.ensemble.RandomForestRegressor(n_estimators=300, n_jobs=-1)
    estimator.fit(threshold_X, threshold_y)
    logger.info(f'ath: score={estimator.score(threshold_X, threshold_y)}')
    return estimator
