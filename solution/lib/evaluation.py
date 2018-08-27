
import numpy as np

import pytoolkit as tk


def log_evaluation(y_val, pred_val, print_fn=None):
    """検証結果をログる。"""
    print_fn = print_fn or tk.log.get(__name__).info

    # 正解率とか
    tk.ml.print_classification_metrics(np.ravel(y_val), np.ravel(pred_val), print_fn=print_fn)

    # 閾値の最適化
    threshold_list = np.linspace(0.1, 0.9, 20)
    score_list = []
    for th in tk.tqdm(threshold_list):
        score = compute_score(np.int32(y_val > 0.5), np.int32(pred_val > th))
        score_list.append(score)
    best_index = np.argmax(score_list)
    print_fn(f'max score: {score_list[best_index]:.3f} (threshold: {threshold_list[best_index]:.3f})')
    print_fn('scores:')
    for th, score in zip(threshold_list, score_list):
        print_fn(f'  threshold={th:.3f}: score={score:.3f}')

    return threshold_list[best_index]


def compute_score(y_true, y_pred):
    """適当スコア算出。"""
    obj = np.sum(y_true, axis=(1, 2, 3)) > 0
    bg = np.logical_not(obj)
    pred_bg = np.sum(y_pred, axis=(1, 2, 3)) == 0
    prec_list = []
    for threshold in np.arange(0.5, 1.0, 0.05):
        inter = np.sum(np.logical_and(y_true, y_pred), axis=(1, 2, 3))
        union = np.sum(np.logical_or(y_true, y_pred), axis=(1, 2, 3))
        iou = inter / np.maximum(union, 1)
        pred_obj = iou > threshold
        match = np.logical_and(obj, pred_obj) + np.logical_and(bg, pred_bg)
        prec_list.append(np.sum(match) / len(y_true))
    return np.mean(prec_list)
