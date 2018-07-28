
import numpy as np

import pytoolkit as tk


def log_evaluation(y_val, pred_val, print_fn=None):
    """検証結果をログる。"""
    print_fn = print_fn or tk.log.get(__name__).info

    # 正解率とか
    tk.ml.print_classification_metrics(np.ravel(y_val), np.ravel(pred_val), print_fn=print_fn)

    # 閾値の最適化
    threshold_list = np.linspace(0.1, 0.9, 20)
    score1_list = []
    score2_list = []
    for th in tk.tqdm(threshold_list):
        score1 = compute_iou_metric(np.int32(y_val > 0.5), np.int32(pred_val > th))
        score2 = compute_score(np.int32(y_val > 0.5), np.int32(pred_val > th))
        score1_list.append(score1)
        score2_list.append(score2)
    best_index1 = np.argmax(score1_list)
    best_index2 = np.argmax(score2_list)
    print_fn(f'max score1: {score1_list[best_index1]:.3f} (threshold: {threshold_list[best_index1]:.3f})')
    print_fn(f'max score2: {score2_list[best_index2]:.3f} (threshold: {threshold_list[best_index2]:.3f})')
    print_fn('scores:')
    for th, score1, score2 in zip(threshold_list, score1_list, score2_list):
        print_fn(f'  threshold={th:.3f}: score1={score1:.3f} score2={score2:.3f}')


def compute_iou_metric(y_true, y_pred):
    return np.mean([_iou_metric_single(t, p) for t, p in zip(y_true, y_pred)])


def _iou_metric_single(y_true, y_pred, print_table=False):
    # src: https://www.kaggle.com/aglotero/another-iou-metric
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(y_true.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(y_true, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def compute_score(y_true, y_pred):
    """適当スコア算出。"""
    obj = np.sum(y_true, axis=(1, 2, 3)) > 0
    bg = np.logical_not(obj)
    empty = np.sum(y_pred, axis=(1, 2, 3)) == 0
    prec_list = []
    for threshold in np.arange(0.5, 1.0, 0.05):
        inter = np.sum(np.logical_and(y_true, y_pred), axis=(1, 2, 3))
        union = np.sum(np.logical_or(y_true, y_pred), axis=(1, 2, 3))
        iou = inter / np.maximum(union, 1)
        detected = iou > threshold
        match = np.logical_and(obj, detected) + np.logical_and(bg, empty)
        prec_list.append(np.sum(match) / len(y_true))
    return np.mean(prec_list)
