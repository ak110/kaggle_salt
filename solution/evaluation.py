
import numpy as np

import pytoolkit as tk


def log_evaluation(y_val, pred_val, print_fn=None):
    """検証結果をログる。"""
    print_fn = print_fn or tk.log.get(__name__).info

    # 正解率とか
    tk.ml.print_classification_metrics(np.ravel(y_val), np.ravel(pred_val), print_fn=print_fn)

    # 閾値の最適化
    threshold_list = np.linspace(0.25, 0.75, 20)
    score_list = []
    for th in threshold_list:
        score = compute_iou_metric(np.int32(y_val > 0.5), np.int32(pred_val > th))
        score_list.append(score)
    best_index = np.argmax(score_list)
    print_fn(f'max score: {score_list[best_index]:.3f} (threshold: {threshold_list[best_index]:.3f})')
    print_fn('scores:')
    for th, score in zip(threshold_list, score_list):
        print_fn(f'  threshold={th:.3f}: score={score:.3f}')


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
