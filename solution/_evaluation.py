
import numpy as np

import pytoolkit as tk


def predict_tta(model, X_t, d_t, mode='ss'):
    """TTAありな予測処理。"""
    assert mode in ('bin', 'ss')

    def _mirror(X, d):
        p1 = model.predict([X, d], verbose=0)
        p2 = model.predict([X[:, :, ::-1, :], d], verbose=0)
        if mode == 'bin':
            return [p1, p2]
        else:
            return [p1, p2[:, :, ::-1, :]]

    z_t = np.zeros_like(d_t)
    pred = np.mean(
        _mirror(X_t, z_t) +
        _mirror(X_t - 8, z_t) +
        _mirror(X_t + 8, z_t) +
        _mirror(X_t / 1.1, z_t) +
        _mirror(X_t * 1.1, z_t) +
        [], axis=0)
    return pred


def log_evaluation(y_val, pred_val, print_fn=None, search_th=False):
    """検証結果をログる。"""
    print_fn = print_fn or tk.log.get(__name__).info

    # 正解率とか
    tk.ml.print_classification_metrics(np.ravel(y_val), np.ravel(pred_val), print_fn=print_fn)

    # 閾値探索＆スコア表示
    if search_th:
        threshold_list = np.linspace(0.3, 0.7, 100)
        score_list = []
        for th in tk.tqdm(threshold_list, desc='threshold'):
            score = compute_score(np.int32(y_val > 0.5), np.int32(pred_val > th))
            score_list.append(score)
        best_index = np.argmax(score_list)
        print_fn('scores:')
        for th, score in zip(threshold_list[::10], score_list[::10]):
            print_fn(f'  threshold={th:.3f}: score={score:.3f}')
        threshold = threshold_list[best_index]
        score = score_list[best_index]
        print_fn(f'max score: {score:.3f} (threshold: {threshold:.3f})')
    else:
        threshold = 0.5
        score = compute_score(np.int32(y_val > 0.5), np.int32(pred_val > threshold))
        print_fn(f'score: {score:.3f} (threshold: {threshold:.3f})')

    # オレオレ指標
    print_metrics(np.int32(y_val > 0.5), np.int32(pred_val > threshold), print_fn=print_fn)

    return threshold


def get_score_fixed_threshold(y_val, pred_val, search_th=True):
    """適当スコア算出。"""
    if search_th:
        threshold_list = np.linspace(0.3, 0.7, 100)
        score_list = []
        for th in tk.tqdm(threshold_list, desc='threshold'):
            score = compute_score(np.int32(y_val > 0.5), np.int32(pred_val > th))
            score_list.append(score)
        best_index = np.argmax(score_list)
        threshold = threshold_list[best_index]
        score = score_list[best_index]
    else:
        threshold = 0.5
        score = compute_score(np.int32(y_val > 0.5), np.int32(pred_val > threshold))
    return score, threshold


def compute_score(y_true, y_pred):
    """適当スコア算出。"""
    obj = np.sum(y_true, axis=(1, 2, 3)) > 0
    empty = np.logical_not(obj)
    pred_empty = np.sum(y_pred, axis=(1, 2, 3)) == 0
    tn = np.logical_and(empty, pred_empty)

    inter = np.sum(np.logical_and(y_true, y_pred), axis=(1, 2, 3))
    union = np.sum(np.logical_or(y_true, y_pred), axis=(1, 2, 3))
    iou = inter / np.maximum(union, 1)

    prec_list = []
    for threshold in np.arange(0.5, 1.0, 0.05):
        pred_obj = iou > threshold
        match = np.logical_and(obj, pred_obj) + tn
        prec_list.append(np.sum(match) / len(y_true))
    return np.mean(prec_list)


def print_metrics(y_true, y_pred, print_fn):
    """オレオレ指標。"""
    obj = np.sum(y_true, axis=(1, 2, 3)) > 0
    empty = np.logical_not(obj)

    # 答えが空でないときのIoUの平均
    inter = np.sum(np.logical_and(y_true, y_pred), axis=(1, 2, 3))
    union = np.sum(np.logical_or(y_true, y_pred), axis=(1, 2, 3))
    iou = inter / np.maximum(union, 1)
    iou_mean = np.mean(iou[obj])
    print_fn(f'IoU mean:  {iou_mean:.3f}')

    # 答えが空の場合の正解率
    pred_empty = np.sum(y_pred, axis=(1, 2, 3)) == 0
    acc_empty = np.sum(np.logical_and(empty, pred_empty)) / np.sum(empty)
    print_fn(f'Acc empty: {acc_empty:.3f}')
