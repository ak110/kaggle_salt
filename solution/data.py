import pathlib

import cv2
import numpy as np
import pandas as pd

import pytoolkit as tk

TRAIN_IMAGE_DIR = pathlib.Path('../input/train/images')
TRAIN_IMAGE_DIR = pathlib.Path('../input/train/images')
TRAIN_MASK_DIR = pathlib.Path('../input/train/masks')
TEST_IMAGE_DIR = pathlib.Path('../input/test/images')
DEPTHS_PATH = pathlib.Path('../input/depths.csv')

X_mean = 120.34612148318793
X_std = 41.06966522016213


def load_train_data():
    names = [p.name for p in TRAIN_IMAGE_DIR.iterdir()]
    prefixes = [p.stem for p in TRAIN_IMAGE_DIR.iterdir()]

    X = [cv2.imread(str(TRAIN_IMAGE_DIR / p), cv2.IMREAD_GRAYSCALE) for p in tk.tqdm(names)]
    X = (np.array(X, dtype=np.float32) - X_mean) / X_std
    X = np.expand_dims(X, axis=3)
    d = _load_depths(prefixes)
    X = np.concatenate([X, d], axis=-1)

    y = [cv2.imread(str(TRAIN_MASK_DIR / p), cv2.IMREAD_GRAYSCALE) for p in tk.tqdm(names)]
    y = np.array(y, dtype=np.float32) / 255  # 0-1
    y = np.expand_dims(y, axis=3)
    return X, y, prefixes


def load_test_data():
    names = [p.name for p in TEST_IMAGE_DIR.iterdir()]
    prefixes = [p.stem for p in TEST_IMAGE_DIR.iterdir()]

    X = [cv2.imread(str(TEST_IMAGE_DIR / p), cv2.IMREAD_GRAYSCALE) for p in tk.tqdm(names)]
    X = (np.array(X, dtype=np.float32) - X_mean) / X_std
    X = np.expand_dims(X, axis=3)
    d = _load_depths(prefixes)
    X = np.concatenate([X, d], axis=-1)
    return X, prefixes


def _load_depths(prefixes):
    df_depths = pd.read_csv(DEPTHS_PATH)
    depths = {row['id']: row['z'] for _, row in df_depths.iterrows()}
    d = np.array([depths[p] for p in tk.tqdm(prefixes)], dtype=np.float32)
    d -= df_depths['z'].mean()
    d /= df_depths['z'].std()
    d = np.repeat(d, 101 * 101).reshape(len(prefixes), 101, 101, 1)
    return d


def save_submission(save_path, pred, prefixes, threshold):
    pred_dict = {prefix: _encode_rl(pred[i, :, :, 0] >= threshold) for i, prefix in enumerate(tk.tqdm(prefixes))}
    df = pd.DataFrame.from_dict(pred_dict, orient='index')
    df.index.names = ['id']
    df.columns = ['rle_mask']
    df.to_csv(str(save_path))


def _encode_rl(img):
    """ランレングス。"""
    img = img.reshape(img.shape[0] * img.shape[1], order='F')
    runs = []  # list of run lengths
    r = 0  # the current run length
    pos = 1  # count starts from 1 per WK
    for c in img:
        if c == 0:
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    z = ''
    for rr in runs:
        z += f'{rr[0]} {rr[1]} '
    return z[:-1]


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


def _check():
    """動作確認用コード。"""
    tk.better_exceptions()

    TRAIN_CSV_PATH = pathlib.Path('../input/train.csv')
    df = pd.read_csv(TRAIN_CSV_PATH)
    df.fillna('', inplace=True)

    _, y, prefixes = load_train_data()
    for i, prefix in enumerate(tk.tqdm(prefixes)):
        rle_true = df.loc[df['id'] == prefix]
        rle_check = _encode_rl(np.round(y[i, :, :, 0]))
        assert rle_true['rle_mask'].values[0] == rle_check, f'error: {prefix}'


if __name__ == '__main__':
    _check()
