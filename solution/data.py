import pathlib

import cv2
import numpy as np
import pandas as pd

import pytoolkit as tk

TRAIN_IMAGE_DIR = pathlib.Path('../input/train/images/')
TRAIN_IMAGE_DIR = pathlib.Path('../input/train/images/')
TRAIN_MASK_DIR = pathlib.Path('../input/train/masks/')
TEST_IMAGE_DIR = pathlib.Path('../input/test/images/')
DEPTHS_PATH = pathlib.Path('../input/depths.csv')


def load_train_data():
    train_names = [p.name for p in TRAIN_IMAGE_DIR.iterdir()]
    train_prefixes = [p.stem for p in TRAIN_IMAGE_DIR.iterdir()]

    X = [cv2.imread(str(TRAIN_IMAGE_DIR / p), cv2.IMREAD_GRAYSCALE) for p in tk.tqdm(train_names)]
    X = np.array(X, dtype=np.float32) / 255
    X = np.expand_dims(X, axis=3)
    d = _load_depths(train_prefixes)
    X = np.concatenate([X, d], axis=-1)

    y = [cv2.imread(str(TRAIN_MASK_DIR / p), cv2.IMREAD_GRAYSCALE) for p in tk.tqdm(train_names)]
    y = np.array(y, dtype=np.float32) / 255
    y = np.expand_dims(y, axis=3)
    return X, y, train_prefixes


def load_test_data():
    test_names = [p.name for p in TRAIN_IMAGE_DIR.iterdir()]
    test_prefixes = [p.stem for p in TRAIN_IMAGE_DIR.iterdir()]

    X = [cv2.imread(str(TRAIN_IMAGE_DIR / p), cv2.IMREAD_GRAYSCALE) for p in tk.tqdm(test_names)]
    X = np.array(X, dtype=np.float32) / 255
    X = np.expand_dims(X, axis=3)
    d = _load_depths(test_prefixes)
    X = np.concatenate([X, d], axis=-1)
    return X, test_prefixes


def _load_depths(prefixes):
    df_depths = pd.read_csv(DEPTHS_PATH)
    depths = {row['id']: row['z'] for _, row in df_depths.iterrows()}
    d = np.array([depths[p] for p in tk.tqdm(prefixes)], dtype=np.float32)
    d -= df_depths['z'].mean()
    d /= df_depths['z'].std()
    d = np.repeat(d, 101 * 101).reshape(len(prefixes), 101, 101, 1)
    return d


def save_submission(pred, prefixes):
    pred_dict = {prefix: _encode_rl(np.round(pred[i, :, :, 0])) for i, prefix in enumerate(tk.tqdm(prefixes))}
    df = pd.DataFrame.from_dict(pred_dict, orient='index')
    df.index.names = ['id']
    df.columns = ['rle_mask']
    df.to_csv('submission.csv')


def _encode_rl(img):
    """ランレングス。"""
    b = img.reshape(img.shape[0] * img.shape[1], order='F')
    runs = []  # list of run lengths
    r = 0  # the current run length
    pos = 1  # count starts from 1 per WK
    for c in b:
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
