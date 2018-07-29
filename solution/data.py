import pathlib

import cv2
import numpy as np
import pandas as pd

import pytoolkit as tk

TRAIN_IMAGE_DIR = pathlib.Path('../input/train/images')
TRAIN_IMAGE_DIR = pathlib.Path('../input/train/images')
TRAIN_MASK_DIR = pathlib.Path('../input/train/masks')
TEST_IMAGE_DIR = pathlib.Path('../input/test/images')
TRAIN_PATH = pathlib.Path('../input/train.csv')
TEST_PATH = pathlib.Path('../input/sample_submission.csv')
DEPTHS_PATH = pathlib.Path('../input/depths.csv')


def load_train_data():
    id_list = pd.read_csv(TRAIN_PATH)['id'].values
    X = np.array([TRAIN_IMAGE_DIR / (id_ + '.png') for id_ in id_list])
    d = _load_depths(id_list)
    y = np.array([TRAIN_MASK_DIR / (id_ + '.png') for id_ in id_list])
    return X, d, y


def load_test_data():
    id_list = pd.read_csv(TEST_PATH)['id'].values
    X = np.array([TEST_IMAGE_DIR / (id_ + '.png') for id_ in id_list])
    d = _load_depths(id_list)
    return X, d


def _load_depths(id_list):
    df_depths = pd.read_csv(DEPTHS_PATH, index_col='id')
    depths = df_depths['z'].to_dict()
    d = np.array([depths[id_] for id_ in id_list], dtype=np.float32)
    d -= df_depths['z'].mean()
    d /= df_depths['z'].std()
    return d


def load_mask(y):
    y = np.array([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255 for p in tk.tqdm(y)])
    y = np.expand_dims(y, axis=-1)
    return y


def save_submission(save_path, pred_test, threshold):
    id_list = pd.read_csv(TEST_PATH)['id'].values
    pred_dict = {id_: _encode_rl(pred_test[i, :, :, 0] >= threshold) for i, id_ in enumerate(tk.tqdm(id_list))}
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
