import pathlib

import cv2
import numpy as np
import pandas as pd

import pytoolkit as tk

TRAIN_IMAGE_DIR = pathlib.Path('../input/train/images')
TRAIN_MASK_DIR = pathlib.Path('../input/train/masks')
TEST_IMAGE_DIR = pathlib.Path('../input/test/images')
TRAIN_PATH = pathlib.Path('../input/train.csv')
TEST_PATH = pathlib.Path('../input/sample_submission.csv')
DEPTHS_PATH = pathlib.Path('../input/depths.csv')
CACHE_DIR = pathlib.Path('cache')


@tk.cache.memorize(CACHE_DIR, compress=3)
def load_train_data():
    """訓練データ。Xは0～255、yは0～1。dは平均0、標準偏差1。"""
    id_list = pd.read_csv(TRAIN_PATH)['id'].values
    X = _load_image([TRAIN_IMAGE_DIR / f'{id_}.png' for id_ in id_list])
    d = _load_depths(id_list)
    y = _load_image([TRAIN_MASK_DIR / f'{id_}.png' for id_ in id_list]) / 255
    return X, d, y


@tk.cache.memorize(CACHE_DIR, compress=3)
def load_test_data():
    """訓練データ。Xは0～255。dは平均0、標準偏差1。"""
    id_list = pd.read_csv(TEST_PATH)['id'].values
    X = _load_image([TEST_IMAGE_DIR / f'{id_}.png' for id_ in id_list])
    d = _load_depths(id_list)
    return X, d


def _load_depths(id_list):
    df_depths = pd.read_csv(DEPTHS_PATH, index_col='id')
    depths = df_depths['z'].to_dict()
    d = np.array([depths[id_] for id_ in id_list], dtype=np.float32)
    d -= df_depths['z'].mean()
    d /= df_depths['z'].std()
    return d


def _load_image(X):
    X = np.array([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) for p in tk.tqdm(X, desc='load')])
    X = np.expand_dims(X, axis=-1)
    return X


def save_submission(save_path, pred):
    """投稿用ファイルを出力。ついでにちょっとだけ結果を分析。"""
    # 投稿用ファイルを出力
    id_list = pd.read_csv(TEST_PATH)['id'].values
    pred_dict = {id_: _encode_rl(pred[i]) for i, id_ in enumerate(tk.tqdm(id_list, desc='encode_rl'))}
    df = pd.DataFrame.from_dict(pred_dict, orient='index')
    df.index.names = ['id']
    df.columns = ['rle_mask']
    df.to_csv(str(save_path))
    # 結果を分析
    pred_bin = np.expand_dims(np.max(pred, axis=(1, 2, 3)).astype(np.uint8), axis=-1)  # 0 or 1
    empty_count = len(pred_bin) - pred_bin.sum()
    logger = tk.log.get(__name__)
    logger.info(f'empty rate: {empty_count}/{len(pred_bin)} = {100 * empty_count / len(pred_bin):.1f}%')


def _encode_rl(img):
    """ランレングス。"""
    img = img.reshape(img.shape[0] * img.shape[1], order='F')
    img = np.concatenate([[0], img, [0]])
    changes = img[1:] != img[:-1]
    rls = np.where(changes)[0] + 1
    rls[1::2] -= rls[::2]
    return ' '.join(str(x) for x in rls)
