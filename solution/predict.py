import pathlib

import cv2
import numpy as np
import pandas as pd

import pytoolkit as tk

MODELS_DIR = pathlib.Path('models')

TRAIN_IMAGE_DIR = pathlib.Path('../input/train/images/')
TRAIN_IMAGE_DIR = pathlib.Path('../input/train/images/')
TRAIN_MASK_DIR = pathlib.Path('../input/train/masks/')
TEST_IMAGE_DIR = pathlib.Path('../input/test/images/')
DEPTHS_PATH = pathlib.Path('../input/depths.csv')


def _main():
    tk.better_exceptions()

    test_names = [p.name for p in TRAIN_IMAGE_DIR.iterdir()]
    test_prefixes = [p.stem for p in TRAIN_IMAGE_DIR.iterdir()]
    X = [cv2.imread(str(TRAIN_IMAGE_DIR / p), cv2.IMREAD_GRAYSCALE) for p in tk.tqdm(test_names)]
    X = np.array(X, dtype=np.float32) / 255
    X = np.expand_dims(X, axis=3)
    df_depths = pd.read_csv(DEPTHS_PATH)
    d = np.array([df_depths.loc[df_depths['id'] == p]['z'].values for p in tk.tqdm(test_prefixes)], dtype=np.float32)
    d -= df_depths['z'].mean()
    d /= df_depths['z'].std()
    d = np.repeat(d, 101 * 101).reshape(len(X), 101, 101, 1)
    X = np.concatenate([X, d], axis=-1)

    with tk.dl.session():
        tk.log.init(MODELS_DIR / 'predict.log')
        _run(X, test_prefixes)


def _run(X, test_prefixes):
    network = tk.dl.models.load_model(MODELS_DIR / 'model.h5')
    gen = tk.image.ImageDataGenerator()
    model = tk.dl.models.Model(network, gen, batch_size=32)
    pred = model.predict(X, verbose=1)

    pred_dict = {prefix: encode_rl(np.round(pred[i, :, :, 0])) for i, prefix in tk.tqdm(enumerate(test_prefixes))}
    df = pd.DataFrame.from_dict(pred_dict, orient='index')
    df.index.names = ['id']
    df.columns = ['rle_mask']
    df.to_csv(str(MODELS_DIR / 'submission.csv'))


def encode_rl(img):
    """ランレングス。"""
    b = img.reshape(img.shape[0] * img.shape[1], order='F')
    runs = []  # list of run lengths
    r = 0  # the current run length
    pos = 1  # count starts from 1 per WK
    for c in b:
        if (c == 0):
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


if __name__ == '__main__':
    _main()
