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
    with tk.dl.session(use_horovod=True):
        tk.log.init(MODELS_DIR / 'train.log')
        _run()


def _run():
    train_names = [p.name for p in TRAIN_IMAGE_DIR.iterdir()]
    train_prefixes = [p.stem for p in TRAIN_IMAGE_DIR.iterdir()]
    X = [cv2.imread(str(TRAIN_IMAGE_DIR / p), cv2.IMREAD_GRAYSCALE) for p in tk.tqdm(train_names)]
    X = np.array(X, dtype=np.float32) / 255
    X = np.expand_dims(X, axis=3)
    y = [cv2.imread(str(TRAIN_MASK_DIR / p), cv2.IMREAD_GRAYSCALE) for p in tk.tqdm(train_names)]
    y = np.array(y, dtype=np.float32) / 255
    y = np.expand_dims(y, axis=3)
    df_depths = pd.read_csv(DEPTHS_PATH)
    d = np.array([df_depths.loc[df_depths['id'] == p]['z'].values for p in tk.tqdm(train_prefixes)], dtype=np.float32)
    d -= df_depths['z'].mean()
    d /= df_depths['z'].std()
    d = np.repeat(d, 101 * 101).reshape(len(X), 101, 101, 1)
    X = np.concatenate([X, d], axis=-1)
    print(X.shape)
    (X_train, y_train), (X_val, y_val) = tk.ml.split(X, y, split_seed=123, validation_split=0.2)

    import keras
    builder = tk.dl.networks.Builder()
    x = inputs = keras.layers.Input(shape=(101, 101, 2))
    x = builder.conv2d(64)(x)
    x = builder.conv2d(64)(x)
    d1 = x
    x = keras.layers.MaxPooling2D(padding='same')(x)  # 51
    x = builder.conv2d(128)(x)
    x = builder.conv2d(128)(x)
    d2 = x
    x = keras.layers.MaxPooling2D(padding='same')(x)  # 26
    x = builder.conv2d(256)(x)
    x = builder.conv2d(256)(x)
    d3 = x
    x = keras.layers.MaxPooling2D(padding='same')(x)  # 13
    x = builder.conv2d(512)(x)
    x = builder.conv2d(512)(x)
    d4 = x

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = builder.dense(16)(x)
    x = builder.act()(x)
    x = builder.dense(512)(x)

    x = keras.layers.Reshape((1, 1, 512))(x)
    d4 = builder.conv2d(512, 1, use_act=False)(d4)
    x = keras.layers.add([d4, x])
    x = builder.conv2d(512)(x)
    x = builder.conv2d(512)(x)

    x = builder.conv2dtr(256, 2, strides=2, use_act=False)(x)
    d3 = builder.conv2d(256, 1, use_act=False)(d3)
    x = keras.layers.add([d3, x])
    x = builder.conv2d(256)(x)
    x = builder.conv2d(256)(x)

    x = builder.conv2dtr(128, 2, strides=2, use_act=False)(x)
    x = keras.layers.Cropping2D(((0, 1), (0, 1)))(x)
    d2 = builder.conv2d(128, 1, use_act=False)(d2)
    x = keras.layers.add([d2, x])
    x = builder.conv2d(128)(x)
    x = builder.conv2d(128)(x)

    x = builder.conv2dtr(64, 2, strides=2, use_act=False)(x)
    x = keras.layers.Cropping2D(((0, 1), (0, 1)))(x)
    d1 = builder.conv2d(64, 1, use_act=False)(d1)
    x = keras.layers.add([d1, x])
    x = builder.conv2d(64)(x)
    x = builder.conv2d(64)(x)

    x = builder.conv2d(1, use_bn=False, activation='sigmoid')(x)
    network = keras.models.Model(inputs, x)

    gen = tk.image.ImageDataGenerator()
    model = tk.dl.models.Model(network, gen, batch_size=32)
    model.compile(sgd_lr=0.5 / 256, loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=300,
        tsv_log_path=MODELS_DIR / 'history.tsv',
        cosine_annealing=True, mixup=True)
    model.save('model.h5')


if __name__ == '__main__':
    _main()
