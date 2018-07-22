import pathlib

import numpy as np

import data
import pytoolkit as tk

MODELS_DIR = pathlib.Path('models')


def _main():
    tk.better_exceptions()
    X, y, _ = data.load_train_data()
    (X_train, y_train), (X_val, y_val) = tk.ml.split(X, y, split_seed=123, validation_split=0.2)

    with tk.dl.session(use_horovod=True):
        tk.log.init(MODELS_DIR / 'train.log')
        _run(X_train, y_train, X_val, y_val)


def _run(X_train, y_train, X_val, y_val):
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
    gen.add(tk.image.RandomFlipLR(probability=0.5, with_output=True))
    gen.add(tk.image.RandomRotate90(probability=0.5, with_output=True))

    model = tk.dl.models.Model(network, gen, batch_size=32)
    model.compile(sgd_lr=0.5 / 256, loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    # 学習
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=300,
        tsv_log_path=MODELS_DIR / 'history.tsv',
        cosine_annealing=True, mixup=True)
    model.save(MODELS_DIR / 'model.h5')

    # 検証
    pred_val = model.predict(X_val)
    tk.ml.print_classification_metrics(np.ravel(y_val), np.ravel(pred_val))


if __name__ == '__main__':
    _main()
