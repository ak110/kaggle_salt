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
    down_list = []
    for stage, filters in enumerate([64, 128, 256, 512, 512]):
        if stage == 0:
            x = builder.conv2d(filters, strides=1, use_act=False)(x)
        else:
            if builder.shape(x)[-2] % 2 != 0:
                x = tk.dl.layers.pad2d()(padding=((0, 1), (0, 1)))(x)
            x = builder.conv2d(filters, strides=2, use_act=False)(x)
        # for _ in range(3):
        #     x = builder.res_block(filters, dropout=0.25)(x)
        # x = builder.bn_act()(x)
        x = builder.conv2d(filters)(x)
        x = builder.conv2d(filters)(x)
        down_list.append(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = builder.dense(32)(x)
    x = builder.act()(x)

    # stage 0: 101
    # stage 1: 51
    # stage 2: 26
    # stage 3: 13
    # stage 4: 7
    for stage, d in list(enumerate(down_list))[::-1]:
        filters = builder.shape(d)[-1]
        if stage == 4:
            x = keras.layers.Reshape((1, 1, -1))(x)
        else:
            x = builder.conv2dtr(filters // 4, 2, strides=2, use_act=False)(x)
        if stage in (0, 1, 3):
            x = keras.layers.Cropping2D(((0, 1), (0, 1)))(x)
        x = builder.conv2d(filters, 1, use_act=False)(x)
        d = builder.conv2d(filters, 1, use_act=False)(d)
        x = keras.layers.add([x, d])
        x = builder.bn_act()(x)
        x = builder.conv2d(filters)(x)
        x = builder.conv2d(filters)(x)

    x = builder.conv2d(1, use_bias=True, use_bn=False, activation='sigmoid')(x)
    network = keras.models.Model(inputs, x)

    gen = tk.image.ImageDataGenerator()
    gen.add(tk.image.RandomFlipLR(probability=0.5, with_output=True))
    gen.add(tk.image.RandomRotate90(probability=0.5, with_output=True))

    model = tk.dl.models.Model(network, gen, batch_size=32)
    model.compile(sgd_lr=0.5 / 256, loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    model.plot(MODELS_DIR / 'model.svg')

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

    # 閾値の最適化
    threshold_list = np.linspace(0.25, 0.75, 20)
    score_list = np.array([data.compute_iou_metric(np.int32(y_val > 0.5), np.int32(pred_val > th))
                           for th in tk.tqdm(threshold_list)])
    best_index = score_list.argmax()
    logger = tk.log.get(__name__)
    logger.info(f'max score = {score_list[best_index]:.3f}')
    logger.info(f'threshold = {threshold_list[best_index]:.3f}')


if __name__ == '__main__':
    _main()
