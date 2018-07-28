#!/usr/bin/env python3
"""256x256でやってみるやつ。"""
import argparse
import pathlib

import numpy as np
import sklearn.externals.joblib as joblib

import data
import evaluation
import pytoolkit as tk

MODELS_DIR = pathlib.Path('models/model_1')
SPLIT_SEED = 123
CV_COUNT = 5
OUTPUT_TYPE = 'mask'


def _train():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv-index', default=0, choices=range(CV_COUNT), type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    args = parser.parse_args()
    with tk.dl.session(use_horovod=True):
        tk.log.init(MODELS_DIR / f'train.fold{args.cv_index}.log')
        _train_impl(args)


def _train_impl(args):
    logger = tk.log.get(__name__)
    logger.info(f'args: {args}')
    X, d, y = data.load_train_data()
    y = data.load_mask(y)

    import model_3
    mf = model_3.load_oofp(X, y)

    ti, vi = tk.ml.cv_indices(X, y, cv_count=CV_COUNT, cv_index=args.cv_index, split_seed=SPLIT_SEED, stratify=False)
    (X_train, y_train), (X_val, y_val) = ([X[ti], d[ti], mf[ti]], y[ti]), ([X[vi], d[vi], mf[vi]], y[vi])
    logger.info(f'cv_index={args.cv_index}: train={len(y_train)} val={len(y_val)}')

    import keras
    builder = tk.dl.networks.Builder()

    inputs = [
        builder.input_tensor((256, 256, 1)),
        builder.input_tensor((1,)),  # depths
        builder.input_tensor((1,)),  # model_3
    ]
    x = inputs[0]
    x = builder.preprocess()(x)
    down_list = []
    for stage, filters in enumerate([32, 64, 128, 256, 512, 512]):
        if stage != 0:
            x = keras.layers.MaxPooling2D(padding='same')(x)
        x = builder.conv2d(filters)(x)
        x = builder.conv2d(filters)(x)
        x = builder.conv2d(filters)(x)
        down_list.append(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = builder.dense(32)(x)
    x = builder.act()(x)
    x = keras.layers.concatenate([x, inputs[1], inputs[2]])
    x = builder.dense(32)(x)
    x = builder.act()(x)
    gate = builder.dense(1, activation='sigmoid')(x)
    x = keras.layers.Reshape((1, 1, -1))(x)

    # stage 0: 256
    # stage 1: 128
    # stage 2: 64
    # stage 3: 32
    # stage 4: 16
    # stage 5: 8
    for stage, d in list(enumerate(down_list))[::-1]:
        filters = builder.shape(d)[-1]
        if stage == 5:
            x = builder.conv2dtr(32, 8, strides=8)(x)
        else:
            x = builder.conv2dtr(filters // 4, 2, strides=2)(x)
        x = builder.conv2d(filters, 1, use_bn=False, use_act=False)(x)
        d = builder.conv2d(filters, 1, use_bn=False, use_act=False)(d)
        x = keras.layers.add([x, d])
        x = builder.bn_act()(x)
        x = keras.layers.Dropout(0.25)(x)
        x = builder.conv2d(filters)(x)
        x = builder.conv2d(filters)(x)

    x = builder.conv2d(1, use_bias=True, use_bn=False, activation='sigmoid')(x)
    x = keras.layers.multiply([x, gate])
    network = keras.models.Model(inputs, x)

    gen = tk.image.generator.Generator(multiple_input=True)
    gen.add(tk.image.LoadImage(grayscale=True), input_index=0)
    gen.add(tk.image.RandomPadding(probability=1, with_output=True), input_index=0)
    gen.add(tk.image.RandomRotate(probability=0.25, with_output=True), input_index=0)
    gen.add(tk.image.RandomCrop(probability=1, with_output=True), input_index=0)
    gen.add(tk.image.RandomFlipLR(probability=0.5, with_output=True), input_index=0)
    gen.add(tk.image.Resize((256, 256), with_output=True), input_index=0)

    model = tk.dl.models.Model(network, gen, batch_size=args.batch_size)
    model.compile(sgd_lr=0.5 / 256, loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    model.plot(MODELS_DIR / 'model.svg', show_shapes=True)
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=args.epochs,
        tsv_log_path=MODELS_DIR / f'history.fold{args.cv_index}.tsv',
        cosine_annealing=True, mixup=True)
    model.save(MODELS_DIR / f'model.fold{args.cv_index}.h5')

    if tk.dl.hvd.is_master():
        pred_val = model.predict(X_val)
        pred_val = np.array([tk.ndimage.resize(p, 101, 101) for p in pred_val])  # リサイズ
        joblib.dump(pred_val, MODELS_DIR / f'pred-val.fold{args.cv_index}.h5')
        evaluation.log_evaluation(y_val, pred_val)


def load_oofp(X, y):
    """out-of-fold predictionを読み込んで返す。"""
    pred = np.empty((len(y), 101, 101, 1), dtype=np.float32)
    for cv_index in range(CV_COUNT):
        _, vi = tk.ml.cv_indices(X, y, cv_count=CV_COUNT, cv_index=cv_index, split_seed=SPLIT_SEED, stratify=False)
        pred[vi] = joblib.load(MODELS_DIR / f'pred-val.fold{cv_index}.h5')
    return pred


if __name__ == '__main__':
    _train()
