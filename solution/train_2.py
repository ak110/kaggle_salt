#!/usr/bin/env python3
import argparse
import pathlib

import sklearn.externals.joblib as joblib

import data
import evaluation
import pytoolkit as tk

MODELS_DIR = pathlib.Path('models/model_2')
SPLIT_SEED = 234
CV_COUNT = 5


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv-index', default=0, choices=range(CV_COUNT), type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    args = parser.parse_args()
    with tk.dl.session(use_horovod=True):
        tk.log.init(MODELS_DIR / f'train.fold{args.cv_index}.log')
        _run(args)


def _run(args):
    logger = tk.log.get(__name__)
    X, d, y = data.load_train_data()
    ti, vi = tk.ml.cv_indices(X, y, cv_count=CV_COUNT, cv_index=args.cv_index, split_seed=SPLIT_SEED, stratify=False)
    (X_train, y_train), (X_val, y_val) = ([X[ti], d[ti]], y[ti]), ([X[vi], d[vi]], y[vi])
    y_train = data.load_mask(y_train)
    y_val = data.load_mask(y_val)
    logger.info(f'cv_index={args.cv_index}: train={len(y_train)} val={len(y_val)}')

    import keras
    builder = tk.dl.networks.Builder()

    inputs = [
        builder.input_tensor((101, 101, 1)),
        builder.input_tensor((1,)),
    ]
    x = inputs[0]
    x = builder.preprocess()(x)
    d = inputs[1]
    d = keras.layers.RepeatVector(101 * 101)(d)
    d = keras.layers.Reshape((101, 101, 1))(d)
    x = keras.layers.concatenate([x, d])
    down_list = []
    for stage, filters in enumerate([64, 128, 256, 512, 512]):
        if stage == 0:
            x = builder.conv2d(filters, strides=1)(x)
        else:
            x = keras.layers.MaxPooling2D(padding='same')(x)
        x = builder.conv2d(filters)(x)
        x = builder.conv2d(filters)(x)
        x = builder.conv2d(filters)(x)
        down_list.append(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    gate = builder.dense(1, activation='sigmoid')(x)
    x = builder.dense(32)(x)
    x = builder.act()(x)
    x = keras.layers.Reshape((1, 1, -1))(x)

    # stage 0: 101
    # stage 1: 51
    # stage 2: 26
    # stage 3: 13
    # stage 4: 7
    for stage, d in list(enumerate(down_list))[::-1]:
        filters = builder.shape(d)[-1]
        if stage == 4:
            x = builder.conv2dtr(32, 7)(x)
        else:
            x = builder.conv2dtr(filters // 4, 2, strides=2)(x)
            if stage in (0, 1, 3):
                x = keras.layers.Cropping2D(((0, 1), (0, 1)))(x)
        x = builder.conv2d(filters, 1, use_act=False)(x)
        d = builder.conv2d(filters, 1, use_act=False)(d)
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
    gen.add(tk.image.Resize((101, 101), with_output=True), input_index=0)
    gen.add(tk.image.RandomFlipLR(probability=0.5, with_output=True), input_index=0)

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
        joblib.dump(pred_val, MODELS_DIR / f'pred-val.fold{args.cv_index}.h5')
        evaluation.log_evaluation(y_val, pred_val)


if __name__ == '__main__':
    _main()
