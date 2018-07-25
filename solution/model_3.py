#!/usr/bin/env python3
"""有無のみを2クラス分類するやつ。"""
import argparse
import pathlib

import numpy as np
import sklearn.externals.joblib as joblib

import data
import pytoolkit as tk

MODELS_DIR = pathlib.Path('models/model_3')
SPLIT_SEED = 345
CV_COUNT = 5
OUTPUT_TYPE = 'bin'


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv-index', default=0, choices=range(CV_COUNT), type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=40, type=int)
    args = parser.parse_args()
    with tk.dl.session(use_horovod=True):
        tk.log.init(MODELS_DIR / f'train.fold{args.cv_index}.log')
        _run(args)


def _run(args):
    logger = tk.log.get(__name__)
    logger.info(f'args: {args}')
    X, d, y = data.load_train_data()
    y = np.max(data.load_mask(y) > 0.5, axis=(1, 2, 3)).astype(np.uint8)  # 0 or 1
    ti, vi = tk.ml.cv_indices(X, y, cv_count=CV_COUNT, cv_index=args.cv_index, split_seed=SPLIT_SEED, stratify=False)
    (X_train, y_train), (X_val, y_val) = ([X[ti], d[ti]], y[ti]), ([X[vi], d[vi]], y[vi])
    logger.info(f'cv_index={args.cv_index}: train={len(y_train)} val={len(y_val)}')

    import keras
    inputs = [
        keras.layers.Input((256, 256, 1)),
        keras.layers.Input((1,)),
    ]
    x = inputs[0]
    x = tk.dl.layers.preprocess()(mode='tf')(x)
    d = inputs[1]
    d = keras.layers.RepeatVector(256 * 256)(d)
    d = keras.layers.Reshape((256, 256, 1))(d)
    x = keras.layers.concatenate([x, x, d])
    base_model = keras.applications.NASNetLarge(include_top=False, input_tensor=x)
    x = base_model.outputs[0]
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1, activation='sigmoid',
                           kernel_initializer='zeros',
                           kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    network = keras.models.Model(inputs, x)

    gen = tk.image.generator.Generator(multiple_input=True)
    gen.add(tk.image.LoadImage(grayscale=True), input_index=0)
    gen.add(tk.image.RandomRotate(probability=0.25), input_index=0)
    gen.add(tk.image.Resize((256, 256)), input_index=0)
    gen.add(tk.image.RandomFlipLR(probability=0.5), input_index=0)

    model = tk.dl.models.Model(network, gen, batch_size=args.batch_size)
    model.compile(sgd_lr=0.1 / 128, loss='binary_crossentropy', metrics=['acc'])
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
        tk.ml.print_classification_metrics(y_val, pred_val)


if __name__ == '__main__':
    _main()
