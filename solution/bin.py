#!/usr/bin/env python3
"""有無のみを2クラス分類するやつ。"""
import argparse
import pathlib

import numpy as np
import sklearn.externals.joblib as joblib

import pytoolkit as tk
from lib import data

MODEL_NAME = pathlib.Path(__file__).stem
MODELS_DIR = pathlib.Path(f'models/{MODEL_NAME}')
REPORTS_DIR = pathlib.Path('reports')
CV_COUNT = 5
INPUT_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 100


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('check', 'train', 'validate', 'predict'))
    parser.add_argument('--cv-index', default=0, choices=range(CV_COUNT), type=int)
    args = parser.parse_args()
    if args.mode == 'check':
        _create_network()[0].summary()
    elif args.mode == 'train':
        with tk.dl.session(use_horovod=True):
            tk.log.init(MODELS_DIR / f'train.fold{args.cv_index}.log')
            _train(args)
    elif args.mode == 'validate':
        tk.log.init(REPORTS_DIR / f'{MODEL_NAME}.txt')
        _validate()
    else:
        assert args.mode == 'predict'  # このモデルは単体では予測できないので処理無し。


@tk.log.trace()
def _train(args):
    logger = tk.log.get(__name__)
    logger.info(f'args: {args}')

    split_seed = int(MODEL_NAME.encode('utf-8').hex(), 16) % 10000000
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / 'split_seed.txt').write_text(str(split_seed))

    X, d, y = data.load_train_data()
    y = np.max(data.load_mask(y) > 0.5, axis=(1, 2, 3)).astype(np.uint8)  # 0 or 1
    ti, vi = tk.ml.cv_indices(X, y, cv_count=CV_COUNT, cv_index=args.cv_index, split_seed=split_seed, stratify=False)
    (X_train, y_train), (X_val, y_val) = ([X[ti], d[ti]], y[ti]), ([X[vi], d[vi]], y[vi])
    logger.info(f'cv_index={args.cv_index}: train={len(y_train)} val={len(y_val)}')

    network, _ = _create_network()

    gen = tk.image.generator.Generator(multiple_input=True)
    gen.add(tk.image.LoadImage(grayscale=True), input_index=0)
    gen.add(tk.image.RandomFlipLR(probability=0.5), input_index=0)
    gen.add(tk.image.RandomPadding(probability=0.25), input_index=0)
    gen.add(tk.image.RandomRotate(probability=0.25), input_index=0)
    gen.add(tk.image.Resize(INPUT_SIZE), input_index=0)

    model = tk.dl.models.Model(network, gen, batch_size=BATCH_SIZE)
    model.compile(sgd_lr=0.1 / 128, loss='binary_crossentropy', metrics=[tk.dl.metrics.binary_accuracy])
    model.summary()
    model.plot(MODELS_DIR / 'model.svg', show_shapes=True)
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=EPOCHS,
        tsv_log_path=MODELS_DIR / f'history.fold{args.cv_index}.tsv',
        cosine_annealing=True, mixup=True)
    model.save(MODELS_DIR / f'model.fold{args.cv_index}.h5')

    if tk.dl.hvd.is_master():
        pred_val = model.predict(X_val)
        joblib.dump(pred_val, MODELS_DIR / f'pred-val.fold{args.cv_index}.pkl')
        tk.ml.print_classification_metrics(y_val, pred_val)


def _create_network():
    """ネットワークを作って返す。"""
    import keras
    inputs = [
        keras.layers.Input(INPUT_SIZE + (1,)),
        keras.layers.Input((1,)),
    ]
    x = inputs[0]
    x = tk.dl.layers.preprocess()(mode='tf')(x)
    x = keras.layers.concatenate([x, x, x])
    base_model = keras.applications.NASNetLarge(include_top=False, input_tensor=x)
    x = base_model.outputs[0]
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.concatenate([x, inputs[1]])
    x = keras.layers.Dense(256, activation='relu',
                           kernel_initializer='he_uniform',
                           kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.Dense(1, activation='sigmoid',
                           kernel_initializer='he_uniform',
                           kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    network = keras.models.Model(inputs, x)
    return network, None


@tk.log.trace()
def load_oofp(X, y):
    """out-of-fold predictionを読み込んで返す。"""
    split_seed = int((MODELS_DIR / 'split_seed.txt').read_text())
    pred = np.empty((len(y), 1), dtype=np.float32)
    for cv_index in range(CV_COUNT):
        _, vi = tk.ml.cv_indices(X, y, cv_count=CV_COUNT, cv_index=cv_index, split_seed=split_seed, stratify=False)
        pred[vi] = joblib.load(MODELS_DIR / f'pred-val.fold{cv_index}.pkl')
    return pred


@tk.log.trace()
def predict(ensemble):
    """予測。"""
    X, d = data.load_test_data()
    pred_list = []
    for cv_index in range(CV_COUNT):
        network = tk.dl.models.load_model(MODELS_DIR / f'model.fold{cv_index}.h5', compile=False)
        gen = tk.image.generator.Generator(multiple_input=True)
        gen.add(tk.image.LoadImage(grayscale=True), input_index=0)
        gen.add(tk.image.Resize(INPUT_SIZE), input_index=0)
        model = tk.dl.models.Model(network, gen, batch_size=BATCH_SIZE)
        pred = model.predict([X, d], verbose=1)
        pred_list.append(pred)
        if not ensemble:
            break
    return pred_list


@tk.log.trace()
def _validate():
    """検証＆閾値決定。"""
    logger = tk.log.get(__name__)
    X, _, y = data.load_train_data()
    y = data.load_mask(y)
    y = np.max(y > 0.5, axis=(1, 2, 3)).astype(np.uint8)  # 0 or 1
    y = np.expand_dims(y, axis=-1)
    pred = load_oofp(X, y)
    tk.ml.print_classification_metrics(y, pred, print_fn=logger.info)


if __name__ == '__main__':
    _main()
