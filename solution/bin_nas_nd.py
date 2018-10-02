#!/usr/bin/env python3
"""有無のみを2クラス分類するやつ。"""
import argparse
import pathlib

import numpy as np
import sklearn.externals.joblib as joblib

import pytoolkit as tk
from lib import data, generator

MODEL_NAME = pathlib.Path(__file__).stem
MODELS_DIR = pathlib.Path(f'models/{MODEL_NAME}')
REPORTS_DIR = pathlib.Path('reports')
CACHE_DIR = pathlib.Path('cache')
CV_COUNT = 5
INPUT_SIZE = (101, 101)
BATCH_SIZE = 16
EPOCHS = 150


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('check', 'train', 'validate', 'predict'))
    parser.add_argument('--cv-index', default=0, choices=range(CV_COUNT), type=int)
    args = parser.parse_args()
    with tk.dl.session(use_horovod=args.mode == 'train'):
        if args.mode == 'check':
            _create_network()[0].summary()
        elif args.mode == 'train':
            tk.log.init(MODELS_DIR / f'train.fold{args.cv_index}.log')
            _train(args)
        elif args.mode == 'validate':
            tk.log.init(REPORTS_DIR / f'{MODEL_NAME}.txt', file_level='INFO')
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

    X, _, y = data.load_train_data()
    y = np.max(y > 0.5, axis=(1, 2, 3)).astype(np.uint8)  # 0 or 1
    ti, vi = tk.ml.cv_indices(X, y, cv_count=CV_COUNT, cv_index=args.cv_index, split_seed=split_seed, stratify=False)
    (X_train, y_train), (X_val, y_val) = (X[ti], y[ti]), (X[vi], y[vi])
    logger.info(f'cv_index={args.cv_index}: train={len(y_train)} val={len(y_val)}')

    network, _ = _create_network()

    gen = generator.create_generator(mode='bin', multiple_input=False)
    model = tk.dl.models.Model(network, gen, batch_size=BATCH_SIZE)
    model.compile(sgd_lr=0.1 / 128, loss='binary_crossentropy', metrics=[tk.dl.metrics.binary_accuracy])
    model.plot(MODELS_DIR / 'model.svg', show_shapes=True)
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=EPOCHS,
        tsv_log_path=MODELS_DIR / f'history.fold{args.cv_index}.tsv',
        cosine_annealing=True, mixup=True)
    model.save(MODELS_DIR / f'model.fold{args.cv_index}.h5', include_optimizer=False)

    if tk.dl.hvd.is_master():
        tk.ml.print_classification_metrics(y_val, model.predict(X_val))


def _create_network():
    """ネットワークを作って返す。"""
    import keras
    builder = tk.dl.networks.Builder()

    inputs = [
        builder.input_tensor(INPUT_SIZE + (1,)),
    ]
    x = inputs[0]
    x = builder.preprocess(mode='tf')(x)
    x = keras.layers.concatenate([x, x, x])
    x = tk.dl.layers.resize2d()((235, 235))(x)
    base_model = keras.applications.NASNetLarge(include_top=False, input_tensor=x)
    x = base_model.outputs[0]
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1, activation='sigmoid',
                           kernel_initializer='zeros',
                           kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    network = keras.models.Model(inputs, x)
    return network, None


@tk.log.trace()
def _validate():
    """検証＆閾値決定。"""
    logger = tk.log.get(__name__)
    X, d, y = data.load_train_data()
    y = np.max(y > 0.5, axis=(1, 2, 3)).astype(np.uint8)  # 0 or 1
    pred = predict_all('val', X, d)
    tk.ml.print_classification_metrics(y, pred, print_fn=logger.info)


def predict_all(data_name, X, d):
    """予測。"""
    _ = d
    cache_path = CACHE_DIR / data_name / f'{MODEL_NAME}.pkl'
    if cache_path.is_file():
        return joblib.load(cache_path)

    if data_name == 'val':
        X_list, vi_list = [], []
        split_seed = int((MODELS_DIR / 'split_seed.txt').read_text())
        for cv_index in range(CV_COUNT):
            _, vi = tk.ml.cv_indices(X, None, cv_count=CV_COUNT, cv_index=cv_index, split_seed=split_seed, stratify=False)
            X_list.append(X[vi])
            vi_list.append(vi)
    else:
        X, _ = data.load_test_data()
        X_list = [X] * CV_COUNT

    gen = tk.generator.SimpleGenerator()
    model = tk.dl.models.Model.load(MODELS_DIR / f'model.fold0.h5', gen, batch_size=BATCH_SIZE, multi_gpu=True)

    pred_list = []
    for cv_index in tk.tqdm(range(CV_COUNT), desc='predict'):
        if cv_index != 0:
            model.load_weights(MODELS_DIR / f'model.fold{cv_index}.h5')

        X_t = X_list[cv_index]
        pred1 = model.predict(X_t, verbose=0)
        pred2 = model.predict(X_t[:, :, ::-1, :], verbose=0)
        pred = np.mean([pred1, pred2], axis=0)
        pred_list.append(pred)

    if data_name == 'val':
        pred = np.empty((len(X), 1), dtype=np.float32)
        for vi, p in zip(vi_list, pred_list):
            pred[vi] = p
    else:
        pred = pred_list

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pred, cache_path, compress=3)
    return pred


if __name__ == '__main__':
    _main()
