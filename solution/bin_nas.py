#!/usr/bin/env python3
import argparse
import pathlib

import numpy as np
import sklearn.externals.joblib as joblib

import _data
import _evaluation
import pytoolkit as tk

MODEL_NAME = pathlib.Path(__file__).stem
MODELS_DIR = pathlib.Path(f'models/{MODEL_NAME}')
CACHE_DIR = pathlib.Path('cache')
CV_COUNT = 5
INPUT_SIZE = (101, 101)
BATCH_SIZE = 16
EPOCHS = 100


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('check', 'train', 'fine', 'validate', 'predict'))
    parser.add_argument('--cv-index', default=0, choices=range(CV_COUNT), type=int)
    args = parser.parse_args()
    with tk.dl.session(use_horovod=args.mode in ('train', 'fine')):
        if args.mode == 'check':
            _create_network()[0].summary()
        elif args.mode == 'train':
            tk.log.init(MODELS_DIR / f'train.fold{args.cv_index}.log')
            _train(args)
        elif args.mode == 'fine':
            tk.log.init(MODELS_DIR / f'fine.fold{args.cv_index}.log')
            _train(args, fine=True)
        elif args.mode == 'validate':
            tk.log.init(MODELS_DIR / 'validate.log')
            _validate()
        elif args.mode == 'predict':
            tk.log.init(MODELS_DIR / 'predict.log')
            _predict()


@tk.log.trace()
def _train(args, fine=False):
    logger = tk.log.get(__name__)
    logger.info(f'args: {args}')

    split_seed = int(MODEL_NAME.encode('utf-8').hex(), 16) % 10000000
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / 'split_seed.txt').write_text(str(split_seed))

    X, y = _data.load_train_data()
    y = np.max(y > 0.5, axis=(1, 2, 3)).astype(np.uint8)  # 0 or 1
    ti, vi = tk.ml.cv_indices(X, y, cv_count=CV_COUNT, cv_index=args.cv_index, split_seed=split_seed, stratify=False)
    (X_train, y_train), (X_val, y_val) = (X[ti], y[ti]), (X[vi], y[vi])
    logger.info(f'cv_index={args.cv_index}: train={len(y_train)} val={len(y_val)}')

    network, _ = _create_network()

    gen = tk.generator.Generator()
    if fine:
        pseudo_size = len(y_train) // 2
        X_train = np.array(list(X_train) + [None] * pseudo_size)
        y_train = np.array(list(y_train) + [None] * pseudo_size)
        X_test = _data.load_test_data()
        _, pi = tk.ml.cv_indices(X_test, np.zeros((len(X_test),)), cv_count=CV_COUNT, cv_index=args.cv_index, split_seed=split_seed, stratify=False)
        pred_test = predict_all('test', None, use_cache=True)[(args.cv_index + 1) % CV_COUNT]  # cross-pseudo-labeling
        gen.add(tk.generator.RandomPickData(X_test[pi], pred_test[pi]))
    gen.add(tk.image.RandomFlipLR(probability=0.5))
    gen.add(tk.image.RandomPadding(probability=0.25, mode='reflect'))
    gen.add(tk.image.RandomRotate(probability=0.25))
    gen.add(tk.image.RandomAugmentors([
        tk.image.RandomBlur(probability=0.125),
        tk.image.RandomUnsharpMask(probability=0.125),
        tk.image.RandomBrightness(probability=0.25),
        tk.image.RandomContrast(probability=0.25),
    ], probability=0.125))
    gen.add(tk.image.Resize((101, 101)))

    model = tk.dl.models.Model(network, gen, batch_size=BATCH_SIZE)
    if fine:
        model.load_weights(MODELS_DIR / f'model.fold{args.cv_index}.h5')
    model.compile(sgd_lr=0.001 / 128 if fine else 0.1 / 128, loss='binary_crossentropy',
                  metrics=[tk.dl.metrics.binary_accuracy])
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=EPOCHS // 3 if fine else EPOCHS,
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
    X, y = _data.load_train_data()
    y = np.max(y > 0.5, axis=(1, 2, 3)).astype(np.uint8)  # 0 or 1
    pred = predict_all('val', X)
    tk.ml.print_classification_metrics(y, pred, print_fn=logger.info)


@tk.log.trace()
def _predict():
    """予測。"""
    X_test = _data.load_test_data()
    predict_all('test', X_test)


def predict_all(data_name, X, use_cache=False):
    """予測。"""
    cache_path = CACHE_DIR / data_name / f'{MODEL_NAME}.pkl'
    if use_cache and cache_path.is_file():
        return joblib.load(cache_path)

    if data_name == 'val':
        X_list, vi_list = [], []
        split_seed = int((MODELS_DIR / 'split_seed.txt').read_text())
        for cv_index in range(CV_COUNT):
            _, vi = tk.ml.cv_indices(X, None, cv_count=CV_COUNT, cv_index=cv_index, split_seed=split_seed, stratify=False)
            X_list.append(X[vi])
            vi_list.append(vi)
    else:
        X = _data.load_test_data()
        X_list = [X] * CV_COUNT

    gen = tk.generator.SimpleGenerator()
    model = tk.dl.models.Model.load(MODELS_DIR / f'model.fold0.h5', gen, batch_size=BATCH_SIZE, multi_gpu=True)

    pred_list = []
    for cv_index in tk.tqdm(range(CV_COUNT), desc='predict'):
        if cv_index != 0:
            model.load_weights(MODELS_DIR / f'model.fold{cv_index}.h5')

        X_t = X_list[cv_index]
        pred = _evaluation.predict_tta(model, X_t, mode='bin')
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
