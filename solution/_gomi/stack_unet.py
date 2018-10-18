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
EPOCHS = 300


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('check', 'train', 'fine', 'validate', 'predict'))
    parser.add_argument('--cv-index', default=0, choices=range(CV_COUNT), type=int)
    args = parser.parse_args()
    with tk.dl.session(use_horovod=args.mode in ('train', 'fine')):
        if args.mode == 'check':
            _create_network(input_dims=8)[0].summary()
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
    X = get_meta_features('val', X)
    ti, vi = tk.ml.cv_indices(X, y, cv_count=CV_COUNT, cv_index=args.cv_index, split_seed=split_seed, stratify=False)
    (X_train, y_train), (X_val, y_val) = (X[ti], y[ti]), (X[vi], y[vi])
    logger.info(f'cv_index={args.cv_index}: train={len(y_train)} val={len(y_val)}')

    network, _ = _create_network(input_dims=X.shape[-1])

    gen = tk.generator.Generator()
    if fine:
        pseudo_size = len(y_train) // 2
        X_train = np.array(list(X_train) + [None] * pseudo_size)
        y_train = np.array(list(y_train) + [None] * pseudo_size)
        X_test = _data.load_test_data()
        X_test = get_meta_features('test', X_test, args.cv_index)  # ←cv_indexはどれがいいとか無いので適当
        _, pi = tk.ml.cv_indices(X_test, np.zeros((len(X_test),)), cv_count=CV_COUNT, cv_index=args.cv_index, split_seed=split_seed, stratify=False)
        pred_test = predict_all('test', None, use_cache=True)[(args.cv_index + 1) % CV_COUNT]  # cross-pseudo-labeling
        gen.add(tk.generator.RandomPickData(X_test[pi], pred_test[pi]))
    gen.add(tk.image.RandomFlipLR(probability=0.5, with_output=True))
    # gen.add(tk.image.Padding(probability=1, with_output=True))
    # gen.add(tk.image.RandomRotate(probability=0.25, with_output=True))
    # gen.add(tk.image.RandomCrop(probability=1, with_output=True))
    # gen.add(tk.image.Resize((101, 101), with_output=True))

    model = tk.dl.models.Model(network, gen, batch_size=BATCH_SIZE)
    if fine:
        model.load_weights(MODELS_DIR / f'model.fold{args.cv_index}.h5')
    model.compile(sgd_lr=0.001 / 128 if fine else 0.01 / 128, loss=tk.dl.losses.lovasz_hinge_elup1, metrics=[tk.dl.metrics.binary_accuracy], clipnorm=10.0)
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=EPOCHS // 3 if fine else EPOCHS,
        cosine_annealing=True, mixup=False)
    model.save(MODELS_DIR / f'model.fold{args.cv_index}.h5', include_optimizer=False)

    if tk.dl.hvd.is_master():
        _evaluation.log_evaluation(y_val, model.predict(X_val))


def _create_network(input_dims):
    """ネットワークを作って返す。"""
    import keras
    builder = tk.dl.networks.Builder()

    inputs = [
        builder.input_tensor(INPUT_SIZE + (input_dims,)),
    ]
    x = inputs[0]
    x = keras.layers.concatenate([x, tk.dl.layers.channel_pair_2d()()(x)])
    x = tk.dl.layers.pad2d()(((5, 6), (5, 6)), mode='reflect')(x)  # 112
    down_list = [x]
    x = builder.conv2d(32)(x)
    x = builder.conv2d(32)(x)
    for stage, filters in enumerate([64, 128, 256, 512]):  # 56 28 14 7
        x = keras.layers.MaxPooling2D()(x)
        x = builder.conv2d(filters)(x)
        x = builder.conv2d(filters)(x)
        x = builder.scse_block(filters)(x)
        down_list.append(x)

    x = builder.conv2d(512)(x)
    x = builder.conv2d(512)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = builder.dense(256)(x)
    x = builder.act()(x)
    x = builder.dense(7 * 7 * 32)(x)
    x = builder.act()(x)
    x = keras.layers.Reshape((1, 1, -1))(x)

    for stage, d in list(enumerate(down_list))[::-1]:
        filters = builder.shape(d)[-1]
        x = tk.dl.layers.subpixel_conv2d()(scale=2 if stage != 4 else 7)(x)
        x = tk.dl.layers.coord_channel_2d()(x_channel=False)(x)
        x = builder.conv2d(filters, 1, use_act=False)(x)
        d = builder.conv2d(filters, 1, use_act=False)(d)
        x = keras.layers.add([x, d])
        x = builder.res_block(filters)(x)
        x = builder.res_block(filters)(x)
        x = builder.bn_act()(x)

    x = keras.layers.Cropping2D(((5, 6), (5, 6)))(x)  # 101
    x = builder.conv2d(1, use_bias=True, use_bn=False, activation='sigmoid')(x)
    network = keras.models.Model(inputs, x)
    return network, None


@tk.log.trace()
def _validate():
    """検証＆閾値決定。"""
    logger = tk.log.get(__name__)
    X, y = _data.load_train_data()
    pred = predict_all('val', X)
    threshold = _evaluation.log_evaluation(y, pred, print_fn=logger.info, search_th=True)
    (MODELS_DIR / 'threshold.txt').write_text(str(threshold))


@tk.log.trace()
def _predict():
    """予測。"""
    logger = tk.log.get(__name__)
    X_test = _data.load_test_data()
    threshold = float((MODELS_DIR / 'threshold.txt').read_text())
    logger.info(f'threshold = {threshold:.3f}')
    pred_list = predict_all('test', X_test)
    pred = np.mean(pred_list, axis=0) > threshold
    _data.save_submission(MODELS_DIR / 'submission.csv', pred)


def predict_all(data_name, X, use_cache=False, child_cv_index=None):
    """予測。"""
    cache_path = CACHE_DIR / data_name / f'{MODEL_NAME}.pkl'
    if use_cache and cache_path.is_file() and child_cv_index is None:
        return joblib.load(cache_path)

    if data_name == 'test' and child_cv_index is None:
        pred = [predict_all(data_name, X, use_cache, i) for i in range(5)]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pred, cache_path, compress=3)
        return pred

    if data_name == 'val':
        X_val = get_meta_features(data_name, X)
        X_list, vi_list = [], []
        split_seed = int((MODELS_DIR / 'split_seed.txt').read_text())
        for cv_index in range(CV_COUNT):
            _, vi = tk.ml.cv_indices(X_val, None, cv_count=CV_COUNT, cv_index=cv_index, split_seed=split_seed, stratify=False)
            X_list.append(X_val[vi])
            vi_list.append(vi)
    else:
        X_test = get_meta_features(data_name, X, child_cv_index)
        X_list = [X_test] * CV_COUNT

    gen = tk.generator.SimpleGenerator()
    model = tk.dl.models.Model.load(MODELS_DIR / f'model.fold0.h5', gen, batch_size=BATCH_SIZE, multi_gpu=True)

    pred_list = []
    for cv_index in tk.tqdm(range(CV_COUNT), desc='predict'):
        if cv_index != 0:
            model.load_weights(MODELS_DIR / f'model.fold{cv_index}.h5')

        X_t = X_list[cv_index]
        pred = np.mean([
            model.predict(X_t, verbose=0),
            model.predict(X_t[:, :, ::-1, :], verbose=0)[:, :, ::-1, :],
        ], axis=0)
        pred_list.append(pred)

    if data_name == 'val':
        pred = np.empty((len(X), 101, 101, 1), dtype=np.float32)
        for vi, p in zip(vi_list, pred_list):
            pred[vi] = p
    else:
        pred = np.mean(pred_list, axis=0)

    if data_name != 'test':
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pred, cache_path, compress=3)
    return pred


def get_meta_features(data_name, X, cv_index=None):
    """子モデルのout-of-fold predictionsを取得。"""
    import bin_nas
    import reg_nas
    import darknet53_coord_hcs  # 0.860
    import darknet53_large2  # 0.859
    import darknet53_resize128  # 0.859
    import darknet53_sepscse  # 0.863

    def _get(m):
        if data_name == 'val':
            return m
        else:
            assert len(m) == 5
            return m[cv_index]

    X = np.concatenate([
        X / 255,
        np.repeat(_get(bin_nas.predict_all(data_name, X, use_cache=True)), 101 * 101).reshape(len(X), 101, 101, 1),
        np.repeat(_get(reg_nas.predict_all(data_name, X, use_cache=True)), 101 * 101).reshape(len(X), 101, 101, 1),
        _get(darknet53_coord_hcs.predict_all(data_name, X, use_cache=True)),
        _get(darknet53_large2.predict_all(data_name, X, use_cache=True)),
        _get(darknet53_resize128.predict_all(data_name, X, use_cache=True)),
        _get(darknet53_sepscse.predict_all(data_name, X, use_cache=True)),
    ], axis=-1)
    return X


if __name__ == '__main__':
    _main()
