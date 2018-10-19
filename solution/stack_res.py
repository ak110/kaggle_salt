#!/usr/bin/env python3
import argparse
import pathlib

import numpy as np
import sklearn.externals.joblib as joblib

import _ath
import _data
import _evaluation
import _mf
import pytoolkit as tk

MODEL_NAME = pathlib.Path(__file__).stem
MODELS_DIR = pathlib.Path(f'models/{MODEL_NAME}')
CACHE_DIR = pathlib.Path('cache')
CV_COUNT = 5
INPUT_SIZE = (101, 101)
BATCH_SIZE = 32
EPOCHS = 32


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('check', 'train', 'validate', 'predict'))
    parser.add_argument('--cv-index', default=0, choices=range(CV_COUNT), type=int)
    args = parser.parse_args()
    with tk.dl.session(use_horovod=args.mode in ('train', 'fine')):
        if args.mode == 'check':
            _create_network(input_dims=8)[0].summary()
        elif args.mode == 'train':
            tk.log.init(MODELS_DIR / f'train.fold{args.cv_index}.log')
            _train(args)
        elif args.mode == 'validate':
            tk.log.init(MODELS_DIR / 'validate.log')
            _validate()
        elif args.mode == 'predict':
            tk.log.init(MODELS_DIR / 'predict.log')
            _predict()


@tk.log.trace()
def _train(args):
    logger = tk.log.get(__name__)
    logger.info(f'args: {args}')

    split_seed = int(MODEL_NAME.encode('utf-8').hex(), 16) % 10000000
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / 'split_seed.txt').write_text(str(split_seed))

    X, y = _data.load_train_data()
    X = _mf.get_meta_features('val', X)
    ti, vi = tk.ml.cv_indices(X, y, cv_count=CV_COUNT, cv_index=args.cv_index, split_seed=split_seed, stratify=False)
    (X_train, y_train), (X_val, y_val) = (X[ti], y[ti]), (X[vi], y[vi])
    logger.info(f'cv_index={args.cv_index}: train={len(y_train)} val={len(y_val)}')

    network, _ = _create_network(input_dims=X.shape[-1])

    gen = tk.generator.Generator()
    gen.add(tk.image.RandomFlipLR(probability=0.5, with_output=True))
    # gen.add(tk.image.Padding(probability=1, with_output=True))
    # gen.add(tk.image.RandomRotate(probability=0.25, with_output=True))
    # gen.add(tk.image.RandomCrop(probability=1, with_output=True))
    # gen.add(tk.image.Resize((101, 101), with_output=True))

    model = tk.dl.models.Model(network, gen, batch_size=BATCH_SIZE)
    model.compile(sgd_lr=0.01 / 128, loss=tk.dl.losses.lovasz_hinge_elup1, metrics=[tk.dl.metrics.binary_accuracy], clipnorm=10.0)
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=EPOCHS,
        reduce_lr_epoch_rates=(0.5, 0.75, 0.875), mixup=False, lr_warmup=False)
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
    x = tk.dl.layers.coord_channel_2d()(x_channel=False)(x)
    x = keras.layers.concatenate([x, tk.dl.layers.channel_pair_2d()()(x)])
    x = builder.conv2d(128, 1, use_act=False)(x)
    x = builder.res_block(128)(x)
    x = builder.res_block(128)(x)
    x = builder.res_block(128)(x)
    x = builder.bn_act()(x)

    a = keras.layers.concatenate([
        keras.layers.GlobalAveragePooling2D()(x),
        keras.layers.GlobalMaxPooling2D()(x),
    ])
    a = builder.dense(32)(a)
    a = builder.dense(builder.shape(x)[-1], activation='sigmoid')(a)
    a = keras.layers.multiply([x, a])
    x = keras.layers.concatenate([x, a])

    x = builder.conv2d(1, 1, use_bias=True, use_bn=False, activation='sigmoid')(x)

    network = keras.models.Model(inputs, x)
    return network, None


@tk.log.trace()
def _validate():
    """検証＆閾値決定。"""
    logger = tk.log.get(__name__)
    X, y = _data.load_train_data()
    pred = predict_all('val', X, use_cache=True)  # TODO: 仮！, use_cache=True
    # 閾値の調整
    pred_bin = joblib.load(CACHE_DIR / 'val' / 'bin_nas.pkl')
    pred_reg = joblib.load(CACHE_DIR / 'val' / 'reg_nas.pkl')
    threshold_X, threshold_y = _ath.create_data(y, pred, pred_bin, pred_reg)
    split_seed = int((MODELS_DIR / 'split_seed.txt').read_text())
    thresholds = np.empty((len(y),))
    for cv_index in range(CV_COUNT):
        ti, vi = tk.ml.cv_indices(X, y, cv_count=CV_COUNT, cv_index=cv_index, split_seed=split_seed, stratify=False)
        ath_estimator = _ath.create_estimator(threshold_X[ti], threshold_y[ti])
        thresholds[vi] = ath_estimator.predict(threshold_X[vi])
    # CVで決めた閾値でevaluate
    _evaluation.log_evaluation(y, pred, print_fn=logger.info, threshold=thresholds)
    # 全体で学習しなおして保存
    ath_estimator = _ath.create_estimator(threshold_X, threshold_y)
    joblib.dump(ath_estimator, MODELS_DIR / 'ath_estimator.pkl')
    # 最後にインチキ閾値でevaluate
    thresholds = ath_estimator.predict(threshold_X)
    _evaluation.log_evaluation(y, pred, print_fn=logger.info, threshold=thresholds)


@tk.log.trace()
def _predict():
    """予測。"""
    X_test = _data.load_test_data()
    pred_list = predict_all('test', X_test)
    pred = np.mean(pred_list, axis=0) > 0.5
    _data.save_submission(MODELS_DIR / 'submission.csv', pred)


def predict_all(data_name, X, use_cache=False, child_cv_index=None):
    """予測。"""
    cache_path = CACHE_DIR / data_name / f'{MODEL_NAME}.pkl'
    if use_cache and cache_path.is_file() and child_cv_index is None:
        return joblib.load(cache_path)

    if data_name == 'test' and child_cv_index is None:
        pred_bin = np.median(joblib.load(CACHE_DIR / 'test' / 'bin_nas.pkl'), axis=0)
        pred_reg = np.median(joblib.load(CACHE_DIR / 'test' / 'reg_nas.pkl'), axis=0)
        ath_estimator = joblib.load(MODELS_DIR / 'ath_estimator.pkl')
        pred = []
        for cci in range(5):
            y_child = []
            for pc in predict_all(data_name, X, use_cache, cci):
                threshold_X = _ath.create_input_data(pc, pred_bin, pred_reg)
                thresholds = ath_estimator.predict(threshold_X)
                y_child.append(pc > np.reshape(thresholds, (len(pc), 1, 1, 1)))
            pred.append(np.mean(y_child, axis=0))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pred, cache_path, compress=3)
        return pred

    if data_name == 'val':
        X_val = _mf.get_meta_features(data_name, X)
        X_list, vi_list = [], []
        split_seed = int((MODELS_DIR / 'split_seed.txt').read_text())
        for cv_index in range(CV_COUNT):
            _, vi = tk.ml.cv_indices(X_val, None, cv_count=CV_COUNT, cv_index=cv_index, split_seed=split_seed, stratify=False)
            X_list.append(X_val[vi])
            vi_list.append(vi)
    else:
        X_test = _mf.get_meta_features(data_name, X, child_cv_index)
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
        pred = pred_list

    if data_name != 'test':
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pred, cache_path, compress=3)
    return pred


if __name__ == '__main__':
    _main()
