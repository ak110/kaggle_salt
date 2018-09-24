#!/usr/bin/env python3
import argparse
import pathlib

import numpy as np
import sklearn.externals.joblib as joblib
from lib import data, evaluation

import pytoolkit as tk

MODEL_NAME = pathlib.Path(__file__).stem
MODELS_DIR = pathlib.Path(f'models/{MODEL_NAME}')
REPORTS_DIR = pathlib.Path('reports')
CV_COUNT = 5
INPUT_SIZE = (101, 101)
BATCH_SIZE = 64
EPOCHS = 30


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('check', 'train', 'validate', 'predict'))
    parser.add_argument('--cv-index', default=0, choices=range(CV_COUNT), type=int)
    args = parser.parse_args()
    with tk.dl.session(use_horovod=args.mode == 'train'):
        if args.mode == 'check':
            _create_network(input_dims=8)[0].summary()
        elif args.mode == 'train':
            tk.log.init(MODELS_DIR / f'train.fold{args.cv_index}.log')
            _train(args)
        elif args.mode == 'validate':
            tk.log.init(REPORTS_DIR / f'{MODEL_NAME}.txt', file_level='INFO')
            _validate()
        else:
            tk.log.init(MODELS_DIR / 'predict.log')
            _predict()


@tk.log.trace()
def _train(args):
    logger = tk.log.get(__name__)
    logger.info(f'args: {args}')

    split_seed = int(MODEL_NAME.encode('utf-8').hex(), 16) % 10000000
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / 'split_seed.txt').write_text(str(split_seed))

    X, d, y = data.load_train_data()
    X = _get_meta_features('val', X, d)
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
    model.compile(sgd_lr=0.01 / 128, loss=tk.dl.losses.lovasz_hinge, metrics=[tk.dl.metrics.binary_accuracy])
    model.plot(MODELS_DIR / 'model.svg', show_shapes=True)
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=EPOCHS,
        tsv_log_path=MODELS_DIR / f'history.fold{args.cv_index}.tsv',
        cosine_annealing=True, mixup=False)
    model.save(MODELS_DIR / f'model.fold{args.cv_index}.h5', include_optimizer=False)

    if tk.dl.hvd.is_master():
        evaluation.log_evaluation(y_val, model.predict(X_val))


def _create_network(input_dims):
    """ネットワークを作って返す。"""
    import keras
    import keras.backend as K
    builder = tk.dl.networks.Builder()

    inputs = [
        builder.input_tensor(INPUT_SIZE + (input_dims,)),
    ]
    x = inputs[0]
    x = keras.layers.concatenate([x, tk.dl.layers.channel_pair_2d()()(x)])
    x = builder.conv2d(128, 1, use_bias=True, use_bn=False)(x)
    x = builder.conv2d(1, 1, use_bias=True, use_bn=False, activation='sigmoid')(x)

    network = keras.models.Model(inputs, x)
    return network, None


@tk.log.trace()
def _validate():
    """検証＆閾値決定。"""
    logger = tk.log.get(__name__)
    X, d, y = data.load_train_data()
    pred = predict_all('val', X, d)
    threshold = evaluation.log_evaluation(y, pred, print_fn=logger.info, search_th=True)
    (MODELS_DIR / 'threshold.txt').write_text(str(threshold))


@tk.log.trace()
def _predict():
    """予測。"""
    logger = tk.log.get(__name__)
    X_test, d_test = data.load_test_data()
    threshold = float((MODELS_DIR / 'threshold.txt').read_text())
    logger.info(f'threshold = {threshold:.3f}')
    pred_list = sum([predict_all('test', X_test, d_test, chilld_cv_index) for chilld_cv_index in range(5)], [])
    pred = np.mean(pred_list, axis=0) > threshold
    data.save_submission(MODELS_DIR / 'submission.csv', pred)


def predict_all(data_name, X, d, chilld_cv_index=None):
    """予測。"""
    if data_name == 'val':
        X_val = _get_meta_features(data_name, X, d)
        X_list, vi_list = [], []
        split_seed = int((MODELS_DIR / 'split_seed.txt').read_text())
        for cv_index in range(CV_COUNT):
            _, vi = tk.ml.cv_indices(X_val, None, cv_count=CV_COUNT, cv_index=cv_index, split_seed=split_seed, stratify=False)
            X_list.append(X_val[vi])
            vi_list.append(vi)
    else:
        X_test = _get_meta_features(data_name, X, d, chilld_cv_index)
        X_list = [X_test] * CV_COUNT

    gen = tk.generator.SimpleGenerator()
    model = tk.dl.models.Model.load(MODELS_DIR / f'model.fold0.h5', gen, batch_size=BATCH_SIZE, multi_gpu=True)

    pred_list = []
    for cv_index in tk.tqdm(range(CV_COUNT), desc='predict'):
        if cv_index != 0:
            model.load_weights(MODELS_DIR / f'model.fold{cv_index}.h5')

        X_t = X_list[cv_index]
        pred1 = model.predict(X_t, verbose=0)
        pred2 = model.predict(X_t[:, :, ::-1, :], verbose=0)[:, :, ::-1, :]
        pred = np.mean([pred1, pred2], axis=0)
        pred_list.append(pred)

    if data_name == 'val':
        pred = np.empty((len(X), 101, 101, 1), dtype=np.float32)
        for vi, p in zip(vi_list, pred_list):
            pred[vi] = p
    else:
        pred = pred_list
    return pred


def _get_meta_features(data_name, X, d, cv_index=None):
    """子モデルのout-of-fold predictionsを取得。"""
    import bin_nas
    import reg_nas
    import darknet53_bu  # 0.856
    import darknet53_bu_bin  # 0.851
    import darknet53_bu_nm  # 0.858
    import darknet53_elup1_nn  # 0.863
    import darknet53_hc_112_b  # 0.854
    import darknet53_in  # 0.854

    def _get(m):
        if data_name == 'val':
            return m
        else:
            assert len(m) == 5
            return m[cv_index]

    X = np.concatenate([
        X / 255,
        np.repeat(d, 101 * 101).reshape(len(X), 101, 101, 1),
        np.repeat(_get(bin_nas.predict_all(data_name, X, d)), 101 * 101).reshape(len(X), 101, 101, 1),
        np.repeat(_get(reg_nas.predict_all(data_name, X, d)), 101 * 101).reshape(len(X), 101, 101, 1),
        _get(darknet53_bu.predict_all(data_name, X, d)),
        _get(darknet53_bu_bin.predict_all(data_name, X, d)),
        _get(darknet53_bu_nm.predict_all(data_name, X, d)),
        _get(darknet53_elup1_nn.predict_all(data_name, X, d)),
        _get(darknet53_hc_112_b.predict_all(data_name, X, d)),
        _get(darknet53_in.predict_all(data_name, X, d)),
    ], axis=-1)
    return X


if __name__ == '__main__':
    _main()
