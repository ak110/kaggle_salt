#!/usr/bin/env python3
import argparse
import pathlib

import numpy as np
import sklearn.ensemble
import sklearn.externals.joblib as joblib
from lib import data, evaluation

import pytoolkit as tk

MODEL_NAME = pathlib.Path(__file__).stem
MODELS_DIR = pathlib.Path(f'models/{MODEL_NAME}')
REPORTS_DIR = pathlib.Path('reports')
CV_COUNT = 5
INPUT_SIZE = (101, 101)
BATCH_SIZE = 64
EPOCHS = 20


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('all', 'train', 'validate', 'predict'), default='all', nargs='?')
    parser.add_argument('--cv-index', default=0, choices=range(CV_COUNT), type=int)
    args = parser.parse_args()
    if args.mode == 'all':
        tk.log.init(MODELS_DIR / f'train.log')
        for cv_index in range(CV_COUNT):
            args.cv_index = cv_index
            _train(args)
        _validate()
        _predict()
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

    model = sklearn.ensemble.RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=3)
    model.fit(X_train.reshape(-1, X.shape[-1]), y_train.flatten())
    joblib.dump(model, MODELS_DIR / f'model.fold{args.cv_index}.pkl')

    if tk.dl.hvd.is_master():
        pred_val = model.predict_proba(X_val.reshape(-1, X.shape[-1])).reshape(-1, 101, 101, 1)
        evaluation.log_evaluation(y_val, pred_val)


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

    model = joblib.load(MODELS_DIR / f'model.fold0.pkl')

    pred_list = []
    for cv_index in tk.tqdm(range(CV_COUNT), desc='predict'):
        if cv_index != 0:
            model.load_weights(MODELS_DIR / f'model.fold{cv_index}.h5')

        X_t = X_list[cv_index]
        pred = model.predict_proba(X_t)
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
    import darknet53_bu_nm  # 0.858
    import darknet53_elup1_nn  # 0.863

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
        _get(darknet53_bu_nm.predict_all(data_name, X, d)),
        _get(darknet53_elup1_nn.predict_all(data_name, X, d)),
    ], axis=-1)
    return X


if __name__ == '__main__':
    _main()
