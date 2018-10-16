#!/usr/bin/env python3
import argparse
import pathlib

import numpy as np

import _data
import _evaluation
import pytoolkit as tk

MODEL_NAME = pathlib.Path(__file__).stem
MODELS_DIR = pathlib.Path(f'models/{MODEL_NAME}')
REPORTS_DIR = pathlib.Path('reports')


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('validate', 'predict', 'all'), nargs='?', default='all')
    args = parser.parse_args()
    if args.mode == 'validate':
        tk.log.init(REPORTS_DIR / f'{MODEL_NAME}.txt', file_level='INFO')
        _validate()
    elif args.mode == 'predict':
        tk.log.init(None)
        _predict()
    else:
        tk.log.init(REPORTS_DIR / f'{MODEL_NAME}.txt', file_level='INFO')
        _validate()
        _predict()


def _validate():
    """検証。"""
    logger = tk.log.get(__name__)
    X_train, y_train = _data.load_train_data()
    pred = predict_all('val', X_train)
    threshold = _evaluation.log_evaluation(y_train, pred, print_fn=logger.info, search_th=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / 'threshold.txt').write_text(str(threshold))


def _predict():
    """予測。"""
    logger = tk.log.get(__name__)
    X_test = _data.load_test_data()
    threshold = float((MODELS_DIR / 'threshold.txt').read_text())
    logger.info(f'threshold = {threshold:.3f}')
    pred = predict_all('test', X_test)
    _data.save_submission(MODELS_DIR / 'submission.csv', pred > threshold)
    _data.save_submission(MODELS_DIR / 'submission_0.45.csv', pred > 0.45)
    _data.save_submission(MODELS_DIR / 'submission_0.50.csv', pred > 0.50)
    _data.save_submission(MODELS_DIR / 'submission_0.55.csv', pred > 0.55)


@tk.log.trace()
def predict_all(data_name, X):
    import stack_dense
    import stack_pad
    import stack_res
    def _get(pred):
        return pred if data_name == 'val' else np.mean(pred, axis=0)
    return np.average([
        _get(stack_dense.predict_all(data_name, X, use_cache=True)),
        _get(stack_pad.predict_all(data_name, X, use_cache=True)),
        _get(stack_res.predict_all(data_name, X, use_cache=True)),
    ], weights=[1.0, 1.5, 1.3], axis=0)


if __name__ == '__main__':
    _main()
