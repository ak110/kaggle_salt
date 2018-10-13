#!/usr/bin/env python3
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
    tk.log.init(REPORTS_DIR / f'{MODEL_NAME}.txt', file_level='INFO')
    threshold = _validate()
    _predict(threshold)


def _validate():
    """検証"""
    logger = tk.log.get(__name__)
    X_train, y_train = _data.load_train_data()
    pred = predict_all('val', X_train)
    threshold = _evaluation.log_evaluation(y_train, pred, print_fn=logger.info, search_th=True)
    return threshold


def _predict(threshold):
    """予測"""
    X_test = _data.load_test_data()
    pred_list = predict_all('test', X_test)
    pred = np.mean(pred_list, axis=0)
    _data.save_submission(MODELS_DIR / 'submission.csv', pred > threshold)
    _data.save_submission(MODELS_DIR / 'submission_0.45.csv', pred > 0.45)
    _data.save_submission(MODELS_DIR / 'submission_0.50.csv', pred > 0.50)
    _data.save_submission(MODELS_DIR / 'submission_0.55.csv', pred > 0.55)


@tk.log.trace()
def predict_all(data_name, X):
    import stack_3x3
    import stack_dense
    import stack_res
    import stack_unet
    return np.average([
        stack_3x3.predict_all(data_name, X, use_cache=True),
        stack_dense.predict_all(data_name, X, use_cache=True),
        stack_res.predict_all(data_name, X, use_cache=True),
        stack_unet.predict_all(data_name, X, use_cache=True),
    ], weights=[1, 1, 1, 1])


if __name__ == '__main__':
    _main()
