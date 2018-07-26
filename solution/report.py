#!/usr/bin/env python3
import argparse
import pathlib

import numpy as np
import sklearn.metrics
import sklearn.externals.joblib as joblib

import data
import evaluation
import model_1
import model_2
import model_3
import model_4
import pytoolkit as tk

MODELS_DIR = pathlib.Path('models')
REPORTS_DIR = pathlib.Path('reports')
MODELS = {m.__name__: m for m in [model_1, model_2, model_3, model_4]}


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('models', default='all', choices=('all',) + tuple(list(MODELS.keys())), nargs='*')
    args = parser.parse_args()
    tk.log.init(None)
    model_names = MODELS.keys() if args.models == ['all'] else args.models

    X, _, y = data.load_train_data()
    y = data.load_mask(y)

    for model_name in model_names:
        _report(model_name, X, y)


def _report(model_name, X, y):
    print(f'{"=" * 32} {model_name} {"=" * 32}')
    logger = tk.log.get(f'report_{model_name}')
    logger.addHandler(tk.log.file_handler(REPORTS_DIR / f'{model_name}.txt'))

    m = MODELS[model_name]
    if m.OUTPUT_TYPE == 'bin':
        y = np.max(y > 0.5, axis=(1, 2, 3)).astype(np.uint8)  # 0 or 1
        y = np.expand_dims(y, axis=-1)

    pred = np.empty(y.shape, dtype=np.float32)
    for cv_index in range(m.CV_COUNT):
        _, vi = tk.ml.cv_indices(X, y, cv_count=m.CV_COUNT, cv_index=cv_index, split_seed=m.SPLIT_SEED, stratify=False)
        pred[vi] = joblib.load(m.MODELS_DIR / f'pred-val.fold{cv_index}.h5')

    if m.OUTPUT_TYPE == 'bin':
        tk.ml.print_classification_metrics(y, pred, print_fn=logger.info)
        logger.info('roc curve:')
        for fpr, tpr, th in zip(*sklearn.metrics.roc_curve(y, pred)):
            logger.info(f'  threshold={th:.3f}: fpr={fpr:.3f} tpr={tpr:.3f}')
    else:
        evaluation.log_evaluation(y, pred, print_fn=logger.info)


if __name__ == '__main__':
    _main()
