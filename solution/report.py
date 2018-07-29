#!/usr/bin/env python3
import argparse
import pathlib

import numpy as np
import sklearn.metrics

import data
import evaluation
import model_bin
import model_large
import model_notr_bn
import model_small
import pytoolkit as tk

MODELS_DIR = pathlib.Path('models')
REPORTS_DIR = pathlib.Path('reports')
MODELS = {m.__name__: m for m in [model_large, model_small, model_bin, model_notr_bn]}


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

    pred = m.load_oofp(X, y)

    if m.OUTPUT_TYPE == 'bin':
        tk.ml.print_classification_metrics(y, pred, print_fn=logger.info)
        logger.info('roc curve:')
        for fpr, tpr, th in zip(*sklearn.metrics.roc_curve(y, pred)):
            logger.info(f'  threshold={th:.3f}: fpr={fpr:.3f} tpr={tpr:.3f}')
    else:
        evaluation.log_evaluation(y, pred, print_fn=logger.info)


if __name__ == '__main__':
    _main()
