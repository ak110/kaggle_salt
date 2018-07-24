#!/usr/bin/env python3
import argparse
import pathlib

import numpy as np
import sklearn.externals.joblib as joblib

import data
import evaluation
import pytoolkit as tk

MODELS_DIR = pathlib.Path('models')
MODELS = {
    'model_1': {
        'cv_count': 5,
        'split_seed': 123,
    },
    'model_2': {
        'cv_count': 5,
        'split_seed': 234,
    },
}


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('models', default=['all'], choices=('all',) + tuple(list(MODELS.keys())), nargs='+')
    args = parser.parse_args()
    tk.log.init(None)
    model_names = MODELS.keys() if args.models == ['all'] else args.models
    for model_name in model_names:
        _report(args, model_name)


def _report(args, model_name):
    logger = tk.log.get(__name__)
    logger.info(f'{"=" * 32} {model_name} {"=" * 32}')

    X, _, y = data.load_train_data()

    pred = np.empty(y.shape)
    for cv_index in range(MODELS[model_name]['cv_count']):
        _, vi = tk.ml.cv_indices(X, y, cv_count=MODELS[model_name]['cv_count'], cv_index=cv_index, split_seed=MODELS[model_name]['split_seed'])
        pred[vi] = joblib.load(MODELS_DIR / model_name / f'pred-val.fold{cv_index}.h5')

    y = data.load_mask(y)
    evaluation.log_evaluation(y, pred)


if __name__ == '__main__':
    _main()
