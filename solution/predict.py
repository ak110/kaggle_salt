#!/usr/bin/env python3
import pathlib

import numpy as np

import data
import model_vgg as last_model
import pytoolkit as tk

MODELS_DIR = pathlib.Path('models')

ENSEMBLE = False
THRESHOLD = 0.268


def _main():
    tk.better_exceptions()
    with tk.dl.session():
        tk.log.init(MODELS_DIR / 'predict.log')
        _run()


@tk.log.trace()
def _run():
    pred_list = last_model.predict(ensemble=ENSEMBLE)
    pred = np.sum([p > THRESHOLD for p in pred_list], axis=0) > len(pred_list) / 2  # hard voting
    data.save_submission(MODELS_DIR / 'submission.csv', pred)


if __name__ == '__main__':
    _main()
