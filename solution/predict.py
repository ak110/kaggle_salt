#!/usr/bin/env python3
import pathlib

import data
import model_1
import pytoolkit as tk

MODELS_DIR = pathlib.Path('models')

THRESHOLD = 0.311


def _main():
    tk.better_exceptions()
    with tk.dl.session():
        tk.log.init(MODELS_DIR / 'predict.log')

        pred = model_1.predict(ensemble=False)[0]  # TODO: ensemble
        data.save_submission(MODELS_DIR / 'submission.csv', pred, THRESHOLD)


if __name__ == '__main__':
    _main()
