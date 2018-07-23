#!/usr/bin/env python3
import pathlib

import data
import evaluation
import pytoolkit as tk

MODELS_DIR = pathlib.Path('models')


def _main():
    tk.better_exceptions()
    _, (X_val, y_val) = data.load_train_data()
    with tk.dl.session():
        tk.log.init(None)
        _run(X_val, y_val)


def _run(X_val, y_val):
    network = tk.dl.models.load_model(MODELS_DIR / 'model.h5', compile=False)
    gen = tk.image.ImageDataGenerator()
    model = tk.dl.models.Model(network, gen, batch_size=32)

    pred_val = model.predict(X_val)
    evaluation.log_evaluation(y_val, pred_val)


if __name__ == '__main__':
    _main()
