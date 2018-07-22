#!/usr/bin/env python3
import pathlib

import numpy as np

import data
import pytoolkit as tk

MODELS_DIR = pathlib.Path('models')


def _main():
    tk.better_exceptions()
    X, y, _ = data.load_train_data()
    _, (X_val, y_val) = tk.ml.split(X, y, split_seed=123, validation_split=0.2)

    with tk.dl.session():
        tk.log.init(None)
        _run(X_val, y_val)


def _run(X_val, y_val):
    logger = tk.log.get(__name__)

    network = tk.dl.models.load_model(MODELS_DIR / 'model.h5', compile=False)
    gen = tk.image.ImageDataGenerator()
    model = tk.dl.models.Model(network, gen, batch_size=32)
    model.compile(sgd_lr=0.5 / 256, loss='binary_crossentropy', metrics=[tk.dl.metrics.mean_iou])

    logger = tk.log.get(__name__)
    # 検証
    loss, mean_iou = model.evaluate(X_val, y_val)
    logger.info(f'loss:     {loss:.4f}')
    logger.info(f'mean_iou: {mean_iou:.4f}')

    # 検証
    pred_val = model.predict(X_val)
    tk.ml.print_classification_metrics(np.ravel(y_val), np.ravel(pred_val))

    # 閾値の最適化
    threshold_list = np.linspace(0.25, 0.75, 20)
    score_list = []
    for th in tk.tqdm(threshold_list):
        score = data.compute_iou_metric(np.int32(y_val > 0.5), np.int32(pred_val > th))
        logger.info(f'threshold={th:.3f}: score={score:.3f}')
        score_list.append(score)
    best_index = np.argmax(score_list)
    logger.info(f'max score: {score_list[best_index]:.3f}')
    logger.info(f'threshold: {threshold_list[best_index]:.3f}')


if __name__ == '__main__':
    _main()
