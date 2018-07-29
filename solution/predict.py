#!/usr/bin/env python3
import pathlib

import data
import pytoolkit as tk

MODELS_DIR = pathlib.Path('models')

THRESHOLD = 0.311


def _main():
    tk.better_exceptions()
    X, d = data.load_test_data()
    with tk.dl.session():
        tk.log.init(MODELS_DIR / 'predict.log')
        _run(X, d)


def _run(X, d):
    network = tk.dl.models.load_model(MODELS_DIR / 'model_1/model.fold0.h5', compile=False)  # TODO:

    import model_3
    mf = model_3.load_oofp(X, y)
    X = [X, d, mf]

    gen = tk.image.generator.Generator(multiple_input=True)
    gen.add(tk.image.LoadImage(grayscale=True), input_index=0)
    gen.add(tk.image.Resize((256, 256)), input_index=0)

    model = tk.dl.models.Model(network, gen, batch_size=32)

    pred = model.predict(X, verbose=1)
    pred = [tk.ndimage.resize(p, 101, 101) for p in tk.tqdm(pred)]
    import utils
    pred = np.array([utils.apply_crf(tk.ndimage.load(x, grayscale=True), p) for x, p in zip(X, tk.tqdm(pred))])
    data.save_submission(MODELS_DIR / 'submission.csv', pred, THRESHOLD)


if __name__ == '__main__':
    _main()
