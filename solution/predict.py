import pathlib

import data
import pytoolkit as tk

MODELS_DIR = pathlib.Path('models')

THRESHOLD = 0.671


def _main():
    tk.better_exceptions()
    X, test_prefixes = data.load_test_data()

    with tk.dl.session():
        tk.log.init(MODELS_DIR / 'predict.log')
        _run(X, test_prefixes)


def _run(X, test_prefixes):
    network = tk.dl.models.load_model(MODELS_DIR / 'model.h5', compile=False)
    gen = tk.image.ImageDataGenerator()
    model = tk.dl.models.Model(network, gen, batch_size=32)
    pred = model.predict(X, verbose=1)

    data.save_submission(MODELS_DIR / 'submission.csv', pred, test_prefixes, THRESHOLD)


if __name__ == '__main__':
    _main()
