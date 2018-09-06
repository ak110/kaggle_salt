#!/usr/bin/env python3
import argparse
import pathlib

import numpy as np
import sklearn.externals.joblib as joblib
from lib import data, evaluation

import pytoolkit as tk

MODEL_NAME = pathlib.Path(__file__).stem
MODELS_DIR = pathlib.Path(f'models/{MODEL_NAME}')
REPORTS_DIR = pathlib.Path('reports')
CACHE_DIR = pathlib.Path('cache')
CV_COUNT = 5
INPUT_SIZE = (101, 101)
BATCH_SIZE = 32
EPOCHS = 300


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('check', 'train', 'validate', 'predict'))
    parser.add_argument('--cv-index', default=0, choices=range(CV_COUNT), type=int)
    args = parser.parse_args()
    if args.mode == 'check':
        _create_network(input_dims=3)[0].summary()  # input_dimsは適当
    elif args.mode == 'train':
        with tk.dl.session(use_horovod=True):
            tk.log.init(MODELS_DIR / f'train.fold{args.cv_index}.log')
            _train(args)
    elif args.mode == 'validate':
        tk.log.init(REPORTS_DIR / f'{MODEL_NAME}.txt')
        _validate()
    else:
        tk.log.init(MODELS_DIR / 'predict.log')
        _predict()


@tk.log.trace()
def _train(args):
    logger = tk.log.get(__name__)
    logger.info(f'args: {args}')

    split_seed = int(MODEL_NAME.encode('utf-8').hex(), 16) % 10000000
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / 'split_seed.txt').write_text(str(split_seed))

    X, d, y = data.load_train_data()
    X = np.array([tk.ndimage.load(x, grayscale=True) for x in tk.tqdm(X, desc='load')])
    X, X_bin = _get_meta_features(X, 'val')
    y = data.load_mask(y)
    ti, vi = tk.ml.cv_indices(X, y, cv_count=CV_COUNT, cv_index=args.cv_index, split_seed=split_seed, stratify=False)
    (X_train, y_train), (X_val, y_val) = ([X[ti], d[ti], X_bin[ti]], y[ti]), ([X[vi], d[vi], X_bin[vi]], y[vi])
    logger.info(f'cv_index={args.cv_index}: train={len(y_train)} val={len(y_val)}')

    network, _ = _create_network(input_dims=X.shape[-1])

    gen = tk.image.generator.Generator(multiple_input=True)
    gen.add(tk.image.LoadImage(grayscale=True), input_index=0)
    gen.add(tk.image.RandomFlipLR(probability=0.5, with_output=True), input_index=0)
    # gen.add(tk.image.Padding(probability=1, with_output=True), input_index=0)
    # gen.add(tk.image.RandomRotate(probability=0.25, with_output=True), input_index=0)
    # gen.add(tk.image.RandomCrop(probability=1, with_output=True), input_index=0)
    # gen.add(tk.image.Resize(INPUT_SIZE), input_index=0)
    # gen.add(tk.generator.ProcessOutput(lambda y: tk.ndimage.resize(y, 101, 101)))

    model = tk.dl.models.Model(network, gen, batch_size=BATCH_SIZE)
    model.compile(sgd_lr=0.5 / 256, loss='binary_crossentropy', metrics=[tk.dl.metrics.binary_accuracy])
    model.summary()
    model.plot(MODELS_DIR / 'model.svg', show_shapes=True)
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=EPOCHS,
        tsv_log_path=MODELS_DIR / f'history.fold{args.cv_index}.tsv',
        cosine_annealing=True, mixup=True)
    model.save(MODELS_DIR / f'model.fold{args.cv_index}.h5')

    if tk.dl.hvd.is_master():
        evaluation.log_evaluation(y_val, model.predict(X_val))


def _create_network(input_dims):
    """ネットワークを作って返す。"""
    import keras
    builder = tk.dl.networks.Builder()

    inputs = [
        builder.input_tensor(INPUT_SIZE + (input_dims,)),
        builder.input_tensor((1,)),  # depths
        builder.input_tensor((1,)),  # bin
    ]
    x = inputs[0]
    x = builder.preprocess()(x)
    t = keras.layers.concatenate([
        keras.layers.Reshape((101, 101, 1))(keras.layers.RepeatVector(101 * 101)(inputs[0])),
        keras.layers.Reshape((101, 101, 1))(keras.layers.RepeatVector(101 * 101)(inputs[1])),
    ])
    x = keras.layers.concatenate([x, t])
    x = builder.conv2d(128)(x)
    x = builder.se_block(128)(x)
    x = builder.conv2d(128)(x)
    x = builder.conv2d(128)(x)
    x = builder.conv2d(1, use_bias=True, use_bn=False, activation='sigmoid')(x)
    network = keras.models.Model(inputs, x)
    return network, None


@tk.log.trace()
def _validate():
    """検証＆閾値決定。"""
    logger = tk.log.get(__name__)
    _, _, y = data.load_train_data()
    y = data.load_mask(y)
    pred = predict_all('val')
    threshold = evaluation.log_evaluation(y, pred, print_fn=logger.info)
    (MODELS_DIR / 'threshold.txt').write_text(str(threshold))


@tk.log.trace()
def _predict():
    """予測。"""
    logger = tk.log.get(__name__)
    threshold = float((MODELS_DIR / 'threshold.txt').read_text())
    logger.info(f'threshold = {threshold:.3f}')
    pred_list = predict_all('test')
    pred = np.sum([p > threshold for p in pred_list], axis=0) > len(pred_list) / 2  # hard voting
    data.save_submission(MODELS_DIR / 'submission.csv', pred)


def predict_all(data_name):
    """予測。"""
    cache_path = CACHE_DIR / data_name / f'{MODEL_NAME}.pkl'
    if cache_path.is_file():
        return joblib.load(cache_path)

    if data_name == 'val':
        X_val, d_val, _ = data.load_train_data()
        X_val = np.array([tk.ndimage.load(x, grayscale=True) for x in tk.tqdm(X_val, desc='load')])
        X_val, X_bin = _get_meta_features(X_val, data_name)
        X_list, vi_list = [], []
        split_seed = int((MODELS_DIR / 'split_seed.txt').read_text())
        for cv_index in range(CV_COUNT):
            _, vi = tk.ml.cv_indices(X_val, None, cv_count=CV_COUNT, cv_index=cv_index, split_seed=split_seed, stratify=False)
            X_list.append([X_val[vi], d_val[vi], X_bin[vi]])
            vi_list.append(vi)
    else:
        X_test, d_test = data.load_test_data()
        X_test = np.array([tk.ndimage.load(x, grayscale=True) for x in tk.tqdm(X_test, desc='load')])
        X_test, X_bin = _get_meta_features(X_test, data_name)
        X_list = [[X_test, d_test, X_bin]] * CV_COUNT

    gen = tk.image.generator.Generator(multiple_input=True)
    gen.add(tk.image.LoadImage(grayscale=True), input_index=0)
    gen.add(tk.image.Resize(INPUT_SIZE), input_index=0)
    model = tk.dl.models.Model.load(MODELS_DIR / f'model.fold0.h5', gen, batch_size=BATCH_SIZE, multi_gpu=True)

    pred_list = []
    for cv_index in tk.tqdm(range(CV_COUNT), desc='predict'):
        if cv_index != 0:
            model.load_weights(MODELS_DIR / f'model.fold{cv_index}.h5')

        X, d, X_bin = X_list[cv_index]
        pred1 = model.predict([X, d, X_bin], verbose=0)
        pred2 = model.predict([X[:, :, ::-1, :], d, X_bin], verbose=0)[:, :, ::-1, :]
        pred = np.mean([pred1, pred2], axis=0)
        pred_list.append(pred)

    if data_name == 'val':
        pred = np.empty((len(X_val), 101, 101, 1), dtype=np.float32)
        for vi, p in zip(vi_list, pred_list):
            pred[vi] = p
    else:
        pred = pred_list

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pred, cache_path)
    return pred


def _get_meta_features(X, data_name):
    """子モデルのout-of-fold predictionsを取得。"""
    import bin as bin_model
    import darknet53
    import darknet53_128
    import darknet53_nr
    import fast
    import nasnet
    import resnet
    import vgg
    import yolo
    X_bin = bin_model.predict_all(data_name)
    X = np.concatenate([
        X,
        darknet53.predict_all(data_name),
        darknet53_128.predict_all(data_name),
        darknet53_nr.predict_all(data_name),
        fast.predict_all(data_name),
        nasnet.predict_all(data_name),
        resnet.predict_all(data_name),
        vgg.predict_all(data_name),
        yolo.predict_all(data_name),
    ], axis=-1)
    return X, X_bin


if __name__ == '__main__':
    _main()
