#!/usr/bin/env python3
import argparse
import pathlib

import numpy as np
import sklearn.externals.joblib as joblib

import bin
import pytoolkit as tk
from lib import data, evaluation

MODELS_DIR = pathlib.Path(f'models/model_{pathlib.Path(__file__).stem}')
REPORTS_DIR = pathlib.Path('reports')
SPLIT_SEED = 567
CV_COUNT = 5
INPUT_SIZE = (202, 202)


def _train():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('check', 'train', 'validate', 'predict'))
    parser.add_argument('--cv-index', default=0, choices=range(CV_COUNT), type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--ensemble', action='store_true', help='予測時にアンサンブルを行うのか否か。')
    args = parser.parse_args()
    if args.mode == 'check':
        _create_network()[0].summary()
    elif args.mode == 'train':
        with tk.dl.session(use_horovod=True):
            tk.log.init(MODELS_DIR / f'train.fold{args.cv_index}.log')
            _train_impl(args)
    elif args.mode == 'validate':
        tk.log.init(REPORTS_DIR / f'{MODELS_DIR.name}.txt')
        _report_impl()
    else:
        assert args.mode == 'predict'
        tk.log.init(MODELS_DIR / 'predict.log')
        _predict_impl(args)


@tk.log.trace()
def _train_impl(args):
    logger = tk.log.get(__name__)
    logger.info(f'args: {args}')
    X, d, y = data.load_train_data()
    mf = bin.load_oofp(X, y)
    y = data.load_mask(y)
    ti, vi = tk.ml.cv_indices(X, y, cv_count=CV_COUNT, cv_index=args.cv_index, split_seed=SPLIT_SEED, stratify=False)
    (X_train, y_train), (X_val, y_val) = ([X[ti], d[ti], mf[ti]], y[ti]), ([X[vi], d[vi], mf[vi]], y[vi])
    logger.info(f'cv_index={args.cv_index}: train={len(y_train)} val={len(y_val)}')

    network, lr_multipliers = _create_network()

    gen = tk.image.generator.Generator(multiple_input=True)
    gen.add(tk.image.LoadImage(grayscale=True), input_index=0)
    gen.add(tk.image.RandomFlipLR(probability=0.5, with_output=True), input_index=0)
    gen.add(tk.image.Padding(probability=1, with_output=True), input_index=0)
    gen.add(tk.image.RandomRotate(probability=0.25, with_output=True), input_index=0)
    gen.add(tk.image.RandomCrop(probability=1, with_output=True), input_index=0)
    gen.add(tk.image.Resize(INPUT_SIZE), input_index=0)
    gen.add(tk.generator.ProcessOutput(lambda y: tk.ndimage.resize(y, 101, 101)))

    model = tk.dl.models.Model(network, gen, batch_size=args.batch_size)
    model.compile(sgd_lr=0.5 / 256, loss='binary_crossentropy', metrics=[tk.dl.metrics.binary_accuracy], lr_multipliers=lr_multipliers)
    model.summary()
    model.plot(MODELS_DIR / 'model.svg', show_shapes=True)
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=args.epochs,
        tsv_log_path=MODELS_DIR / f'history.fold{args.cv_index}.tsv',
        cosine_annealing=True, mixup=True)
    model.save(MODELS_DIR / f'model.fold{args.cv_index}.h5')

    if tk.dl.hvd.is_master():
        pred_val = model.predict(X_val)
        joblib.dump(pred_val, MODELS_DIR / f'pred-val.fold{args.cv_index}.pkl')
        evaluation.log_evaluation(y_val, pred_val)


def _create_network():
    """ネットワークを作って返す。"""
    import keras
    builder = tk.dl.networks.Builder()

    inputs = [
        builder.input_tensor(INPUT_SIZE + (1,)),
        builder.input_tensor((1,)),  # depths
        builder.input_tensor((1,)),  # bin
    ]
    x = inputs[0]
    x = x_in = builder.preprocess(mode='div255')(x)
    x = keras.layers.concatenate([x, x, x])
    base_network = tk.applications.darknet53.darknet53(include_top=False, input_tensor=x)
    lr_multipliers = {l: 0.1 for l in base_network.layers}
    down_list = []
    down_list.append(x_in)  # stage 0: 202
    down_list.append(base_network.get_layer(name='add_1').output)  # stage 1: 101
    down_list.append(base_network.get_layer(name='add_3').output)  # stage 2: 50
    down_list.append(base_network.get_layer(name='add_11').output)  # stage 3: 25
    down_list.append(base_network.get_layer(name='add_19').output)  # stage 4: 12
    down_list.append(base_network.get_layer(name='add_23').output)  # stage 5: 6
    x = base_network.outputs[0]
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = builder.dense(64)(x)
    x = builder.act()(x)
    x = keras.layers.concatenate([x, inputs[1], inputs[2]])
    x = builder.dense(256)(x)
    x = builder.act()(x)
    gate = builder.dense(1, activation='sigmoid')(x)
    x = keras.layers.Reshape((1, 1, -1))(x)
    x = builder.conv2dtr(256, 3, strides=3)(x)

    for stage, (d, filters) in list(enumerate(zip(down_list, [16, 32, 64, 128, 256, 512])))[::-1][:5]:
        x = tk.dl.layers.subpixel_conv2d()(scale=2)(x)
        if stage in (3, 1):
            x = builder.conv2dtr(filters, 2, padding='valid')(x)
        x = builder.dwconv2d(5)(x)
        x = builder.conv2d(filters, 1, use_act=False)(x)
        d = builder.conv2d(filters, 1, use_act=False)(d)
        x = keras.layers.add([x, d])
        x = builder.res_block(filters, dropout=0.25)(x)
        x = builder.res_block(filters, dropout=0.25)(x)
        x = builder.bn_act()(x)
    x = builder.conv2d(1, use_bias=True, use_bn=False, activation='sigmoid')(x)
    x = keras.layers.multiply([x, gate])
    network = keras.models.Model(inputs, x)
    return network, lr_multipliers


@tk.log.trace()
def load_oofp(X, y):
    """out-of-fold predictionを読み込んで返す。"""
    pred = np.empty((len(y), 101, 101, 1), dtype=np.float32)
    for cv_index in range(CV_COUNT):
        _, vi = tk.ml.cv_indices(X, y, cv_count=CV_COUNT, cv_index=cv_index, split_seed=SPLIT_SEED, stratify=False)
        pred[vi] = joblib.load(MODELS_DIR / f'pred-val.fold{cv_index}.pkl')
    return pred


@tk.log.trace()
def predict(ensemble):
    """予測。"""
    X, d = data.load_test_data()
    mf_list = bin.predict(ensemble)
    pred_list = []
    for mf in mf_list:
        X = [X, d, mf]
        for cv_index in range(CV_COUNT):
            network = tk.dl.models.load_model(MODELS_DIR / f'model.fold{cv_index}.h5', compile=False)
            gen = tk.image.generator.Generator(multiple_input=True)
            gen.add(tk.image.LoadImage(grayscale=True), input_index=0)
            gen.add(tk.image.Resize(INPUT_SIZE), input_index=0)
            model = tk.dl.models.Model(network, gen, batch_size=32)
            pred = model.predict(X, verbose=1)
            pred_list.append(pred)
            if not ensemble:
                break
    return pred_list


@tk.log.trace()
def _report_impl():
    """検証＆閾値決定。"""
    logger = tk.log.get(__name__)
    X, _, y = data.load_train_data()
    y = data.load_mask(y)
    pred = load_oofp(X, y)
    threshold = evaluation.log_evaluation(y, pred, print_fn=logger.info)
    (MODELS_DIR / 'threshold.txt').write_text(str(threshold))


@tk.log.trace()
def _predict_impl(args):
    """予測。"""
    logger = tk.log.get(__name__)
    threshold = float((MODELS_DIR / 'threshold.txt').read_text())
    logger.info(f'threshold = {threshold:.3f}')
    pred_list = predict(ensemble=args.ensemble)
    pred = np.sum([p > threshold for p in pred_list], axis=0) > len(pred_list) / 2  # hard voting
    data.save_submission(MODELS_DIR / 'submission.csv', pred)


if __name__ == '__main__':
    _train()
