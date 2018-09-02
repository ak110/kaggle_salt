#!/usr/bin/env python3
import argparse
import pathlib

import numpy as np
import sklearn.externals.joblib as joblib

import pytoolkit as tk
from lib import data, evaluation

MODEL_NAME = pathlib.Path(__file__).stem
MODELS_DIR = pathlib.Path(f'models/{MODEL_NAME}')
REPORTS_DIR = pathlib.Path('reports')
SPLIT_SEED = int(MODEL_NAME.encode('utf-8').hex(), 16) % 10000000
CV_COUNT = 5
INPUT_SIZE = (227, 227)


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('check', 'train', 'validate', 'predict'))
    parser.add_argument('--cv-index', default=0, choices=range(CV_COUNT), type=int)
    parser.add_argument('--batch-size', default=12, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--ensemble', action='store_true', help='予測時にアンサンブルを行うのか否か。')
    args = parser.parse_args()
    if args.mode == 'check':
        _create_network()[0].summary()
    elif args.mode == 'train':
        with tk.dl.session(use_horovod=True):
            tk.log.init(MODELS_DIR / f'train.fold{args.cv_index}.log')
            _train(args)
    elif args.mode == 'validate':
        tk.log.init(REPORTS_DIR / f'{MODEL_NAME}.txt')
        _validate()
    else:
        assert args.mode == 'predict'
        tk.log.init(MODELS_DIR / 'predict.log')
        _predict(args)


@tk.log.trace()
def _train(args):
    logger = tk.log.get(__name__)
    logger.info(f'args: {args}')
    X, d, y = data.load_train_data()
    y = data.load_mask(y)
    ti, vi = tk.ml.cv_indices(X, y, cv_count=CV_COUNT, cv_index=args.cv_index, split_seed=SPLIT_SEED, stratify=False)
    (X_train, y_train), (X_val, y_val) = ([X[ti], d[ti]], y[ti]), ([X[vi], d[vi]], y[vi])
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
    ]
    x = inputs[0]
    x = x_in = builder.preprocess(mode='tf')(x)
    x = keras.layers.concatenate([x, x, x])
    base_network = keras.applications.NASNetLarge(include_top=False, input_tensor=x)
    lr_multipliers = {l: 0.1 for l in base_network.layers}
    down_list = []
    down_list.append(x_in)  # stage 0: 227
    down_list.append(base_network.get_layer(name='activation_4').output)  # stage 1: 113
    down_list.append(base_network.get_layer(name='activation_12').output)  # stage 2: 57
    down_list.append(base_network.get_layer(name='activation_95').output)  # stage 3: 29
    down_list.append(base_network.get_layer(name='activation_178').output)  # stage 4: 15
    down_list.append(base_network.get_layer(name='activation_260').output)  # stage 5: 8
    x = base_network.outputs[0]
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = builder.dense(64)(x)
    x = builder.act()(x)
    x = keras.layers.concatenate([x, inputs[1], inputs[2]])
    x = builder.dense(256)(x)
    x = builder.act()(x)
    x = keras.layers.Reshape((1, 1, -1))(x)
    x = builder.conv2dtr(256, 4, strides=4)(x)

    for stage, (d, filters) in list(enumerate(zip(down_list, [16, 32, 64, 128, 256, 512])))[::-1]:
        x = builder.conv2dtr(filters // 4, 3, strides=2, padding='valid' if stage == 0 else 'same')(x)
        if stage in (4, 3, 2, 1):
            x = builder.dwconv2d(2, padding='valid')(x)
        x = builder.conv2d(filters, 1, use_act=False)(x)
        d = builder.conv2d(filters, 1, use_act=False)(d)
        x = keras.layers.add([x, d])
        x = builder.res_block(filters, dropout=0.25)(x)
        x = builder.res_block(filters, dropout=0.25)(x)
        x = builder.bn_act()(x)
    x = tk.dl.layers.resize2d()((101, 101))(x)
    x = builder.conv2d(64, use_act=False)(x)
    x = builder.res_block(64, dropout=0.25)(x)
    x = builder.res_block(64, dropout=0.25)(x)
    x = builder.bn_act()(x)
    x = builder.conv2d(1, use_bias=True, use_bn=False, activation='sigmoid')(x)
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
    pred_list = []
    for cv_index in range(CV_COUNT):
        network = tk.dl.models.load_model(MODELS_DIR / f'model.fold{cv_index}.h5', compile=False)
        gen = tk.image.generator.Generator(multiple_input=True)
        gen.add(tk.image.LoadImage(grayscale=True), input_index=0)
        gen.add(tk.image.Resize(INPUT_SIZE), input_index=0)
        model = tk.dl.models.Model(network, gen, batch_size=32)
        pred = model.predict([X, d], verbose=1)
        pred_list.append(pred)
        if not ensemble:
            break
    return pred_list


@tk.log.trace()
def _validate():
    """検証＆閾値決定。"""
    logger = tk.log.get(__name__)
    X, _, y = data.load_train_data()
    y = data.load_mask(y)
    pred = load_oofp(X, y)
    threshold = evaluation.log_evaluation(y, pred, print_fn=logger.info)
    (MODELS_DIR / 'threshold.txt').write_text(str(threshold))


@tk.log.trace()
def _predict(args):
    """予測。"""
    logger = tk.log.get(__name__)
    threshold = float((MODELS_DIR / 'threshold.txt').read_text())
    logger.info(f'threshold = {threshold:.3f}')
    pred_list = predict(ensemble=args.ensemble)
    pred = np.sum([p > threshold for p in pred_list], axis=0) > len(pred_list) / 2  # hard voting
    data.save_submission(MODELS_DIR / 'submission.csv', pred)


if __name__ == '__main__':
    _main()
