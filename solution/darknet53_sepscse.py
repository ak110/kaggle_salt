#!/usr/bin/env python3
import argparse
import pathlib

import numpy as np
import sklearn.externals.joblib as joblib

import _data
import _evaluation
import pytoolkit as tk

MODEL_NAME = pathlib.Path(__file__).stem
MODELS_DIR = pathlib.Path(f'models/{MODEL_NAME}')
REPORTS_DIR = pathlib.Path('reports')
CACHE_DIR = pathlib.Path('cache')
CV_COUNT = 5
INPUT_SIZE = (101, 101)
BATCH_SIZE = 16
EPOCHS = 300


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('check', 'train', 'fine', 'validate', 'predict'))
    parser.add_argument('--cv-index', default=0, choices=range(CV_COUNT), type=int)
    args = parser.parse_args()
    with tk.dl.session(use_horovod=args.mode in ('train', 'fine')):
        if args.mode == 'check':
            _create_network()[0].summary()
        elif args.mode == 'train':
            tk.log.init(MODELS_DIR / f'train.fold{args.cv_index}.log')
            _train(args)
        elif args.mode == 'fine':
            tk.log.init(MODELS_DIR / f'fine.fold{args.cv_index}.log')
            _train(args, fine=True)
        elif args.mode == 'validate':
            tk.log.init(REPORTS_DIR / f'{MODEL_NAME}.txt', file_level='INFO')
            _validate()
        elif args.mode == 'predict':
            tk.log.init(MODELS_DIR / 'predict.log')
            _predict()


@tk.log.trace()
def _train(args, fine=False):
    logger = tk.log.get(__name__)
    logger.info(f'args: {args}')

    split_seed = int(MODEL_NAME.encode('utf-8').hex(), 16) % 10000000
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / 'split_seed.txt').write_text(str(split_seed))

    X, y = _data.load_train_data()
    ti, vi = tk.ml.cv_indices(X, y, cv_count=CV_COUNT, cv_index=args.cv_index, split_seed=split_seed, stratify=False)
    (X_train, y_train), (X_val, y_val) = (X[ti], y[ti]), (X[vi], y[vi])
    logger.info(f'cv_index={args.cv_index}: train={len(y_train)} val={len(y_val)}')

    network, lr_multipliers = _create_network()

    gen = tk.generator.Generator()
    if fine:
        pseudo_size = len(y_train) // 2
        X_train = [np.array(list(X_train[0]) + [None] * pseudo_size), np.array(list(X_train[1]) + [None] * pseudo_size)]
        y_train = np.array(list(y_train) + [None] * pseudo_size)
        X_test = _data.load_test_data()
        _, pi = tk.ml.cv_indices(X_test, np.zeros((len(X_test),)), cv_count=CV_COUNT, cv_index=args.cv_index, split_seed=split_seed, stratify=False)
        pred_test = predict_all('test', X_test, use_cache=True)[(args.cv_index + 1) % CV_COUNT]  # pseudo-labeling
        gen.add(tk.generator.RandomPickData(X_test[pi], pred_test[pi]))
        lr_multipliers = None
    gen.add(tk.image.RandomFlipLR(probability=0.5, with_output=True))
    gen.add(tk.image.Padding(probability=1, with_output=True))
    gen.add(tk.image.RandomRotate(probability=0.25, with_output=True))
    gen.add(tk.image.RandomCrop(probability=1, with_output=True))
    gen.add(tk.image.RandomAugmentors([
        tk.image.RandomBlur(probability=0.125),
        tk.image.RandomUnsharpMask(probability=0.125),
        tk.image.RandomBrightness(probability=0.25),
        tk.image.RandomContrast(probability=0.25),
    ], probability=0.125))
    gen.add(tk.image.Resize((101, 101), with_output=True))

    model = tk.dl.models.Model(network, gen, batch_size=BATCH_SIZE)
    if fine:
        model.load_weights(MODELS_DIR / f'model.fold{args.cv_index}.h5')
    model.compile(sgd_lr=0.01 / 128 if fine else 0.1 / 128, loss=tk.dl.losses.lovasz_hinge_elup1,
                  metrics=[tk.dl.metrics.binary_accuracy], lr_multipliers=lr_multipliers, clipnorm=10.0)
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=EPOCHS // 3 if fine else EPOCHS,
        cosine_annealing=True, mixup=False)
    model.save(MODELS_DIR / f'model.fold{args.cv_index}.h5', include_optimizer=False)

    if tk.dl.hvd.is_master():
        _evaluation.log_evaluation(y_val, model.predict(X_val))


def _create_network():
    """ネットワークを作って返す。"""
    import keras
    builder = tk.dl.networks.Builder()

    inputs = [
        builder.input_tensor(INPUT_SIZE + (1,)),
    ]
    x = inputs[0]
    x = builder.preprocess(mode='div255')(x)
    x = tk.dl.layers.pad2d()(((5, 6), (5, 6)), mode='reflect')(x)  # 112
    x = keras.layers.concatenate([x, x, x])
    base_network = tk.applications.darknet53.darknet53(include_top=False, input_tensor=x, for_small=True)
    lr_multipliers = {l: 0.1 for l in base_network.layers}
    down_list = []
    down_list.append(base_network.get_layer(name='add_1').output)  # stage 1: 112
    down_list.append(base_network.get_layer(name='add_3').output)  # stage 2: 56
    down_list.append(base_network.get_layer(name='add_11').output)  # stage 3: 28
    down_list.append(base_network.get_layer(name='add_19').output)  # stage 4: 14
    down_list.append(base_network.get_layer(name='add_23').output)  # stage 5: 7

    x = base_network.outputs[0]
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = builder.dense(256)(x)
    x = builder.act()(x)
    x = builder.dense(7 * 7 * 32)(x)
    x = builder.act()(x)
    x = keras.layers.Reshape((1, 1, -1))(x)

    up_list = []
    for stage, (d, filters) in list(enumerate(zip(down_list, [32, 64, 128, 256, 512])))[::-1]:
        x = tk.dl.layers.subpixel_conv2d()(scale=2 if stage != 4 else 7)(x)
        x = tk.dl.layers.coord_channel_2d()(x_channel=False)(x)
        d = builder.scse_block(builder.shape(d)[-1])(d)
        x = builder.conv2d(filters, 1, use_act=False)(x)
        d = builder.conv2d(filters, 1, use_act=False)(d)
        x = keras.layers.add([x, d])
        x = builder.res_block(filters)(x)
        x = builder.res_block(filters)(x)
        x = builder.bn_act()(x)
        up_list.append(builder.conv2d(32, 1)(x))

    x = keras.layers.concatenate([
        tk.dl.layers.resize2d()((112, 112))(up_list[0]),
        tk.dl.layers.resize2d()((112, 112))(up_list[1]),
        tk.dl.layers.resize2d()((112, 112))(up_list[2]),
        tk.dl.layers.resize2d()((112, 112))(up_list[3]),
        up_list[4],
    ])  # 112
    x = tk.dl.layers.coord_channel_2d()(x_channel=False)(x)

    x = builder.conv2d(64, use_act=False)(x)
    x = builder.res_block(64)(x)
    x = builder.res_block(64)(x)
    x = builder.res_block(64)(x)
    x = builder.bn_act()(x)

    x = keras.layers.Cropping2D(((5, 6), (5, 6)))(x)  # 101
    x = builder.conv2d(1, use_bias=True, use_bn=False, activation='sigmoid')(x)
    network = keras.models.Model(inputs, x)
    return network, lr_multipliers


@tk.log.trace()
def _validate():
    """検証＆閾値決定。"""
    logger = tk.log.get(__name__)
    X, y = _data.load_train_data()
    pred = predict_all('val', X)
    threshold = _evaluation.log_evaluation(y, pred, print_fn=logger.info, search_th=True)
    (MODELS_DIR / 'threshold.txt').write_text(str(threshold))


@tk.log.trace()
def _predict():
    """予測。"""
    logger = tk.log.get(__name__)
    X_test = _data.load_test_data()
    threshold = float((MODELS_DIR / 'threshold.txt').read_text())
    logger.info(f'threshold = {threshold:.3f}')
    pred_list = predict_all('test', X_test)
    pred = np.mean(pred_list, axis=0) > threshold
    _data.save_submission(MODELS_DIR / 'submission.csv', pred)


def predict_all(data_name, X, use_cache=False):
    """予測。"""
    cache_path = CACHE_DIR / data_name / f'{MODEL_NAME}.pkl'
    if use_cache and cache_path.is_file():
        return joblib.load(cache_path)

    if data_name == 'val':
        X_list, vi_list = [], []
        split_seed = int((MODELS_DIR / 'split_seed.txt').read_text())
        for cv_index in range(CV_COUNT):
            _, vi = tk.ml.cv_indices(X, None, cv_count=CV_COUNT, cv_index=cv_index, split_seed=split_seed, stratify=False)
            X_list.append(X[vi])
            vi_list.append(vi)
    else:
        X_list = [X] * CV_COUNT

    gen = tk.generator.SimpleGenerator()
    model = tk.dl.models.Model.load(MODELS_DIR / f'model.fold0.h5', gen, batch_size=BATCH_SIZE, multi_gpu=True)

    pred_list = []
    for cv_index in tk.tqdm(range(CV_COUNT), desc='predict'):
        if cv_index != 0:
            model.load_weights(MODELS_DIR / f'model.fold{cv_index}.h5')

        X_t, d_t = X_list[cv_index]
        pred = _evaluation.predict_tta(model, X_t, d_t)
        pred_list.append(pred)

    if data_name == 'val':
        pred = np.empty((len(X), 101, 101, 1), dtype=np.float32)
        for vi, p in zip(vi_list, pred_list):
            pred[vi] = p
    else:
        pred = pred_list

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pred, cache_path, compress=3)
    return pred


if __name__ == '__main__':
    _main()
