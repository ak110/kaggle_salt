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
CACHE_DIR = pathlib.Path('cache')
CV_COUNT = 5
INPUT_SIZE = (101, 101)
BATCH_SIZE = 16
EPOCHS = 300


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('check', 'train', 'validate', 'predict'))
    parser.add_argument('--cv-index', default=0, choices=range(CV_COUNT), type=int)
    args = parser.parse_args()
    with tk.dl.session(use_horovod=args.mode == 'train'):
        if args.mode == 'check':
            _create_network()[0].summary()
        elif args.mode == 'train':
            tk.log.init(MODELS_DIR / f'train.fold{args.cv_index}.log')
            _train(args)
        elif args.mode == 'validate':
            tk.log.init(REPORTS_DIR / f'{MODEL_NAME}.txt', file_level='INFO')
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
    ti, vi = tk.ml.cv_indices(X, y, cv_count=CV_COUNT, cv_index=args.cv_index, split_seed=split_seed, stratify=False)
    (X_train, y_train), (X_val, y_val) = ([X[ti], d[ti]], y[ti]), ([X[vi], d[vi]], y[vi])
    logger.info(f'cv_index={args.cv_index}: train={len(y_train)} val={len(y_val)}')

    network, lr_multipliers = _create_network()

    gen = tk.generator.Generator(multiple_input=True)
    gen.add(tk.image.LoadImage(grayscale=True), input_index=0)
    gen.add(tk.image.RandomFlipLR(probability=0.5, with_output=True), input_index=0)
    gen.add(tk.image.Padding(probability=1, with_output=True), input_index=0)
    gen.add(tk.image.RandomRotate(probability=0.25, with_output=True), input_index=0)
    gen.add(tk.image.RandomCrop(probability=1, with_output=True), input_index=0)
    gen.add(tk.image.Resize(INPUT_SIZE), input_index=0)
    gen.add(tk.generator.ProcessOutput(lambda y: tk.ndimage.resize(y, 101, 101)))

    model = tk.dl.models.Model(network, gen, batch_size=BATCH_SIZE)
    model.compile(sgd_lr=0.1 / 128, loss=mixed_loss, metrics=[tk.dl.metrics.binary_accuracy], lr_multipliers=lr_multipliers)
    model.plot(MODELS_DIR / 'model.svg', show_shapes=True)
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=EPOCHS,
        tsv_log_path=MODELS_DIR / f'history.fold{args.cv_index}.tsv',
        cosine_annealing=True, mixup=True)
    model.save(MODELS_DIR / f'model.fold{args.cv_index}.h5')

    if tk.dl.hvd.is_master():
        evaluation.log_evaluation(y_val, model.predict(X_val))


def _create_network():
    """ネットワークを作って返す。"""
    import keras
    builder = tk.dl.networks.Builder()

    inputs = [
        builder.input_tensor(INPUT_SIZE + (1,)),
        builder.input_tensor((1,)),  # depths
    ]
    x = inputs[0]
    x = keras.layers.UpSampling2D()(x)  # 202
    x = x_in = tk.dl.layers.pad2d()(((11, 11), (11, 11)), mode='reflect')(x)  # 224
    x = keras.layers.concatenate([x, x, x])
    x = builder.preprocess(mode='caffe')(x)
    base_network = keras.applications.ResNet50(include_top=False, input_tensor=x, pooling='avg')
    lr_multipliers = {l: 0.1 for l in base_network.layers}
    down_list = []
    down_list.append(x_in)  # stage 0: 224
    down_list.append(base_network.get_layer(name='max_pooling2d_1').input)  # stage 1: 112
    down_list.append(base_network.get_layer(name='res3a_branch2a').input)  # stage 2: 55
    down_list.append(base_network.get_layer(name='res4a_branch2a').input)  # stage 3: 28
    down_list.append(base_network.get_layer(name='res5a_branch2a').input)  # stage 4: 14
    down_list.append(base_network.get_layer(name='activation_49').output)  # stage 5: 7

    x = base_network.outputs[0]
    x = builder.dense(64)(x)
    x = builder.act()(x)
    x = keras.layers.concatenate([x, inputs[1]])
    x = builder.dense(64)(x)
    x = builder.act()(x)
    x = keras.layers.Reshape((1, 1, -1))(x)

    for stage, (d, filters) in list(enumerate(zip(down_list, [16, 32, 64, 128, 256, 512])))[::-1]:
        if stage == len(down_list) - 1:
            x = builder.conv2dtr(32, 7, strides=7)(x)
        else:
            if stage == 1:
                x = builder.conv2dtr(filters // 4, 2, strides=1, padding='valid')(x)
            x = builder.conv2dtr(filters // 4, 3, strides=2)(x)
            if stage == 2:
                x = builder.dwconv2d(2, padding='valid')(x)
        x = builder.conv2d(filters, 1, use_act=False)(x)
        d = builder.conv2d(filters, 1, use_act=False)(d)
        x = keras.layers.add([x, d])
        x = builder.res_block(filters, dropout=0.25)(x)
        x = builder.res_block(filters, dropout=0.25)(x)
        x = builder.bn_act()(x)
        x = builder.scse_block(filters)(x)

    x = keras.layers.Cropping2D(((11, 11), (11, 11)))(x)  # 202
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.AveragePooling2D()(x)  # 101
    x = builder.conv2d(1, use_bias=True, use_bn=False, activation='sigmoid')(x)
    network = keras.models.Model(inputs, x)
    return network, lr_multipliers


@tk.log.trace()
def _validate():
    """検証＆閾値決定。"""
    logger = tk.log.get(__name__)
    X, d, y = data.load_train_data()
    pred = predict_all('val', X, d)
    threshold = evaluation.log_evaluation(y, pred, print_fn=logger.info)
    (MODELS_DIR / 'threshold.txt').write_text(str(threshold))


@tk.log.trace()
def _predict():
    """予測。"""
    logger = tk.log.get(__name__)
    threshold = float((MODELS_DIR / 'threshold.txt').read_text())
    logger.info(f'threshold = {threshold:.3f}')
    X_test, d_test = data.load_test_data()
    pred_list = predict_all('test', X_test, d_test)
    pred = np.sum([p > threshold for p in pred_list], axis=0) > len(pred_list) / 2  # hard voting
    data.save_submission(MODELS_DIR / 'submission.csv', pred)


def predict_all(data_name, X, d):
    """予測。"""
    cache_path = CACHE_DIR / data_name / f'{MODEL_NAME}.pkl'
    if cache_path.is_file():
        return joblib.load(cache_path)

    if data_name == 'val':
        X_list, vi_list = [], []
        split_seed = int((MODELS_DIR / 'split_seed.txt').read_text())
        for cv_index in range(CV_COUNT):
            _, vi = tk.ml.cv_indices(X, None, cv_count=CV_COUNT, cv_index=cv_index, split_seed=split_seed, stratify=False)
            X_list.append([X[vi], d[vi]])
            vi_list.append(vi)
    else:
        X, d = data.load_test_data()
        X_list = [[X, d]] * CV_COUNT

    gen = tk.generator.SimpleGenerator()
    model = tk.dl.models.Model.load(MODELS_DIR / f'model.fold0.h5', gen, batch_size=BATCH_SIZE, multi_gpu=True)

    pred_list = []
    for cv_index in tk.tqdm(range(CV_COUNT), desc='predict'):
        if cv_index != 0:
            model.load_weights(MODELS_DIR / f'model.fold{cv_index}.h5')

        X_t, d_t = X_list[cv_index]
        pred1 = model.predict([X_t, d_t], verbose=0)
        pred2 = model.predict([X_t[:, :, ::-1, :], d_t], verbose=0)[:, :, ::-1, :]
        pred = np.mean([pred1, pred2], axis=0)
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


def mixed_loss(y_true, y_pred):
    """BCE+Lovasz hinge"""
    import keras.backend as K
    loss1 = K.binary_crossentropy(y_true, y_pred)
    loss2 = lovasz_hinge(y_true, y_pred)
    return (loss1 + loss2) / 2


def lovasz_hinge(y_true, y_pred):
    """Binary Lovasz hinge loss"""
    import keras.backend as K
    from lovasz_softmax import lovasz_losses_tf
    logit = K.log(y_pred / (1 - y_pred + K.epsilon()))
    return lovasz_losses_tf.lovasz_hinge(logit, y_true)


if __name__ == '__main__':
    _main()
