#!/usr/bin/env python3
"""late submit用ベースライン。"""
import argparse
import pathlib

import numpy as np
import sklearn.externals.joblib as joblib

import _data
import _evaluation
import pytoolkit as tk

MODEL_NAME = pathlib.Path(__file__).stem
MODELS_DIR = pathlib.Path(f'models/{MODEL_NAME}')
CACHE_DIR = pathlib.Path('cache')
CV_COUNT = 5
INPUT_SIZE = (101, 101)
BATCH_SIZE = 16
EPOCHS = 300


def _main():
    tk.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('check', 'train', 'fine', 'validate', 'predict'))
    parser.add_argument('--cv-index', default=0, choices=[0], type=int)
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
            tk.log.init(MODELS_DIR / 'validate.log')
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
        X_train = np.array(list(X_train) + [None] * pseudo_size)
        y_train = np.array(list(y_train) + [None] * pseudo_size)
        X_test = _data.load_test_data()
        _, pi = tk.ml.cv_indices(X_test, np.zeros((len(X_test),)), cv_count=CV_COUNT, cv_index=args.cv_index, split_seed=split_seed, stratify=False)
        #pred_test = predict_all('test', None, use_cache=True)[(args.cv_index + 1) % CV_COUNT]  # cross-pseudo-labeling
        import stack_res
        pred_test = stack_res.predict_all('test', None, use_cache=True)[(args.cv_index + 1) % CV_COUNT]  # cross-pseudo-labeling
        gen.add(tk.generator.RandomPickData(X_test[pi], pred_test[pi]))
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
    model.compile(optimizer='adadelta', loss=lovasz_hinge,
                  metrics=[tk.dl.metrics.binary_accuracy]) # , lr_multipliers=lr_multipliers
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=EPOCHS // 3 if fine else EPOCHS,
        cosine_annealing=False, mixup=False)
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
        x = builder.conv2d(filters, 1, use_act=False)(x)
        d = tk.dl.layers.coord_channel_2d()(x_channel=False)(d)
        d = builder.conv2d(filters, 1, use_act=False)(d)
        x = keras.layers.add([x, d])
        x = builder.res_block(filters)(x)
        x = builder.res_block(filters)(x)
        x = builder.bn_act()(x)
        x = builder.scse_block(filters)(x)
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
    split_seed = int((MODELS_DIR / 'split_seed.txt').read_text())
    ti, vi = tk.ml.cv_indices(X, y, cv_count=CV_COUNT, cv_index=0, split_seed=split_seed, stratify=False)
    X_val, y_val = X[vi], y[vi]
    pred_val = predict_all('val', X_val)
    threshold = _evaluation.log_evaluation(y_val, pred_val, print_fn=logger.info, search_th=True)
    (MODELS_DIR / 'threshold.txt').write_text(str(threshold))


@tk.log.trace()
def _predict():
    """予測。"""
    logger = tk.log.get(__name__)
    X_test = _data.load_test_data()
    threshold = float((MODELS_DIR / 'threshold.txt').read_text())
    logger.info(f'threshold = {threshold:.3f}')
    pred = predict_all('test', X_test) > threshold
    _data.save_submission(MODELS_DIR / 'submission.csv', pred)


def predict_all(data_name, X, use_cache=False):
    """予測。"""
    gen = tk.generator.SimpleGenerator()
    model = tk.dl.models.Model.load(MODELS_DIR / f'model.fold0.h5', gen, batch_size=BATCH_SIZE, multi_gpu=True)
    pred = _evaluation.predict_tta(model, X)
    return pred

import tensorflow

def lovasz_hinge(y_true, y_pred):
    """Lovasz hinge loss"""
    K = tensorflow.keras.backend
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    logits = K.log(y_pred / (1 - y_pred))
    def loss_per_image(t):
        log, lab = t
        log = K.reshape(log, (-1,))
        lab = K.reshape(lab, (-1,))
        lab = K.cast(lab, log.dtype)  # logits.dtypeにしてたかも (どちらでも同じ)
        signs = 2. * lab - 1.
        errors = 1. - log * tensorflow.stop_gradient(signs)
        errors_sorted, perm = tensorflow.nn.top_k(errors, k=K.shape(errors)[0], name="sort")
        gt_sorted = tensorflow.gather(lab, perm)
        gts = K.sum(gt_sorted)  # tensorflow.reduce_sumにしてたかも (どちらでも同じ)
        intersection = gts - tensorflow.cumsum(gt_sorted)
        union = gts + tensorflow.cumsum(1. - gt_sorted)
        jaccard = 1. - intersection / union
        grad = tensorflow.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
        a = tensorflow.nn.elu(errors_sorted) + 1
        loss = tensorflow.tensordot(a, tensorflow.stop_gradient(grad), 1, name="loss_per_image")
        return loss
    losses = tensorflow.map_fn(loss_per_image, (logits, y_true), dtype=tensorflow.float32)
    return K.mean(losses)


if __name__ == '__main__':
    _main()
