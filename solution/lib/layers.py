
import keras
import keras.backend as K
import numpy as np

import pytoolkit as tk


class CustomPreProcess(keras.layers.Layer):
    """入力の前処理。depth方向が重要っぽいのでそれっぽくする。入力は1チャンネルで出力は3チャンネル。"""

    def __init__(self, mode, **kwargs):
        assert mode in ('div255', 'caffe')
        self.mode = mode
        super().__init__(**kwargs)

    def get_config(self):
        config = {'mode': self.mode}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        assert input_shape[-1] == 1
        return input_shape[:-1] + (3,)

    def call(self, inputs, **kwargs):
        x = inputs / 255
        d = K.ones_like(x) * np.linspace(0, 1, 101).reshape((1, 101, 1, 1))
        x = K.concatenate([x, d, x * d], axis=-1)
        if self.mode == 'caffe':
            x = K.bias_add(x[..., ::-1] * 255, K.constant(np.array([-103.939, -116.779, -123.68])))
        return x
