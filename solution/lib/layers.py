
import keras
import keras.backend as K
import numpy as np

import pytoolkit as tk


class CustomPreProcess(keras.layers.Layer):
    """入力の前処理。depth方向が重要っぽいのでそれっぽくする。入力は1チャンネルで出力は3チャンネル。"""

    def call(self, inputs, **kwargs):
        x = inputs / 255
        d = np.ones((101, 101, 1)) * np.linspace(0, 1, 101).reshape((101, 1, 1))
        xd = x * d
        return K.concatenate([x, d, xd], axis=-1)

    def compute_output_shape(self, input_shape):
        assert input_shape[-1] == 1
        return input_shape[:-1] + (3,)
