#!/usr/bin/env python3

import numpy as np

import bin_nas
import darknet53_coord_hcs
import darknet53_expert_large
import darknet53_expert_small
import darknet53_large2
import darknet53_resize128
import darknet53_sepscse
import reg_nas


def get_meta_features(data_name, X, cv_index=None):
    """子モデルのout-of-fold predictionsを取得。"""
    X = np.concatenate([
        X / 255,
        np.repeat(_get(bin_nas.predict_all(data_name, X, use_cache=True), data_name, cv_index), 101 * 101).reshape(len(X), 101, 101, 1),
        np.repeat(_get(reg_nas.predict_all(data_name, X, use_cache=True), data_name, cv_index), 101 * 101).reshape(len(X), 101, 101, 1),
        _get(darknet53_coord_hcs.predict_all(data_name, X, use_cache=True), data_name, cv_index),
        _get(darknet53_expert_large.predict_all(data_name, X, use_cache=True), data_name, cv_index),
        _get(darknet53_expert_small.predict_all(data_name, X, use_cache=True), data_name, cv_index),
        _get(darknet53_large2.predict_all(data_name, X, use_cache=True), data_name, cv_index),
        _get(darknet53_resize128.predict_all(data_name, X, use_cache=True), data_name, cv_index),
        _get(darknet53_sepscse.predict_all(data_name, X, use_cache=True), data_name, cv_index),
    ], axis=-1)
    return X


def _get(m, data_name, cv_index):
    if data_name == 'val':
        return m
    else:
        assert len(m) == 5
        return m[cv_index]
