"""
pip install -U pydensecrf
"""

import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils
import sklearn.externals.joblib as joblib

import pytoolkit as tk


def apply_crf_all(X, pred):
    jobs = [joblib.delayed(apply_crf, check_pickle=False)(x, p) for x, p in zip(X, pred)]
    with joblib.Parallel(backend='threading') as parallel:
        pred = parallel(jobs)
    return np.array(pred)


def apply_crf(original_image_path, mask_image):
    """
    Function which returns the labelled image after applying CRF
    """
    original_image = tk.ndimage.load(original_image_path, grayscale=True)

    n_labels = 2
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    mask_flat = np.ravel(mask_image)
    sm = np.array([1 - mask_flat, mask_flat])
    U = pydensecrf.utils.unary_from_softmax(sm)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(10)
    # MAP = np.argmax(Q, axis=0)  # class
    MAP = np.array(Q)[1, ...]  # probability

    return MAP.reshape((original_image.shape[0], original_image.shape[1], 1))
