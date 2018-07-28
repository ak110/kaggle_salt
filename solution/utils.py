"""
pip install -U pydensecrf
"""
import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils


def apply_crf(original_image, mask_image):
    """
    Function which returns the labelled image after applying CRF
    """

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
