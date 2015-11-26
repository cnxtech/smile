from conf import paths
import scipy.io
import numpy as np

def load_train():
    """ Loads training data. """
    tr_set = scipy.io.loadmat(file_name = paths.TR_SET_PATH)
    tr_identity = tr_set['tr_identity']
    tr_labels = tr_set['tr_labels']
    tr_images = tr_set['tr_images']

    return tr_identity, tr_labels, tr_images
