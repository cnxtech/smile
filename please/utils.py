from conf import paths
import scipy.io
import numpy as np

def load_train():
    """ Loads training data. """
    tr_set = scipy.io.loadmat(file_name = paths.TR_SET)
    tr_identity = reshape_labels(tr_set['tr_identity'])
    tr_labels = reshape_labels(tr_set['tr_labels'])
    tr_images = reshape_images(tr_set['tr_images'])

    return tr_identity, tr_labels, tr_images

def load_test():
    """ Loads training data. """
    test_set = scipy.io.loadmat(file_name = paths.TEST_SET)
    test_images = reshape_images(test_set['public_test_images'])

    return test_images

def reshape_labels(labels):
    return labels.reshape(-1)

def reshape_images(images):
    height, width, count = images.shape
    return images.reshape(height * width, count).T
