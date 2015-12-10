from conf import paths
import scipy.io
import numpy as np

def load_train():
    """ Loads all training data. """
    tr_set = scipy.io.loadmat(file_name = paths.TR_SET)
    tr_identity = tr_set['tr_identity']
    tr_labels = tr_set['tr_labels']
    tr_images = tr_set['tr_images']

    return tr_identity, tr_labels, tr_images

def load_unlabeled():
    """ Loads all unlabeled data."""
    unlabeled_set = scipy.io.loadmat(file_name = paths.UNLABELED_SET)
    unlabeled_images = unlabeled_set['unlabeled_images']

    return unlabeled_images

def load_test():
    """ Loads training data. """
    test_set = scipy.io.loadmat(file_name = paths.TEST_SET)
    test_images = test_set['public_test_images']

    # hidden_set = scipy.io.loadmat(file_name = paths.HIDDEN_SET)
    # hidden_images = hidden_set['hidden_test_images']

    return test_images

