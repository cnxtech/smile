from conf import paths
import scipy.io
import numpy as np

def load_train():
    """ Loads all training data. """
    tr_identity1, tr_labels1, tr_images1 = _load_train_helper(paths.TR_SET1)
    tr_identity2, tr_labels2, tr_images2 = _load_train_helper(paths.TR_SET2)
    tr_identity3, tr_labels3, tr_images3 = _load_train_helper(paths.TR_SET3)
    tr_identity = np.concatenate((tr_identity1, tr_identity2, tr_identity3), axis=0)
    tr_labels = np.concatenate((tr_labels1, tr_labels2, tr_labels3), axis=0)
    tr_images = np.concatenate((tr_images1, tr_images2, tr_images3), axis=2)
    return tr_identity, tr_labels, tr_images

def _load_train_helper(file_path):
    """ Loads each training data. """
    tr_set = scipy.io.loadmat(file_name = file_path)
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

