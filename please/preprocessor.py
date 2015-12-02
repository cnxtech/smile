import utils
from skimage import exposure
import numpy as np
import random

def load_train():
    tr_identity, tr_labels, tr_images = utils.load_train()

    pc_tr_identity = reshape_labels(tr_identity)
    pc_tr_labels = reshape_labels(tr_labels)
    pc_tr_images = expose_images(reshape_images(tr_images))

    return pc_tr_identity, pc_tr_labels, pc_tr_images

def load_unlabeled():
    unlabeled_images = utils.load_unlabeled()
    pc_unlabeled_images = reshape_images(unlabeled_images)
    return pc_unlabeled_images

def load_train_and_valid():
    tr_identity, tr_labels, tr_images = load_train()

    image_map = {}
    unique_identity = np.unique(tr_identity)
    zipped_images = zip(tr_identity, tr_labels, tr_images)
    random.shuffle(zipped_images)

    for (identity, label, image) in zipped_images:
        if identity not in image_map:
            image_map[identity] = []
        image_map[identity].append((label, image))

    return extract_data(image_map, 0.7)

def load_test():
    test_images = utils.load_test()

    pc_test_images = reshape_images(test_images)
    # pc_hidden_images = reshape_images(hidden_images)

    # combined_images = np.concatenate((pc_test_images, pc_hidden_images))
    combined_images = pc_test_images
    pc_combined_images = expose_images(combined_images)

    return pc_combined_images

def extract_data(image_map, tr_prob):
    tr_data = []
    valid_data = []

    for identity, labelled_images in image_map.items():
        next_data = tr_data

        if (len(tr_data) != 0 or len(valid_data) != 0):
            tr_ratio = len(tr_data) / float(len(valid_data) + len(tr_data))
            next_data = tr_data if tr_ratio < tr_prob else valid_data

        for (label, image) in labelled_images:
            next_data.append([identity, label, image])

    data = [zip(*tr_data), zip(*valid_data)]
    return [np.array(item) for sublist in data for item in sublist]

def reshape_labels(labels):
    return labels.reshape(-1)

def reshape_images(images):
    height, width, count = images.shape
    return images.reshape(height * width, count).T

def expose_images(images):
    return exposure.equalize_hist(images)
