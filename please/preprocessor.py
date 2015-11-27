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
    test_images, hidden_images = utils.load_test()

    pc_test_images = reshape_images(test_images)
    pc_hidden_images = reshape_images(hidden_images)

    combined_images = np.concatenate((pc_test_images, pc_hidden_images))
    pc_combined_images = expose_images(combined_images)

    return pc_combined_images

def extract_data(image_map, tr_prob):
    tr_data = []
    valid_data = []

    for identity, labelled_images in image_map.items():
        for (label, image) in labelled_images:
            data = [identity, label, image]
            if random.random() < tr_prob:
                tr_data.append(data)
            else:
                valid_data.append(data)

    data = [zip(*tr_data), zip(*valid_data)]
    return [item for sublist in data for item in sublist]

def reshape_labels(labels):
    return labels.reshape(-1)

def reshape_images(images):
    height, width, count = images.shape
    return images.reshape(height * width, count).T

def expose_images(images):
    return exposure.equalize_hist(images)
