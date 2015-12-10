import utils
import numpy as np
import random

def load_train(is_gabor):
    tr_identity, tr_labels, tr_images = utils.load_train()

    pc_tr_identity = reshape_labels(tr_identity)
    pc_tr_labels = reshape_labels(tr_labels)    
    pc_tr_images = reshape_images(tr_images, is_gabor)

    return pc_tr_identity, pc_tr_labels, pc_tr_images

def load_unlabeled(is_gabor):
    unlabeled_images = utils.load_unlabeled()
    pc_unlabeled_images = reshape_images(unlabeled_images, is_gabor)
    return pc_unlabeled_images

def load_train_and_valid(is_gabor):
    tr_identity, tr_labels, tr_images = load_train(is_gabor)

    image_map = {}
    unique_identity = np.unique(tr_identity)
    zipped_images = zip(tr_identity, tr_labels, tr_images)
    # make images not sorted by id
    random.shuffle(zipped_images)

    for (identity, label, image) in zipped_images:
        if identity not in image_map:
            image_map[identity] = []
        image_map[identity].append((label, image))

    return extract_data(image_map, 0.8)

def load_test(is_gabor):
    test_images = utils.load_test()
    pc_test_images = reshape_images(test_images, is_gabor)
    return pc_test_images

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

def reshape_images(images, is_gabor):
    if is_gabor:
        return images.T
    else: 
        height, width, count = images.shape
        return images.reshape(height * width, count).T

def normalize_images(images):
    images = np.copy(images.astype(np.float64))
    for i in range(images.shape[0]):
        image = images[i]
        images[i] = (image - image.min()) / float(image.max() - image.min())

    return images

