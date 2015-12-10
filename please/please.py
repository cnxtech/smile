import models
import submission
import plot_digits
import preprocessor
import os.path
import numpy as np
from sklearn.metrics import classification_report

import utils
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel

def please():
    target_names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    tr_identity, tr_labels, tr_images, valid_identity, valid_labels, valid_images = preprocessor.load_train_and_valid()
    test_images = preprocessor.load_test()
    # unlabeled_images = preprocessor.load_unlabeled()

    kernels = create_gabor_filters(
            [1.0/x for x in range(3, 15, 3)],
            [x*np.pi*0.125 for x in range(8)],
            np.pi, np.pi)

    features = compute_all_filter_responses(np.concatenate((tr_images, valid_images)), kernels)
    features = features.reshape(features.shape[0], 32 * 32 * 32)

    model = models.SVM()
    model.fit(features, np.concatenate((tr_labels, valid_labels)))

    # valid_features = compute_all_filter_responses(valid_images, kernels)
    # valid_features = valid_features.reshape(valid_features.shape[0], 32 * 32 * 32)
    # valid_predictions = model.predict(valid_features)

    # print(classification_report(valid_labels, valid_predictions, target_names=target_names))

    test_features = compute_all_filter_responses(test_images, kernels)
    test_features = test_features.reshape(test_features.shape[0], 32 * 32 * 32)
    test_predictions = model.predict(test_features)

    submission.output(test_predictions)
    return

    model = models.SVM()
    model.fit(tr_images, tr_labels)
    # train_predictions = model.predict(tr_images)
    valid_predictions = model.predict(valid_images)
    # test_predictions = model.predict(test_images)

    print "########### Model " + model.getName() + " ##################\n"
    print "########### Valid ##################\n"
    print(classification_report(valid_labels, valid_predictions, target_names=target_names))
    # print "\n########### Train ##################\n"
    # print(classification_report(tr_labels, train_predictions, target_names=target_names))

    # submission.output(test_predictions)


def create_gabor_filters(frequencies, thetas, sigmaX, sigmaY):
    kernels = []
    for frequency in frequencies:
        for theta in thetas:
            # keep it real
            kernel = np.real(gabor_kernel(frequency, theta, sigmaX, sigmaY))
            kernels.append(kernel)

    return np.array(kernels)

def compute_all_filter_responses(images, kernels):
    num_samples = images.shape[0]
    num_kernels = kernels.shape[0]

    features = np.empty((num_samples, num_kernels, images.shape[1], images.shape[2]), dtype=np.double)
    for i, image in enumerate(images):
        features[i] = convolve_filters(image, kernels)

    return features

def convolve_filters(image, kernels):
    num_kernels = kernels.shape[0]

    features = np.empty((num_kernels, image.shape[0], image.shape[1]), dtype=np.double)
    for k, kernel in enumerate(kernels):
        features[k] = ndi.convolve(image, kernel, mode='nearest')

    return features

if __name__ == '__main__':
    please() # work
