import models
import submission
import plot_digits
import preprocessor
import utils
import os.path
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel

def gabor_please(enable_PCA, use_all, submit_test_prediction):
    print "Using Gabor please, PCA: %s, use_all:%s, submit_test_prediction: %s" % (str(enable_PCA), str(use_all), str(submit_test_prediction))
    tr_identity, tr_labels, tr_images, valid_identity, valid_labels, valid_images = preprocessor.load_train_and_valid(is_gabor=True)
    test_images = preprocessor.load_test(is_gabor=True)

    kernels = create_gabor_filters(
            [1.0/x for x in range(3, 15, 3)],
            [x*np.pi*0.125 for x in range(8)],
            np.pi, np.pi)

    model = models.SVM()

    train_features = compute_all_filter_responses(tr_images, kernels)
    train_features = train_features.reshape(train_features.shape[0], 32 * 32 * 32)
    valid_features = compute_all_filter_responses(valid_images, kernels)
    valid_features = valid_features.reshape(valid_features.shape[0], 32 * 32 * 32)
    test_features = compute_all_filter_responses(test_images, kernels)
    test_features = test_features.reshape(test_features.shape[0], 32 * 32 * 32)

    ########## compute using both train and valid
    if(use_all):
        all_features = compute_all_filter_responses(np.concatenate((tr_images, valid_images)), kernels)
        all_features = all_features.reshape(all_features.shape[0], 32 * 32 * 32)
        if (enable_PCA):
            all_features, train_features, valid_features, test_features = PCA_Preprocess(all_features, train_features, valid_features, test_features)
        model.fit(all_features, np.concatenate((tr_labels, valid_labels)))
    else:
        if(enable_PCA):
            _train_features, train_features, valid_features, test_features = PCA_Preprocess(train_features, train_features, valid_features, test_features)
        model.fit(train_features, tr_labels)

    train_predictions = model.predict(train_features)
    valid_predictions = model.predict(valid_features)

    printClassificationRate(model, valid_labels, valid_predictions, tr_labels, train_predictions)

    if(submit_test_prediction):
        test_predictions = model.predict(test_features)
        submission.output(test_predictions)

def PCA_Preprocess(pca_fit_data, train_features, valid_features, test_features):
    pca = models.PCA()
    pca.fit(pca_fit_data)
    transformed_pca_fit_dat = pca.transform(pca_fit_data)
    transformed_train_features = pca.transform(train_features)
    transformed_valid_features = pca.transform(valid_features)
    transformed_test_features = pca.transform(test_features)
    print "PCA n components: %d" % (pca.get_n_components())
    return transformed_pca_fit_dat, transformed_train_features, transformed_valid_features, transformed_test_features
    

def please(submit_test_prediction):
    tr_identity, tr_labels, tr_images, valid_identity, valid_labels, valid_images = preprocessor.load_train_and_valid(is_gabor=False)
    
    model = models.SVM()
    model.fit(tr_images, tr_labels)
    train_predictions = model.predict(tr_images)
    valid_predictions = model.predict(valid_images)

    printClassificationRate(model, valid_labels, valid_predictions, tr_labels, train_predictions)

    if(submit_test_prediction):
        test_images = preprocessor.load_test(is_gabor=False)
        test_predictions = model.predict(test_images)
        submission.output(test_predictions)

def PCA_please(enable_PCA_cache, submit_test_prediction):
    tr_identity, tr_labels, tr_images, valid_identity, valid_labels, valid_images = preprocessor.load_train_and_valid(is_gabor=False)
    unlabeled_images = preprocessor.load_unlabeled(is_gabor=False)
    
    if(enable_PCA_cache):        
        pca_filename = 'pca.pkl'
        if os.path.isfile(pca_filename):
            pca = joblib.load(pca_filename)
        else: 
            # save the PCA
            pca = models.PCA()
            pca.fit(unlabeled_images)
            joblib.dump(pca, pca_filename)
    else:
        pca = models.PCA()
        pca.fit(unlabeled_images)

    pca_tr_images = pca.transform(tr_images)
    pca_valid_images = pca.transform(valid_images)

    model = models.SVM()
    model.fit(pca_tr_images, tr_labels)
    train_predictions = model.predict(pca_tr_images)
    valid_predictions = model.predict(pca_valid_images)

    printClassificationRate(model, valid_labels, valid_predictions, tr_labels, train_predictions)
    
    if(submit_test_prediction):
        test_images = preprocessor.load_test(is_gabor=False)
        pca_test_images = pca.transform(test_images)
        test_predictions = model.predict(pca_test_images)
        submission.output(test_predictions)

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

def printClassificationRate(model, valid_labels, valid_predictions, tr_labels, train_predictions):
    target_names = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    print "########### Model " + model.getName() + " ##################\n"
    print "########### Valid ##################\n"
    print(classification_report(valid_labels, valid_predictions, target_names=target_names))
    print "\n########### Train ##################\n"
    print(classification_report(tr_labels, train_predictions, target_names=target_names))

if __name__ == '__main__':
    # please(False) 
    gabor_please(enable_PCA=True, use_all=True, submit_test_prediction=True)
