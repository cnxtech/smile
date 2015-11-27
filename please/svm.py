import numpy as np
from sklearn import svm

class SVM:
    def __init__(self):
        self.classifier = svm.SVC(
            kernel='linear',
            decision_function_shape='ovr'
        )

    def fit(self, data, target):
        self.classifier.fit(data, target)

    def predict(self, data):
        return self.classifier.predict(data)
