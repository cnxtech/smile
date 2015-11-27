import numpy as np
from sklearn import neighbors

class KNeighbors:
    def __init__(self, K):
        self.K = K
        self.classifier = neighbors.KNeighborsClassifier(K)

    def fit(self, data, target):
        self.classifier.fit(data, target)

    def predict(self, data):
        return self.classifier.predict(data)
