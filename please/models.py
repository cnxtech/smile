from sklearn import neighbors
from sklearn import svm
from sklearn import ensemble

class KNeighbors:
    def __init__(self):
        self.classifier = neighbors.KNeighborsClassifier(3)

    def fit(self, data, target):
        self.classifier.fit(data, target)

    def predict(self, data):
        return self.classifier.predict(data)

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

class AdaBoost:
    def __init__(self):
        self.classifier = ensemble.AdaBoostClassifier(
            base_estimator=svm.SVC(probability=True,kernel='linear', decision_function_shape='ovr')
        )

    def fit(self, data, target):
        self.classifier.fit(data, target)

    def predict(self, data):
        return self.classifier.predict(data)
