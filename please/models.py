from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA as sklearn_PCA

class KNN:
    def __init__(self):
        self.classifier = neighbors.KNeighborsClassifier(3)

    def fit(self, data, target):
        self.classifier.fit(data, target)

    def predict(self, data):
        return self.classifier.predict(data)

    def getName(self):
        return self.__class__.__name__

class SVM:
    def __init__(self):
        self.classifier = svm.SVC(
            C=1.0,
            kernel='linear',
            decision_function_shape='ovr',
        )

    def fit(self, data, target):
        self.classifier.fit(data, target)

    def predict(self, data):
        return self.classifier.predict(data)

    def getName(self):
        return self.__class__.__name__


class AdaBoost:
    # def __init__(self):
    #     self.classifier = ensemble.AdaBoostClassifier(
    #         base_estimator=svm.SVC(probability=True,kernel='linear', decision_function_shape='ovr')
    #     )

    def __init__(self):
        self.classifier = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=7),
            algorithm="SAMME.R",
            n_estimators=100
        )

    def fit(self, data, target):
        self.classifier.fit(data, target)

    def predict(self, data):
        return self.classifier.predict(data)

    def getName(self):
        return self.__class__.__name__

## this is used only on the unlabeled data to get PCA
# then we are going to use the model to compress labeled data, to get rid of the noise
class PCA:
    def __init__(self):
        self.classifier = sklearn_PCA(n_components='mle')

    def fit(self, unlabeld_data):
        self.classifier = self.classifier.fit(unlabeld_data)

    def transform(self, labeled_data):
        return self.classifier.transform(labeled_data)

    def inverse_transform(self, data):
        '''
            decompress our data, and then we can see if we still preserve human expression
        '''
        return self.classifier.inverse_transform(data)

    def getName(self):
        return self.__class__.__name__
