import os

class Paths:
    def __init__(self):
        self.ROOT = '../data'

    def get_output(self):
        return os.path.join(self.ROOT, 'submission.csv')
    OUTPUT = property(get_output)

    def get_unlabeled_set(self):
        return os.path.join(self.ROOT, 'unlabeled_images.mat')
    UNLABELED_SET = property(get_unlabeled_set)

    def get_tr_set1(self):
        return os.path.join(self.ROOT, 'labeled_images.mat')
    TR_SET1 = property(get_tr_set1)

    def get_tr_set2(self):
        return os.path.join(self.ROOT, 'labeled_images-2.mat')
    TR_SET2 = property(get_tr_set2)

    def get_tr_set3(self):
        return os.path.join(self.ROOT, 'labeled_images-3.mat')
    TR_SET3 = property(get_tr_set3)

    def get_test_set(self):
        return os.path.join(self.ROOT, 'public_test_images.mat')
    TEST_SET = property(get_test_set)

    def get_hidden_set(self):
        return os.path.join(self.ROOT, 'hidden_test_images.mat')
    HIDDEN_SET = property(get_hidden_set)

paths = Paths()
