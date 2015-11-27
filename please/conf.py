import os

class Paths:
    def __init__(self):
        self.ROOT = '../data'

    def get_output(self):
        return os.path.join(self.ROOT, 'submission.csv')
    OUTPUT = property(get_output)

    def get_tr_set(self):
        return os.path.join(self.ROOT, 'labeled_images.mat')
    TR_SET = property(get_tr_set)

    def get_test_set(self):
        return os.path.join(self.ROOT, 'public_test_images.mat')
    TEST_SET = property(get_test_set)

    def get_hidden_set(self):
        return os.path.join(self.ROOT, 'hidden_test_images.mat')
    HIDDEN_SET = property(get_hidden_set)

paths = Paths()
