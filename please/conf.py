import os

class Paths:
    def __init__(self):
        self.DATA_PATH = '../data'

    def get_tr_set(self):
        return os.path.join(self.DATA_PATH, 'labeled_images.mat')
    TR_SET_PATH = property(get_tr_set)

paths = Paths()
