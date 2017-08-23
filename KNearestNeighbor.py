import numpy as np

class KNearestNeighbor(object):
    def __init__(self, k):
        self._k = k
        self._training_data = None
        self._training_labels = None

    def train(self, data, labels):
        self._training_data = data
        self._training_labels = labels

    def predict(self, features):
        target = np.repeat([features], len(self._training_data), axis=0)
        dists = np.sum(np.square(self._training_data - target), axis=1)
        indices = dists.argsort()[:self._k]
        labels = map(lambda x: self._training_labels[x], indices)

        return np.bincount(labels).argmax()

