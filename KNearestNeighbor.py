
class KNearestNeighbor(object):
    def __init__(self, k):
        self._k = k
        self._training_data = None
        self._training_labels = None

    def train(self, data, labels):
        self._training_data = data
        self._training_labels = labels

    def _distance(self, a, b):
        sum = 0

        for i, j in zip(a, b):
            sum += (int(i)-int(j))**2

        return sum

    def predict(self, features):
        minimum_distance = float('Inf')
        label = None

        for i in range(len(self._training_data)):
            distance = self._distance(features, self._training_data[i])

            if minimum_distance > distance:
                minimum_distance = distance
                label = self._training_labels[i]

        return label