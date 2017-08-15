import numpy as np
import operator


class KNearestNeighbor(object):
    def __init__(self, k):
        self._k = k
        self._training_data = None
        self._training_labels = None

    def train(self, data, labels):
        self._training_data = data
        self._training_labels = labels

    def _distance(self, a, b):
        return np.linalg.norm(a-b)

    def _keep(self, items, new_item):
        if len(items) < self._k:
            items.append(new_item)
            return items

        i = 0
        for item in items:
            if items[i]['distance'] > new_item['distance']:
                items.insert(i, new_item)
                break

            i += 1

        return items[:self._k]

    def _choose_majority(self, items):
        options = {}

        for item in items:
            label = item['label']
            if label not in options:
                options[label] = 1
                continue

            options[label] += 1

        return max(options.iteritems(), key=operator.itemgetter(1))[0]

    def predict(self, features):
        candidates = []

        for i in range(len(self._training_data)):
            distance = self._distance(features, self._training_data[i])
            candidates = self._keep(candidates, {
                'label': self._training_labels[i],
                'distance': distance
            })

        return self._choose_majority(candidates)
