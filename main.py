#!/usr/bin/env python

from KNearestNeighbor import KNearestNeighbor
from time import time
import numpy as np

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


data = None
labels = None

for i in range(1, 6):
    dict = unpickle('cifar-10-batches-py/data_batch_' + str(i))

    if data is None:
        data = dict['data']
        labels = dict['labels']
    else:
        data = np.concatenate((data, dict['data']))
        labels = np.concatenate((labels, dict['labels']))

classifier = KNearestNeighbor(1)

classifier.train(data, labels)

success = 0
failure = 0
times = []
test_data_dict = unpickle('cifar-10-batches-py/test_batch')
test_data = test_data_dict['data']
test_labels = test_data_dict['labels']

try:
    for i in range(len(test_data)):
        if i % 100 == 0:
            print "%d predictions have been made" % i

        features = test_data[i]
        label = test_labels[i]
        start = time()
        prediction = classifier.predict(features)
        end = time()

        times.append(end-start)

        if prediction != label:
            failure += 1
        else:
            success += 1
finally:
    rate = 100*success / float(success + failure)
    avg_time = reduce(lambda a, b: a+b, times) / len(times)

    print "Success rate is %f%%" % rate
    print "Average time to make a prediction is %f" % avg_time
