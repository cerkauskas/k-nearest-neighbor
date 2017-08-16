#!/usr/bin/env python

from KNearestNeighbor import KNearestNeighbor
from time import time
import csv
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
    # else:
    #     data = np.concatenate((data, dict['data']))
    #     labels = np.concatenate((labels, dict['labels']))

test_data_dict = unpickle('cifar-10-batches-py/test_batch')
test_data = test_data_dict['data']
test_labels = test_data_dict['labels']

for k in range(1, 150):
    classifier = KNearestNeighbor(k)

    classifier.train(data, labels)

    try:
        success = 0
        failure = 0
        times = []

        for i in range(len(test_data)):
            if i % 20000 == 0:
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
        with open('results.csv', 'a') as result_file:
            writer = csv.writer(result_file).writerow([
                k, success, failure, rate, avg_time
            ])

    print "k=%d is calculated" % k
