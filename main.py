#!/usr/bin/env python
import ctypes

from KNearestNeighbor import KNearestNeighbor
import time
import csv
import numpy as np
import multiprocessing

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


dict = unpickle('cifar-10-batches-py/data_batch_1')

data = dict['data']
labels = dict['labels']

test_data_dict = unpickle('cifar-10-batches-py/test_batch')
test_data = test_data_dict['data']
test_labels = test_data_dict['labels']


def Worker(k):

    classifier = KNearestNeighbor(k)

    classifier.train(data, labels)
    success = 0
    failure = 0
    times = []

    try:

        for i in range(len(test_data)):
            if i % 20000 == 0:
                print "%d predictions have been made" % i

            features = test_data[i]
            label = test_labels[i]
            start = time.time()
            prediction = classifier.predict(features)
            end = time.time()

            times.append(end-start)

            if prediction != label:
                failure += 1
            else:
                success += 1
    finally:
        rate = 100*success / float(success + failure)
        avg_time = reduce(lambda a, b: a+b, times) / len(times)
        print avg_time

    print "k=%d is calculated" % k


global_start = time.time()

Worker(20)

global_end = time.time()

print "Script took %f" % (global_end - global_start)
