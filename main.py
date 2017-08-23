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

        queue.put([
            k, success, failure, rate, avg_time
        ])

    print "k=%d is calculated" % k


def Reader():
    while True:
        item = queue.get()
        if item == 'DONE':
            break

        with file('results.csv', 'a') as f:
            csv.writer(f).writerow(item)


global_start = time.time()
queue = multiprocessing.Queue()
reader_p = multiprocessing.Process(target=Reader)
reader_p.daemon = True
reader_p.start()

k = 87
k_max = 111
simultaneus_processes = multiprocessing.cpu_count() - 1
while True:
    if k > k_max:
        for p in multiprocessing.active_children():
            if p != reader_p:
                p.join()

        queue.put('DONE')

        break

    active_children = len(multiprocessing.active_children())

    if active_children < simultaneus_processes - 1:
        worker = multiprocessing.Process(target=Worker, args=(k, ))
        worker.start()

        print "k=%d started" % k
        k += 1

global_end = time.time()

print "Script took %f" % (global_end - global_start)
