#!/usr/bin/env python

from KNearestNeighbor import KNearestNeighbor
from time import time


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

classifier = KNearestNeighbor(1)

dict = unpickle('cifar-10-batches-py/data_batch_1')

data = dict['data']
labels = dict['labels']

classifier.train(data, labels)

success = 0
failure = 0
times = []
for i in range(len(data)):
    if i % 100 == 0:
        print "%d predictions have been made" % i
    features = data[i]
    label = labels[i]

    start = time()
    prediction = classifier.predict(data[i])
    end = time()
    times.append(end-start)

    if prediction != label:
        failure += 1
    else:
        success += 1

rate = 100*success / float(success + failure)
avg_time = reduce(lambda a, b: a+b, times) / len(times)

print "Success rate is %f" % rate
print "Average time to make a prediction is %f" % avg_time