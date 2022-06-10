from functools import reduce
from operator import add
from collections import Counter
import random, copy
from .common import rdindices


def volume(distrb, dataset, labels):
    drep = Counter(reduce(add, distrb))  # nb of label repetitions
    extracted_labels = [x for _, x in dataset]
    dsamples = Counter(extracted_labels)  # nb of samples per label
    divisions = {label: divide(dsamples[label], drep[label]) for label in labels}
    indices = {label: [] for label in labels}
    for i, label in enumerate(extracted_labels):
        indices[label].append(i)
    return [rdindices(indices[label], divisions[label]) for label in labels]


def label(nworkers, labels, minlabels, balanded=False):
    if balanded:
        # number of labels to be added on workers
        l = len(labels)
        r = (nworkers * minlabels) % l
        distrb = [minlabels + (i < l - r) for i in range(nworkers)]
        total_distrb = sum(distrb)
        tokens = {label: minlabels for label in labels}
    else:
        distrb = [minlabels for _ in range(nworkers)]
        total_distrb = sum(distrb)
        # number of labels which are not in the distribution
        k = len(labels) - total_distrb % len(labels)
        low_labels = random.sample(labels, k)
        tokens = {label: minlabels - (label in low_labels) for label in labels}
    clabels = copy.copy(labels)

    def random_label():
        label = random.choice(clabels)
        tokens[label] -= 1
        if tokens[label] == 0:
            clabels.pop(clabels.index(label))
        return label

    return tuple(tuple(random_label() for _ in range(n)) for n in distrb)


def divide(size, nfractions):
    rnglist = [random.random() for _ in range(nfractions)]
    total = sum(rnglist)
    divl = [size * (1 + (e - (total / nfractions))) / nfractions for e in rnglist]
    p, r = zip(*map(lambda x: divmod(x, 1), divl))
    distribution = list(map(int, p))
    distribution[0] += int(round(sum(r), 0))
    return distribution