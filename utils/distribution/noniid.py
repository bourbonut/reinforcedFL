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
    return {label: rdindices(indices[label], divisions[label]) for label in labels}


def label(nworkers, labels, minlabels, balanced=False):
    l = len(labels)
    if balanced:
        # number of labels to be added on workers
        if nworkers * minlabels >= l * (minlabels - 1):
            d = (nworkers * minlabels) % l
            if d == 0:
                distrb = [minlabels for i in range(nworkers)]
            else:
                p, r = divmod(l - (nworkers * minlabels) % l, nworkers)
                distrb = [minlabels + p + (i < r) for i in range(nworkers)]
        else:
            p, r = divmod(minlabels * l, nworkers)
            distrb = [p + (i < r) for i in range(nworkers)]
        # distrib has k * (l * minlabels) elements
        k = sum(distrb) // (l * minlabels)
        tokens = {label: k * minlabels for label in labels}
    else:
        distrb = [minlabels for _ in range(nworkers)]
        # number of labels which are not in the distribution
        p, r = divmod(minlabels * (l - minlabels), 10)
        low_labels = random.sample(labels, r)
        tokens = {label: minlabels - p - (label in low_labels) for label in labels}
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
