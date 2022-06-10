from functools import reduce
from itertools import accumulate, zip_longest
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


def label(nworkers, labels, *args, **kwargs):
    return [labels for _ in range(nworkers)]


def divide(size, nfractions):
    p, r = divmod(size, nfractions)
    return [p + (i < r) for i in range(nfractions)]
