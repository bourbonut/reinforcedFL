from itertools import accumulate, zip_longest, groupby
import random


def rdindices(indices, sizes):
    random.shuffle(indices)
    iterator = zip_longest(
        accumulate(sizes, initial=0), accumulate(sizes), fillvalue=-1
    )
    return [indices[a:b] for a, b in iterator][:-1]


def sort_per_label(list_, iterable=False, key=None):
    if iterable:
        return dict(groupby(sorted(list_, key=key), key=key))
    else:
        return {
            label: list(values)
            for label, values in groupby(sorted(list_, key=key), key=key)
        }
