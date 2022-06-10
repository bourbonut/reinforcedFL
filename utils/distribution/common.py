from itertools import accumulate, zip_longest
import random


def rdindices(indices, sizes):
    random.shuffle(indices)
    iterator = zip_longest(
        accumulate(sizes, initial=0), accumulate(sizes), fillvalue=-1
    )
    return [indices[a:b] for a, b in iterator][:-1]
