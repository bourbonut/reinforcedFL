"""
Functions which deal with how participation is managed
"""
import random

def entirely(indices, *args, **kwargs):
    """
    All workers participate to the training
    """
    return indices

def randomly(indices, k, *args, **kwargs):
    """
    Randomly, a sequence of workers of size `k`
    participate to the training
    """
    return random.sample(indices, k)

# TODO: Function from a model and input
