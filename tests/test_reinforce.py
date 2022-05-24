from evaluator.model import *


def func(x):
    return x + 1


def test_cannary():
    assert func(3) == 4
