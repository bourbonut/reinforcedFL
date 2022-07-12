from utils.path import *


def test_root():
    assert not ("utils" in str(ROOT_PATH)) and Path().absolute() == ROOT_PATH


def test_experiments():
    create(EXP_PATH)
    assert EXP_PATH.exists()


def test_iterate():
    path = iterate(EXP_PATH)
    assert not (path.exists())
