from pathlib import Path

from reinforcedFL.utils.path import EXP_PATH, ROOT_PATH, create, iterate


def test_root():
    assert "utils" not in str(ROOT_PATH) and Path().absolute() == ROOT_PATH


def test_experiments():
    create(EXP_PATH)
    assert EXP_PATH.exists()


def test_iterate():
    path = iterate(EXP_PATH)
    assert not (path.exists())
