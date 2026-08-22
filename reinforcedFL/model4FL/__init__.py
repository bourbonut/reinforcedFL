"""
Module which holds all models which are used for learning on local data :
- MNIST model
- FashionMNIST model
"""

from . import fashionmnist, mnist  # cifar10

__all__ = ["mnist", "fashionmnist"]  # "cifar10"
