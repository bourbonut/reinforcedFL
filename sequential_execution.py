#!/home/osboxes/anaconda3/envs/gym/bin/python
from utils import *
from torchvision import datasets
from torchvision.transforms import ToTensor


isdownloaded = not (DATA_PATH.exists())
mnist_dataset = {}
mnist_dataset["training"] = datasets.MNIST(
    root="data", train=True, download=isdownloaded, transform=ToTensor()
)
mnist_dataset["test"] = datasets.MNIST(root="data", train=False, transform=ToTensor())
