"""
This file allows to manage configuration path,
to refresh data to change the distribution and
to select if the program should run on GPU or
CPU with multithreading.
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("configuration", help="configuration path", type=str)
parser.add_argument("--refresh", help="refresh data of workers", dest="refresh", action="store_true")
parser.add_argument("--gpu", help="run the program on GPU, else CPU (multithreading)", dest="gpu", action="store_true")

args = parser.parse_args()

