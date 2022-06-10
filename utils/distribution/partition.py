"""
Functions to partition data
"""

import random, pickle, torch, copy
from functools import reduce
from operator import add
from utils.distribution import noniid
from utils.distribution import iid

MODULES = {"iid": iid, "noniid": iid}


class WorkerDataset(torch.utils.data.Dataset):
    """
    Simple class for local dataset
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def generate(
    path,
    datatrain,
    datatest,
    nworkers,
    label_distrb="iid",
    minlabels=3,
    balanded=False,
    volume_distrb="iid",
):
    msg = "Training data must have the same labels as testing data"
    assert datatrain.classes == datatest.classes, msg
    msg = "Training data must have the same number of labels as testing data"
    get_nb_labels = lambda x: len(list(x.class_to_idx.values()))
    assert get_nb_labels(datatrain) == get_nb_labels(datatest), msg

    # Load functions for distribution
    label = MODULES[label_distrb].label
    volume = MODULES[volume_distrb].volume

    # Values which are equal to the label of classes
    labels = list(datatrain.class_to_idx.values())
    # Distribution of labels
    distrb = label(nworkers, labels, minlabels, balanded=balanded)
    # Generate training indices
    train_indices = volume(distrb, datatrain, labels)
    # Generate testing indices
    test_indices = volume(distrb, datatest, labels)

    def train_pickup(label):
        indices = train_indices[label][0]
        train_indices[label].pop(0)
        return indices

    def test_pickup(label):
        indices = test_indices[label][0]
        test_indices[label].pop(0)
        return indices

    for worker_labels in distrb:
        # Worker training indices
        wktrain_indices = reduce(add, map(train_pickup, worker_labels))
        # Worker testing indices
        wktest_indices = reduce(add, map(test_pickup, worker_labels))

        # Generate data
        worker_data = [
            WorkerDataset([datatrain[i] for i in wktrain_indices]),
            WorkerDataset([datatest[i] for i in wktest_indices]),
        ]

        # Now put it all in an npz
        name_file = "worker-" + str(i + 1) + ".pkl"
        with open(path / name_file, "wb") as file:
            pickle.dump(worker_data, file)
        print("Data for node {} saved".format(i + 1))
