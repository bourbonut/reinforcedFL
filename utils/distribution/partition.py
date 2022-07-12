"""
Functions to partition data
"""

import random, pickle, torch, copy
from functools import reduce
from operator import add, itemgetter
from utils.plot import stacked
from utils.distribution.common import sort_per_label
import importlib


class WorkerDataset(torch.utils.data.Dataset):
    """
    Simple class for local dataset
    """

    def __init__(self, labels, data):
        self.data = data
        self.labels = labels
        # To get easier samples per label
        self.classified = sort_per_label(data, key=itemgetter(1))
        # Number of samples per label
        self.amount = {label: len(self.classified[label]) for label in self.classified}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class AugmentedDataset(torch.utils.data.Dataset):
    """
    Class for augmented dataset
    """

    def __init__(self, dataset, k, noise=100):
        """
        Return a new dataset with same distribution
        and k times the total size

        Parameters:
            
            dataset (torch.utils.data.Dataset): 
                the dataset which is going to be augmented

            k (float):      the percentage for augmentation (must be greater than 1)
            noise (int):    the amount of noise added 
        """
        assert k > 1, "k must be greater than 1 (e.g. `k = 1.5`)"
        self.noise = noise
        self.k = k
        sorted_data=sort_per_label(dataset, key=itemgetter(1))
        self.data = reduce(add, map(self.augment, sorted_data.values())) 
        random.shuffle(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data) 
    
    def make_noise(self, sample):
        """
        Add noise to the given sample
        """
        x, y = sample
        noise = torch.randint(0, self.noise, x.size())
        return [x + noise, y]
    
    def augment(self, samples):
        """
        Add random samples from the given samples, 
        make noise on all samples and return samples
        """
        size = len(samples)
        p, r = divmod(self.k, 1)
        new_samples = (p - 1) * samples
        augmented_size = int(r * size)
        new_samples += random.sample(samples, augmented_size)
        total_samples = samples + new_samples
        return [self.make_noise(sample) for sample in total_samples]

def generate(
    path,
    datatrain,
    datatest,
    nworkers,
    label_distrb="iid",
    minlabels=3,
    balanced=False,
    volume_distrb="iid",
    save2png=False,
    k=None,
    noise=10,
):
    """
    Generate local dataset for workers

    Parameters:

        path (PosixPath):       path where data of workers are saved
        datatrain (Dataset):    dataset for training
        datatest (Dataset):     dataset for testing
        nworkers (int):         number of workers
        label_distrb (str):     "iid" or "noniid"
        minlabels (int):        See function `label` in `utils/distribution/noniid.py`
        balanced (bool):        See function `label` in `utils/distribution/noniid.py`
        volume_distrb (str):    "iid" or "noniid"
        save2png (bool):        if True, save a stacked chart of the distribution
        k (float):              if specified, k > 1 and increases the size of the dataset
        noise (int):            for data augmentation, parameter for noise
    """
    msg = "Training data must have the same labels as testing data"
    assert datatrain.classes == datatest.classes, msg
    msg = "Training data must have the same number of labels as testing data"
    get_nb_labels = lambda x: len(list(x.class_to_idx.values()))
    assert get_nb_labels(datatrain) == get_nb_labels(datatest), msg

    # Values which are equal to the label of classes
    labels = list(datatrain.class_to_idx.values())

    # Data augmentation
    if k is not None:
        datatrain = AugmentedDataset(datatrain, k, noise=noise)
        datatest = AugmentedDataset(datatest, k, noise=noise) 

    # Load functions for distribution
    get = lambda distrb: importlib.import_module(f"utils.distribution.{distrb}")
    label = getattr(get(label_distrb), "label")
    volume = getattr(get(volume_distrb), "volume")

    # Distribution of labels
    distrb = label(nworkers, labels, minlabels, balanced=balanced)
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

    distrb_per_wk = {"Worker {}".format(i): [0] * len(labels) for i in range(nworkers)}
    for k, worker_labels in enumerate(distrb):
        # Worker training indices
        indices_per_labels = tuple(map(train_pickup, worker_labels))
        wktrain_indices = reduce(add, indices_per_labels)
        # Worker testing indices
        wktest_indices = reduce(add, map(test_pickup, worker_labels))
        random.shuffle(wktrain_indices)
        random.shuffle(wktest_indices)

        # Generate data
        worker_data = [
            WorkerDataset(worker_labels, [datatrain[i] for i in wktrain_indices]),
            WorkerDataset(worker_labels, [datatest[i] for i in wktest_indices]),
        ]

        if save2png:  # y_stacked of the function `stacked`
            for label, indices in zip(worker_labels, indices_per_labels):
                distrb_per_wk["Worker " + str(k)][label] += len(indices)
        # Now put it all in an npz
        name_file = "worker-" + str(k + 1) + ".pkl"
        with open(path / name_file, "wb") as file:
            pickle.dump(worker_data, file)
        print("Data for worker {} saved".format(k + 1))

    if save2png:
        stacked(
            labels,
            distrb_per_wk,
            path / "distribution.png",
            title="Distribution of labels between workers",
        )
