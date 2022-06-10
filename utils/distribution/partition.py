"""
Functions to partition data
"""

import random, pickle, torch, copy


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


def generate(datatrain, datatest, nworkers):
    # rnglist = [random.random() for _ in range(nfractions)]
    pass


def generate_IID_parties(dataset, k_nodes, path, **kwargs):
    """
    Generate IID data (random shuffle) for each node.

    Parameters :
        dataset (dict[str, VisionDataset]):
            "training" for training data and "test" for test data
            where data must be `VisionDataset`
        k_nodes (int):  Number of node
        path (Path):    Folder to save data for nodes
    """

    msg = "Training data and test data have not the same number of labels"
    assert dataset["training"].classes == dataset["test"].classes, msg
    size_train = len(dataset["training"])
    size_test = len(dataset["test"])
    num_labels = len(dataset["training"].classes)
    mtr, mte = (size_train // k_nodes, size_test // k_nodes)  # samples per node

    # Shuffle indices to select random samples
    train_indices = list(range(size_train))
    test_indices = list(range(size_test))
    random.shuffle(train_indices)
    random.shuffle(test_indices)

    print("Informations:")
    print("Training dataset:")
    print(dataset["training"])
    print("Number of samples per node (training):", mtr)
    print("\n Test dataset:")
    print(dataset["test"])
    print("Number of samples per node (test):", mte)
    print("\nGeneration of data for nodes (total = {} nodes) ...".format(k_nodes))
    for i in range(k_nodes):
        rtr = size_train % k_nodes if i + 1 == k_nodes else 0
        rte = size_test % k_nodes if i + 1 == k_nodes else 0
        # Select indices for each node and get the data
        node_train_indices = train_indices[mtr * i : mtr * (i + 1) + rtr]
        node_test_indices = test_indices[mte * i : mte * (i + 1) + rte]

        # Generate data
        node_data = [None, None]
        for j, key in enumerate(dataset):
            data = dataset[key]
            indices = node_train_indices if key == "training" else node_test_indices
            node_data[j] = WorkerDataset([data[idx] for idx in indices])

        # Now put it all in an npz
        name_file = "nodes-" + str(i + 1) + ".pkl"
        with open(path / name_file, "wb") as file:
            pickle.dump(node_data, file)
        print("Data for node {} saved".format(i + 1))
