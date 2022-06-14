from utils.path import create, iterate, data_path_key, EXP_PATH, DATA_PATH
from utils.distribution import *
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import pickle, pytest

create(EXP_PATH)
nworkers = 4
isdownloaded = not (DATA_PATH.exists())
mnist_dataset = {}
datatrain = datasets.MNIST(
    root="data", train=True, download=isdownloaded, transform=ToTensor()
)
datatest = datasets.MNIST(root="data", train=False, transform=ToTensor())


def test_check_data():
    assert DATA_PATH.exists()


def test_iid_label():
    labels = iid.label(nworkers, list(datatrain.class_to_idx.values()))
    assert all((len(list(datatrain.class_to_idx.values())) == len(x) for x in labels))


def test_iid_divide():
    result = iid.divide(15000, 7)
    assert sum(result) == 15000


def test_iid_volume():
    labels = list(datatrain.class_to_idx.values())
    distrb = iid.label(nworkers, labels)
    result = iid.volume(distrb, datatrain, labels)
    result = list(result.values())
    assert len(result) == len(labels)
    assert all(len(result[i]) == nworkers for i in range(len(result)))
    indices = set()
    for idcs in result:
        for elements in idcs:
            sidcs = set(elements)
            assert len(sidcs) == len(elements)
            assert len(sidcs.intersection(indices)) == 0
            indices = indices.union(sidcs)


def test_noniid_label_unbalanced():
    labels = noniid.label(nworkers, list(datatrain.class_to_idx.values()), 3)
    s = set()
    for label in labels:
        s = s.union(label)
        assert len(label) == 3
    assert len(s) == len(list(datatrain.class_to_idx.values()))


def test_noniid_label_balanced():
    labels = noniid.label(nworkers, list(datatrain.class_to_idx.values()), 3, True)
    s = set()
    for label in labels:
        s = s.union(label)
    assert len(s) == len(list(datatrain.class_to_idx.values()))
    assert any(len(label) != 3 for label in labels)


def test_noniid_volume():
    labels = list(datatrain.class_to_idx.values())
    distrb = noniid.label(nworkers, labels, 3, True)
    result = noniid.volume(distrb, datatrain, labels)
    result = list(result.values())
    assert len(result) == len(labels)
    assert all(len(result[i]) >= 3 for i in range(len(result)))
    indices = set()
    for idcs in indices:
        sidcs = set(idcs)
        assert len(sidcs) == len(idcs)
        assert len(sidcs.intersection(indices)) == 0
        indices = indices.union(sidcs)

@pytest.mark.slow
def test_generate_IID():
    wk_data_path = data_path_key("MNIST", "IID", nworkers) / "workers"
    create(wk_data_path)
    generate(wk_data_path, datatrain, datatest, nworkers)
    assert wk_data_path.exists()
    assert len(list(wk_data_path.iterdir())) == nworkers
    ref = len(list(datatrain.class_to_idx.values()))
    for filename in wk_data_path.iterdir():
        with open(filename, "rb") as file:
            wkdatatrain, wkdatatest = pickle.load(file)
        tr_extracted_labels = set([x for _, x in wkdatatrain])
        te_extracted_labels = set([x for _, x in wkdatatest])
        assert len(tr_extracted_labels) == ref
        assert len(te_extracted_labels) == ref

@pytest.mark.slow
def test_generate_nonIID_label_balanced():
    nworkers = 7
    wk_data_path = data_path_key("MNIST", "nonIID-label-balanced", nworkers) / "workers"
    create(wk_data_path)
    generate(
        wk_data_path,
        datatrain,
        datatest,
        nworkers,
        label_distrb="noniid",
        minlabels=3,
        balanced=True,
    )
    assert wk_data_path.exists()
    assert len(list(wk_data_path.iterdir())) == nworkers

@pytest.mark.slow
def test_generate_nonIID_label_unbalanced():
    nworkers = 7
    wk_data_path = (
        data_path_key("MNIST", "nonIID-label-unbalanded", nworkers) / "workers"
    )
    create(wk_data_path)
    generate(
        wk_data_path,
        datatrain,
        datatest,
        nworkers,
        label_distrb="noniid",
        minlabels=3,
        balanced=False,
    )
    assert wk_data_path.exists()
    assert len(list(wk_data_path.iterdir())) == nworkers

@pytest.mark.slow
def test_generate_nonIID_volume():
    nworkers = 7
    wk_data_path = data_path_key("MNIST", "nonIID-volume", nworkers) / "workers"
    create(wk_data_path)
    generate(wk_data_path, datatrain, datatest, nworkers, volume_distrb="noniid")
    assert wk_data_path.exists()
    assert len(list(wk_data_path.iterdir())) == nworkers

@pytest.mark.slow
def test_generate_nonIID():
    nworkers = 7
    wk_data_path = data_path_key("MNIST", "nonIID", nworkers) / "workers"
    create(wk_data_path)
    generate(
        wk_data_path,
        datatrain,
        datatest,
        nworkers,
        label_distrb="noniid",
        minlabels=3,
        balanced=False,
        volume_distrb="noniid",
    )
    assert wk_data_path.exists()
    assert len(list(wk_data_path.iterdir())) == nworkers

@pytest.mark.slow
def test_open_dataset():
    nodes_data_path = data_path_key("MNIST", "IID", nworkers) / "workers"
    if nodes_data_path.exists():
        batch_size = 64
        filename = next(nodes_data_path.iterdir())
        with open(filename, "rb") as file:
            wkdatatrain, wkdatatest = pickle.load(file)
        trainloader = DataLoader(wkdatatrain, batch_size=batch_size, num_workers=1)
        testloader = DataLoader(wkdatatest, batch_size=batch_size, num_workers=1)
        sample_train, label_train = next(iter(trainloader))
        assert sample_train.shape == (batch_size, 1, 28, 28)
        assert label_train.shape == (batch_size,)
        sample_test, label_test = next(iter(testloader))
        assert sample_test.shape == (batch_size, 1, 28, 28)
        assert label_test.shape == (batch_size,)
