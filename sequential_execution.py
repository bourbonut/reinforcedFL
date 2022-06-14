from utils import *
from core import *
from torchvision import datasets
from torchvision.transforms import ToTensor
from model4FL.mnist import ModelMNIST, extras
import pickle, torch
import streamlit as st
from itertools import starmap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
NWORKERS = 4
ROUNDS = 10
EPOCHS = 3
ON_GPU = True
PARTITION_TYPE = "nonIID"
label_distrb = "noniid"
volume_distrb = "noniid"
minlabels = 3
balanced = True

if ON_GPU:
    from core.sequential_gpu import train, evaluate
else:
    from core.parallel import train, evaluate

st.header("Federated Reinforcement Learning")
st.subheader("Informations")
st.markdown(
    "This experiment is going to run on {} with **{} rounds** with **{} workers** which are going to be trained on **{} epochs**.".format(
        "GPU" if ON_GPU else "CPU (multithreading)", ROUNDS, NWORKERS, EPOCHS
    )
)
msg = (
    "and "
    + ("**balanced**" if balanced else "**unbalanced**")
    + (
        " with an average of {} labels per worker".format(minlabels)
        if label_distrb == "noniid"
        else ""
    )
)
st.markdown("The distribution of **labels** is **{}**{}.".format(label_distrb, msg))
st.markdown("The distribution of **volume** is **{}**.".format(volume_distrb))

# Get the dataset
with st.spinner("Opening the dataset"):
    isdownloaded = not (DATA_PATH.exists())
    datatrain = datasets.MNIST(
        root="data", train=True, download=isdownloaded, transform=ToTensor()
    )
    datatest = datasets.MNIST(root="data", train=False, transform=ToTensor())
st.success("OK")

nclasses = len(datatrain.classes)  # for the model
size_traindata = len(datatrain)  # for aggregation
size_testdata = len(datatest)  # for aggregation

# Get path of data for workers and generate them
wk_data_path = EXP_PATH / tracker(
    NWORKERS, label_distrb, volume_distrb, minlabels, balanced
)
exists = False
with st.spinner("Generate data for workers"):
    if not (wk_data_path.exists()):
        exists = True
        create(wk_data_path)
        generate(
            wk_data_path,
            datatrain,
            datatest,
            NWORKERS,
            label_distrb=label_distrb,
            volume_distrb=volume_distrb,
            minlabels=minlabels,
            balanced=balanced,
            save2png=True,
        )
if exists:
    st.success("OK")
else:
    st.success("Already done")

# Experiment path
exp_path = iterate(EXP_PATH)

# Initialization of the server
with st.spinner("Initialization of the server"):
    server = FederatedAveraging(
        ModelMNIST(nclasses).to(device), size_traindata, size_testdata
    )
st.success("OK")
# Initialization of workers
with st.spinner("Initialization of the workers"):
    models = (ModelMNIST(nclasses) for _ in range(4))
    workers = tuple(
        Node(model.to(device), wk_data_path / "worker-{}.pkl".format(i + 1))
        for i, model in enumerate(models)
    )
st.success("OK")

# Plot stacked chart
st.image(str(wk_data_path / "distribution.png"))

global_accs = []
placeholder = st.empty()
# Main loop
for r in range(ROUNDS):
    # Workers download the global model
    for worker in workers:
        worker.communicatewith(server)

    # Workers evaluate accuracy of the global model
    # on their local data
    accuracies = evaluate(workers)
    avg_acc = server.global_accuracy(accuracies)
    global_accs.append(avg_acc)
    with placeholder:
        st.line_chart(global_accs)

    # Training loop of workers
    for e in range(EPOCHS):
        curr_path = exp_path / "round{}".format(r) / "epoch{}".format(e)
        create(curr_path, verbose=False)
        train(workers, curr_path)

    # Server downloads all local updates
    for worker in workers:
        server.communicatewith(worker)
    server.update()

with open(exp_path / "result.pkl", "wb") as file:
    pickle.dump(global_accs, file)
