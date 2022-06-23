from utils import *
from core import *
from torchvision import datasets
from torchvision.transforms import ToTensor
import model4FL
import pickle, torch
import streamlit as st
from itertools import starmap
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_path = ROOT_PATH / "configurations"
if not(config_path.exists()):
    raise RuntimeError("Create a `configurations` folder. Then add json files with parameters (see README.md for more information)")

configuration = st.selectbox("Choose the configuration", tuple((file.name for file in config_path.glob('*.json'))))
ON_GPU = st.checkbox("Run on GPU")
REFRESH = st.checkbox("Refresh data distribution")

clicked = st.button("Start")
if clicked:
    # Parameters
    with open(config_path / configuration, "r") as file:
        parameters = json.load(file)

    ROUNDS = parameters["rounds"]
    NWORKERS = parameters["nworkers"]
    EPOCHS = parameters["epochs"]
    volume_distrb = parameters["volume_distrb"]
    label_distrb = parameters["label_distrb"]
    minlabels = parameters.get("minlabels", 3)
    balanced = parameters.get("balanced", True)

    # Loading model, optimizer (in extras)
    if hasattr(model4FL, parameters["model"]):
        module = getattr(model4FL, parameters["model"])
        Model = getattr(module, "Model")
        extras = getattr(module, "extras")
    else:
        raise ImportError(
            "Not found '{}' module in 'model4FL' module".format(parameters["model"])
        )

    if ON_GPU:
        from core.sequential_gpu import train, evaluate
    else:
        from core.parallel import train, evaluate

    # Introduction
    st.header("Federated Reinforcement Learning")
    st.subheader("Information")
    st.table({key.title(): [str(parameters[key]).upper()] for key in parameters})

    # Get the dataset
    with st.spinner("Opening the dataset"):
        datatrain, datatest = dataset(parameters["dataset"])
    st.success("Dataset opened.")

    nclasses = len(datatrain.classes)  # for the model
    size_traindata = len(datatrain)  # for aggregation
    size_testdata = len(datatest)  # for aggregation

    # Get path of data for workers and generate them
    wk_data_path = EXP_PATH / tracker(
        parameters["dataset"], NWORKERS, label_distrb, volume_distrb, minlabels, balanced
    )
    exists = True
    with st.spinner("Generate data for workers"):
        if not (wk_data_path.exists()) or REFRESH:
            exists = False
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
        st.warning("Data for workers are already generated.")
    else:
        st.success("Data for workers are generated successfully.")

    # Experiment path
    exp_path = iterate(EXP_PATH)

    # Initialization of the server
    with st.spinner("Initialization of the server"):
        server = FederatedAveraging(
            Model(nclasses).to(device), size_traindata, size_testdata
        )
    st.success("The server is successfully initialized.")
    # Initialization of workers
    with st.spinner("Initialization of the workers"):
        models = (Model(nclasses) for _ in range(NWORKERS))
        workers = tuple(
            Node(model.to(device), wk_data_path / "worker-{}.pkl".format(i + 1))
            for i, model in enumerate(models)
        )
    st.success("Workers are successfully initialized.")

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
            st.image(
                topng(
                    chart(
                        range(1, len(global_accs) + 1),
                        {"Average accuracy": global_accs},
                        title="Evolution of the average accuracy per round",
                        x_title="Rounds",
                        y_title="Accuracy (in %)",
                    )
                )
            )

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

    st.balloons()
