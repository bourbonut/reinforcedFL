from utils import *
from core import *
import model4FL
import pickle, torch, json
import streamlit as st
from core.federated_learning import aggregation, worker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_path = ROOT_PATH / "configurations"
if not config_path.exists():
    raise RuntimeError(
        "Create a `configurations` folder. Then add json files with parameters (see README.md for more information)"
    )

# Introduction
st.header("Federated Reinforcement Learning")
st.subheader("Information")

configuration = st.selectbox(
    "Choose the configuration",
    tuple((file.name for file in config_path.glob("*.json"))),
)
# Parameters
with open(config_path / configuration, "r") as file:
    parameters = json.load(file)

st.table({key.title(): [str(parameters[key]).upper()] for key in parameters})
ON_GPU = st.checkbox("Run on GPU")
REFRESH = st.checkbox("Refresh data distribution")

clicked = st.button("Start")
if clicked:
    NEXPS = parameters.get("nexps", 1)
    ROUNDS = parameters["rounds"]
    NWORKERS = parameters["nworkers"]
    EPOCHS = parameters["epochs"]
    server_class = getattr(aggregation, parameters.get("server", "FederatedAveraging"))
    worker_class = getattr(worker, parameters.get("worker", "Worker"))
    volume_distrb = parameters["volume_distrb"]
    label_distrb = parameters["label_distrb"]
    minlabels = parameters.get("minlabels", 3)
    balanced = parameters.get("balanced", True)
    participation = parameters.get("participation", "all")
    participate = PARTICIPATION.get(participation, "all")

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

    # Get the dataset
    with st.spinner("Opening the dataset"):
        datatrain, datatest = dataset(parameters["dataset"])
    st.success("Dataset opened.")

    nclasses = len(datatrain.classes)  # for the model
    size_traindata = len(datatrain)  # for aggregation
    size_testdata = len(datatest)  # for aggregation

    # Get path of data for workers and generate them
    wk_data_path = EXP_PATH / tracker(
        parameters["dataset"],
        NWORKERS,
        label_distrb,
        volume_distrb,
        minlabels,
        balanced,
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
        server = server_class(
            Model(nclasses).to(device), size_traindata, size_testdata
        )
    st.success("The server is successfully initialized.")

    # Initialization of workers
    with st.spinner("Initialization of the workers"):
        models = (Model(nclasses) for _ in range(NWORKERS))
        workers = tuple(
            worker_class(model.to(device), wk_data_path / f"worker-{i+1}.pkl")
            for i, model in enumerate(models)
        )
    st.success("Workers are successfully initialized.")

    # Plot stacked chart
    st.image(str(wk_data_path / "distribution.png"))

    # Global accuracies : first list for training
    # second list for testing
    global_accs = [[], []]
    placeholder = st.empty() # for streamlit
    # Main loop
    for iexp in range(NEXPS):
        for r in range(ROUNDS):
            # Workers download the global model
            for worker in workers:
                worker.communicatewith(server)

            # Workers evaluate accuracy of the global model
            # on their local data
            accuracies = evaluate(workers)
            avg_acc = server.global_accuracy(accuracies)
            global_accs[1].append(avg_acc)

            # Update the line chart for testing average accuracy
            with placeholder:
                st.image(toplot(global_accs))

            # Training loop of workers
            for e in range(EPOCHS):
                curr_path = exp_path / f"round{r}" / f"epoch{e}"
                create(curr_path, verbose=False)
                train(workers, curr_path)

            accuracies = evaluate(workers, True)
            avg_acc = server.global_accuracy(accuracies, True)
            global_accs[0].append(avg_acc)

            # Update the line chart for training average accuracy
            with placeholder:
                st.image(toplot(global_accs))

            # Server downloads all local updates
            for worker in workers:
                server.communicatewith(worker)
            server.update()

    with open(exp_path / "result.pkl", "wb") as file:
        pickle.dump(global_accs, file)

    st.balloons()
