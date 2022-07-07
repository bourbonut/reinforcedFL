from utils import *
from core import *
import model4FL
import pickle, torch, json
from core.federated_learning import aggregation, worker
import argparse
from rich import print as rich_print
from rich.markdown import Markdown
from rich.table import Table
from rich.console import Console, Group
from rich.align import Align
from rich.live import Live
from rich.panel import Panel
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(dest="environment", help="environment path")
parser.add_argument(dest="distribution", help="distribution path")
parser.add_argument(dest="model", help="model path")
parser.add_argument(
    "--refresh", action="store_true", dest="refresh", help="Refresh data distribution"
)
parser.add_argument("--gpu", action="store_true", dest="gpu", help="Run on GPU")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Introduction
arguments = ["environment", "distribution", "model"]
parameters = {key: None for key in arguments}

console = Console()
console.print(Markdown("# Federated Reinforcement Learning"))

panel = Panel("", title="Information")
tables = []
with Live(panel, auto_refresh=False) as live:
    for argument in arguments:
        path = Path(getattr(args, argument))
        with open(path, "r") as file:
            parameters[argument] = json.load(file)
        table = Table(title=argument + " information")
        for element in parameters[argument]:
            table.add_column(element)
        table.add_row(*map(str, parameters[argument].values()))
        align = Align.center(table)
        tables.append(align)
        panel.renderable = Group(*tables)
        live.refresh()

NEXPS = parameters["environment"].get("nexps", 1)
ROUNDS = parameters["environment"]["rounds"]
NWORKERS = parameters["environment"]["nworkers"]
EPOCHS = parameters["environment"]["epochs"]
server_class = getattr(aggregation, parameters["model"]["server_class"])
worker_class = getattr(worker, parameters["model"]["worker_class"])

ON_GPU = args.gpu
REFRESH = args.refresh

# Loading model, optimizer (in extras)
if hasattr(model4FL, parameters["model"]["task_model"]):
    module = getattr(model4FL, parameters["model"]["task_model"])
    Model = getattr(module, "Model")
    extras = getattr(module, "extras")
else:
    raise ImportError(
        f"Not found \"{parameters['model']['task_model']}\" module in 'model4FL' module"
    )

panel = Panel("", title="Initialization")
texts = []
with Live(panel, auto_refresh=False) as live:
    if ON_GPU:
        from core.sequential_gpu import train, evaluate

        texts.append(Align.center("The program is running on GPU"))
        panel.renderable = Group(*texts)
        live.refresh()
    else:
        from core.parallel import train, evaluate

    # Get the dataset
    dataname = parameters["environment"]["dataset"]
    texts.append(Align.center("[cyan]Opening the dataset[/]"))
    panel.renderable = Group(*texts)
    live.refresh()
    datatrain, datatest = dataset(dataname)
    texts[-1] = Align.center("[green]Dataset opened.[/]")
    panel.renderable = Group(*texts)
    live.refresh()

    nclasses = len(datatrain.classes)  # for the model
    # for aggregation
    size_traindata = parameters["distribution"].get("k", 1) * len(datatrain)
    size_testdata = parameters["distribution"].get("k", 1) * len(datatest)

    # Get path of data for workers and generate them
    wk_data_path = EXP_PATH / tracker(dataname, NWORKERS, **parameters["distribution"])
    exists = True
    texts.append(Align.center("[cyan]Generate data for workers[/]"))
    panel.renderable = Group(*texts)
    live.refresh()
    if not (wk_data_path.exists()) or REFRESH:
        exists = False
        create(wk_data_path)
        generate(
            wk_data_path,
            datatrain,
            datatest,
            NWORKERS,
            save2png=True,
            **parameters["distribution"],
        )
    if exists:
        texts[-1] = Align.center("[yellow]Data for workers are already generated.[/]")
        panel.renderable = Group(*texts)
        live.refresh()
    else:
        texts[-1] = Align.center(
            "[green]Data for workers are generated successfully.[/]"
        )
        panel.renderable = Group(*texts)
        live.refresh()

    # Experiment path
    exp_path = iterate(EXP_PATH)
    create(exp_path, verbose=False)
    # Save configuration
    with open(exp_path / "configuration.json", "w") as file:
        json.dump(parameters, file)

    # Initialization of the server
    texts.append(Align.center("[cyan]Initialization of the server[/]"))
    panel.renderable = Group(*texts)
    live.refresh()
    server = server_class(
        Model(nclasses).to(device), size_traindata, size_testdata, **parameters["model"]
    )
    texts[-1] = Align.center("[green]The server is successfully initialized.[/]")
    panel.renderable = Group(*texts)
    live.refresh()

    # Initialization of workers
    texts.append(Align.center("[cyan]Initialization of the workers[/]"))
    panel.renderable = Group(*texts)
    live.refresh()
    models = (Model(nclasses) for _ in range(NWORKERS))
    batch_size = parameters["model"].get("batch_size", 64)
    workers = tuple(
        worker_class(
            model.to(device),
            wk_data_path / f"worker-{i+1}.pkl",
            batch_size=batch_size,
        )
        for i, model in enumerate(models)
    )
    texts[-1] = Align.center("[green]Workers are successfully initialized.[/]")
    panel.renderable = Group(*texts)
    live.refresh()

# Create a directory
create(exp_path / "agent", verbose=False)
# Global accuracies : first list for training
# second list for testing
global_accs = [[], []]
tables = []
panel = Panel("", title="Experiment")

# Main loop
with Live(panel, auto_refresh=False, vertical_overflow="fold") as live:
    for iexp in range(NEXPS):
        table = Table(
            "Round",
            "Training accuracies",
            "Testing accuracies",
            title=f"Experiment {iexp}",
        )
        tables.append(Align.center(table))
        panel.renderable = Group(*tables)
        live.refresh()
        for r in range(ROUNDS):
            # Workers download the global model
            for worker in workers:
                worker.communicatewith(server)

            # Workers evaluate accuracy of the global model
            # on their local data
            accuracies = evaluate(workers)
            avg_acc = server.global_accuracy(accuracies)
            global_accs[1].append(avg_acc)

            # Training loop of workers
            for e in range(EPOCHS):
                # No save of loss evolution
                # curr_path = exp_path / f"round{r}" / f"epoch{e}"
                # create(curr_path, verbose=False)
                # train(workers, curr_path)
                train(workers)

            accuracies = evaluate(workers, True)
            avg_acc = server.global_accuracy(accuracies, True)
            global_accs[0].append(avg_acc)

            # Update the table for training average accuracy
            table.add_row(
                str(r + 1),
                "{:.2%}".format(avg_acc),
                "{:.2%}".format(global_accs[1][-1]),
            )
            live.refresh()

            # Server downloads all local updates
            for worker in workers:
                server.communicatewith(worker)
            server.update()

        # Reset the server
        server.reset(exp_path / "agent" / f"loss-rl-{iexp}.png")
        server.global_model = Model(nclasses).to(device)

        # Reset workers
        for worker in workers:
            worker.model = Model(nclasses).to(device)

        # Save results
        with open(exp_path / f"global_accs-{iexp}.pkl", "wb") as file:
            pickle.dump(global_accs, file)

        global_accs[0].clear()
        global_accs[1].clear()

server.finish(exp_path / "agent")
console.print("Finished.")
