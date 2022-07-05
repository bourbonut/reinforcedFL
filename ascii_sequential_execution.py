from utils import *
from core import *
import model4FL
import pickle, torch, json
from core.federated_learning import aggregation, worker
import argparse
from rich import print as rich_print
from rich.markdown import Markdown
from rich.table import Table
from rich.console import Console
from rich.live import Live
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
arguments = ["environment", "model", "distribution"]
parameters = {key: None for key in arguments}

console = Console()
console.print(Markdown("# Federated Reinforcement Learning"))
console.print(Markdown("## Information"))
for argument in arguments:
    path = Path(getattr(args, argument))
    with open(path, "r") as file:
        parameters[argument] = json.load(file)
    table = Table(title=argument + " information")
    for element in parameters[argument]:
        table.add_column(element)
    table.add_row(*map(str, parameters[argument].values()))
    console.print(table)

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

if ON_GPU:
    from core.sequential_gpu import train, evaluate
    console.print("The program is running on GPU")
else:
    from core.parallel import train, evaluate

# Get the dataset
dataname = parameters["environment"]["dataset"]
with Live("[cyan]Opening the dataset[/]") as live:
    datatrain, datatest = dataset(dataname)
    live.update("[green]Dataset opened.[/]")

nclasses = len(datatrain.classes)  # for the model
# for aggregation
size_traindata = parameters["distribution"].get("k", 1) * len(
    datatrain
)
size_testdata = parameters["distribution"].get("k", 1) * len(
    datatest
)

# Get path of data for workers and generate them
wk_data_path = EXP_PATH / tracker(dataname, NWORKERS, **parameters["distribution"])
exists = True
with Live("[cyan]Generate data for workers[/]") as live:
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
        live.update("[yellow]Data for workers are already generated.[/]")
    else:
        live.update("[green]Data for workers are generated successfully.[/]")

# Experiment path
exp_path = iterate(EXP_PATH)
create(exp_path, verbose=False)
# Save configuration
with open(exp_path / "configuration.json", "w") as file:
    json.dump(parameters, file)

# Initialization of the server
with Live("[cyan]Initialization of the server[/]") as live:
    server = server_class(Model(nclasses).to(device), size_traindata, size_testdata)
    live.update("[green]The server is successfully initialized.[/green]")

# Initialization of workers
with Live("[cyan]Initialization of the workers[/]") as live:
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
    live.update("[green]Workers are successfully initialized.[/]")

console.print("")
# Global accuracies : first list for training
# second list for testing
global_accs = [[], []]
# table = Table("Training accuracies", "Testing accuracies")
table = Table()
table.add_column("Training accuracies")
table.add_column("Testing accuracies")

# Main loop
with Live(table, auto_refresh=False) as live:
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

            # Training loop of workers
            for e in range(EPOCHS):
                curr_path = exp_path / f"round{r}" / f"epoch{e}"
                create(curr_path, verbose=False)
                train(workers, curr_path)

            accuracies = evaluate(workers, True)
            avg_acc = server.global_accuracy(accuracies, True)
            global_accs[0].append(avg_acc)

            # Update the table for training average accuracy
            table.add_row("{:.2%}".format(avg_acc), "{:.2%}".format(global_accs[1][-1]))
            live.refresh()

            # Server downloads all local updates
            for worker in workers:
                server.communicatewith(worker)
            server.update()

with open(exp_path / "result.pkl", "wb") as file:
    pickle.dump(global_accs, file)

console.print("Finished.")
