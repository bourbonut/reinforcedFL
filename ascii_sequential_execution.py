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
from time import perf_counter
import random

parser = argparse.ArgumentParser()
parser.add_argument(dest="environment", help="environment path")
parser.add_argument(dest="distribution", help="distribution path")
parser.add_argument(dest="model", help="model path")
parser.add_argument(
    "--refresh", action="store_true", dest="refresh", help="Refresh data distribution"
)
parser.add_argument("--gpu", action="store_true", dest="gpu", help="Run on GPU")
args = parser.parse_args()

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
device = torch.device("cuda" if ON_GPU and torch.cuda.is_available() else "cpu")

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
        Model(nclasses, device).to(device),
        [],
        [],
        **parameters["model"],
    )
    texts[-1] = Align.center("[green]The server is successfully initialized.[/]")
    panel.renderable = Group(*texts)
    live.refresh()

    # Initialization of the scheduler (participation agent)
    texts.append(Align.center("[cyan]Initialization of the scheduler[/]"))
    # scheduler = Scheduler(
    #    parameters["model"]["ninput"], parameters["model"]["noutput"], device, exp_path / "scheduler",
    #)
    texts[-1] = Align.center("[green]The scheduler is successfully initialized.[/]")
    panel.renderable = Group(*texts)
    live.refresh()

    # Initialization of workers
    texts.append(Align.center("[cyan]Initialization of the workers[/]"))
    panel.renderable = Group(*texts)
    live.refresh()
    models = (Model(nclasses, device) for _ in range(NWORKERS))
    batch_size = parameters["model"].get("batch_size", 64)
    workers = tuple(
        worker_class(
            model.to(device),
            wk_data_path / f"worker-{i+1}.pkl",
            EPOCHS,
            batch_size=batch_size,
        )
        for i, model in enumerate(models)
    )
    server.n = [len(worker._train) for worker in workers]
    server.t = [len(worker._test) for worker in workers]
    texts[-1] = Align.center("[green]Workers are successfully initialized.[/]")
    panel.renderable = Group(*texts)
    live.refresh()

# Create a directory
create(exp_path / "agent", verbose=False)
create(exp_path / "scheduler", verbose=False)

# Global accuracies : first list for training
# second list for testing
global_accs = []
state = []

# Panel
console.print(Align.center(Markdown("## Experiments\n")))
# Main loop
for iexp in range(NEXPS):
    table = Table(
        "Round",
        "Training accuracies [%]",
        "Testing accuracies [%]",
        "Duration \[s]",
        "Losses",
        title=f"Experiment {iexp}",
    )

    # Global initial accuracies
    pair = evaluate(workers, full=True)
    accuracies, singular_accuracies = zip(*pair)
    server.update_delta(accuracies)
    # server.collects_global_accuracies(singular_accuracies)

    align = Align.center(table)
    with Live(align, auto_refresh=False, vertical_overflow="fold") as live:
        for r in range(ROUNDS):
            start = perf_counter()

            # Selection of future participants
            # indices_participants = scheduler.select_next_partipants(state)
            indices_participants = random.sample(list(range(len(workers))), len(workers) // 10)
            participants = [workers[i] for i in indices_participants]

            # Workers download the global model
            for worker in participants:
                worker.communicatewith(server)

            # Training loop of workers
            train(participants)

            # Workers evaluate accuracy of global model
            # on their local training data
            pair = evaluate(participants, True, full=True)
            accuracies, singular_accuracies = zip(*pair)
            tr_avg_acc = server.compute_glb_acc(accuracies, indices_participants, True)
            # server.collects_training_accuracies(singular_accuracies)

            # Server downloads all local updates
            for worker in participants:
                server.communicatewith(worker)

            server.update(indices_participants)

            for worker in workers:
                worker.communicatewith(server)

            # Workers evaluate accuracy of the global model
            # on their local testing data
            pair = evaluate(workers, full=True)
            accuracies, singular_accuracies = zip(*pair)
            te_avg_acc = server.compute_glb_acc(accuracies, list(range(len(workers))))
            # server.collects_global_accuracies(singular_accuracies, indices_participants)
            duration = perf_counter() - start

            # Train the agent
            server.train_agent(accuracies)

            # Update the table
            table.add_row(
                str(r + 1),
                f"{tr_avg_acc:.2%}",
                f"{te_avg_acc:.2%}",
                f"{duration:.3f} s",
                f"{server.batch_loss}",
            )
            live.refresh()

            # Update global_accs
            global_accs.append((tr_avg_acc, te_avg_acc))

    # Reset the server
    server.reset(exp_path / "agent" / f"loss-rl-{iexp}.png")
    server.global_model = Model(nclasses, device).to(device)
    # scheduler.reset()
    state.clear()

    # Reset workers
    for worker in workers:
        worker.model = Model(nclasses, device).to(device)

    # Save results
    with open(exp_path / f"global_accs-{iexp}.pkl", "wb") as file:
        pickle.dump(global_accs, file)

    global_accs.clear()

server.finish(exp_path / "agent")
console.print("Finished.")
