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
from copy import copy
import statistics

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
        Model(nclasses, device).to(device), [], [], **parameters["model"]
    )
    texts[-1] = Align.center("[green]The server is successfully initialized.[/]")
    panel.renderable = Group(*texts)
    live.refresh()

    # Initialization of the scheduler (participation agent)
    texts.append(Align.center("[cyan]Initialization of the scheduler[/]"))
    scheduler = Scheduler(
        **parameters["model"], device=device, path=exp_path / "scheduler"
    )
    texts[-1] = Align.center("[green]The scheduler is successfully initialized.[/]")
    panel.renderable = Group(*texts)
    live.refresh()

    # Initialization of workers
    texts.append(Align.center("[cyan]Initialization of the workers[/]"))
    panel.renderable = Group(*texts)
    live.refresh()
    models = (Model(nclasses, device) for _ in range(NWORKERS))
    batch_size = parameters["model"].get("batch_size", 64)
    with open(wk_data_path / "allpartitions.pkl", "rb") as file:
        allpartitions = pickle.load(file)
    workers = tuple(
        worker_class(
            *allpartitions[i],
            model.to(device),
            wk_data_path / f"worker-{i+1}.pkl",
            EPOCHS,
            batch_size=batch_size,
        )
        for i, model in enumerate(models)
    )
    server.n = [len(worker._train) for worker in workers]
    server.t = [len(worker._test) for worker in workers]
    a = [
        worker.computation_speed * (worker.a // worker.batch_size) * worker.epochs
        for worker in workers
    ]
    b = [worker.NB_PARAMS * 1e-6 * 32 / worker.network[0][0] for worker in workers]
    c = [worker.NB_PARAMS * 1e-6 * 32 / worker.network[1][0] for worker in workers]
    alltimes = [statistics.mean(m) for m in zip(a, b, c)]
    scheduler.mean = torch.tensor([[statistics.mean(x) for x in (a, b, c)]] * NWORKERS)
    a = [worker.STD for worker in workers]
    b = [worker.NB_PARAMS * 1e-6 * 32 / worker.network[0][1] for worker in workers]
    c = [worker.NB_PARAMS * 1e-6 * 32 / worker.network[1][1] for worker in workers]
    scheduler.std = torch.tensor([[statistics.mean(x) for x in (a, b, c)]] * NWORKERS)
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
new_state = []
old_action = []

# alltimes = []
ten_best = sorted(alltimes)[: int(len(workers) * 0.1)]
best_indices = set([alltimes.index(i) for i in ten_best])
sx = sorted(alltimes)
# print(", ".join((f"{x:>8_.3f}" for x in sx)))
# print(best_indices)

break_now = False

times = []

# Panel
console.print(Align.center(Markdown("## Experiments\n")))
# Main loop
for iexp in range(NEXPS):
    table = Table(
        "Round",
        "Duration \[s]",
        "Losses",
        "Time (computation & communication) \[s]",
        "Quality (ratio & size & tested)",
        title=f"Experiment {iexp}",
    )
    already_selected = set()

    align = Align.center(table)
    with Live(align, auto_refresh=False, vertical_overflow="fold") as live:
        start = perf_counter()
        selection = scheduler.select_next_partipants(state, [])
        old_action = copy(selection)
        indices_participants = [i for i in range(NWORKERS) if selection[i]]
        # print(f"{indices_participants = }")
        participants = [workers[i] for i in indices_participants]

        state.clear()
        for i, worker in enumerate(workers):
            if i in indices_participants:
                state.extend(worker.compute_times())
            else:
                state.extend([0.0, 0.0, 0.0])

        means = [
            statistics.mean(filter(lambda x: x != 0.0, state[i::3])) for i in range(3)
        ]
        state = [
            (random.random() * 0.2 + 0.9) * means[i % 3] if s == 0.0 else s
            for i, s in enumerate(state)
        ]

        # print("Update state:", state)
        # print(f"{state = }")
        server.update(indices_participants)

        duration = perf_counter() - start

        iterator = zip(selection, scheduler.grouped(state))
        max_time = max((sel * sum(time) for sel, time in iterator))
        table.add_row(
            str(1),
            f"{duration:.3f} s",
            "Random round",
            f"{max_time:.3f} s",
            "Random round",
        )
        times.append(max_time)
        live.refresh()
        for r in range(1, ROUNDS):
            start = perf_counter()

            # Selection of future participants
            action = selection = scheduler.select_next_partipants(
                state, old_action, debug=alltimes
            )
            indices_participants = [i for i in range(NWORKERS) if selection[i]]
            already_selected.update(set(indices_participants))
            # print(f"{indices_participants = }")
            participants = [workers[i] for i in indices_participants]
            if len(participants) == 0:
                break_now = True
                break

            new_state.clear()
            for i, worker in enumerate(workers):
                if i in indices_participants:
                    new_state.extend(worker.compute_times())
                else:
                    new_state.extend(state[3 * i : 3 * (i + 1)])

            # print(f"Reward of round {r + 1}")
            reward = scheduler.compute_reward(selection, new_state)
            old_action = list(action)
            scheduler.memory.push(state, action, 0, new_state, reward)

            duration = perf_counter() - start

            # print(f"{list(scheduler.grouped(new_state)) = }")
            # max_time = max((sum(time) for time in scheduler.grouped(new_state)))
            iterator = zip(selection, scheduler.grouped(new_state))
            max_time = max((sel * sum(time) for sel, time in iterator))
            state = copy(new_state)
            num = len(best_indices.intersection(indices_participants))
            ratio = num / len(best_indices)
            # Update the table
            table.add_row(
                str(r + 1),
                f"{duration:.3f} s",
                f"{scheduler.agent.losses}",
                f"{max_time:.3f} s",
                f"{(ratio, len(indices_participants), len(already_selected))}",
            )
            times.append(max_time)
            live.refresh()

    if break_now:
        break

    scheduler.update()
    scheduler.reset()
    state.clear()
    old_action.clear()


scheduler.finish()
with open(exp_path / "scheduler" / "times.pkl", "wb") as file:
    pickle.dump(times, file)

console.print("Finished.")
