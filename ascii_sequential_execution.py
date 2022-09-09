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
from copy import copy, deepcopy
import statistics
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
    # if "path_model" in parameters["model"]:
    #     path, number = parameters["model"]["path_model"].values()
    #     path = Path().absolute() / path
    #     parameters["model"]["path_model"] = [path / f"actor-{number}.pt", path / f"critic-{number}.pt"]
    scheduler = Scheduler(
        **parameters["model"],
        device=device,
        path=exp_path / "scheduler",
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

a = [worker.computation_speed * (len(worker._train) // worker.batch_size) * worker.epochs for worker in workers]
b = [worker.NB_PARAMS * 1e-6 * 32 / worker.network[0][0] for worker in workers]
c = [worker.NB_PARAMS * 1e-6 * 32 / worker.network[1][0] for worker in workers]
alltimes = [statistics.mean(m) for m in zip(a, b, c)]

ten_best = sorted(alltimes)[: int(len(workers) * 0.1)]
best_indices = set([alltimes.index(i) for i in ten_best])
sx = sorted(alltimes)

# Create a directory
create(exp_path / "agent", verbose=False)
create(exp_path / "scheduler", verbose=False)

# Global accuracies : first list for training
# second list for testing
global_accs = []
state = []
new_state = []
old_action = []

break_now = False
times = []

te_avg_acc = 0

# Panel
console.print(Align.center(Markdown("## Experiments\n")))

# =========================================================================================================

# Training

# Main loop
for iexp in range(NEXPS):
    table = Table(
        "Round",
        "Training accuracies [%]",
        "Testing accuracies [%]",
        "Duration \[s]",
        "Time (computation & communication) \[s]",
        "Quality (ratio & size & tested)",
        title=f"Experiment {iexp}",
    )
    already_selected = set()

    align = Align.center(table)
    with Live(align, auto_refresh=False, vertical_overflow="fold") as live:
        start = perf_counter()
        selection, _ = scheduler.select_next_partipants(state, [])
        old_action = list(selection)
        indices_participants = [i for i in range(len(workers)) if selection[i]]
        participants = [workers[i] for i in indices_participants]

        # for worker in participants:
        #     worker.communicatewith(server)

        # train(participants)
        # pair = evaluate(participants, True, full=True)
        # accuracies, _ = zip(*pair)
        # tr_avg_acc = server.compute_glb_acc(accuracies, indices_participants, True)
        tr_avg_acc = 0

        for worker in participants:
            server.communicatewith(worker)

        state.clear()
        for i, worker in enumerate(workers):
            if i in indices_participants:
                state.extend(worker.compute_times())
            else:
                state.extend([0.0, 0.0, 0.0])

        means = [statistics.mean(filter(lambda x: x!=0., state[i::3])) for i in range(3)]
        state = [(random.random() * 0.2 + 0.9) * means[i%3] if s==0. else s for i, s in enumerate(state)]

        server.update(indices_participants)

        # for worker in workers:
        #     worker.communicatewith(server)

        # pair = evaluate(workers, full=True)
        # accuracies, _ = zip(*pair)
        # te_avg_acc = server.compute_glb_acc(accuracies, list(range(len(workers))))
        te_avg_acc = 0
        duration = perf_counter() - start

        max_time = max((sel * sum(time) for sel, time in zip(selection, scheduler.grouped(state))))
        table.add_row(
            str(1),
            f"{tr_avg_acc:.2%}",
            f"{te_avg_acc:.2%}",
            f"{duration:.3f} s",
            f"{max_time:.3f} s",
            "Random round",
        )
        times.append(max_time)
        live.refresh()
        global_accs.append((tr_avg_acc, te_avg_acc))
        
        for r in range(1, ROUNDS):
            start = perf_counter()

            # Selection of future participants
            selection, action = scheduler.select_next_partipants(state, old_action, debug=alltimes)
            indices_participants = [i for i in range(len(workers)) if selection[i]]
            already_selected.update(set(indices_participants))

            participants = [workers[i] for i in indices_participants]
            if len(participants) <= 10: # 10:
                break_now = True
                break

            # Workers download the global model
            # for worker in participants:
            #     worker.communicatewith(server)

            # # Training loop of workers
            # train(participants)

            # # Workers evaluate accuracy of global model
            # # on their local training data
            # pair = evaluate(participants, True, full=True)
            # accuracies, _ = zip(*pair)
            # tr_avg_acc = server.compute_glb_acc(accuracies, indices_participants, True)
            tr_avg_acc = 0
            # server.collects_training_accuracies(singular_accuracies)

            # # Server downloads all local updates
            # for worker in participants:
            #     server.communicatewith(worker)

            new_state.clear()
            for i, worker in enumerate(workers):
                if i in indices_participants:
                    new_state.extend(worker.compute_times())
                else:
                    new_state.extend(state[3 * i: 3 * (i + 1)])

            # server.update(indices_participants)
            reward = scheduler.compute_reward(selection, new_state)
            scheduler.memory.push(
                    scheduler.normalize_all(state).tolist(),
                    action,
                    0,
                    scheduler.normalize_all(new_state).tolist(),
                    reward,
            )
            scheduler.update()
            old_action = list(action)

            # for worker in workers:
            #     worker.communicatewith(server)

            # # Workers evaluate accuracy of the global model
            # # on their local testing data
            # pair = evaluate(workers, full=True)
            # accuracies, _ = zip(*pair)
            # te_avg_acc = server.compute_glb_acc(accuracies, list(range(len(workers))))
            te_avg_acc = 0
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
                f"{tr_avg_acc:.2%}",
                f"{te_avg_acc:.2%}",
                f"{duration:.3f} s",
                f"{max_time:.3f} s",
                f"{(ratio, len(indices_participants), len(already_selected))}",
            )
            live.refresh()
            times.append(max_time)

            # Update global_accs
            global_accs.append((tr_avg_acc, te_avg_acc))
            r += 1

    # Reset the server
    # server.reset(exp_path / "agent" / f"loss-rl-{iexp}.png")
    # server.global_model = Model(nclasses, device).to(device)
    scheduler.reset()
    state.clear()
    old_action.clear()

    # Reset workers
    # for worker in workers:
    #    worker.model = Model(nclasses, device).to(device)

    # Save results
    # with open(exp_path / f"global_accs-training-{iexp}.pkl", "wb") as file:
    #     pickle.dump(global_accs, file)

    with open(exp_path / f"times-training-{iexp}.pkl", "wb") as file:
        pickle.dump(times, file)
        
    global_accs.clear()
    if break_now:
        break

server.reset()
server.global_model = Model(nclasses, device).to(device)
for worker in workers:
   worker.model = Model(nclasses, device).to(device)
scheduler.finish()
# =========================================================================================================

# Evaluation
scheduler.agent.set_eval()

global_accs = []
state = []
new_state = []
old_action = []

break_now = False
times = []

static_times = [[worker.compute_times() for worker in workers] for _ in range(50)]

# Main loop
for iexp in range(NEXPS):
    table = Table(
        "Round",
        "Training accuracies [%]",
        "Testing accuracies [%]",
        "Duration \[s]",
        "Time (computation & communication) \[s]",
        title=f"Experiment {iexp}",
    )
    already_selected = set()

    align = Align.center(table)
    with Live(align, auto_refresh=False, vertical_overflow="fold") as live:
        start = perf_counter()
        selection, _ = scheduler.select_next_partipants(state, [])
        old_action = list(selection)
        indices_participants = [i for i in range(len(workers)) if selection[i]]
        participants = [workers[i] for i in indices_participants]

        for worker in participants:
            worker.communicatewith(server)

        train(participants)
        pair = evaluate(participants, True, full=True)
        accuracies, _ = zip(*pair)
        tr_avg_acc = server.compute_glb_acc(accuracies, indices_participants, True)

        for worker in participants:
            server.communicatewith(worker)

        local_times = static_times[0]
        state.clear()
        for i, worker in enumerate(workers):
            if i in indices_participants:
                state.extend(local_times[i])
            else:
                state.extend([0.0, 0.0, 0.0])

        means = [statistics.mean(filter(lambda x: x!=0., state[i::3])) for i in range(3)]
        state = [(random.random() * 0.2 + 0.9) * means[i%3] if s==0. else s for i, s in enumerate(state)]

        server.update(indices_participants)

        for worker in workers:
            worker.communicatewith(server)

        pair = evaluate(workers, full=True)
        accuracies, _ = zip(*pair)
        te_avg_acc = server.compute_glb_acc(accuracies, list(range(len(workers))))
        duration = perf_counter() - start

        max_time = max((sel * sum(time) for sel, time in zip(selection, scheduler.grouped(state))))
        table.add_row(
            str(1),
            f"{tr_avg_acc:.2%}",
            f"{te_avg_acc:.2%}",
            f"{duration:.3f} s",
            f"{max_time:.3f} s",
        )
        times.append(max_time)
        live.refresh()
        global_accs.append((tr_avg_acc, te_avg_acc))
        
        # for r in range(1, ROUNDS):
        r = 1
        while te_avg_acc < 0.95: # 0.95:
            start = perf_counter()

            # Selection of future participants
            selection, action = scheduler.select_next_partipants(state, old_action)
            indices_participants = [i for i in range(len(workers)) if selection[i]]
            already_selected.update(set(indices_participants))

            participants = [workers[i] for i in indices_participants]
            if len(participants) == 0:
                break_now = True
                break

            # Workers download the global model
            for worker in participants:
                worker.communicatewith(server)

            # Training loop of workers
            train(participants)

            # Workers evaluate accuracy of global model
            # on their local training data
            pair = evaluate(participants, True, full=True)
            accuracies, _ = zip(*pair)
            tr_avg_acc = server.compute_glb_acc(accuracies, indices_participants, True)
            # server.collects_training_accuracies(singular_accuracies)

            # Server downloads all local updates
            for worker in participants:
                server.communicatewith(worker)

            server.update(indices_participants)

            local_times = static_times[r]
            new_state.clear()
            for i, worker in enumerate(workers):
                if i in indices_participants:
                    new_state.extend(local_times[i])
                else:
                    new_state.extend(state[3 * i: 3 * (i + 1)])

            for worker in workers:
                worker.communicatewith(server)

            # Workers evaluate accuracy of the global model
            # on their local testing data
            pair = evaluate(workers, full=True)
            accuracies, _ = zip(*pair)
            te_avg_acc = server.compute_glb_acc(accuracies, list(range(len(workers))))
            duration = perf_counter() - start

            max_time = max((sel * sum(time) for sel, time in zip(selection, scheduler.grouped(new_state))))
            state = copy(new_state)
            # Update the table
            table.add_row(
                str(r + 1),
                f"{tr_avg_acc:.2%}",
                f"{te_avg_acc:.2%}",
                f"{duration:.3f} s",
                f"{max_time:.3f} s",
            )
            live.refresh()
            times.append(max_time)

            # Update global_accs
            global_accs.append((tr_avg_acc, te_avg_acc))
            r += 1

    # Reset the server
    # server.reset(exp_path / "agent" / f"loss-rl-{iexp}.png")
    # server.global_model = Model(nclasses, device).to(device)
    # scheduler.reset()
    # state.clear()
    # old_action.clear()

    # Reset workers
    # for worker in workers:
    #    worker.model = Model(nclasses, device).to(device)

    # Save results
    with open(exp_path / f"global_accs-evaluation-{iexp}.pkl", "wb") as file:
        pickle.dump(global_accs, file)

    with open(exp_path / f"times-evaluation-{iexp}.pkl", "wb") as file:
        pickle.dump(times, file)
        
    global_accs.clear()
    break

server.reset()
server.global_model = Model(nclasses, device).to(device)
for worker in workers:
   worker.model = Model(nclasses, device).to(device)
# =======================================================================================================

# Federated Averaging

global_accs = []
state = []
new_state = []
old_action = []

break_now = False
times = []

# Main loop
for iexp in range(NEXPS):
    table = Table(
        "Round",
        "Training accuracies [%]",
        "Testing accuracies [%]",
        "Duration \[s]",
        "Time (computation & communication) \[s]",
        title=f"Experiment {iexp}",
    )
    already_selected = set()

    align = Align.center(table)
    with Live(align, auto_refresh=False, vertical_overflow="fold") as live:
        start = perf_counter()
        # selection, _ = scheduler.select_next_partipants(state, [])
        # old_action = list(selection)
        # indices_participants = [i for i in range(len(workers)) if selection[i]]
        indices_participants = random.sample(list(range(len(workers))), int(0.1 * len(workers)))
        selection = [1 if i in indices_participants else 0 for i in range(NWORKERS)]
        participants = [workers[i] for i in indices_participants]

        for worker in participants:
            worker.communicatewith(server)

        train(participants)
        pair = evaluate(participants, True, full=True)
        accuracies, _ = zip(*pair)
        tr_avg_acc = server.compute_glb_acc(accuracies, indices_participants, True)

        for worker in participants:
            server.communicatewith(worker)

        local_times = static_times[0]
        state.clear()
        for i, worker in enumerate(workers):
            if i in indices_participants:
                state.extend(local_times[0])
            else:
                state.extend([0.0, 0.0, 0.0])

        means = [statistics.mean(filter(lambda x: x!=0., state[i::3])) for i in range(3)]
        state = [(random.random() * 0.2 + 0.9) * means[i%3] if s==0. else s for i, s in enumerate(state)]

        server.update(indices_participants)

        for worker in workers:
            worker.communicatewith(server)

        pair = evaluate(workers, full=True)
        accuracies, _ = zip(*pair)
        te_avg_acc = server.compute_glb_acc(accuracies, list(range(len(workers))))
        duration = perf_counter() - start

        max_time = max((sel * sum(time) for sel, time in zip(selection, scheduler.grouped(state))))
        table.add_row(
            str(1),
            f"{tr_avg_acc:.2%}",
            f"{te_avg_acc:.2%}",
            f"{duration:.3f} s",
            f"{max_time:.3f} s",
        )
        times.append(max_time)
        live.refresh()
        global_accs.append((tr_avg_acc, te_avg_acc))
        
        # for r in range(1, ROUNDS):
        r = 1
        while te_avg_acc < 0.95: # 0.95:
            start = perf_counter()

            # Selection of future participants
            # selection, action = scheduler.select_next_partipants(state, old_action)
            # indices_participants = [i for i in range(len(workers)) if selection[i]]
            indices_participants = random.sample(list(range(len(workers))), int(0.1 * len(workers)))
            selection = [1 if i in indices_participants else 0 for i in range(NWORKERS)]
            already_selected.update(set(indices_participants))

            participants = [workers[i] for i in indices_participants]
            if len(participants) == 0:
                break_now = True
                break

            # Workers download the global model
            for worker in participants:
                worker.communicatewith(server)

            # Training loop of workers
            train(participants)

            # Workers evaluate accuracy of global model
            # on their local training data
            pair = evaluate(participants, True, full=True)
            accuracies, _ = zip(*pair)
            tr_avg_acc = server.compute_glb_acc(accuracies, indices_participants, True)
            # server.collects_training_accuracies(singular_accuracies)

            # Server downloads all local updates
            for worker in participants:
                server.communicatewith(worker)

            server.update(indices_participants)

            local_times = static_times[r]
            new_state.clear()
            for i, worker in enumerate(workers):
                if i in indices_participants:
                    new_state.extend(local_times[i])
                else:
                    new_state.extend(state[3 * i: 3 * (i + 1)])

            # old_action = list(action)

            for worker in workers:
                worker.communicatewith(server)

            # Workers evaluate accuracy of the global model
            # on their local testing data
            pair = evaluate(workers, full=True)
            accuracies, _ = zip(*pair)
            te_avg_acc = server.compute_glb_acc(accuracies, list(range(len(workers))))
            # server.collects_global_accuracies(singular_accuracies, indices_participants)
            duration = perf_counter() - start

            #print(f"{list(scheduler.grouped(new_state)) = }")
            #max_time = max((sum(time) for time in scheduler.grouped(new_state)))
            max_time = max((sel * sum(time) for sel, time in zip(selection, scheduler.grouped(new_state))))
            state = copy(new_state)
            # Update the table
            table.add_row(
                str(r + 1),
                f"{tr_avg_acc:.2%}",
                f"{te_avg_acc:.2%}",
                f"{duration:.3f} s",
                f"{max_time:.3f} s",
            )
            live.refresh()
            times.append(max_time)

            # Update global_accs
            global_accs.append((tr_avg_acc, te_avg_acc))
            r += 1

    # Reset the server
    # server.reset(exp_path / "agent" / f"loss-rl-{iexp}.png")
    # server.global_model = Model(nclasses, device).to(device)
    # scheduler.reset()
    # state.clear()
    # old_action.clear()

    # Reset workers
    # for worker in workers:
    #    worker.model = Model(nclasses, device).to(device)

    # Save results
    with open(exp_path / f"global_accs-fedavg-{iexp}.pkl", "wb") as file:
        pickle.dump(global_accs, file)

    with open(exp_path / f"times-fedavg-{iexp}.pkl", "wb") as file:
        pickle.dump(times, file)
        
    global_accs.clear()
    break
server.finish(exp_path / "agent")
scheduler.finish()
console.print("Finished.")
