import torch as t
import numpy as np
from ml_logger import logger
from params_proto import cli_parse
from playground.maml.maml_torch.archive import paper_metrics as metrics
from playground.maml.maml_torch.tasks import Sine
from waterbear import DefaultBear

amp_tasks = [(amp, Sine(npts=100, amp=amp)) for amp in np.arange(5.0, 10.5, 0.5)]
phase_tasks = [(phi0, Sine(npts=100, phi0=phi0)) for phi0 in np.arange(1.0, 2.25, 0.25) * np.pi]

device = t.device('cuda' if t.cuda.is_available() else 'cpu')


def amp_test(model, epoch=None, grad_step=None):
    for amp, task in amp_tasks:
        proper = t.tensor(task.proper()).to(device)
        _loss, _ = metrics.comp_loss(*proper, model)
        logger.log_keyvalue(epoch, f"test-grad-{grad_step}-amp-{amp:.1f}", _loss.item(), silent=True)


def phase_test(model, epoch=None, grad_step=None):
    for phi0, task in phase_tasks:
        proper = t.tensor(task.proper()).to(device)
        _loss, _ = metrics.comp_loss(*proper, model)
        logger.log_keyvalue(epoch, f"test-grad-{grad_step}-phi0-{phi0:.2f}", _loss.item(), silent=True)


def all_tests(*a, **b):
    amp_test(*a, **b)
    phase_test(*a, **b)


def launch_training():
    from playground.maml.maml_torch.maml_multi_step import maml, G

    np.random.seed(G.seed)
    t.manual_seed(G.seed)
    t.cuda.manual_seed(G.seed)

    from datetime import datetime

    now = datetime.now()
    G.log_prefix = f"{now:%Y-%m-%d}/new-maml-torch/out-of-distribution"
    G.n_epochs = 70000  # from cbfinn universality paper
    G.n_gradient_steps = 5
    G.test_grad_steps = [1, 5]
    G.test_interval = 5
    G.save_interval = 100  # save the weights ever 100 epoch.

    logger.configure(log_directory=G.log_dir, prefix=G.log_prefix)
    logger.log_params(G=vars(G))

    maml(test_fn=all_tests)


@cli_parse
class Args:
    log_dir = "/Users/ge/machine_learning/berkeley-playground/ins-runs"
    log_prefix = "2018-06-03/new-maml-torch-uniform-xs/smaller-batch/analysis"
    weight_path = "../modules/79000_FunctionalMLP.pkl"
    grad_steps = 40
    k_shot = 5
    learning_rate = 0.001


def adapt_and_test():
    import os
    import dill
    from playground.maml.maml_torch.maml_multi_step import FunctionalMLP

    logger.configure(log_directory=Args.log_dir, prefix=Args.log_prefix)
    logger.log_params(Args=vars(Args))

    # load weights
    with open(os.path.join(Args.log_dir, Args.log_prefix, Args.weight_path), 'rb') as f:
        weights = dill.load(f)
    model = FunctionalMLP(1, 1)

    losses = DefaultBear(list)
    for amp, task in amp_tasks:
        model.params.update(
            {k: t.tensor(v, requires_grad=True, dtype=t.double).to(device) for k, v in weights[0].items()})
        sgd = t.optim.SGD(model.parameters(), lr=Args.learning_rate)
        proper = t.tensor(task.proper()).to(device)
        samples = t.tensor(task.samples(Args.k_shot)).to(device)

        for grad_ind in range(Args.grad_steps):
            with t.no_grad():
                xs, labels = proper
                ys = model(xs.unsqueeze(-1))
                loss = model.criteria(ys, labels.unsqueeze(-1))
                logger.log(grad_ind, loss=loss.item(), silent=grad_ind != Args.grad_steps - 1)
                losses[f"amp-{amp:.2f}-loss"].append(loss.item())

            xs, labels = samples
            ys = model(xs.unsqueeze(-1))
            loss = model.criteria(ys, labels.unsqueeze(-1))
            sgd.zero_grad()
            loss.backward()
            sgd.step()
        # losses = np.array([v for k, v in losses.items()])

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.title(f'Learning Curves')
    for amp, task in amp_tasks:
        plt.plot(losses[f"amp-{amp:.2f}-loss"], label=f"amp {amp:.2f}")
    plt.legend()
    logger.log_pyplot(None, key=f"losses/learning_curves_amp.png", fig=fig)
    plt.close()

    average_losses = np.array([
        losses[f"amp-{amp:.2f}-loss"] for amp, task in amp_tasks
    ])
    fig = plt.figure()
    plt.title(f'Learning Curves Averaged amp ~ [5 - 10]')
    plt.plot(average_losses.mean(0))
    plt.ylim(0, 28)
    logger.log_pyplot(None, key=f"losses/learning_curves_amp_all.png", fig=fig)
    plt.close()



if __name__ == "__main__":
    # launch_training()
    adapt_and_test()
