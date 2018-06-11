from collections import defaultdict

import torch as t
import numpy as np
from ml_logger import logger
from params_proto import cli_parse
from playground.maml.maml_torch.tasks import Sine
from tqdm import tqdm

device = t.device('cuda' if t.cuda.is_available() else 'cpu')


@cli_parse
class Args:
    log_dir = "/Users/ge/machine_learning/berkeley-playground/ins-runs"
    log_prefix = "2018-06-09/maml_baselines/analysis"
    weight_path = "../modules/79000_FunctionalMLP.pkl"
    grad_steps = 40
    k_shot = 5
    test_k_shot = 5
    learning_rate = 0.001
    batch_size = 600


def thunk(**kwargs):
    task = Sine(**kwargs)
    return task.samples(Args.k_shot), task.samples(Args.test_k_shot)


# use these as the standard testing data.
def generate_amps_data():
    amps = np.arange(5.0, 10.25, 0.25)
    return amps, {amp: [thunk(amp=amp)
                        for _ in range(Args.batch_size)
                        ] for amp in amps}


def generate_phi0_data():
    phi0s = np.arange(1, 5) * np.pi
    return phi0s, {phi0: [thunk(phi0=phi0)
                          for _ in range(Args.batch_size)
                          ] for phi0 in phi0s}


def adapt_and_test(model):
    import os
    import dill

    # load weights
    with open(os.path.join(Args.log_dir, Args.log_prefix, Args.weight_path), 'rb') as f:
        weights = dill.load(f)

    def adaptation_losses(few_shot_xs, few_shot_labels, test_xs, test_labels):
        """
        returns mse-loss with len(grad_steps + 1) in the format of
                train_loss, test_loss
        """
        few_shot_xs = t.tensor(few_shot_xs).to(device)
        few_shot_labels = t.tensor(few_shot_labels).to(device)
        test_xs = t.tensor(test_xs).to(device)
        test_labels = t.tensor(test_labels).to(device)
        model.params.update(
            {k: t.tensor(v, requires_grad=True).to(device) for k, v in weights[0].items()})
        sgd = t.optim.SGD(model.parameters(), lr=Args.learning_rate)
        for grad_ind in range(Args.grad_steps + 1):
            with t.no_grad():
                if hasattr(model, "is_recurrent") and model.is_recurrent:
                    h0 = model.h0_init()
                    ys, ht = model(test_xs.unsqueeze(-1).unsqueeze(-1), h0)
                    # ys : Size(5, batch_n == 1, 1).
                    ys = ys.squeeze(-1)
                else:
                    ys = model(test_xs.unsqueeze(-1))

                test_loss = model.criteria(ys, test_labels.unsqueeze(-1))

            if hasattr(model, "is_recurrent") and model.is_recurrent:
                h0 = model.h0_init()
                ys, ht = model(few_shot_xs.unsqueeze(-1).unsqueeze(-1), h0)
                # ys : Size(5, batch_n == 1, 1).
                ys = ys.squeeze(-1)
            else:
                ys = model(few_shot_xs.unsqueeze(-1))
            loss = model.criteria(ys, few_shot_labels.unsqueeze(-1))
            sgd.zero_grad()
            loss.backward()
            sgd.step()

            yield loss.item(), test_loss.item()

    losses = defaultdict(lambda: 0)
    amps, data = generate_amps_data()
    for amp in tqdm(amps, desc='amps'):
        sample_batch = data[amp]
        for few_shots, proper_samples in sample_batch:
            losses[amp] += np.array([*adaptation_losses(*few_shots, *proper_samples)])
        losses[amp] /= len(sample_batch)

    import matplotlib.pyplot as plt
    fig_adaptation = plt.figure()
    plt.title(f'Loss Curves amp [5 - 10]')
    for amp in amps:
        # note: pick the test_loss instead of the train_loss
        plt.plot(losses[amp][:, 1], label=f"amp {amp:.2f}")
    # plt.legend()
    plt.ylim(0, 28)  # from Chelsea's paper

    fig_out_of_distribution = plt.figure()
    plt.title(f'Out-of-Distribution amp [5 - 10]')
    final_loss = [losses[amp][-1, 1] for amp in amps]
    plt.plot(amps, final_loss)
    # plt.ylim(0, 18)  # from Chelsea's paper
    return fig_adaptation, fig_out_of_distribution


if __name__ == "__main__":
    import os

    # launch_training()
    logger.configure(log_directory=Args.log_dir, prefix=Args.log_prefix)
    # logger.log_params(Args=vars(Args))
    root_dir = "/Users/ge/machine_learning/berkeley-playground/ins-runs/2018-06-09/maml-baselines"
    # data_paths = {"mlp": "sinusoid-maml-mlp/debug/modules/28000_FunctionalMLP.pkl",
    #               "rnn": "sinusoid-maml-rnn/debug/modules/28000_FunctionalRNN.pkl"}
    data_paths = {"rnn": "sinusoid-maml-rnn/debug/modules/28000_FunctionalRNN.pkl"}
    for key, path in data_paths.items():
        Args.weight_path = os.path.join(root_dir, path)
        from playground.maml.maml_torch.maml_multi_step import FunctionalMLP, FunctionalRNN

        if key == 'mlp':
            model = FunctionalMLP(1, 1)
        elif key == 'rnn':
            model = FunctionalRNN(1, 1, 10)
            
        fig_adaptation, fig_out = adapt_and_test(model)
        logger.log_pyplot(None, key=f"{key}-adaptation.png", fig=fig_adaptation)
        logger.log_pyplot(None, key=f"{key}-out-of-distribution.png", fig=fig_out)
