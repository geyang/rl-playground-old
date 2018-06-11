import copy
from collections import defaultdict
import numpy as np
import sys
from tqdm import trange, tnrange
import torch as t

if 'IPython' in sys.modules:
    trange = tnrange

import torch_helpers as h


def _sgd(model, xs, ys, lr):
    _ys = model(t.tensor(xs).unsqueeze(dim=-1))
    loss = model.criteria(_ys, t.tensor(ys).unsqueeze(dim=-1))
    model.zero_grad()
    loss.backward(t.tensor([1.]))
    model.step(lr)
    return loss, _ys


LINE_STYLES = ['-', '--', '-.', ':']


def comp_ys(xs, model):
    # convert to torch.float32 (single precision) to work with weight matrices.
    return model(xs.float().unsqueeze(dim=-1))


def comp_loss(xs, ys, model):
    if hasattr(model, 'is_recurrent') and model.is_recurrent:
        raise NotImplementedError('recurrent unit has not been implemented -- Ge')
    else:
        _ys = comp_ys(xs, model)
        # convert to torch.float32 (single precision) to work with weight matrices.
        loss = model.criteria(_ys, ys.float().unsqueeze(dim=-1))
        return loss, _ys


def fine_tuning_statistics(Problem, model, runs, k=5, n_steps=10, k_eval=100):
    log = defaultdict(list)
    for run_ind in trange(runs):
        model = copy.deepcopy(model)
        problem = Problem()

        proper = problem.proper(k_eval)
        xs, ys = problem.samples(k)
        losses = []
        for ft_ind in range(n_steps + 1):
            loss, _ys = comp_loss(*proper, model, volatile=True)
            losses.append(loss.data.numpy())
            if ft_ind < n_steps:
                _sgd(model, xs, ys, lr=0.01)
        log['loss'].append(losses)
    return log


def smooth2(y, radius=20):
    if len(y) < 2 * radius + 1:
        return np.ones_like(y) * y.mean()
    kernel = np.ones(2 * radius + 1)
    out = np.zeros_like(y)
    out[:] = np.nan
    out[radius:-radius] = np.convolve(y, kernel, mode='valid') / \
                          np.convolve(np.ones_like(y), kernel, mode='valid')
    return out


def smooth(x, window_length=41, polyorder=3):
    try:
        return signal.savgol_filter(x, window_length=window_length, polyorder=polyorder)
    except Exception as e:
        return x


def monit_loss(cache, clear=True, label='loss', color='grey', window=True):
    import matplotlib.pyplot as plt
    if clear:
        plt.cla()
    loss = np.array(cache)
    loss = loss.squeeze(axis=-1)
    plt.plot(smooth2(loss), '-', color=color, alpha=0.6, label=label)
    if window:
        plt.plot(loss, edgecolor='pink', facecolor='red', alpha=0.10, label=label + 'min/max')
    plt.ylim(0, 4)
    plt.ylabel('Mean-Squared Loss')


from scipy import signal


def monit_batch_loss(cache, clear=True, label='loss', color='red', window=True):
    import matplotlib.pyplot as plt
    if clear:
        plt.cla()
    plt.title('loss over time')

    loss = np.array(cache)
    try:
        loss = loss.squeeze(axis=-1)
    except:
        return
    plt.plot(smooth2(loss.mean(axis=-1)), '-', color=color, alpha=0.6, label=label)
    if window:
        ln, *_ = loss.shape
        plt.fill_between(range(ln), loss.min(axis=-1), loss.max(axis=-1), edgecolor='pink',
                         facecolor='red', alpha=0.10, label=label + ' min/max')
    plt.ylim(0, 4)
    plt.ylabel('Mean-Squared Loss')


def plot_fit(title, problem, model, k_shot=5, n_steps=10):
    import matplotlib.pyplot as plt
    model = copy.deepcopy(model)
    plt.cla()
    plt.title(title)
    proper = problem.proper()
    plt.plot(*proper, color='red', label='ground truth')

    ys = comp_ys(proper[0], model)
    plt.plot(proper[0], ys.squeeze(dim=-1).data.numpy(), ':', label='pre-update', color='green', alpha=-.2)

    samples = problem.samples(k_shot)
    plt.scatter(*samples, marker='^', label='k samples')

    log = defaultdict(list)
    for ft_ind in range(n_steps + 1):

        loss, _ys = comp_loss(*proper, model, volatile=True)
        log['loss'].append(loss.data.numpy())

        if ft_ind == 1 or ft_ind == n_steps:
            label = '{} grad step'.format(ft_ind) + ('' if ft_ind == 1 else "s")
            line_style = '-.' if ft_ind == 1 else "--"
            plt.plot(proper[0], _ys.squeeze(dim=-1).data.numpy(), ls=line_style, label=label)

        if ft_ind < n_steps:
            _sgd(model, *samples, lr=0.01)

    o = lambda a: [a[i] for i in [4, 1, 3, 0, 2]]
    handles, labels = plt.gca().get_legend_handles_labels()  # reverse legend order
    plt.legend(o(handles), o(labels), loc='upper center', ncol=3, framealpha=0, edgecolor='none')
    plt.ylim(-5.5, 8.5)


def plot_loss_vs_fine_tuning(fig, title, problem, model, k=5, n_steps=10):
    import matplotlib.pyplot as plt
    model = copy.deepcopy(model)
    ax = fig.add_subplot(121)
    plt.cla()
    plt.title(title.format(k=k))
    proper = problem.proper()
    plt.plot(*proper, color='red', label='ground truth')

    ys = comp_ys(proper[0], model)
    plt.plot(proper[0], ys.squeeze(dim=-1).data.numpy(), ':', label='pre-update', color='green', alpha=-.2)

    samples = problem.samples(k)
    plt.scatter(*samples, marker='^', label='samples')

    log = defaultdict(list)
    for ft_ind in range(n_steps + 1):

        loss, _ys = comp_loss(*proper, model, volatile=True)
        log['loss'].append(loss.data.numpy())

        if ft_ind == 1 or ft_ind == n_steps:
            label = '{} grad step'.format(ft_ind) + ('' if ft_ind == 1 else "s")
            line_style = '-.' if ft_ind == 1 else "--"
            plt.plot(proper[0], _ys.squeeze(dim=-1).data.numpy(), ls=line_style, label=label)

        if ft_ind < n_steps:
            _sgd(model, *samples, lr=0.01)

    o = lambda a: [a[i] for i in [4, 1, 3, 0, 2]]
    handles, labels = ax.get_legend_handles_labels()  # reverse legend order
    plt.legend(o(handles), o(labels), loc='upper center', ncol=3, framealpha=0, edgecolor='none')
    plt.ylim(-5.5, 8.5)

    fig.add_subplot(122)
    plt.title('Fine-tuned Losses')
    plt.cla()
    plt.plot(np.array(log['loss']).squeeze(axis=-1), label="MAML")
    plt.plot([0] * (n_steps + 1), 's-', label="oracle", color="red")
    # todo: add pre-trained loss
    # plt.plot(, 'o-', label="pre-trained, step", color="blue")
    plt.xlabel('Fine-tune Steps')
    plt.ylabel('Mean-Squared Loss')
    plt.legend(loc='upper right', ncol=2, framealpha=1, edgecolor='none')
    plt.ylim(-0.25, 3)

    plt.tight_layout()
    plt.show()

    return log


def _loss(param, model):
    pass
