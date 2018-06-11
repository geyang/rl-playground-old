import dill
from collections import deque
from params_proto import cli_parse, Proto
from tqdm import tqdm
from waterbear import DefaultBear


@cli_parse
class Args:
    run_dir = Proto(None, help="path to the data.pkl file saved by the maml_torch.py")
    test_n_steps = [1, 5]


def load(path='data.pkl'):
    with open(path, 'rb') as f:
        while True:
            try:
                yield dill.load(f)
            except EOFError:
                break


import numpy as np


class Reducer:
    def __init__(self):
        self.cache = []

    def append(self, x):
        self.cache.append(x)

    def mean(self):
        return sum(self.cache) / float(len(self.cache))


class TimeSeries:
    def __init__(self, data_constructor=list):
        self.x = []
        self.y = data_constructor()

    def append(self, step, datum):
        self.x.append(step)
        self.y.append(datum)


class RollingAverage:
    def __init__(self, window=400):
        self.cache = deque(maxlen=window)
        self.data = []

    def append(self, x):
        self.cache.append(x)
        self.data.append(self.mean())

    def mean(self):
        return sum(self.cache) / float(len(self.cache))


def smooth(y, radius=40):
    import numpy as np
    if len(y) < 2 * radius + 1:
        # return np.ones_like(y) * y.mean()
        return y
    kernel = np.ones(2 * radius + 1)
    out = np.zeros_like(y)
    out[:] = np.nan
    out[radius:-radius] = np.convolve(y, kernel, mode='valid') / \
                          np.convolve(np.ones_like(y), kernel, mode='valid')
    return out


def plot_learning_curve():
    amps = np.arange(5.0, 10.5, 0.5)
    phi0s = np.arange(1.0, 2.25, 0.25) * np.pi

    start = 0
    xs = []
    keys = set()
    data = DefaultBear(TimeSeries)
    for i, entry in enumerate(tqdm(load(Args.run_dir))):
        if i < start:
            continue
        _step = entry['_step']
        _timestamp = entry['_timestamp']
        for k, v in entry.items():
            keys.add(k)
            if k in ['_step', '_timestamp']:
                continue
            data[k].append(_step, v)

    print(f"data includes the following keys:\n{keys}")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 4))
    plt.title('Sinusoidal')
    step = 1
    for g in [step]:
        ys = np.array([data[f'{_}-grad-{g}-loss'].y for _ in range(10)])
        plt.plot(data[f'0-grad-{g}-loss'].x, smooth(ys.mean(axis=0)), label='train')
    for g in [step]:
        ys = np.array([data[f'{_}-grad-{g}-test-loss'].y for _ in range(10)])
        plt.plot(data[f'0-grad-{g}-test-loss'].x, smooth(ys.mean(axis=0)), label='test')
    plt.ylim(0, 5)
    plt.legend()
    plt.show()
    return fig


if __name__ == "__main__":
    import os
    from ml_logger import logger

    root_dir = "/Users/ge/machine_learning/berkeley-playground/ins-runs/2018-06-10"
    paths = "debug/maml-baselines-fixed/sinusoid-maml-mlp/data.pkl".split(' ')
    logger.configure(log_directory=root_dir, prefix='analysis')

    # for k, path in data_paths.items():
    for path in paths:
        Args.run_dir = os.path.join(root_dir, path)
        dirname = os.path.dirname(Args.run_dir)
        fig = plot_learning_curve()
        # logger.log_pyplot(0, fig, key=f'{dirname}/learning_curve.png', namespace='')
